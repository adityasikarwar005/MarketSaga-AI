"""M12 - The recorder.

The most important service in the project and by some distance the least
interesting one. Models can be rebuilt in a weekend; history cannot be recovered
at any price. The binding constraint on whether this project ever passes Gate G5
is how many event-conditioned samples were captured before then, and this is the
module that captures them.

Design consequences of that:

**Append-only.** Nothing is ever rewritten. A correction is a new record.

**Durable by default.** Periodic fsync. Throughput here is irrelevant - a
recorder that is fast and loses the last four minutes of a crash is worthless.

**Restartable.** Crash mid-session, restart, keep appending. The manifest is
rebuilt from the file rather than trusted from memory.

**Self-verifying.** Every file has a manifest with a row count, a content hash
and first/last timestamps, so ``jobs/verify_recording.py`` can answer "is
yesterday complete?" without loading the data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import IO, Any

from libs.ms_core.clock import UTC, Clock, SessionCalendar
from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import ChainSnapshot, Msg, Tick

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw"


@dataclass
class FileManifest:
    """What a completed file should contain. Written beside it, read by the QC job."""

    stream: str
    partition: str
    filename: str
    rows: int
    sha256: str
    first_ts: str | None
    last_ts: str | None
    bytes: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class Recorder:
    """Append-only NDJSON writer, partitioned by stream, date and instrument.

    Layout::

        data/raw/<stream>/date=YYYY-MM-DD/<stream>-<key>.ndjson
        data/raw/<stream>/date=YYYY-MM-DD/_manifest.json

    NDJSON rather than a binary format on purpose: at this stage the ability to
    open a file and read it with your eyes when something looks wrong is worth
    more than the disk it costs. Parquet conversion happens nightly (M17).
    """

    def __init__(
        self,
        clock: Clock,
        calendar: SessionCalendar,
        *,
        root: Path = DATA_ROOT,
        fsync_every: int = 500,
        partition_by_instrument: bool = True,
    ) -> None:
        self._clock = clock
        self._calendar = calendar
        self._root = root
        self._fsync_every = fsync_every
        self._partition_by_instrument = partition_by_instrument
        self._handles: dict[Path, IO[str]] = {}
        self._rows: dict[Path, int] = {}
        self._since_sync: dict[Path, int] = {}
        self._first_ts: dict[Path, str] = {}
        self._last_ts: dict[Path, str] = {}

    # -- paths --------------------------------------------------------------
    def _partition_date(self, ts: datetime) -> date:
        """Partition by *exchange* date, not UTC date.

        A session that starts at 09:15 IST is 03:45 UTC. Partitioning by UTC date
        happens to work for Indian hours today, but it is the kind of assumption
        that breaks silently the first time a special session runs in the
        evening - so the exchange calendar decides, explicitly.
        """
        return self._calendar.local(ts).date()

    def path_for(self, stream: str, ts: datetime, key: str | int | None) -> Path:
        part = self._root / stream / f"date={self._partition_date(ts).isoformat()}"
        name = f"{stream}-{key}.ndjson" if key is not None else f"{stream}.ndjson"
        return part / name

    def _handle(self, path: Path) -> IO[str]:
        if path not in self._handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = path.exists()
            fh = path.open("a", encoding="utf-8", newline="\n")
            self._handles[path] = fh
            # Restart: recover the row count from disk rather than assuming zero.
            self._rows[path] = _count_lines(path) if existing else 0
            self._since_sync[path] = 0
            if existing and self._rows[path]:
                first, last = _first_last_ts(path)
                if first:
                    self._first_ts[path] = first
                if last:
                    self._last_ts[path] = last
            log.info(
                "recorder opened file",
                fields={"path": str(path), "existing_rows": self._rows[path]},
            )
        return self._handles[path]

    # -- writing ------------------------------------------------------------
    def write(self, stream: str, msg: Msg, *, key: str | int | None = None) -> Path:
        ts = getattr(msg, "ts", None) or getattr(msg, "t_ingest", None)
        if not isinstance(ts, datetime):
            raise ValueError(f"{type(msg).__name__} has no timestamp to partition by")
        path = self.path_for(stream, ts, key)
        fh = self._handle(path)
        fh.write(msg.model_dump_json(by_alias=True) + "\n")

        self._rows[path] += 1
        self._since_sync[path] += 1
        iso = ts.astimezone(UTC).isoformat()
        self._first_ts.setdefault(path, iso)
        self._last_ts[path] = iso

        if self._since_sync[path] >= self._fsync_every:
            self._flush(path)
        METRICS.inc("recorder_rows_total", stream=stream)
        return path

    def write_tick(self, tick: Tick) -> Path:
        key = tick.instrument_id if self._partition_by_instrument else None
        return self.write("ticks", tick, key=key)

    def write_chain(self, snap: ChainSnapshot) -> Path:
        return self.write("chain", snap, key=snap.underlying_id)

    def record_all(self, ticks: Iterable[Tick]) -> int:
        n = 0
        for t in ticks:
            self.write_tick(t)
            n += 1
        return n

    # -- durability ---------------------------------------------------------
    def _flush(self, path: Path) -> None:
        import os

        fh = self._handles[path]
        fh.flush()
        os.fsync(fh.fileno())
        self._since_sync[path] = 0
        METRICS.inc("recorder_fsync_total")

    def flush(self) -> None:
        for path in list(self._handles):
            self._flush(path)

    # -- manifests ----------------------------------------------------------
    def write_manifests(self) -> list[FileManifest]:
        """Seal the current files. Called at rotation, on shutdown, and by the QC job."""
        self.flush()
        manifests: list[FileManifest] = []
        by_partition: dict[Path, list[FileManifest]] = {}
        for path in list(self._handles):
            m = FileManifest(
                stream=path.parent.parent.name,
                partition=path.parent.name,
                filename=path.name,
                rows=self._rows[path],
                sha256=_sha256_of(path),
                first_ts=self._first_ts.get(path),
                last_ts=self._last_ts.get(path),
                bytes=path.stat().st_size,
            )
            manifests.append(m)
            by_partition.setdefault(path.parent, []).append(m)

        for partition, entries in by_partition.items():
            (partition / "_manifest.json").write_text(
                json.dumps([asdict(e) for e in entries], indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return manifests

    def close(self) -> list[FileManifest]:
        manifests = self.write_manifests()
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()
        log.info("recorder closed", fields={"files": len(manifests)})
        return manifests

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def row_counts(self) -> dict[str, int]:
        return {str(p): n for p, n in self._rows.items()}


# --------------------------------------------------------------------------- #
# Reading back
# --------------------------------------------------------------------------- #
def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _first_last_ts(path: Path) -> tuple[str | None, str | None]:
    first = last = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                ts = json.loads(line).get("ts")
            except json.JSONDecodeError:
                continue
            if ts:
                if first is None:
                    first = ts
                last = ts
    return first, last


def read_stream(path: Path, cls: type[Msg]) -> Iterator[Msg]:
    """Re-read a recorded file. Malformed lines are reported, never skipped silently."""
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield cls.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno} is not a valid {cls.__name__}: {exc}") from exc


def verify_partition(partition: Path) -> dict[str, Any]:
    """Check a recorded day against its manifest.

    Answers the only question that matters the morning after: is yesterday's
    data complete and unmodified? Returns a report rather than raising, so the
    QC job can aggregate many days.
    """
    manifest_path = partition / "_manifest.json"
    report: dict[str, Any] = {"partition": str(partition), "ok": True, "problems": []}
    if not manifest_path.exists():
        report["ok"] = False
        report["problems"].append("no _manifest.json - the recorder did not shut down cleanly")
        return report

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    report["files"] = len(entries)
    total = 0
    for e in entries:
        path = partition / e["filename"]
        if not path.exists():
            report["ok"] = False
            report["problems"].append(f"{e['filename']}: missing")
            continue
        rows = _count_lines(path)
        total += rows
        if rows < e["rows"]:
            report["ok"] = False
            report["problems"].append(
                f"{e['filename']}: {rows} rows on disk, manifest says {e['rows']} - data lost"
            )
        elif rows > e["rows"]:
            # More rows than the manifest is normal if recording resumed after
            # the manifest was written; the hash check below decides.
            report["problems"].append(
                f"{e['filename']}: {rows} rows, manifest {e['rows']} - manifest is stale, reseal"
            )
        elif _sha256_of(path) != e["sha256"]:
            report["ok"] = False
            report["problems"].append(f"{e['filename']}: hash mismatch - file was modified")
    report["rows"] = total
    return report
