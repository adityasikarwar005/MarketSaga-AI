"""M17 - Parquet archive and DuckDB research layer.

The recorder writes NDJSON because it must be durable, append-only and readable
by eye when something looks wrong at 09:16. Research needs the opposite: wide
columnar scans over months. So the archive converts, nightly, and the two
formats coexist rather than compete.

Three properties the research layer depends on:

**Idempotent.** Archiving a day twice produces byte-identical Parquet. That is
what makes a dataset version meaningful, and it is asserted in CI.

**Versioned.** A dataset version is a manifest of file hashes. A model records
the version it trained on, so "which data produced this model" has an exact
answer months later rather than an approximate one.

**Point-in-time by construction.** Every read goes through a function that takes
an ``as_of`` and filters on it. There is no code path that returns rows from the
future, because the leakage that follows from one is invisible until live
trading disagrees with the backtest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import Candle, ChainSnapshot, Tick
from libs.ms_data.recorder import DATA_ROOT, read_stream

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_ROOT = REPO_ROOT / "data" / "parquet"
DATASET_DIR = REPO_ROOT / "data" / "datasets"


class MissingDependency(ImportError):
    """Raised when the infra extra is not installed."""


def _require(module: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency(
            f"{module} is needed for the research layer: pip install -e '.[infra]'"
        ) from exc


# --------------------------------------------------------------------------- #
# Flattening
# --------------------------------------------------------------------------- #
TICK_COLUMNS = (
    "ts", "instrument_id", "ltp", "qty", "bid", "ask",
    "bid_qty", "ask_qty", "oi", "seq", "src", "trace_id",
)  # fmt: skip

CANDLE_COLUMNS = (
    "ts", "instrument_id", "tf", "o", "h", "l", "c", "v",
    "oi", "trades", "vwap", "synthetic", "version", "trace_id",
)  # fmt: skip

CHAIN_COLUMNS = (
    "ts", "underlying_id", "spot", "completeness", "expiry", "strike",
    "opt_type", "ltp", "bid", "ask", "volume", "oi", "oi_change", "iv", "trace_id",
)  # fmt: skip


def tick_rows(ticks: Iterable[Tick]) -> list[dict[str, Any]]:
    return [
        {
            "ts": t.ts,
            "instrument_id": t.instrument_id,
            "ltp": t.ltp,
            "qty": t.qty,
            "bid": t.bid,
            "ask": t.ask,
            "bid_qty": t.bid_qty,
            "ask_qty": t.ask_qty,
            "oi": t.oi,
            "seq": t.seq,
            "src": t.src,
            "trace_id": t.trace_id,
        }
        for t in ticks
    ]


def candle_rows(candles: Iterable[Candle]) -> list[dict[str, Any]]:
    return [
        {
            "ts": c.ts,
            "instrument_id": c.instrument_id,
            "tf": c.tf,
            "o": c.o,
            "h": c.h,
            "l": c.low,
            "c": c.c,
            "v": c.v,
            "oi": c.oi,
            "trades": c.trades,
            "vwap": c.vwap,
            "synthetic": c.synthetic,
            "version": c.version,
            "trace_id": c.trace_id,
        }
        for c in candles
    ]


def chain_rows(snapshots: Iterable[ChainSnapshot]) -> list[dict[str, Any]]:
    """One row per quote, snapshot fields denormalised onto each.

    Denormalised on purpose: columnar storage compresses the repeated snapshot
    fields to almost nothing, and every research query wants quotes as a flat
    table rather than a nested structure it has to explode first.
    """
    out: list[dict[str, Any]] = []
    for snap in snapshots:
        for q in snap.quotes:
            out.append(
                {
                    "ts": snap.ts,
                    "underlying_id": snap.underlying_id,
                    "spot": snap.spot,
                    "completeness": snap.completeness,
                    "expiry": q.expiry,
                    "strike": q.strike,
                    "opt_type": q.opt_type.value,
                    "ltp": q.ltp,
                    "bid": q.bid,
                    "ask": q.ask,
                    "volume": q.volume,
                    "oi": q.oi,
                    "oi_change": q.oi_change,
                    "iv": q.iv,
                    "trace_id": snap.trace_id,
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Archiving
# --------------------------------------------------------------------------- #
@dataclass
class ArchiveResult:
    stream: str
    day: date
    files: list[Path] = field(default_factory=list)
    rows: int = 0
    source_files: int = 0

    @property
    def ok(self) -> bool:
        return self.rows > 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _write_parquet(rows: list[dict[str, Any]], columns: Sequence[str], out: Path) -> Path:
    pa = _require("pyarrow")
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])

    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {col: [r.get(col) for r in rows] for col in columns},
    )
    # Determinism matters (the dataset version is a content hash), and pyarrow is
    # deterministic given identical input - including its dictionary encoding, so
    # there is no need to disable it. Disabling it forces PLAIN encoding, which
    # DuckDB cannot skip over during filter pushdown ("PlainSkip not
    # implemented"), and every chain query then fails. The idempotency test in
    # tests/unit/test_archive.py is what keeps this honest.
    pq.write_table(
        table,
        out,
        compression="zstd",
        compression_level=3,
        write_statistics=True,
        store_schema=True,
    )
    return out


def archive_day(
    stream: str,
    day: date,
    *,
    raw_root: Path = DATA_ROOT,
    parquet_root: Path = PARQUET_ROOT,
) -> ArchiveResult:
    """Convert one day of one stream from NDJSON to Parquet.

    Idempotent: existing output is overwritten with identical bytes, so a failed
    or repeated run is harmless.
    """
    # Validate up front. Discovering an unknown stream halfway through a loop
    # means a typo in a cron entry silently archives nothing on the days a
    # partition happens to be missing, and raises on the days it is not.
    if stream not in ("ticks", "candles", "chain"):
        raise ValueError(f"unknown stream {stream!r}")

    partition = raw_root / stream / f"date={day.isoformat()}"
    result = ArchiveResult(stream=stream, day=day)
    if not partition.exists():
        return result

    sources = sorted(partition.glob("*.ndjson"))
    result.source_files = len(sources)

    columns: Sequence[str]
    for src in sources:
        if stream == "ticks":
            rows = tick_rows(read_stream(src, Tick))
            columns = TICK_COLUMNS
            key = src.stem.split("-", 1)[-1]
            out = parquet_root / stream / f"date={day.isoformat()}" / f"instrument={key}.parquet"
        elif stream == "candles":
            rows = candle_rows(read_stream(src, Candle))
            columns = CANDLE_COLUMNS
            key = src.stem.split("-", 1)[-1]
            out = parquet_root / stream / f"date={day.isoformat()}" / f"instrument={key}.parquet"
        else:  # chain
            rows = chain_rows(read_stream(src, ChainSnapshot))
            columns = CHAIN_COLUMNS
            key = src.stem.split("-", 1)[-1]
            out = parquet_root / stream / f"date={day.isoformat()}" / f"underlying={key}.parquet"
        if not rows:
            continue
        _write_parquet(rows, columns, out)
        result.files.append(out)
        result.rows += len(rows)

    METRICS.inc("archive_rows_total", value=float(result.rows), stream=stream)
    log.info(
        "archived",
        fields={
            "stream": stream,
            "day": day.isoformat(),
            "rows": result.rows,
            "files": len(result.files),
        },
    )
    return result


def archive_all(
    day: date,
    *,
    streams: Sequence[str] = ("ticks", "candles", "chain"),
    raw_root: Path = DATA_ROOT,
    parquet_root: Path = PARQUET_ROOT,
) -> list[ArchiveResult]:
    return [archive_day(s, day, raw_root=raw_root, parquet_root=parquet_root) for s in streams]


def available_days(stream: str, *, raw_root: Path = DATA_ROOT) -> list[date]:
    root = raw_root / stream
    if not root.exists():
        return []
    out = []
    for part in root.glob("date=*"):
        try:
            out.append(date.fromisoformat(part.name.removeprefix("date=")))
        except ValueError:
            continue
    return sorted(out)


# --------------------------------------------------------------------------- #
# Dataset versions
# --------------------------------------------------------------------------- #
@dataclass
class DatasetVersion:
    """An exact, checkable description of what data a model saw.

    Six months after a model is trained, "which data was this fitted on?" must
    have an exact answer. A date range is not one - files get re-archived,
    backfilled and corrected. A manifest of content hashes is.
    """

    name: str
    created_at: str
    streams: list[str]
    start: str
    end: str
    files: dict[str, str] = field(default_factory=dict)  # relative path -> sha256

    @property
    def digest(self) -> str:
        blob = json.dumps(self.files, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["digest"] = self.digest
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetVersion:
        return cls(
            name=d["name"],
            created_at=d["created_at"],
            streams=list(d["streams"]),
            start=d["start"],
            end=d["end"],
            files=dict(d["files"]),
        )


def build_dataset_version(
    name: str,
    start: date,
    end: date,
    now: datetime,
    *,
    streams: Sequence[str] = ("candles", "chain"),
    parquet_root: Path = PARQUET_ROOT,
    dataset_dir: Path = DATASET_DIR,
) -> DatasetVersion:
    files: dict[str, str] = {}
    for stream in streams:
        root = parquet_root / stream
        if not root.exists():
            continue
        for part in sorted(root.glob("date=*")):
            try:
                day = date.fromisoformat(part.name.removeprefix("date="))
            except ValueError:
                continue
            if not (start <= day <= end):
                continue
            for f in sorted(part.glob("*.parquet")):
                files[str(f.relative_to(parquet_root)).replace("\\", "/")] = _sha256(f)

    version = DatasetVersion(
        name=name,
        created_at=now.isoformat(),
        streams=list(streams),
        start=start.isoformat(),
        end=end.isoformat(),
        files=files,
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / f"{name}.json").write_text(
        json.dumps(version.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return version


def verify_dataset_version(
    name: str, *, parquet_root: Path = PARQUET_ROOT, dataset_dir: Path = DATASET_DIR
) -> tuple[bool, list[str]]:
    """Confirm the data on disk still matches what a model was trained on."""
    path = dataset_dir / f"{name}.json"
    if not path.exists():
        return False, [f"no dataset version named {name!r}"]
    version = DatasetVersion.from_dict(json.loads(path.read_text(encoding="utf-8")))
    problems: list[str] = []
    for rel, expected in version.files.items():
        f = parquet_root / rel
        if not f.exists():
            problems.append(f"{rel}: missing")
        elif _sha256(f) != expected:
            problems.append(f"{rel}: content changed since the version was created")
    return not problems, problems


# --------------------------------------------------------------------------- #
# Research layer
# --------------------------------------------------------------------------- #
class Research:
    """DuckDB over Parquet. Analytical scans, not point lookups.

    Every method takes an explicit time window and, where relevant, an ``as_of``.
    There is deliberately no ``read_everything()``: a research API that makes it
    easy to load the whole archive makes it equally easy to leak the future into
    a feature.
    """

    def __init__(self, parquet_root: Path = PARQUET_ROOT) -> None:
        self._root = parquet_root
        self._duckdb = _require("duckdb")
        self._con = self._duckdb.connect(":memory:")

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> Research:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _glob(self, stream: str) -> str:
        return str(self._root / stream / "date=*" / "*.parquet").replace("\\", "/")

    def _has(self, stream: str) -> bool:
        root = self._root / stream
        return root.exists() and any(root.rglob("*.parquet"))

    def sql(self, query: str) -> pd.DataFrame:
        """Escape hatch for ad-hoc analysis. Prefer the typed helpers."""
        return self._con.execute(query).df()

    def candles(
        self,
        start: datetime,
        end: datetime,
        *,
        instruments: Sequence[int] | None = None,
        tf: int = 1,
        include_synthetic: bool = True,
    ) -> pd.DataFrame:
        """Candles in ``[start, end)``.

        Right-open, matching the candle bucket convention exactly - a subtly
        different window here would misalign every label built on top of it.
        """
        if not self._has("candles"):
            return self._empty(CANDLE_COLUMNS)
        where = [
            f"ts >= TIMESTAMPTZ '{start.isoformat()}'",
            f"ts < TIMESTAMPTZ '{end.isoformat()}'",
            f"tf = {int(tf)}",
        ]
        if instruments:
            where.append(f"instrument_id IN ({','.join(str(int(i)) for i in instruments)})")
        if not include_synthetic:
            where.append("synthetic = FALSE")
        return self.sql(
            f"SELECT * FROM read_parquet('{self._glob('candles')}') "
            f"WHERE {' AND '.join(where)} ORDER BY ts, instrument_id"
        )

    def ticks(
        self, start: datetime, end: datetime, *, instruments: Sequence[int] | None = None
    ) -> pd.DataFrame:
        if not self._has("ticks"):
            return self._empty(TICK_COLUMNS)
        where = [
            f"ts >= TIMESTAMPTZ '{start.isoformat()}'",
            f"ts < TIMESTAMPTZ '{end.isoformat()}'",
        ]
        if instruments:
            where.append(f"instrument_id IN ({','.join(str(int(i)) for i in instruments)})")
        return self.sql(
            f"SELECT * FROM read_parquet('{self._glob('ticks')}') "
            f"WHERE {' AND '.join(where)} ORDER BY ts, instrument_id"
        )

    def chain_as_of(
        self, as_of: datetime, underlying_id: int, *, max_age_seconds: float = 30.0
    ) -> pd.DataFrame:
        """The most recent chain snapshot strictly before ``as_of``.

        The point-in-time primitive for every options feature. Anything that
        reaches forward - even by one snapshot - is look-ahead bias, and it will
        look like a working model right up until real money is involved.

        Right-open, matching the feature layer exactly. Research and serving
        disagreeing about the boundary is the same skew by a different route.
        """
        if not self._has("chain"):
            return self._empty(CHAIN_COLUMNS)
        return self.sql(f"""
            WITH candidates AS (
                SELECT * FROM read_parquet('{self._glob("chain")}')
                WHERE underlying_id = {int(underlying_id)}
                  AND ts < TIMESTAMPTZ '{as_of.isoformat()}'
                  AND ts >= TIMESTAMPTZ '{as_of.isoformat()}' - INTERVAL '{int(max_age_seconds)} seconds'
            )
            SELECT * FROM candidates
            WHERE ts = (SELECT max(ts) FROM candidates)
            ORDER BY expiry, strike, opt_type
            """)

    def sessions(self, stream: str = "candles") -> list[date]:
        if not self._has(stream):
            return []
        df = self.sql(
            f"SELECT DISTINCT CAST(ts AS DATE) AS d FROM read_parquet('{self._glob(stream)}') ORDER BY d"
        )
        return [r.date() if hasattr(r, "date") else r for r in df["d"].tolist()]

    def coverage(self) -> pd.DataFrame:
        """Rows and sessions per stream. The first thing to look at each morning."""
        rows = []
        for stream in ("ticks", "candles", "chain"):
            if not self._has(stream):
                rows.append(
                    {"stream": stream, "rows": 0, "sessions": 0, "first": None, "last": None}
                )
                continue
            df = self.sql(f"""SELECT count(*) AS rows,
                           count(DISTINCT CAST(ts AS DATE)) AS sessions,
                           min(ts) AS first, max(ts) AS last
                    FROM read_parquet('{self._glob(stream)}')""")
            rec = df.iloc[0].to_dict()
            rec["stream"] = stream
            rows.append(rec)
        pd_mod = _require("pandas")
        return pd_mod.DataFrame(rows)[["stream", "rows", "sessions", "first", "last"]]

    def _empty(self, columns: Sequence[str]) -> pd.DataFrame:
        pd_mod = _require("pandas")
        return pd_mod.DataFrame({c: [] for c in columns})
