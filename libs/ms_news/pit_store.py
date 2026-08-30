"""M25 - the point-in-time event store.

The contract is one line: ``store.as_of(t)`` returns everything knowable at
``t``. Nothing more, nothing less. Every backtest that touches news reads
through it, so the ways it can be wrong are the ways a strategy can be
imaginary.

``as_of`` gates on availability, never on the publish time
------------------------------------------------------------

The obvious implementation filters ``t_event <= t`` and is the bug this module
exists to prevent. An item published at 11:00:00 that this system did not finish
processing until 11:00:04 was **not knowable at 11:00:01**, and a backtest that
acts on it at 11:00:01 is trading on information it did not have. Four seconds
is not a rounding error at an event horizon measured in seconds - it is the
entire edge, and it shows up in results as a strategy that is astonishingly good
at the open.

So the filter is :meth:`Timed.known_at`, which reads ``t_available`` and falls
back to ``t_ingest``. ``t_event`` is what the store sorts and reports by, and
never what it gates on.

Backfill lives in its own partition, and is excluded by default
-----------------------------------------------------------------

An item collected live has a ``t_available`` that was **measured**: the pipeline
really did finish with it at that moment. An item fetched from an archive six
months later has a ``t_available`` that was **reconstructed**, and however
carefully it was reconstructed it is a claim about a moment this system was not
present for.

Mixing the two gives a corpus where the same field means two different things,
and the difference is invisible at query time. So they are separate partitions,
``as_of`` reads live only unless asked otherwise, and the result says which
partitions produced it. Wanting backfill is legitimate; getting it by accident
is not.

Append-only, and corrections are new rows
-------------------------------------------

Same rule as the paper ledger. A publisher revising a story does not mutate the
stored item - a superseding row is appended and the earlier one stays. A store
that edits in place cannot answer ``as_of`` for any time before the edit, which
is the one question it exists to answer.

Deliberately not a vector index yet
-------------------------------------

The plan lists pgvector or FAISS on embeddings. There are no embeddings until
M27 exists, and an index over a field nothing populates is a dependency to
install and a schema to migrate in exchange for nothing. The query surface below
is by class, entity, window and ``as_of`` - the four the plan names - and the
embedding index is added when there is something to index. This is the same
reasoning as `ms_ops/runner.py` and `ms_ops/registry.py`: the part with
correctness requirements is testable without standing up a service.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from libs.ms_core.obs import METRICS, get_logger
from libs.ms_core.schemas import RawNewsItem

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIT_ROOT = REPO_ROOT / "data" / "news"


class Partition(StrEnum):
    """Where an item's ``t_available`` came from.

    Not a storage detail. It is the difference between a measured availability
    and a reconstructed one, and a backtest that cannot tell them apart is
    reporting on a corpus it does not understand.
    """

    LIVE = "live"  # collected as it happened; t_available was observed
    BACKFILL = "backfill"  # fetched later; t_available was reconstructed


class PitStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredItem:
    """One row: the item, which partition it came from, and what supersedes it."""

    item: RawNewsItem
    partition: Partition
    supersedes: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "partition": self.partition.value,
                "supersedes": self.supersedes,
                "item": json.loads(self.item.model_dump_json()),
            },
            sort_keys=True,
        )

    @staticmethod
    def from_json(raw: str) -> StoredItem:
        d = json.loads(raw)
        return StoredItem(
            item=RawNewsItem.model_validate(d["item"]),
            partition=Partition(d["partition"]),
            supersedes=d.get("supersedes"),
        )


@dataclass(frozen=True)
class AsOfResult:
    """What a query returned, and what it was allowed to look at.

    ``partitions`` is carried on the result rather than left to the caller to
    remember. A number computed from live-only data and one computed with
    backfill mixed in are not comparable, and the only reliable moment to record
    which was which is the moment it is produced.
    """

    as_of: datetime
    partitions: tuple[Partition, ...]
    items: tuple[RawNewsItem, ...]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def digest(self) -> str:
        """Stable hash of this result, for reproducibility checks.

        Over ids in the order returned, so it catches both a changed membership
        and a changed ordering. Two runs of the same query on the same store
        must agree, or nothing built on top of it can be reproduced.
        """
        h = hashlib.sha256()
        h.update(self.as_of.isoformat().encode())
        h.update(b"|")
        h.update(",".join(p.value for p in self.partitions).encode())
        for item in self.items:
            h.update(b"|")
            h.update(item.id.encode())
        return h.hexdigest()[:16]

    def render(self) -> str:
        parts = "+".join(p.value for p in self.partitions)
        return (
            f"  as_of {self.as_of.isoformat()} [{parts}]: "
            f"{len(self.items):,} items, digest {self.digest}"
        )


class PitEventStore:
    """Append-only news store with a point-in-time read.

    One file per partition per day. Postgres is an index built from these when
    it exists; the files are the truth, for the same reason the paper ledger
    keeps its own.
    """

    def __init__(self, root: Path = PIT_ROOT) -> None:
        self.root = root

    def path(self, day: datetime, partition: Partition) -> Path:
        return self.root / partition.value / f"{day.date().isoformat()}.ndjson"

    # -- writing ------------------------------------------------------------
    def append(
        self,
        item: RawNewsItem,
        *,
        partition: Partition = Partition.LIVE,
        supersedes: str | None = None,
    ) -> Path:
        """Store one item. Never updates an existing row.

        Partitioned by ``t_ingest`` rather than ``t_event``: the file an item
        lands in should be the day this system saw it, so a late-arriving item
        about last week does not silently rewrite last week's file.
        """
        if item.t_available is not None and item.t_available < item.t_ingest:
            raise PitStoreError(
                f"t_available {item.t_available.isoformat()} precedes t_ingest "
                f"{item.t_ingest.isoformat()}. Availability before arrival is not "
                "a possible state, and storing it would make as_of() return the "
                "item before it existed."
            )
        p = self.path(item.t_ingest, partition)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(StoredItem(item=item, partition=partition, supersedes=supersedes).to_json())
            fh.write("\n")
        METRICS.inc("pit_items_appended", partition=partition.value)
        return p

    def append_many(
        self, items: Iterable[RawNewsItem], *, partition: Partition = Partition.LIVE
    ) -> int:
        written = 0
        for item in items:
            self.append(item, partition=partition)
            written += 1
        return written

    # -- reading ------------------------------------------------------------
    def _rows(self, partitions: Sequence[Partition]) -> list[StoredItem]:
        rows: list[StoredItem] = []
        for partition in partitions:
            directory = self.root / partition.value
            if not directory.exists():
                continue
            for f in sorted(directory.glob("*.ndjson")):
                for line in f.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(StoredItem.from_json(line))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # A truncated final line is what a killed writer leaves.
                        # Counted, so a file that is losing rows is visible rather
                        # than quietly shorter than it should be.
                        METRICS.inc("pit_rows_unreadable", partition=partition.value)
                        log.warning("unreadable pit row", fields={"file": str(f)})
        return rows

    def as_of(
        self,
        t: datetime,
        *,
        partitions: Sequence[Partition] = (Partition.LIVE,),
        since: datetime | None = None,
        source_id: int | None = None,
        lang: str | None = None,
    ) -> AsOfResult:
        """Everything knowable at ``t``. Nothing more, nothing less.

        The gate is :meth:`RawNewsItem.known_at`, which reads ``t_available``
        and falls back to ``t_ingest``. Filtering on ``t_event`` instead would
        admit items published before ``t`` that this system had not finished
        with - the exact lookahead this store exists to make impossible.

        Superseded rows are dropped: if a later row names an earlier one in
        ``supersedes`` **and that later row is itself knowable at t**, the
        earlier one is replaced. A correction that had not arrived yet cannot
        retroactively censor what was believed at the time.
        """
        rows = self._rows(tuple(partitions))
        visible = [r for r in rows if r.item.known_at(t)]
        superseded = {r.supersedes for r in visible if r.supersedes}

        items = [
            r.item
            for r in visible
            if r.item.id not in superseded
            and (since is None or r.item.t_event >= since)
            and (source_id is None or r.item.source_id == source_id)
            and (lang is None or r.item.lang == lang)
        ]
        # Sorted by event time then id: deterministic across runs and across
        # file order, which is what makes the digest reproducible.
        items.sort(key=lambda i: (i.t_event, i.id))
        METRICS.observe("pit_as_of_items", float(len(items)))
        return AsOfResult(as_of=t, partitions=tuple(partitions), items=tuple(items))

    def days(self, partition: Partition = Partition.LIVE) -> list[str]:
        directory = self.root / partition.value
        if not directory.exists():
            return []
        return sorted(p.stem for p in directory.glob("*.ndjson"))

    def counts(self) -> dict[str, int]:
        return {p.value: len(self._rows([p])) for p in Partition}
