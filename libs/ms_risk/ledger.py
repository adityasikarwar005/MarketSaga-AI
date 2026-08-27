"""M93 - the paper ledger and outcome resolver.

The audit spine. Every prediction the system makes, and what actually happened,
in one row. Never deleted.

Predictions and outcomes belong in the same row
-----------------------------------------------

Keeping them in two tables joined by an id is the obvious design and the wrong
one here. The join is the thing that goes missing: a prediction whose outcome was
never written looks identical to one whose outcome row was lost, and both look
identical to one that was never supposed to have an outcome. Three very different
situations, one empty join.

In one row, "unresolved" is a state the row itself carries, with a reason
attached. `verify` can then say *why* each unresolved prediction is unresolved,
which is the difference between an audit and a count.

Why the file is the source of truth and the database is an index
----------------------------------------------------------------

The ledger must never lose a write, and the moment it most needs to be written is
the moment something has gone wrong - which is also when the database is most
likely to be unreachable. So an append-only file takes the write, and Postgres is
loaded from it.

This is the same shape the recorder already uses (ndjson, then parquet, then
DuckDB), and for the same reason: the durable thing should be the thing with the
fewest moving parts.

Resolution happens at the horizon, not when convenient
------------------------------------------------------

An outcome resolved early measures a different quantity than the one predicted,
and one resolved late quietly includes information the prediction never had.
:func:`resolve` refuses to resolve a prediction before its horizon has elapsed,
and records the resolution time so a late one is visible rather than invisible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

LEDGER_ROOT = Path(__file__).resolve().parents[2] / "data" / "ledger"


class Resolution(StrEnum):
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    UNRESOLVABLE = "UNRESOLVABLE"


class LedgerError(RuntimeError):
    pass


@dataclass
class LedgerEntry:
    """One prediction and, eventually, its outcome. The same row throughout."""

    # ---- the prediction, written when the signal is made -------------------
    entry_id: str
    reco_id: str
    ts_predicted: datetime
    symbol: str
    event_class: str
    horizon_minutes: int
    p_up: float
    p_lower: float | None
    entry_price: float
    lots: int
    is_long: bool
    mode: str

    # ---- the outcome, written back at the horizon --------------------------
    resolution: Resolution = Resolution.PENDING
    ts_resolved: datetime | None = None
    exit_price: float | None = None
    gross_pnl: float | None = None
    costs: float | None = None
    unresolved_reason: str = ""

    @property
    def horizon_ends(self) -> datetime:
        return self.ts_predicted + timedelta(minutes=self.horizon_minutes)

    @property
    def net_pnl(self) -> float | None:
        if self.gross_pnl is None:
            return None
        return self.gross_pnl - (self.costs or 0.0)

    @property
    def was_right(self) -> bool | None:
        """Directionally. None while unresolved - not False.

        A pending prediction is not a wrong one, and defaulting it to False
        makes every hit rate drift down as the day goes on.
        """
        if self.exit_price is None:
            return None
        moved_up = self.exit_price > self.entry_price
        return moved_up == (self.p_up > 0.5)

    @property
    def resolution_lag(self) -> timedelta | None:
        """How late the outcome was written, past the horizon.

        Recorded because a systematically late resolution is a systematically
        different measurement, and it is invisible unless someone looks.
        """
        if self.ts_resolved is None:
            return None
        return self.ts_resolved - self.horizon_ends

    def to_json(self) -> str:
        def encode(o: Any) -> Any:
            if isinstance(o, datetime):
                return o.isoformat()
            if isinstance(o, StrEnum):
                return o.value
            raise TypeError(f"cannot serialise {type(o).__name__}")

        return json.dumps(asdict(self), default=encode, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> LedgerEntry:
        raw = json.loads(line)
        raw["ts_predicted"] = datetime.fromisoformat(raw["ts_predicted"])
        if raw.get("ts_resolved"):
            raw["ts_resolved"] = datetime.fromisoformat(raw["ts_resolved"])
        raw["resolution"] = Resolution(raw["resolution"])
        return cls(**raw)


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
class PaperLedger:
    """Append-only per day, with resolution written as a superseding record.

    Resolution does **not** edit the original line. It appends a second record
    for the same ``entry_id``, and the reader takes the last one. Editing in
    place would mean the file's history no longer shows what was believed at
    prediction time, which is the one thing this file exists to preserve - and
    it is exactly the kind of edit that gets made with good intentions.
    """

    def __init__(self, root: Path = LEDGER_ROOT) -> None:
        self.root = root

    def path(self, day: datetime) -> Path:
        return self.root / f"{day.date().isoformat()}.ndjson"

    def append(self, entry: LedgerEntry) -> Path:
        p = self.path(entry.ts_predicted)
        self.root.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(entry.to_json() + chr(10))
        return p

    def _lines(self, day: datetime) -> Iterator[LedgerEntry]:
        p = self.path(day)
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield LedgerEntry.from_json(line)

    def entries(self, day: datetime) -> list[LedgerEntry]:
        """Latest record per entry_id, in prediction order.

        Later records supersede earlier ones. The earlier ones stay on disk -
        that is the audit trail, and reading only the latest is a convenience
        for callers rather than a deletion.
        """
        latest: dict[str, LedgerEntry] = {}
        for entry in self._lines(day):
            latest[entry.entry_id] = entry
        return sorted(latest.values(), key=lambda e: e.ts_predicted)

    def history(self, day: datetime, entry_id: str) -> list[LedgerEntry]:
        """Every version of one row, oldest first. What was believed, and when."""
        return [e for e in self._lines(day) if e.entry_id == entry_id]

    def days(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.ndjson"))


def resolve(
    ledger: PaperLedger,
    entry: LedgerEntry,
    *,
    now: datetime,
    exit_price: float | None,
    costs: float = 0.0,
    reason: str = "",
) -> LedgerEntry:
    """Write the outcome back, refusing to do it early.

    An outcome resolved before its horizon measures a different quantity than the
    one predicted. This raises rather than resolving, because the resulting number
    would be perfectly plausible and quietly about something else.

    ``exit_price=None`` records an honest failure to resolve, and requires a
    reason. A prediction with no outcome and no explanation is the row that makes
    every later report unauditable.
    """
    if now < entry.horizon_ends:
        raise LedgerError(
            f"{entry.entry_id} has a {entry.horizon_minutes}m horizon ending at "
            f"{entry.horizon_ends.isoformat()}, and resolution was attempted at "
            f"{now.isoformat()}. An outcome resolved early measures a different "
            "quantity than the one predicted."
        )
    if entry.resolution is not Resolution.PENDING:
        raise LedgerError(
            f"{entry.entry_id} is already {entry.resolution.value}. A corrected "
            "outcome is a new entry that supersedes this one, not an edit."
        )

    if exit_price is None:
        if not reason.strip():
            raise LedgerError(
                "an unresolvable prediction needs a documented reason. A row with "
                "no outcome and no explanation is what makes every later report "
                "unauditable - it cannot be told apart from one that was lost."
            )
        resolved = LedgerEntry(
            **{
                **asdict_shallow(entry),
                "resolution": Resolution.UNRESOLVABLE,
                "ts_resolved": now,
                "unresolved_reason": reason.strip(),
            }
        )
    else:
        direction = 1 if entry.is_long else -1
        gross = direction * (exit_price - entry.entry_price) * entry.lots
        resolved = LedgerEntry(
            **{
                **asdict_shallow(entry),
                "resolution": Resolution.RESOLVED,
                "ts_resolved": now,
                "exit_price": exit_price,
                "gross_pnl": gross,
                "costs": costs,
            }
        )

    ledger.append(resolved)
    log.info(
        "ledger resolved",
        fields={
            "entry_id": entry.entry_id,
            "resolution": resolved.resolution.value,
            "net": resolved.net_pnl,
            "lag_s": (resolved.resolution_lag or timedelta()).total_seconds(),
        },
    )
    return resolved


def asdict_shallow(entry: LedgerEntry) -> dict[str, Any]:
    """Field values without recursing into datetimes and enums."""
    return {f: getattr(entry, f) for f in entry.__dataclass_fields__}


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass
class LedgerAudit:
    """Whether the day's ledger can be believed."""

    day: str
    total: int = 0
    resolved: int = 0
    unresolvable: int = 0
    pending: int = 0
    overdue: list[str] = field(default_factory=list)
    late_resolutions: list[tuple[str, float]] = field(default_factory=list)

    @property
    def accounted_for(self) -> int:
        return self.resolved + self.unresolvable

    @property
    def complete(self) -> bool:
        """Every prediction has an outcome or a documented reason.

        Pending is not a failure while the horizon is still open. Overdue is.
        """
        return not self.overdue

    def render(self) -> str:
        lines = [
            f"  {self.day}: {self.total} predictions, {self.resolved} resolved, "
            f"{self.unresolvable} unresolvable (with reasons), {self.pending} pending",
        ]
        if self.overdue:
            lines.append(
                f"    {len(self.overdue)} OVERDUE - past the horizon with no outcome "
                "and no reason:"
            )
            lines += [f"      {e}" for e in self.overdue[:10]]
        if self.late_resolutions:
            worst = max(s for _, s in self.late_resolutions)
            lines.append(
                f"    {len(self.late_resolutions)} resolved late, worst {worst:.0f}s "
                "past the horizon - a systematically late resolution is a "
                "systematically different measurement"
            )
        lines.append(
            "    -> auditable" if self.complete else "    -> NOT auditable: see overdue above"
        )
        return "\n".join(lines)


def verify(
    ledger: PaperLedger, day: datetime, *, now: datetime, late_after_seconds: float = 300.0
) -> LedgerAudit:
    """Every prediction has an outcome, or a documented reason, or time left.

    The three-way split is the point. A count of "resolved" alone cannot tell a
    healthy day from one where half the outcomes were quietly never written.
    """
    audit = LedgerAudit(day=day.date().isoformat())
    for entry in ledger.entries(day):
        audit.total += 1
        if entry.resolution is Resolution.RESOLVED:
            audit.resolved += 1
            lag = entry.resolution_lag
            if lag is not None and lag.total_seconds() > late_after_seconds:
                audit.late_resolutions.append((entry.entry_id, lag.total_seconds()))
        elif entry.resolution is Resolution.UNRESOLVABLE:
            audit.unresolvable += 1
        else:
            audit.pending += 1
            if now > entry.horizon_ends:
                audit.overdue.append(
                    f"{entry.entry_id} ({entry.event_class}) horizon ended "
                    f"{entry.horizon_ends.isoformat()}"
                )
    return audit


def to_rows(entries: Sequence[LedgerEntry]) -> list[dict[str, Any]]:
    """Flatten for loading into the database index.

    The file stays the source of truth. This is a projection of it, and if the
    two ever disagree the file wins - which is why nothing here computes anything
    the file does not already contain.
    """
    return [
        {
            "entry_id": e.entry_id,
            "reco_id": e.reco_id,
            "ts_predicted": e.ts_predicted,
            "ts_resolved": e.ts_resolved,
            "symbol": e.symbol,
            "event_class": e.event_class,
            "horizon_minutes": e.horizon_minutes,
            "p_up": e.p_up,
            "p_lower": e.p_lower,
            "entry_price": e.entry_price,
            "exit_price": e.exit_price,
            "lots": e.lots,
            "is_long": e.is_long,
            "mode": e.mode,
            "resolution": e.resolution.value,
            "gross_pnl": e.gross_pnl,
            "costs": e.costs,
            "net_pnl": e.net_pnl,
            "was_right": e.was_right,
            "unresolved_reason": e.unresolved_reason,
        }
        for e in entries
    ]
