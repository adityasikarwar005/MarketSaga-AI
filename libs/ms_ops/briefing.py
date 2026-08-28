"""M95 - the morning briefing, and the two words in its gate that do the work.

The plan calls this "the first thing you ship to a real user (yourself)", and
its gate is *20 consecutive automated briefings*. Both halves of that phrase are
load-bearing, and one of them is easy to lose.

Consecutive, not twenty
-------------------------

Counting briefing records gives twenty for a system that ran on twenty scattered
days across three months. The gate is asking whether the thing runs **every
trading morning without anyone touching it**, which is a different question and
the one that predicts whether it will still be running in June.

:func:`consecutive_streak` walks trading days backwards from the most recent
session and stops at the first one with no briefing. A holiday is not a miss -
skipping it is correct behaviour - so the walk is calendar-aware rather than a
subtraction on dates. A streak that counts calendar days would break every
weekend and never reach twenty at all, which is the failure that gets noticed;
one that counts records reaches twenty while proving nothing, which is the
failure that does not.

Published before the open, scored after it
--------------------------------------------

A briefing carries a forecast, and a forecast is only a forecast if it was
written down before the outcome. :meth:`Briefing.publish` refuses to run once
the session has opened, and the check is against an injected clock rather than
the wall - a replayed morning must be able to produce the briefing it would have
produced, and a briefing stamped with today's time while forecasting a day in
March is not evidence of anything.

The score is written as a **separate, later record** that points back at the
briefing by id. Filling a score into the briefing itself would make the file a
place where a published forecast can be edited after the answer is known, which
is exactly what the append-only ledger elsewhere in this project exists to stop.

An uncalibrated band is labelled in the briefing
--------------------------------------------------

``GapForecast`` already knows whether its intervals went through conformal
calibration. That flag has to survive into the text a person reads, because a
90% band that is not a 90% band is the single most misleading thing a briefing
can contain - it is wrong in a direction that makes the reader more confident.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from libs.ms_core.clock import Clock
from libs.ms_core.obs import get_logger
from libs.ms_models.gap import GapForecast

log = get_logger(__name__)

# The gate's number. Twenty trading mornings is a month of them.
GATE_STREAK = 20

# A briefing is due before the market opens. 06:30 IST, per MA1's rhythm.
BRIEFING_HOUR_IST = 6
BRIEFING_MINUTE_IST = 30


class BriefingError(RuntimeError):
    pass


class Delivery(StrEnum):
    MARKDOWN = "markdown"
    PUSH = "push"
    EMAIL = "email"


@dataclass(frozen=True)
class KeyLevel:
    """One price worth naming, and why it is worth naming.

    ``why`` is required. A level with no reason is a number on a chart, and the
    briefing's whole job is to be a thing a person can argue with.
    """

    name: str
    price: float
    why: str

    def render(self) -> str:
        return f"    {self.name:<22} {self.price:>10.2f}   {self.why}"


@dataclass(frozen=True)
class CalendarItem:
    when: str  # exchange-local, e.g. "11:00 IST"
    what: str
    expected_impact: str = ""

    def render(self) -> str:
        tail = f"  ({self.expected_impact})" if self.expected_impact else ""
        return f"    {self.when:<12} {self.what}{tail}"


@dataclass
class Briefing:
    """One morning's briefing. Immutable once published.

    ``briefing_id`` is derived from the session and the forecast, so the same
    inputs give the same id and a re-run cannot quietly create a second record
    for one morning.
    """

    session: date
    published_at: datetime
    forecast: GapForecast
    overnight_events: list[str] = field(default_factory=list)
    key_levels: list[KeyLevel] = field(default_factory=list)
    calendar: list[CalendarItem] = field(default_factory=list)
    delivery: Delivery = Delivery.MARKDOWN
    notes: list[str] = field(default_factory=list)

    @property
    def briefing_id(self) -> str:
        payload = json.dumps(
            {
                "session": self.session.isoformat(),
                "median": round(self.forecast.median_gap, 10),
                "band": [round(v, 10) for v in self.forecast.forecast.interval()],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def render(self) -> str:
        lo, hi = self.forecast.forecast.interval()
        band_label = "90% band" if self.forecast.calibrated else "90% band [UNCALIBRATED]"
        lines = [
            f"# Morning briefing - {self.session}",
            f"_published {self.published_at.isoformat()} · id {self.briefing_id}_",
            "",
            "## Gap forecast",
            f"    direction : {self.forecast.direction}",
            f"    median    : {self.forecast.median_gap:+.3%}",
            f"    {band_label:<10}: [{lo:+.3%}, {hi:+.3%}]",
            f"    window    : {self.forecast.window_hours:.1f} hours"
            + ("  (after a weekend or holiday)" if self.forecast.spans_weekend else ""),
        ]
        if not self.forecast.calibrated:
            lines.append(
                "    NOTE: these intervals did not go through conformal calibration. "
                "The band is narrower than 90% and will look more confident than it is."
            )
        if self.forecast.drivers:
            lines.append("    drivers   : " + "; ".join(self.forecast.drivers[:4]))

        lines += ["", "## Overnight"]
        lines += (
            [f"    - {e}" for e in self.overnight_events]
            if self.overnight_events
            else ["    nothing recorded"]
        )

        lines += ["", "## Key levels"]
        lines += (
            [level.render() for level in self.key_levels] if self.key_levels else ["    none set"]
        )

        lines += ["", "## Today"]
        lines += (
            [item.render() for item in self.calendar]
            if self.calendar
            else ["    nothing scheduled"]
        )

        if self.notes:
            lines += ["", "## Notes"] + [f"    - {n}" for n in self.notes]
        return "\n".join(lines)


def publish(
    *,
    session: date,
    forecast: GapForecast,
    clock: Clock,
    session_open: datetime,
    overnight_events: Sequence[str] = (),
    key_levels: Sequence[KeyLevel] = (),
    calendar: Sequence[CalendarItem] = (),
    delivery: Delivery = Delivery.MARKDOWN,
) -> Briefing:
    """Write the briefing, or refuse because the session has already opened.

    The clock is injected. A briefing stamped with the wall clock while
    forecasting a morning in March is not a record of a forecast, and a replayed
    day has to be able to produce the briefing it would have produced.
    """
    now = clock.now()
    if now >= session_open:
        raise BriefingError(
            f"the {session} session opened at {session_open.isoformat()} and it is "
            f"now {now.isoformat()}. A gap forecast published after the open is not "
            "a forecast, and a briefing that will publish one is a briefing whose "
            "score means nothing."
        )
    if forecast.session != session:
        raise BriefingError(
            f"the forecast is for {forecast.session} and the briefing is for {session}"
        )

    briefing = Briefing(
        session=session,
        published_at=now,
        forecast=forecast,
        overnight_events=list(overnight_events),
        key_levels=list(key_levels),
        calendar=list(calendar),
        delivery=delivery,
    )
    if not forecast.calibrated:
        briefing.notes.append(
            "the gap intervals are uncalibrated; treat the band as indicative and "
            "do not size against it"
        )
    log.info(
        "briefing published",
        fields={"session": session.isoformat(), "id": briefing.briefing_id},
    )
    return briefing


# --------------------------------------------------------------------------- #
# Scoring, as a separate later record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BriefingScore:
    """What actually happened, recorded against a briefing that already exists.

    Written as its own record pointing back by ``briefing_id`` rather than
    filled into the briefing. Filling it in would make the briefing file a place
    where a published forecast can be edited once the answer is known.
    """

    briefing_id: str
    session: date
    scored_at: datetime
    actual_gap: float
    predicted_median: float
    band: tuple[float, float]

    @property
    def error(self) -> float:
        return self.actual_gap - self.predicted_median

    @property
    def covered(self) -> bool:
        lo, hi = self.band
        return lo <= self.actual_gap <= hi

    @property
    def direction_right(self) -> bool:
        """Only meaningful when the forecast committed to a direction.

        A flat call is neither right nor wrong about direction, and counting it
        as wrong makes an honest abstention look like a miss.
        """
        if abs(self.predicted_median) <= 0.0005:
            return False
        return (self.actual_gap > 0) == (self.predicted_median > 0)

    @property
    def committed(self) -> bool:
        return abs(self.predicted_median) > 0.0005

    def render(self) -> str:
        cover = "covered" if self.covered else "OUTSIDE the band"
        if self.committed:
            direction = "direction right" if self.direction_right else "direction wrong"
        else:
            direction = "no direction called"
        return (
            f"    {self.session}  actual {self.actual_gap:+.3%}  "
            f"predicted {self.predicted_median:+.3%}  "
            f"err {self.error:+.3%}  {cover}  {direction}"
        )


def score_against_open(
    briefing: Briefing, *, actual_gap: float, clock: Clock, session_open: datetime
) -> BriefingScore:
    """Score a briefing once the open has happened, and not before."""
    now = clock.now()
    if now < session_open:
        raise BriefingError(
            f"the {briefing.session} session opens at {session_open.isoformat()} and "
            f"it is {now.isoformat()}. There is no actual open to score against yet."
        )
    return BriefingScore(
        briefing_id=briefing.briefing_id,
        session=briefing.session,
        scored_at=now,
        actual_gap=actual_gap,
        predicted_median=briefing.forecast.median_gap,
        band=briefing.forecast.forecast.interval(),
    )


# --------------------------------------------------------------------------- #
# The streak, which is the part of the gate that gets lost
# --------------------------------------------------------------------------- #
TradingDayTest = Callable[[date], bool]


def consecutive_streak(
    sessions: Sequence[date], *, through: date, is_trading_day: TradingDayTest, limit: int = 400
) -> int:
    """Trading mornings in an unbroken run ending at ``through``.

    Walks backwards one trading day at a time and stops at the first that has no
    briefing. A holiday is skipped rather than counted as a miss - not running
    on a closed day is correct behaviour, and treating it as a break means the
    streak resets every Diwali.

    ``len(sessions)`` is the number this replaces. Twenty records spread over
    three months satisfies "twenty briefings" and says nothing about whether the
    thing runs unattended every morning, which is what the gate is asking.
    """
    have = set(sessions)
    if through not in have:
        return 0

    streak = 0
    cursor = through
    for _ in range(limit):
        if not is_trading_day(cursor):
            cursor = date.fromordinal(cursor.toordinal() - 1)
            continue
        if cursor not in have:
            break
        streak += 1
        cursor = date.fromordinal(cursor.toordinal() - 1)
    return streak


@dataclass
class BriefingLog:
    """Append-only record of briefings and their scores.

    Two files rather than one, for the same reason the score is a separate
    record: the briefings file is what was claimed, the scores file is what
    happened, and nothing writes to the first one after the morning it belongs
    to.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def scores_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_scores" + self.path.suffix)

    def record(self, briefing: Briefing) -> None:
        if briefing.session in self.sessions():
            raise BriefingError(
                f"a briefing for {briefing.session} is already recorded. Publishing a "
                "second one for the same morning would let a forecast be replaced "
                "after the fact."
            )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "session": briefing.session.isoformat(),
                        "published_at": briefing.published_at.isoformat(),
                        "briefing_id": briefing.briefing_id,
                        "median": briefing.forecast.median_gap,
                        "band": list(briefing.forecast.forecast.interval()),
                        "calibrated": briefing.forecast.calibrated,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def record_score(self, score: BriefingScore) -> None:
        with self.scores_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "briefing_id": score.briefing_id,
                        "session": score.session.isoformat(),
                        "scored_at": score.scored_at.isoformat(),
                        "actual_gap": score.actual_gap,
                        "predicted_median": score.predicted_median,
                        "band": list(score.band),
                        "covered": score.covered,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def sessions(self) -> list[date]:
        if not self.path.exists():
            return []
        return [
            date.fromisoformat(json.loads(line)["session"])
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def scored_sessions(self) -> list[date]:
        if not self.scores_path.exists():
            return []
        return [
            date.fromisoformat(json.loads(line)["session"])
            for line in self.scores_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def unscored(self) -> list[date]:
        """Briefings that were published and never checked against the open.

        A briefing nobody scored is a claim nobody tested, and a run of those is
        the shape of a system that looks operational and is not.
        """
        return sorted(set(self.sessions()) - set(self.scored_sessions()))


@dataclass
class GateG4Briefing:
    """The briefing half of Gate G4: twenty consecutive automated mornings."""

    streak: int
    published: int
    unscored: list[date]
    required: int = GATE_STREAK

    @property
    def passes(self) -> bool:
        return self.streak >= self.required and not self.unscored

    def render(self) -> str:
        lines = [
            f"  briefing streak: {self.streak} consecutive trading mornings "
            f"(need {self.required}); {self.published} published in total"
        ]
        if self.published > self.streak:
            lines.append(
                f"    {self.published} briefings exist but only {self.streak} are "
                "consecutive. The gate asks whether this runs every trading morning "
                "unattended, which a scattered count cannot answer."
            )
        if self.unscored:
            lines.append(
                f"    {len(self.unscored)} briefing(s) never scored against the open "
                f"({self.unscored[0]} .. {self.unscored[-1]}). A forecast nobody "
                "checked is a claim nobody tested."
            )
        lines.append(f"  -> {'PASSES' if self.passes else 'does not pass'}")
        return "\n".join(lines)


def evaluate_gate(
    log: BriefingLog, *, through: date, is_trading_day: TradingDayTest
) -> GateG4Briefing:
    sessions = log.sessions()
    return GateG4Briefing(
        streak=consecutive_streak(sessions, through=through, is_trading_day=is_trading_day),
        published=len(sessions),
        unscored=log.unscored(),
    )
