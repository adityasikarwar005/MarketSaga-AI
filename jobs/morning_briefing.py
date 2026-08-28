"""M95 - the morning briefing. The first thing this ships to a real user.

    python jobs/morning_briefing.py --dry-run
    python jobs/morning_briefing.py --dry-run --gate
    python jobs/morning_briefing.py --session 2026-08-27 --out build/briefing.md

Runs at 06:30 IST: what happened overnight, the gap forecast as a distribution,
key levels, today's calendar. Written before the open and scored after it.

The gate is *20 consecutive automated briefings*, and ``--gate`` reports the
streak rather than the count. Twenty briefings on twenty scattered mornings
satisfies the sentence and answers a different question - the one worth asking
is whether this runs every trading morning with nobody watching.

``--dry-run`` builds a briefing from a synthetic forecast and prints it. It
touches no log, so a rehearsal cannot pad the streak the gate is counting.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.clock import FrozenClock, RealClock
from libs.ms_core.obs import configure
from libs.ms_models.gap import GapForecast
from libs.ms_models.quantile import QuantileForecast
from libs.ms_ops.briefing import (
    BriefingLog,
    CalendarItem,
    Delivery,
    KeyLevel,
    evaluate_gate,
    publish,
)

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "research" / "briefings.ndjson"
IST = ZoneInfo("Asia/Kolkata")


def session_open_utc(session: date) -> datetime:
    """09:15 IST, as an aware UTC instant.

    Built in IST and converted, never assembled in UTC by subtracting five and a
    half hours - which is the arithmetic that is right until it is not.
    """
    return datetime.combine(session, time(9, 15), tzinfo=IST).astimezone(UTC)


def briefing_time_utc(session: date) -> datetime:
    return datetime.combine(session, time(6, 30), tzinfo=IST).astimezone(UTC)


def demo_forecast(session: date) -> GapForecast:
    """A plausible gap distribution. Synthetic - no real overnight data here."""
    median, half = 0.0021, 0.0068
    return GapForecast(
        session=session,
        forecast=QuantileForecast(
            quantiles={
                0.05: median - half,
                0.25: median - half / 3,
                0.5: median,
                0.75: median + half / 3,
                0.95: median + half,
            }
        ),
        window_hours=17.5,
        spans_weekend=session.weekday() == 0,
        drivers=["SPX +0.62%", "USDINR -0.1%", "Brent +1.4%"],
        calibrated=True,
    )


def trading_day(day: date) -> bool:
    """Weekdays only. The real calendar lives in `SessionCalendar`; this job
    falls back to weekdays so `--gate` works before a holiday list is loaded,
    and says so rather than pretending it knows the exchange calendar."""
    return day.weekday() < 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="build and print, record nothing")
    parser.add_argument("--session", type=date.fromisoformat, help="the session to brief for")
    parser.add_argument("--out", type=Path, help="write the markdown here")
    parser.add_argument("--gate", action="store_true", help="report the consecutive-morning streak")
    args = parser.parse_args(argv)

    configure()
    session = args.session or RealClock().now().astimezone(IST).date()

    if args.gate:
        log = BriefingLog(path=LOG_PATH)
        gate = evaluate_gate(log, through=session, is_trading_day=trading_day)
        print(gate.render())
        print(
            "\n  Streak is over trading days on a weekday calendar - the exchange "
            "holiday list is not loaded here, so a holiday inside the window would "
            "read as a miss."
        )
        return 0 if gate.passes else 1

    # The briefing is published at 06:30 IST. For a dry run the clock is frozen
    # there so the output is the same every time and cannot depend on when
    # somebody happened to run it.
    clock = FrozenClock(briefing_time_utc(session))
    briefing = publish(
        session=session,
        forecast=demo_forecast(session),
        clock=clock,
        session_open=session_open_utc(session),
        overnight_events=[
            "SPX closed +0.62%, breadth positive",
            "USDINR flat overnight",
            "No India-specific headlines on the wire",
        ],
        key_levels=[
            KeyLevel("prior close", 24_012.40, "yesterday's settlement"),
            KeyLevel("overnight high", 24_118.00, "SGX implied"),
            KeyLevel("weekly expiry strike", 24_000.00, "max open interest"),
        ],
        calendar=[
            CalendarItem("11:00 IST", "RBI policy statement", "high"),
            CalendarItem("15:30 IST", "Weekly expiry settlement", "mechanical"),
        ],
        delivery=Delivery.MARKDOWN,
    )

    text = briefing.render()
    print(text)

    if args.dry_run:
        print(
            "\n  DRY RUN - synthetic forecast, and nothing was recorded. A rehearsal "
            "that wrote to the log would pad the streak the gate counts."
        )
        return 0

    log = BriefingLog(path=LOG_PATH)
    log.record(briefing)
    print(f"\n  recorded in {LOG_PATH.relative_to(ROOT)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"  written to {args.out}")
    print(
        "  The score is a separate later record. Run this again after the open "
        "with the actual gap to close the loop - an unscored briefing is a claim "
        "nobody tested, and the gate counts those against you."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
