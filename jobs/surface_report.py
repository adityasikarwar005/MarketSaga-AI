"""M70 - the surface report: fit RMSE per expiry, and the rejected-quote rate.

    python jobs/surface_report.py --date 2026-08-27                      the F&O bhavcopy
    python jobs/surface_report.py --date 2026-08-31 --source recorded    a live recorded book
    python jobs/surface_report.py --date 2026-08-31 --source recorded --at 11:30
    python jobs/surface_report.py --demo                                 synthetic, no data needed

Two numbers per expiry, and the second one is the one that gets ignored.

**Fit RMSE** says how well the smile describes the quotes it was given. It is
the number people look at, and on its own it is close to useless: a surface
fitted to the six quotes that survived a bad feed will have a superb RMSE.

**Rejected-quote rate** says how many quotes it was *not* given. A chain that
normally rejects 15% and today rejects 60% has a feed problem, a liquidity
event, or a stale snapshot - and the surface fitted from the survivors looks
entirely normal. This is why the report prints them together and why the
rejection reasons are broken out: "spread too wide" on every strike is a
different morning from "OI too low" on every strike.

The report also prints the arbitrage verdict, because a surface can have an
excellent RMSE, a healthy rejection rate, and still imply a negative
probability density.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core import config as cfgmod
from libs.ms_core.clock import SessionCalendar
from libs.ms_core.obs import configure
from libs.ms_core.schemas import ChainSnapshot, OptionQuote, OptionType
from libs.ms_data.bhavcopy import (
    BhavcopyStore,
    ConvexityCheck,
    chain_from_records,
    convexity_of,
    negative_butterflies,
)
from libs.ms_data.holidays import HolidayStore
from libs.ms_data.recorder import DATA_ROOT
from libs.ms_options.pricing import OptType, black76_price
from libs.ms_options.surface import PriceBasis, Surface, SVIParams, fit_surface

IST = ZoneInfo("Asia/Kolkata")

# One snapshot is about 40 KB, and a full session is a few hundred megabytes
# by the close. Reading the last 40 MB reaches roughly the final thousand
# snapshots, which is every instant anybody fits and none of the morning.
RECORDED_TAIL_BYTES = 40_000_000

ROOT = Path(__file__).resolve().parents[1]
CHAIN_DIR = ROOT / "data" / "raw"

# A plausible Nifty smile, used only by --demo.
# Calibrated to a plausible Nifty term structure: ~17% at a week, 13% at a
# month, 12% at two. A demo that prints a 60% ATM vol trains the reader to
# ignore the number, which is the one thing the report exists to stop.
DEMO_SMILE = SVIParams(a=0.0006, b=0.0057, rho=-0.42, m=0.015, sigma=0.115)
DEMO_SPOT = 24000.0
DEMO_R = 0.065


def demo_chain(ts: datetime) -> ChainSnapshot:
    """A synthetic chain across three expiries, with a wide strike ladder.

    Total variance grows with maturity by construction, so a calendar violation
    in the output is the fitter's doing rather than the fixture's.
    """
    quotes: list[OptionQuote] = []
    for days, scale in ((7, 0.45), (28, 1.0), (56, 1.7)):
        expiry = ts + timedelta(days=days)
        t = days / 365.0
        f = DEMO_SPOT * math.exp(DEMO_R * t)
        params = SVIParams(
            a=DEMO_SMILE.a * scale,
            b=DEMO_SMILE.b * scale,
            rho=DEMO_SMILE.rho,
            m=DEMO_SMILE.m,
            sigma=DEMO_SMILE.sigma,
        )
        # The ladder spans a fixed number of standard deviations, not a fixed
        # percentage. A +-28% ladder is +-8 sigma on a 13-vol weekly, so every
        # strike past the first few is a 0-delta quote the filters throw away -
        # and the report then reads as a 70% rejection rate rather than as a
        # badly chosen fixture.
        atm_vol = math.sqrt(params.total_variance(0.0) / t)
        span = 3.5 * atm_vol * math.sqrt(t)
        for i in range(25):
            k = -span + 2.0 * span * i / 24
            strike = round(DEMO_SPOT * math.exp(k) / 50.0) * 50.0
            vol = math.sqrt(params.total_variance(math.log(strike / f)) / t)
            for opt, kind in ((OptionType.CE, OptType.CALL), (OptionType.PE, OptType.PUT)):
                mid = black76_price(f, strike, vol, t, DEMO_R, kind)
                if mid <= 0.10:
                    continue
                # Wider spreads on the wings, as a real chain has - so the
                # rejection rate in the report is not identically zero.
                half = mid * (0.004 + 0.05 * abs(k))
                quotes.append(
                    OptionQuote(
                        expiry=expiry,
                        strike=strike,
                        opt_type=opt,
                        bid=max(mid - half, 0.05),
                        ask=mid + half,
                        ltp=mid,
                        oi=max(int(8000 * math.exp(-8 * k * k)), 1),
                    )
                )
    return ChainSnapshot(ts=ts, underlying_id=1, spot=DEMO_SPOT, quotes=quotes)


@dataclass(frozen=True)
class Provenance:
    """Where the chain came from, in the numbers a reader needs to judge it."""

    symbol: str
    day: date
    listed: int
    traded: int
    dropped_untraded: int
    convexity: ConvexityCheck
    settles_at_close: int

    def render(self) -> list[str]:
        pct = self.dropped_untraded / self.listed if self.listed else 0.0
        at_close = self.settles_at_close / self.traded if self.traded else 0.0
        return [
            f"  REAL chain: {self.symbol} F&O bhavcopy for {self.day}.",
            f"    {self.listed:,} contracts listed, {self.traded:,} traded, "
            f"{self.dropped_untraded:,} dropped for never trading ({pct:.0%}).",
            "",
            "  READ THIS BEFORE THE NUMBERS BELOW.",
            "    A bhavcopy carries no book, so this is fitted on SETTLEMENT",
            f"    prices - and on {at_close:.0%} of the contracts kept here, NSE set",
            "    settlement TO THE CLOSE, which is the last traded price. That is",
            "    the path the mid rule exists to forbid. On contracts that did not",
            "    trade, settlement is NSE's own model output instead, which is why",
            "    they are dropped. There is no third kind of row.",
            "",
            "    The cost is measurable in the marks themselves, before fitting:",
            *self.convexity.render(),
            "    Those prices were never simultaneously available, so a surface",
            "    fitted on them inherits the arbitrage in its inputs. Treat what",
            "    follows as a measurement of daily marks, NOT as a surface any",
            "    consumer should price against.",
        ]


def load_chain(day: date, *, symbol: str = "NIFTY") -> tuple[ChainSnapshot, Provenance]:
    """One session's real option chain, from the NSE F&O bhavcopy.

    ``traded_only`` is not optional here and is deliberately not a flag. More
    than half of a listed chain never trades, and every one of those contracts
    still carries a settlement price NSE computed from its own model. Fitting
    SVI to those points fits the exchange's model, converges beautifully, and
    reports a smile that nothing traded at.
    """
    cfg = cfgmod.load(warn=False)
    cal = SessionCalendar.from_config(cfg.sessions).with_holidays(HolidayStore().as_calendar_dict())
    dayfile = BhavcopyStore(calendar=cal).get(day)

    listed = sum(1 for r in dayfile.records if r.symbol == symbol)
    if not listed:
        present = sorted({r.symbol for r in dayfile.records})[:12]
        raise SystemExit(
            f"  no contracts named {symbol!r} in the {day} bhavcopy. "
            f"Present, first twelve: {', '.join(present)}"
        )

    records, dropped = dayfile.options(symbol, traded_only=True)
    if not records:
        raise SystemExit(f"  {listed} {symbol} contracts listed on {day}, none traded.")

    ids = {i.symbol: i.id for i in cfg.instruments.instruments}
    if symbol not in ids:
        raise SystemExit(
            f"  {symbol!r} has no id in config/instruments.yaml. An id invented "
            "here would attribute this chain to an instrument that does not exist."
        )

    # The chain is a photograph of the close, which is what settlement marks.
    # available_at() answers a different question - when a replay may release
    # the file - and using it here would put every time-to-expiry a day short.
    _, session_end = cal.session_bounds(day)

    chain = chain_from_records(
        records,
        underlying_id=ids[symbol],
        ts=session_end,
        price_field="settlement",
    )
    return chain, Provenance(
        symbol=symbol,
        day=day,
        listed=listed,
        traded=len(records),
        dropped_untraded=dropped,
        convexity=negative_butterflies(records),
        settles_at_close=sum(1 for r in records if r.close == r.settlement),
    )


@dataclass(frozen=True)
class RecordedProvenance:
    """A chain read from the recorder's own partitions, not from the archive.

    The counterpart to :class:`Provenance`, and almost every warning that one
    carries is absent here - which is the whole point and is worth stating
    rather than leaving as an absence. A recorded snapshot is a two-sided book
    captured at one instant, so the mid rule holds and `PriceBasis.MID` applies
    unchanged. There is no settlement mark, no exchange model output standing in
    for a contract that never traded, and no set of prices assembled across a
    session and called simultaneous.

    What it is not is a photograph of the close. It is one instant chosen out of
    a session, so a fit from it is not comparable day-over-day with a bhavcopy
    fit, and two instants from the same morning are not independent observations
    of anything.
    """

    symbol: str
    day: date
    token: int
    snapshot_ts: datetime
    quotes: int
    strikes: int
    expiries: list[date]
    completeness: float
    snapshots_available: int
    missing_book: int
    convexity: ConvexityCheck

    def render(self) -> list[str]:
        return [
            f"  REAL chain: {self.symbol} recorded live book for {self.day}.",
            f"    snapshot {self.snapshot_ts.astimezone(IST):%H:%M:%S} IST, "
            f"one of {self.snapshots_available:,} on file, completeness "
            f"{self.completeness:.0%}.",
            f"    {self.quotes:,} quotes across {self.strikes} strikes and "
            f"{len(self.expiries)} expiries "
            f"({', '.join(str(e) for e in self.expiries)}).",
            f"    {self.missing_book:,} quotes carry no two-sided book.",
            "",
            "  Convexity of the marks themselves, before any model touches them -",
            "  the same measurement, by the same function, that the bhavcopy path",
            "  reports. This is the comparison M70 exists to make:",
            *self.convexity.render(),
            "",
            "  This is the input M70 said it needed and did not have.",
            "    Every price below is a MID from a book quoted at one instant, so",
            "    none of the settlement caveats apply: no exchange model output",
            "    standing in for a contract that never traded, and no prices",
            "    assembled across a session and called simultaneous.",
            "",
            "    Two things it is still not. It is one instant out of a session,",
            "    not a close, so this fit is not comparable day-over-day with a",
            "    bhavcopy fit. And the instrument is the vendor's token",
            f"    ({self.token}), not the id in config/instruments.yaml - the live",
            "    path and the research layer have not been reconciled.",
        ]


def load_recorded_chain(
    day: date,
    *,
    symbol: str = "NIFTY",
    root: Path | None = None,
    at: time | None = None,
) -> tuple[ChainSnapshot, RecordedProvenance]:
    """One snapshot out of a live recording, for M70 to fit.

    The archive loader above is the only way M70 could reach real data, and a
    bhavcopy has no book - which is exactly the finding M70 reported and could
    not act on. This is the other door.

    ``at`` picks the last snapshot at or before that exchange-local time;
    without it the most recent complete snapshot is used. Snapshots with no
    quotes are skipped rather than returned: the first few of a session are
    empty while the builder fills, and fitting one would report "no expiry had
    enough usable quotes" for a recording that is fine.
    """
    from jobs.live_state import discover_token, tail_lines

    base = root or DATA_ROOT
    token = discover_token(base, day)
    if token is None:
        raise SystemExit(
            f"  no recorded chain for {day} under {base}. Record one with "
            "`python tasks.py record angel --chain`, or pass --source bhavcopy."
        )

    path = base / "chain" / f"date={day.isoformat()}" / f"chain-{token}.ndjson"
    snaps: list[ChainSnapshot] = []
    for line in tail_lines(path, RECORDED_TAIL_BYTES):
        try:
            snap = ChainSnapshot.model_validate_json(line)
        except Exception:
            continue  # a half-written final line while the recorder is running
        if snap.quotes:
            snaps.append(snap)
    if not snaps:
        raise SystemExit(
            f"  {path.name} holds no snapshot with quotes in its last "
            f"{RECORDED_TAIL_BYTES // 1_000_000} MB."
        )

    if at is not None:
        cutoff = datetime.combine(day, at, tzinfo=IST)
        eligible = [s for s in snaps if s.ts.astimezone(IST) <= cutoff]
        if not eligible:
            raise SystemExit(
                f"  no snapshot at or before {at:%H:%M} IST; the earliest on file is "
                f"{snaps[0].ts.astimezone(IST):%H:%M:%S}."
            )
        chosen = eligible[-1]
    else:
        chosen = snaps[-1]

    strikes = {q.strike for q in chosen.quotes}
    return chosen, RecordedProvenance(
        symbol=symbol,
        day=day,
        token=token,
        snapshot_ts=chosen.ts,
        quotes=len(chosen.quotes),
        strikes=len(strikes),
        expiries=sorted({q.expiry.astimezone(IST).date() for q in chosen.quotes}),
        completeness=chosen.completeness,
        snapshots_available=len(snaps),
        missing_book=sum(1 for q in chosen.quotes if q.mid is None),
        convexity=convexity_of(
            [
                (q.expiry.astimezone(IST).date(), q.opt_type, q.strike, q.mid)
                for q in chosen.quotes
                if q.mid is not None
            ],
            price_label="quoted at",
        ),
    )


def report(surface: Surface, *, now: datetime) -> str:
    lines = [surface.render(), ""]

    if not surface.slices:
        return "\n".join([*lines, "  no expiry had enough usable quotes to fit"])

    lines.append("  per-expiry summary")
    lines.append(
        f"    {'expiry':<12} {'T':>8} {'quotes':>7} {'rmse(vp)':>9} {'reject':>8}  {'atm':>7}"
    )
    worst_reject = 0.0
    reasons: dict[str, int] = {}
    for _e, sl in sorted(surface.slices.items(), key=lambda kv: kv[1].t):
        worst_reject = max(worst_reject, sl.stats.reject_rate)
        for reason, n in sl.stats.rejected.items():
            reasons[reason.value] = reasons.get(reason.value, 0) + n
        lines.append(
            f"    {sl.expiry.date()!s:<12} {sl.t:>8.4f} {sl.n_quotes:>7} "
            f"{sl.rmse * 100:>9.3f} {sl.stats.reject_rate:>7.1%}  {sl.level:>6.2%}"
        )

    dominant = max(reasons, key=lambda r: reasons[r]) if reasons else "nothing"
    lines.append("")
    lines.append(f"  Worst rejection rate {worst_reject:.0%}, dominated by {dominant}.")
    lines.append(
        "  The absolute level is not the signal. Half a full chain being rejected "
        "is normal - the exchange lists strikes far past the liquid band and the "
        "delta filter is supposed to drop them. What means something is this "
        "number moving against its own history, and a shift in which reason "
        "dominates: every strike failing on spread is a different morning from "
        "every strike failing on open interest. Feed it to the drift monitors "
        "(MA3) rather than reading it as a threshold here."
    )

    arb = surface.arbitrage_report()
    lines.append("")
    lines.append(arb.render())
    if not arb.clean:
        lines.append(
            "  This surface will not be served. A good RMSE and a healthy "
            "rejection rate do not make a surface arbitrage-free, and the "
            "violation is invisible in any single option price."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, help="the recorded day to fit")
    parser.add_argument("--demo", action="store_true", help="fit a synthetic chain instead")
    parser.add_argument("--symbol", default="NIFTY", help="underlying to fit (default NIFTY)")
    parser.add_argument(
        "--source",
        choices=("bhavcopy", "recorded"),
        default="bhavcopy",
        help="where the chain comes from; `recorded` reads a live snapshot off the recorder",
    )
    parser.add_argument(
        "--at",
        type=time.fromisoformat,
        help="recorded only: the last snapshot at or before this exchange-local time (HH:MM)",
    )
    parser.add_argument("--strict", action="store_true", help="exit non-zero if arbitrage is found")
    args = parser.parse_args(argv)

    configure()

    print()
    if args.demo:
        now = datetime(2026, 8, 27, 6, 0, tzinfo=UTC)
        chain = demo_chain(now)
        basis = PriceBasis.MID
        print("  SYNTHETIC chain with a known smile - not real market data.")
    elif args.date and args.source == "recorded":
        chain, rec_prov = load_recorded_chain(args.date, symbol=args.symbol, at=args.at)
        now = chain.ts
        # MID, not SETTLEMENT. A recorded snapshot is a two-sided book quoted at
        # one instant, which is the basis the surface was written for and the
        # one the archive could never supply.
        basis = PriceBasis.MID
        for line in rec_prov.render():
            print(line)
    elif args.date:
        if args.at is not None:
            parser.error("--at applies to --source recorded; a bhavcopy has one price per day")
        now = datetime.combine(args.date, datetime.min.time(), tzinfo=UTC)
        chain, prov = load_chain(args.date, symbol=args.symbol)
        basis = PriceBasis.SETTLEMENT
        for line in prov.render():
            print(line)
    else:
        parser.error("give --date <YYYY-MM-DD> or --demo")

    print()
    surface = fit_surface(chain, now=chain.ts, basis=basis)
    print(report(surface, now=now))

    if args.strict and not surface.arbitrage_report().clean:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
