"""M70 - the surface report: fit RMSE per expiry, and the rejected-quote rate.

    python jobs/surface_report.py --date 2026-08-27
    python jobs/surface_report.py --demo           # a synthetic chain, no data needed

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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.obs import configure
from libs.ms_core.schemas import ChainSnapshot, OptionQuote, OptionType
from libs.ms_options.pricing import OptType, black76_price
from libs.ms_options.surface import Surface, SVIParams, fit_surface

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


def load_chain(day: date) -> ChainSnapshot:
    raise SystemExit(
        f"No recorded chain for {day}. Chain capture lands with the recorder "
        f"(look in {CHAIN_DIR.relative_to(ROOT)}), and nothing has been recorded "
        "on real option data yet.\n\n"
        "Run `python jobs/surface_report.py --demo` to exercise the fitter on a "
        "synthetic chain with a known smile. Be clear when reporting which one "
        "produced a number: only the overnight gap forecaster has ever run on "
        "real market data."
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
    parser.add_argument("--strict", action="store_true", help="exit non-zero if arbitrage is found")
    args = parser.parse_args(argv)

    configure()

    if args.demo:
        now = datetime(2026, 8, 27, 6, 0, tzinfo=UTC)
        chain = demo_chain(now)
        print("  SYNTHETIC chain with a known smile - not real market data.")
    elif args.date:
        now = datetime.combine(args.date, datetime.min.time(), tzinfo=UTC)
        chain = load_chain(args.date)
    else:
        parser.error("give --date <YYYY-MM-DD> or --demo")

    surface = fit_surface(chain, now=chain.ts)
    print(report(surface, now=now))

    if args.strict and not surface.arbitrage_report().clean:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
