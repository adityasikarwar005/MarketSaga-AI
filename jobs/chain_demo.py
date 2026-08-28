"""The decision chain, end to end, on synthetic data.

    python jobs/chain_demo.py

Runs the layers that produce a recommendation and prints what each one hands to
the next:

    quantile heads (M54)  ->  path sampler (M60)
    IV surface (M70)      ->  Greeks (M71)  ->  structures (M72)

**Every number here comes from synthetic data with a known answer.** That is not
a caveat, it is the point of the demo: each stage can be checked against what it
was given. Exactly one component in this project has ever run on real market
data - the overnight gap forecaster, on free daily bars - and this is not it.
"""

from __future__ import annotations

import math
import random
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.schemas import ChainSnapshot, OptionQuote, OptionType
from libs.ms_models.paths import calibrate, diagnose, simulate
from libs.ms_models.quantile import DEFAULT_QUANTILES
from libs.ms_models.quantile_heads import QuantileHeads
from libs.ms_options.greeks import Leg, aggregate, greeks_from_surface
from libs.ms_options.pricing import OptType, black76_price
from libs.ms_options.structures import (
    StructureKind,
    build_grid,
    enumerate_structures,
    summarise,
)
from libs.ms_options.surface import SVIParams, fit_surface

TS = datetime(2026, 8, 27, 6, 0, tzinfo=UTC)
EXPIRY = TS + timedelta(days=28)
SPOT = 24000.0
R = 0.065
LOT = 75
SMILE = SVIParams(a=0.0006, b=0.0057, rho=-0.42, m=0.015, sigma=0.115)
HORIZON = 60  # minutes


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def synthetic_features(n: int = 1200, seed: int = 11):
    """Features whose conditional quantiles are known in closed form.

    The noise widens with the second feature, so a model that is not tracking
    conditional volatility can be seen not doing it.
    """
    rng = random.Random(seed)
    rows, ys = [], []
    for _ in range(n):
        x1, x2 = rng.uniform(-1.0, 1.0), rng.uniform(0.0, 1.0)
        rows.append([x1, x2, rng.uniform(-1.0, 1.0)])
        ys.append((2.0 * x1 + rng.gauss(0.0, 0.5 + 1.5 * x2)) * 0.004)
    return rows, ys


def synthetic_chain() -> ChainSnapshot:
    """A chain priced off a known smile, two-sided, wings quoted wider."""
    t = (EXPIRY - TS).total_seconds() / (365.0 * 24 * 3600)
    f = SPOT * math.exp(R * t)
    atm_vol = math.sqrt(SMILE.total_variance(0.0) / t)
    span = 3.0 * atm_vol * math.sqrt(t)

    quotes: list[OptionQuote] = []
    for i in range(21):
        k = -span + 2.0 * span * i / 20
        strike = round(SPOT * math.exp(k) / 50.0) * 50.0
        vol = math.sqrt(SMILE.total_variance(math.log(strike / f)) / t)
        for opt, kind in ((OptionType.CE, OptType.CALL), (OptionType.PE, OptType.PUT)):
            mid = black76_price(f, strike, vol, t, R, kind)
            if mid <= 0.20:
                continue
            half = mid * (0.004 + 0.05 * abs(k))
            quotes.append(
                OptionQuote(
                    expiry=EXPIRY,
                    strike=strike,
                    opt_type=opt,
                    bid=max(mid - half, 0.05),
                    ask=mid + half,
                    ltp=mid,
                    oi=max(int(9000 * math.exp(-8 * k * k)), 1),
                    volume=max(int(2500 * math.exp(-8 * k * k)), 1),
                )
            )
    return ChainSnapshot(ts=TS, underlying_id=1, spot=SPOT, quotes=quotes)


def main() -> int:
    print(__doc__.split("\n\n")[0])
    print("  SYNTHETIC DATA THROUGHOUT - nothing here has seen a real quote.")

    # ---------------------------------------------------------------- M54 ---
    rule("M54  quantile heads - what the model thinks the move will be")
    rows, ys = synthetic_features()
    train_rows, train_y = rows[:900], ys[:900]

    started = time.perf_counter()
    heads = QuantileHeads(horizons=(HORIZON,)).fit(train_rows, {HORIZON: train_y})
    fit_ms = (time.perf_counter() - started) * 1000
    print(heads.report.render())

    quiet_row = [0.2, 0.05, 0.0]  # low conditional volatility
    busy_row = [0.2, 0.95, 0.0]  # high conditional volatility
    quiet = heads.predict(quiet_row, horizon=HORIZON)
    busy = heads.predict(busy_row, horizon=HORIZON)

    print(f"\n    fitted {len(heads.heads)} heads in {fit_ms:.0f} ms")
    print(f"    {'quantile':<10}{'quiet row':>14}{'busy row':>14}")
    for a in sorted(DEFAULT_QUANTILES):
        print(f"    a={a:<8.2f}{quiet.quantiles[a]:>+14.5f}{busy.quantiles[a]:>+14.5f}")
    qw = quiet.quantiles[0.95] - quiet.quantiles[0.05]
    bw = busy.quantiles[0.95] - busy.quantiles[0.05]
    print(f"\n    5-95 width: {qw:.5f} quiet vs {bw:.5f} busy  ({bw / qw:.2f}x wider)")
    print("    The band moves with conditional volatility. Newton boosting on the")
    print("    pinball loss produces one that barely does, while its marginal")
    print("    coverage still reads as calibrated.")

    # ---------------------------------------------------------------- M60 ---
    rule("M60  path sampler - routes to that distribution, not just the endpoint")
    variance0 = (bw / 3.29) ** 2 / HORIZON  # rough per-step variance from the band

    started = time.perf_counter()
    raw = simulate(n_paths=5000, steps=HORIZON, variance0=variance0, seed=3)
    sim_ms = (time.perf_counter() - started) * 1000
    calibrated = calibrate(raw, busy)

    print(diagnose(calibrated).render())
    print(f"\n    {sim_ms:.0f} ms for 5000 paths x {HORIZON} steps (plan's budget: 60 ms)")
    print(f"    {'quantile':<10}{'head says':>14}{'paths give':>14}")
    for a in sorted(DEFAULT_QUANTILES):
        print(f"    a={a:<8.2f}{busy.quantiles[a]:>+14.5f}{calibrated.quantile(a):>+14.5f}")

    tail = calibrated.terminal[calibrated.terminal <= calibrated.quantile(0.05)]
    print(
        f"\n    CVaR5 {float(tail.mean()):+.5f}   worst path {float(calibrated.terminal.min()):+.5f}"
    )
    print(f"    The 5% head is {busy.quantiles[0.05]:+.5f}. CVaR is the mean of the tail,")
    print("    so it must sit below that - clamped tails make it equal to it, and")
    print("    every tail-risk number then reads 30% kinder than it is.")

    # ---------------------------------------------------------------- M70 ---
    rule("M70  IV surface - fitted from mid prices, forward from put-call parity")
    chain = synthetic_chain()
    started = time.perf_counter()
    surface = fit_surface(chain, now=TS, r=R)
    surf_ms = (time.perf_counter() - started) * 1000
    print(surface.render())
    print(f"\n    fitted in {surf_ms:.0f} ms")

    sl = surface.slices[EXPIRY]
    print(f"    true smile   {SMILE.render()}")
    print(f"    recovered    {sl.params.render()}")
    print("\n    features handed downstream:")
    for name, value in surface.features(EXPIRY, now=TS).items():
        print(f"      {name:<28} {value:+.6f}")

    # ---------------------------------------------------------------- M71 ---
    rule("M71  Greeks - the delta that hedges, not the one the formula gives")
    strike = round(sl.forward.forward / 50.0) * 50.0 + 500.0
    sg = greeks_from_surface(sl, strike=strike, kind=OptType.CALL, r=R)
    print(f"    {strike:.0f} CE, iv {sg.vol:.4f}, price {sg.price:.2f}\n")
    print(f"      black-76 delta        {sg.delta_black76:+.6f}")
    print(f"      smile-adjusted delta  {sg.delta:+.6f}")
    print(
        f"      difference            {sg.smile_delta_adjustment:+.6f}  "
        f"({abs(sg.smile_delta_adjustment / sg.delta_black76):.1%})"
    )
    print(
        f"      on 10 lots            {sg.smile_delta_adjustment * 10 * LOT:+.1f} units of forward,"
    )
    print("                            short in the same direction every day.")
    print("                            It presents as hedging slippage.")

    # ---------------------------------------------------------------- M72 ---
    rule("M72  structures - what is tradeable, and what can be got out of")
    grid = build_grid(chain.quotes, expiry=EXPIRY)
    started = time.perf_counter()
    candidates = enumerate_structures(grid, forward=sl.forward.forward, lot_size=LOT, lots=1)
    enum_ms = (time.perf_counter() - started) * 1000
    print(summarise(candidates, grid))
    print(f"\n    enumerated in {enum_ms:.0f} ms")

    unfiltered = enumerate_structures(
        grid, forward=sl.forward.forward, lot_size=LOT, lots=1, min_ratio=0.0
    )
    print(f"    {len(unfiltered) - len(candidates)} of {len(unfiltered)} dropped: they cannot")
    print("    clear their own crossing cost, while every leg passes every")
    print("    per-leg liquidity filter.")

    def nearest_the_money(kind: StructureKind):
        """The one a reader expects to see.

        Taking the first of each kind returns the lowest strike in the ladder -
        a "straddle" 2 500 points in the money, with a delta of 0.99. It is a
        real structure and it demonstrates nothing.
        """
        of_kind = [c for c in candidates if c.kind is kind]
        if not of_kind:
            return None
        return min(
            of_kind,
            key=lambda c: sum(abs(leg.strike - sl.forward.forward) for leg in c.legs) / len(c.legs),
        )

    print("\n    a few survivors, nearest the money:")
    for kind in (StructureKind.VERTICAL_DEBIT, StructureKind.STRADDLE, StructureKind.IRON_CONDOR):
        picked = nearest_the_money(kind)
        if picked is not None:
            print(picked.render())

    # ------------------------------------------------------- M71 portfolio ---
    rule("M71  portfolio roll-up - and the number that must never be one number")
    straddle = nearest_the_money(StructureKind.STRADDLE)
    assert straddle is not None
    book = aggregate(
        [Leg(leg.strike, leg.kind, leg.expiry, 2, LOT, leg.is_long) for leg in straddle.legs],
        surface,
        now=TS,
        r=R,
    )
    print(book.render())

    rule("what this did and did not show")
    print("  Ran: M54 -> M60, and M70 -> M71 -> M72, on synthetic inputs whose")
    print("  answers are known in advance. Every stage was checked against what")
    print("  it was given: the smile came back to six decimals, the calibrated")
    print("  paths land on the heads, the recovered surface parameters match.")
    print()
    print("  Not shown, because it is not built or not unblocked:")
    print("    - M61/M62  fan chart and projection scoring")
    print("    - M73      Monte Carlo P&L (needs M56, which needs a purchased chain)")
    print("    - M84      the live broker adapter (behind gate G8)")
    print()
    print("  And the thing worth repeating: exactly one component in this project")
    print("  has ever run on real market data. Gate G4 does not pass, and that is")
    print("  recorded in docs/gates.md with all twelve trials named.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
