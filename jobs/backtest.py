"""Run the simulator and report what it found.

    python jobs/backtest.py --demo
    python jobs/backtest.py --start 2026-01-01 --end 2026-06-30 --delay 3 --costs full

The only number anyone is allowed to quote about profitability comes from here,
and the report is written to make that number hard to over-read: the gates it
does or does not clear are printed next to it, and so is every reason the result
might be an artefact rather than an edge.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.obs import configure
from libs.ms_sim.fills import Quote
from libs.ms_sim.simulator import (
    MIN_DISCOVERY_DELAY,
    Intent,
    LatencyModel,
    SimConfig,
    SimEvent,
    Simulator,
)

# Gate G7, from the plan. Printed whether or not it passes.
G7_MIN_PROFIT_FACTOR = 1.2
G7_MAX_DRAWDOWN = 0.12
G7_MIN_MONTHS = 12


def demo_world(n_events: int, seed: int):
    """A synthetic world with a known, small edge and realistic costs.

    Synthetic on purpose: this job exists before there is intraday data, and a
    demo that pretends otherwise would be the first lie in a module built to
    prevent them. Every number below is a property of this generator.
    """
    rng = random.Random(seed)
    t0 = datetime(2026, 1, 5, 4, 0, tzinfo=UTC)
    classes = ["geo.military", "macro.rate_decision", "corp.earnings", "noise.rumour"]
    edge = {
        "geo.military": 0.58,
        "macro.rate_decision": 0.54,
        "corp.earnings": 0.51,
        "noise.rumour": 0.50,
    }

    events, paths = [], {}
    for _i in range(n_events):
        t = t0 + timedelta(minutes=rng.randint(0, 60 * 24 * 300))
        klass = rng.choice(classes)
        sym = f"NIFTY{rng.choice([23800, 24000, 24200])}CE"
        key = (sym, t)
        base = 100.0 + rng.gauss(0, 5)
        drift = 1.0 if rng.random() < edge[klass] else -1.0
        paths[key] = (base, drift * abs(rng.gauss(0, 3.0)))
        events.append(SimEvent(t_event=t, event_class=klass, symbol=sym, payload={"base": base}))

    def book_at(symbol: str, at: datetime):
        # A book that drifts with the event that is nearest in the past.
        best = None
        for (sym, t), (base, move) in paths.items():
            if sym != symbol or t > at:
                continue
            if best is None or t > best[0]:
                best = (t, base, move)
        if best is None:
            return None
        _, base, move = best
        elapsed = min(30.0, (at - best[0]).total_seconds() / 60.0)
        price = base + move * (elapsed / 30.0)
        return Quote(bid=max(0.5, price - 0.6), ask=price + 0.6, bid_size=500, ask_size=500)

    def bars_after(symbol: str, ts: datetime):
        q = book_at(symbol, ts)
        if q is None:
            return
        price = q.mid
        for i in range(1, 31):
            at = ts + timedelta(minutes=i)
            nxt = book_at(symbol, at)
            if nxt is None:
                return
            hi, lo = max(price, nxt.mid) + 0.4, min(price, nxt.mid) - 0.4
            yield at, hi, lo
            price = nxt.mid

    return events, book_at, bars_after


def report(blotter, *, months: float, latency: LatencyModel) -> None:
    print()
    print(blotter.render())
    print()
    print("  GATE G7")
    pf = blotter.profit_factor
    checks = [
        (f"profit factor {pf:.2f} >= {G7_MIN_PROFIT_FACTOR}", pf >= G7_MIN_PROFIT_FACTOR),
        (
            f"max drawdown {blotter.max_drawdown:.1%} <= {G7_MAX_DRAWDOWN:.0%}",
            blotter.max_drawdown <= G7_MAX_DRAWDOWN,
        ),
        (f"{months:.1f} months >= {G7_MIN_MONTHS}", months >= G7_MIN_MONTHS),
    ]
    for label, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}]  {label}")
    print(f"  -> Gate G7 {'PASSES' if all(o for _, o in checks) else 'does not pass'}")

    print()
    print("  Before quoting any of this:")
    if not latency.measured:
        print("    - the latency model is UNMEASURED. Feed it real decision timings.")
    if blotter.tie_broken_share > 0.2:
        print(
            f"    - {blotter.tie_broken_share:.0%} of exits came from the tie rule, " "not the data"
        )
    print(
        f"    - the discovery delay was {MIN_DISCOVERY_DELAY.total_seconds():.0f}s, "
        "the floor. A real wire is slower."
    )
    print("    - this is one path. A single equity curve cannot say whether it was luck;")
    print("      that is what combinatorial purged CV and the deflated Sharpe are for.")


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Event-driven backtest (M82)")
    p.add_argument("--demo", action="store_true", help="synthetic world with a known edge")
    p.add_argument("--events", type=int, default=1200)
    p.add_argument("--delay", type=float, default=3.0, help="discovery delay in seconds")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    if not args.demo:
        print()
        print("  No recorded intraday data yet, so there is nothing real to replay.")
        print("  Run with --demo to exercise the machinery on a synthetic world")
        print("  whose edge is known, or connect a broker feed and record sessions.")
        return 1

    events, book_at, bars_after = demo_world(args.events, args.seed)
    config = SimConfig(
        discovery_delay=timedelta(seconds=args.delay),
        latency=LatencyModel(seed=args.seed),
    )

    def strategy(event: SimEvent, quote: Quote, at: datetime) -> Intent | None:
        if event.event_class == "noise.rumour":
            return None
        mid = quote.mid
        return Intent(symbol=event.symbol, is_long=True, qty=1, target=mid * 1.03, stop=mid * 0.985)

    def cost_of(intent: Intent, price: float) -> float:
        # Four orders' worth of charges plus a crossed spread, per lot.
        return 0.9 * intent.qty + 1.2 * intent.qty

    sim = Simulator(config, book_at=book_at, bars_after=bars_after, cost_of=cost_of)
    blotter = sim.run(events, strategy)

    span_days = (max(e.t_event for e in events) - min(e.t_event for e in events)).days
    report(blotter, months=span_days / 30.4, latency=config.latency)
    print()
    print(f"  {sim.skipped_no_book} events skipped for want of a book")
    return 0


if __name__ == "__main__":
    sys.exit(main())
