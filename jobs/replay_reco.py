"""Reconstruct a decision from its stored record.

    python jobs/replay_reco.py --list
    python jobs/replay_reco.py --reco-id reco-0001

Six months from now the question will be why the system lost money on some class
of event, and the only way to answer it is to read what the system believed when
it acted. This is the tool that reads it back.

It deliberately reconstructs from the **stored file** and nothing else - not from
the models, not from the feature store, not from the database. If a record cannot
explain itself without those, it is not a record, it is a pointer to state that
has since changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.obs import configure
from libs.ms_risk.recommendation import RecommendationStore


def show(rec: dict[str, Any]) -> None:
    t, f = rec["trigger"], rec["forecast"]
    print()
    print(f"  {rec['reco_id']}   [{rec['mode']}]  {rec['action']}")
    print(f"  emitted {rec['ts_emitted']}   trace {rec['trace_id']}")

    print()
    print("  TRIGGER")
    print(f"    {t['event_class']}  severity {t['severity']}  novelty {t['novelty']}")
    print(f"    credibility {t['credibility']}  sources {', '.join(t['sources']) or 'none'}")
    print(f"    seen {t['seconds_since_event']}s after the event")

    print()
    print("  FORECAST")
    ci = f"  CI {f['p_up_ci']}" if f.get("p_up_ci") else "  (no interval - point estimate)"
    print(f"    p_up {f['p_up']}{ci}   horizon {f['horizon_min']}m")
    if f.get("quantiles"):
        qs = "  ".join(f"{k}={v:+.4f}" for k, v in sorted(f["quantiles"].items()))
        print(f"    {qs}")
    if f.get("sigma_forecast") and f.get("implied_move"):
        ratio = f["sigma_forecast"] / f["implied_move"]
        verdict = "worth buying" if ratio > 1 else "already priced - nothing to buy"
        print(f"    forecast {f['sigma_forecast']:.4f} vs implied {f['implied_move']:.4f}")
        print(f"    edge ratio {ratio:.2f}x  ->  {verdict}")

    if rec.get("structure"):
        s = rec["structure"]
        print()
        print("  STRUCTURE")
        print(f"    {s['type']}   debit {s['net_debit']}  breakeven {s['breakeven']}")
        for leg in s["legs"]:
            print(f"      {leg['side']:<4} {leg['qty']} x {leg['strike']} {leg['opt']}")
        print(f"    max profit {s['max_profit']}   max loss {s['max_loss']}")

    if rec.get("sizing"):
        z = rec["sizing"]
        print()
        print("  SIZING")
        print(
            f"    Kelly {z['kelly_frac']:.4f} -> used {z['used_frac']:.4f} "
            f"-> {z['lots']} lot(s), {z['capital_at_risk']:,.0f} at risk"
        )

    if rec.get("exits"):
        e = rec["exits"]
        print()
        print("  EXITS")
        print(f"    target {e['target']}   stop {e['stop']}   time {e['time_stop']}")
        if e.get("vol_stop"):
            print(f"    vol: {e['vol_stop']}")

    print()
    print("  WHY  (ordered - each reason builds on the one before)")
    for i, why in enumerate(rec["explanation"], 1):
        print(f"    {i}. {why}")

    if rec.get("rejected_because"):
        print()
        print("  BLOCKED BY")
        for r in rec["rejected_because"]:
            print(f"    - {r}")

    print()
    print(f"  gates passed  : {', '.join(rec['gates_passed']) or 'none recorded'}")
    print(f"  model versions: {rec['model_versions']}")
    print(f"  kill level    : {rec['kill_level']}")
    print()
    print("  Reconstructed entirely from the stored file. No model, no feature")
    print("  store, no database was consulted - a record that needs them to")
    print("  explain itself is a pointer to state that has since changed.")


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Replay a stored recommendation (M77)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--reco-id", help="the record to reconstruct")
    g.add_argument("--list", action="store_true", help="what is stored")
    args = p.parse_args()

    store = RecommendationStore()
    if args.list:
        ids = store.ids()
        if not ids:
            print(f"  nothing stored under {store.root}")
            return 0
        print(f"  {len(ids)} recommendation(s) in {store.root}")
        for i in ids:
            print(f"    {i}")
        return 0

    try:
        show(store.load(args.reco_id))
    except FileNotFoundError as exc:
        print(f"  {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
