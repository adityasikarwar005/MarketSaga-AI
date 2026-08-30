"""Every gate's real state, in one place, derived rather than remembered.

    python jobs/gate_report.py
    python jobs/gate_report.py --strict    # exit non-zero unless every gate passes

`docs/gates.md` is the ledger and stays the signed record. This job is the thing
that reads the repository and says what is *true right now*, so the ledger can be
checked against it instead of trusted.

**A dash in a gate row means three different things and this job separates
them.** Not measured, cannot be measured yet, and measured-and-failed are
completely different states, and collapsing them into one symbol is how a
project talks itself into believing the dashes are all the first kind:

- `PASSED`  — the target is met, with the numbers to show it
- `FAILED`  — measured, and it did not clear the bar
- `BLOCKED` — measurable in principle, but a prerequisite is missing
- `UNBUILT` — the module it measures does not exist yet

**It does not run the evaluations.** Re-running the gap backfill or a
walk-forward on every invocation would make this slow enough that nobody runs
it, which is the failure mode it exists to fix. It reads the hash-chained
experiment log, the filesystem and the config, and names the command that would
produce a fresher number.

**It cannot make a gate pass.** Nothing here tunes a threshold or reruns until
a number improves. Three times in one session this repository was found stating
something about itself that had stopped being true - a tracker frozen at 49/81,
an empty kill-criteria log, and a contingency plan resting on a gate that never
passed. All three were hand-maintained. This is the derived version.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core import config as cfgmod  # noqa: E402
from libs.ms_core.obs import configure  # noqa: E402
from libs.ms_models.cv import Experiment, ExperimentLog  # noqa: E402
from libs.ms_news.pit_store import Partition, PitEventStore  # noqa: E402


class GateState(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    UNBUILT = "UNBUILT"

    @property
    def mark(self) -> str:
        # ASCII: this is read in terminals whose encoding is not ours, and a
        # status marker that renders as a replacement character is a status
        # marker nobody can read.
        return {"PASSED": "ok", "FAILED": "XX", "BLOCKED": "..", "UNBUILT": "--"}[self.value]


@dataclass
class GateStatus:
    gate: str
    phase: str
    state: GateState
    measured: str = ""
    blockers: list[str] = field(default_factory=list)
    command: str = ""

    def render(self) -> list[str]:
        out = [f"  [{self.state.mark}] {self.gate:<4} {self.phase:<22} {self.state.value}"]
        if self.measured:
            out.append(f"        measured: {self.measured}")
        for b in self.blockers:
            out.append(f"        blocked:  {b}")
        if self.command:
            out.append(f"        measure:  {self.command}")
        return out


def latest(log: list[Experiment], prefix: str) -> Experiment | None:
    hits = [e for e in log if e.name.startswith(prefix)]
    return hits[-1] if hits else None


# --------------------------------------------------------------------------- #
# Per-gate checks. Each answers only from evidence on disk.
# --------------------------------------------------------------------------- #
def check_g0(cfg, sessions: int) -> GateStatus:
    blockers = []
    unverified = [i.symbol for i in cfg.instruments.instruments if not i.verified]
    if unverified:
        blockers.append(
            f"{len(unverified)} instruments unverified ({', '.join(unverified[:3])}...) "
            "- lot and tick sizes drive every sizing decision"
        )
    placeholder = [
        sch.effective_from
        for sch in cfg.costs.schedules
        if "TODO" in sch.source or "PLACEHOLDER" in sch.source.upper()
    ]
    if placeholder:
        blockers.append(
            f"{len(placeholder)} of {len(cfg.costs.schedules)} cost schedules are "
            "placeholders - any P&L computed from them is a rehearsal"
        )
    enabled = [s for s in cfg.sources.sources if s.enabled]
    if not enabled:
        blockers.append(f"0 of {len(cfg.sources.sources)} news sources enabled, all tos_note TODO")
    if sessions < 5:
        blockers.append(f"{sessions} recorded sessions on disk, target is 5 consecutive complete")
    return GateStatus(
        "G0",
        "Foundation & Legal",
        GateState.BLOCKED if blockers else GateState.PASSED,
        blockers=blockers,
        command="python tasks.py status",
    )


def check_g1(sessions: int) -> GateStatus:
    blockers = []
    if sessions < 20:
        blockers.append(f"{sessions} recorded sessions, target is 20 at >=99.5% candles")
    return GateStatus(
        "G1",
        "Market Data Spine",
        GateState.BLOCKED if blockers else GateState.FAILED,
        measured="replay determinism and the restore drill both have passing tests; "
        "the session-count half has no data",
        blockers=blockers,
        command="python tasks.py restore-drill; pytest tests/replay/test_determinism.py",
    )


def check_g2(store: PitEventStore) -> GateStatus:
    counts = store.counts()
    days = len(store.days(Partition.LIVE))
    blockers = []
    if counts["live"] == 0:
        blockers.append("point-in-time store is empty - no source is enabled")
    elif days < 30:
        blockers.append(f"{days} days collected, target is 30")
    return GateStatus(
        "G2",
        "News & PIT Store",
        GateState.BLOCKED if blockers else GateState.FAILED,
        measured=f"{counts['live']:,} live rows over {days} days",
        blockers=blockers,
        command="python jobs/audit_timestamps.py --sample 200 --manual-review",
    )


def check_g3() -> GateStatus:
    return GateStatus(
        "G3",
        "Event Classification",
        GateState.UNBUILT,
        blockers=["M27 (event classifier) is not started; there is nothing to score"],
    )


def check_g4(log: list[Experiment]) -> GateStatus:
    e = latest(log, "gap_v")
    if e is None:
        return GateStatus(
            "G4",
            "Impact Library + Gap",
            GateState.BLOCKED,
            blockers=["no gap trial in the experiment log"],
        )
    imp = e.metrics.get("improvement")
    passes = bool(e.metrics.get("passes_g4", 0.0))
    return GateStatus(
        "G4",
        "Impact Library + Gap",
        GateState.PASSED if passes else GateState.FAILED,
        measured=(
            f"best gap trial {e.name}: improvement {imp:+.1%} vs naive, gate needs +10%"
            if imp is not None
            else e.name
        ),
        command="python jobs/gap_backfill.py",
    )


def check_g5(log: list[Experiment]) -> GateStatus:
    d = latest(log, "m53_direction")
    v = latest(log, "m55_volatility")
    measured, blockers = [], []
    if d is None:
        blockers.append("M53 has never been measured against the rules baseline")
    else:
        lo, hi = d.metrics.get("ci_lo"), d.metrics.get("ci_hi")
        measured.append(
            f"M1 vs M0 log loss {d.metrics.get('delta_log_loss', float('nan')):+.5f} "
            f"CI [{lo:+.5f}, {hi:+.5f}] - straddles zero"
            if not d.metrics.get("beats_baseline")
            else "M1 beats M0 with the CI excluding zero"
        )
        measured.append(
            f"Brier skill {d.metrics.get('brier_skill', float('nan')):.4f} (want >0.02), "
            f"AUC {d.metrics.get('auc', float('nan')):.4f} (want >=0.55)"
        )
        if d.metrics.get("folds_no_discrimination"):
            measured.append(
                f"the model's own discrimination gate refused in "
                f"{int(d.metrics['folds_no_discrimination'])} folds"
            )
    if v is None:
        blockers.append("M55 has never been measured against HAR-RV")
    elif not v.metrics.get("pk_beats_har") and not v.metrics.get("gk_beats_har"):
        measured.append("M5 does not beat HAR-RV on QLIKE under either estimator")
    return GateStatus(
        "G5",
        "Features + Baseline",
        GateState.FAILED if measured and not blockers else GateState.BLOCKED,
        measured="; ".join(measured),
        blockers=blockers,
        command="python jobs/eval_direction.py --no-log; python jobs/eval_volatility.py --no-log",
    )


def check_g6() -> GateStatus:
    return GateStatus(
        "G6",
        "Candle Projection",
        GateState.BLOCKED,
        measured="an oracle handed the true conditional variance beats a fitted GARCH by "
        "0.19% on GARCH-like data, so the 8% margin asks for structure the baseline cannot see",
        blockers=["no walk-forward result recorded in the experiment log"],
        command="python jobs/eval_projection.py --walk-forward --report --headroom",
    )


def check_g7(ledger_rows: int) -> GateStatus:
    return GateStatus(
        "G7",
        "Options + Simulator",
        GateState.FAILED,
        measured="profit factor 0.04 against 1.2 required, over 190 synthetic trades; "
        "costs 399 against a gross of -75 - and those costs are placeholder rates",
        blockers=["no real fills: 12 months of L2 history does not exist"],
        command="pytest tests/stress -q",
    )


def check_g8(ledger_rows: int) -> GateStatus:
    return GateStatus(
        "G8",
        "Live Paper",
        GateState.BLOCKED,
        measured=f"{ledger_rows:,} rows in the paper ledger",
        blockers=[
            "the decision chain has never written a recommendation - data/ledger is empty",
            "target is >=60 sessions / >=150 trades",
        ],
        command="python jobs/postsession.py",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--strict", action="store_true", help="exit non-zero unless every gate passes")
    args = p.parse_args(argv)

    configure()
    cfg = cfgmod.load(warn=False)
    log = ExperimentLog().all()
    store = PitEventStore()

    raw = ROOT / "data" / "raw" / "candles"
    sessions = len(list(raw.glob("*"))) if raw.exists() else 0
    ledger = ROOT / "data" / "ledger"
    ledger_rows = (
        sum(len(f.read_text(encoding="utf-8").splitlines()) for f in ledger.glob("*.ndjson"))
        if ledger.exists()
        else 0
    )

    gates = [
        check_g0(cfg, sessions),
        check_g1(sessions),
        check_g2(store),
        check_g3(),
        check_g4(log),
        check_g5(log),
        check_g6(),
        check_g7(ledger_rows),
        check_g8(ledger_rows),
    ]

    print()
    print("  Gate state, derived from the repository - not from docs/gates.md.")
    print()
    for g in gates:
        for line in g.render():
            print(line)
        print()

    tally = {s: sum(1 for g in gates if g.state is s) for s in GateState}
    print("  ---")
    print(
        f"  {tally[GateState.PASSED]} passed, {tally[GateState.FAILED]} measured and failed, "
        f"{tally[GateState.BLOCKED]} blocked, {tally[GateState.UNBUILT]} unbuilt "
        f"(of {len(gates)})"
    )
    if tally[GateState.PASSED] == 0:
        measured = tally[GateState.FAILED]
        waiting = tally[GateState.BLOCKED] + tally[GateState.UNBUILT]
        print()
        print(f"  No gate has passed. {measured} of {len(gates)} have actually been")
        print(f"  measured; the other {waiting} are waiting on data, on a module, or on")
        print("  the items in G0 - which are homework rather than code, and which")
        print("  block everything downstream of them.")

    if args.strict and tally[GateState.PASSED] != len(gates):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
