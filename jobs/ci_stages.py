"""MA4 - the CI stages, as a thing you can run locally.

The plan names six checks that must pass on every commit: unit, determinism,
parity, leakage lint, AP-hack and stress. They live here rather than only in a
workflow file for one reason: **a check that exists only inside CI is a check
nobody can reproduce when it fails.** The workflow calls this module, and so can
a person, and both get the same answer.

Each stage is separate on purpose. ``pytest -q`` over everything gives one
number, and one number cannot say *which* guarantee broke. "Determinism failed"
and "the stress suite failed" call for entirely different reactions, and
collapsing them means reading a traceback to find out which one happened.

On the stage the plan calls "AP-hack"
--------------------------------------

The label is not defined anywhere else in ``megaplan.md``, and the reading taken
here is the null-test suite - ``tests/ml/test_no_signal.py`` - which asks every
fitted component what it reports when there is nothing to find. That is the
check that catches a result being manufactured rather than measured, which is
what the surrounding text treats as the anti-pattern worth a CI stage of its
own. If the intent was something else, this is the place to change it, and the
mapping is stated rather than assumed so the difference is visible.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _py() -> str:
    """The interpreter to run stages with.

    Prefers the repo's ``.venv`` when there is one, and falls back to the
    interpreter already running - which is what a CI runner has, where the
    dependencies are installed into the environment rather than into a venv.

    Returning the venv path unconditionally is what this module did first, and
    it failed on every CI run with ``No such file or directory: .venv/bin/python``
    - a module whose entire premise is that a check should be reproducible
    outside CI, written so that it only worked inside a local checkout.
    """
    for candidate in (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


@dataclass(frozen=True)
class Stage:
    """One guarantee, and the command that establishes it."""

    name: str
    argv: tuple[str, ...]
    guarantee: str
    # Infrastructure stages need Postgres and Redis. A run without them should
    # say the stage was skipped, never report it green - a skipped check that
    # renders as a pass is worse than no check at all.
    needs_infra: bool = False


STAGES: tuple[Stage, ...] = (
    Stage(
        "unit",
        ("-m", "pytest", "tests/unit", "-q", "-m", "not infra"),
        "the individual pieces do what their contracts say",
    ),
    Stage(
        "determinism",
        ("-m", "pytest", "tests/replay", "-q"),
        "the same input replays to the same output, twice",
    ),
    Stage(
        "parity",
        ("-m", "pytest", "tests/parity", "-q"),
        "the offline and online paths compute the same features",
    ),
    Stage(
        "leakage-lint",
        ("-m", "pytest", "tests/parity/test_offline_online.py", "-q", "-k", "lint or leak"),
        "no feature can reach past its context to see the future",
    ),
    Stage(
        "ap-hack",
        ("-m", "pytest", "tests/ml/test_no_signal.py", "-q"),
        "every fitted component reports nothing when there is nothing to find",
    ),
    Stage(
        "stress",
        ("-m", "pytest", "tests/stress", "-q"),
        "the ten blueprint scenarios behave as specified before implementation",
    ),
)


@dataclass
class StageResult:
    stage: Stage
    returncode: int
    seconds: float
    skipped: bool = False
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.skipped or self.returncode == 0

    def render(self) -> str:
        if self.skipped:
            mark = "SKIP"
        elif self.ok:
            mark = " ok "
        else:
            mark = "FAIL"
        return f"  [{mark}] {self.stage.name:<14} {self.seconds:>6.1f}s   {self.stage.guarantee}"


@dataclass
class CiReport:
    results: list[StageResult] = field(default_factory=list)

    @property
    def failed(self) -> list[StageResult]:
        return [r for r in self.results if not r.ok]

    @property
    def skipped(self) -> list[StageResult]:
        return [r for r in self.results if r.skipped]

    @property
    def green(self) -> bool:
        return not self.failed

    def render(self) -> str:
        lines = ["", "CI stages:"] + [r.render() for r in self.results]
        total = sum(r.seconds for r in self.results)
        lines.append(f"  {'':<7}{'total':<14} {total:>6.1f}s")
        if self.failed:
            lines.append("")
            lines.append(
                f"  {len(self.failed)} stage(s) failed: "
                + ", ".join(r.stage.name for r in self.failed)
            )
            lines.append(
                "  Each stage is a different guarantee. Read which one broke before "
                "reading the traceback - they call for different fixes."
            )
        if self.skipped:
            lines.append(
                f"  {len(self.skipped)} stage(s) skipped for missing infrastructure. "
                "Skipped is not passed."
            )
        return "\n".join(lines)


def run_stage(stage: Stage, *, have_infra: bool) -> StageResult:
    if stage.needs_infra and not have_infra:
        return StageResult(stage, 0, 0.0, skipped=True)
    started = time.monotonic()
    proc = subprocess.run([_py(), *stage.argv], cwd=ROOT, capture_output=True, text=True)
    return StageResult(
        stage,
        proc.returncode,
        time.monotonic() - started,
        output=(proc.stdout + proc.stderr),
    )


def run_all(stages: Sequence[Stage] = STAGES, *, have_infra: bool = False) -> CiReport:
    report = CiReport()
    for stage in stages:
        result = run_stage(stage, have_infra=have_infra)
        report.results.append(result)
        print(result.render(), flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MA4 CI stages.")
    parser.add_argument("--stage", action="append", help="run only these stages, by name")
    parser.add_argument("--have-infra", action="store_true", help="Postgres and Redis are up")
    parser.add_argument("--list", action="store_true", help="list the stages and exit")
    args = parser.parse_args(argv)

    if args.list:
        for s in STAGES:
            print(f"  {s.name:<14} {s.guarantee}")
        return 0

    chosen = STAGES
    if args.stage:
        by_name = {s.name: s for s in STAGES}
        unknown = [n for n in args.stage if n not in by_name]
        if unknown:
            print(f"unknown stage(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"known: {', '.join(s.name for s in STAGES)}", file=sys.stderr)
            return 2
        chosen = tuple(by_name[n] for n in args.stage)

    report = run_all(chosen, have_infra=args.have_infra)
    print(report.render())

    for result in report.failed:
        print(f"\n--- {result.stage.name} output ---")
        print(result.output.strip()[-4000:])

    return 0 if report.green else 1


if __name__ == "__main__":
    raise SystemExit(main())
