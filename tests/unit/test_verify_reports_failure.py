"""`tasks.py verify` could not fail, and had never been able to.

CLAUDE.md calls it "what CI runs. A module is not done until this is green."
It was green unconditionally. Every stage ran with `check=False`, the return
codes were dropped, and the dispatcher called the command without reading what
it returned - so lint failures, type errors and a red test suite all exited 0.

Only GitHub Actions could go red, because it runs the same tools as separate
steps. That is what the pipeline-red-for-four-pushes note in CLAUDE.md is
describing from the other end, and it re-frames the ruff 0.16.4/0.16.5 incident
in test_toolchain_pins.py: the version drift was real, but a matching ruff
would not have saved that push either.

These tests drive the stage helpers with a stubbed `run`, so they assert the
wiring rather than the tools. Shelling out to a real failing lint would test
ruff, take a minute, and still not prove the exit code survives the dispatcher.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

import tasks

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("lint", "typecheck", "test")


@pytest.fixture
def failing_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "run", lambda *a, **k: 1)


@pytest.fixture
def passing_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "run", lambda *a, **k: 0)


class TestStagesReportTheirResult:
    @pytest.mark.parametrize("stage", STAGES)
    def test_a_failing_tool_makes_the_stage_nonzero(self, stage: str, failing_run: None) -> None:
        assert getattr(tasks, stage)() != 0, (
            f"tasks.{stage}() returned 0 while its tool exited 1. The stage is "
            "swallowing the failure, which is how verify came to be green "
            "unconditionally."
        )

    @pytest.mark.parametrize("stage", STAGES)
    def test_a_passing_tool_makes_the_stage_zero(self, stage: str, passing_run: None) -> None:
        assert getattr(tasks, stage)() == 0


class TestVerify:
    def test_verify_fails_when_a_stage_fails(self, failing_run: None) -> None:
        assert tasks.verify() == 1

    def test_verify_passes_when_every_stage_passes(self, passing_run: None) -> None:
        assert tasks.verify() == 0

    def test_every_stage_runs_even_after_one_fails(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Stopping at the first failure means one fix per round trip.

        `check=False` was always the right call; discarding the result was not.
        """
        seen: list[str] = []

        def spy(*args: Any, **kwargs: Any) -> int:
            seen.append(next((a for a in args if a in ("ruff", "black", "mypy", "pytest")), "?"))
            return 1

        monkeypatch.setattr(tasks, "run", spy)
        assert tasks.verify() == 1
        assert {"ruff", "mypy", "pytest"} <= set(
            seen
        ), f"verify stopped early: it only reached {seen}."
        assert "lint, typecheck, test" in capsys.readouterr().out


class TestDispatcher:
    def test_the_entry_point_exits_with_the_command_result(self) -> None:
        """A stage can return 1 and still exit 0 if the caller ignores it.

        That is what happened, so the wiring is asserted on the source rather
        than trusted. The first version of this test looked for any `sys.exit`
        inside the `__main__` block and passed against the broken dispatcher,
        because the usage error above it already calls `sys.exit(1)`. It has to
        be the dispatch call specifically that is wrapped.
        """
        tree = ast.parse((ROOT / "tasks.py").read_text(encoding="utf-8"))
        main = next(
            n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.dump(n.test)
        )
        wrapped = [
            call
            for call in ast.walk(main)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "exit"
            and any(
                isinstance(inner, ast.Subscript)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "COMMANDS"
                for arg in call.args
                for inner in ast.walk(arg)
            )
        ]
        assert wrapped, (
            "the __main__ block dispatches into COMMANDS without sys.exit() "
            "around the call, so every exit code the stages return is discarded "
            "at the last step - which is the bug this file exists for."
        )
