"""The lint toolchain is pinned in three places, and they must agree.

`ruff` and `black` decide whether a commit and a build pass. Their versions now
live in `pyproject.toml`, in `.pre-commit-config.yaml`, and in whatever the
developer actually installed. Three copies of a number drift, and this one
drifted already: on 2026-08-30 the working copy had ruff 0.16.4 and CI resolved
0.16.5, which turned a green `tasks.py verify` into a red pipeline nineteen
seconds later on a rule unrelated to the change.

Pinning stops CI and the working copy diverging. It does nothing about the
copies diverging from each other, which is what these tests are for: bumping one
place alone fails here rather than surfacing as a formatting fight or a red
pipeline weeks later.

The installed version is checked too. A pin that nobody has installed is a
comment.
"""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PINNED = ("ruff", "black", "mypy")
# ruff and black also run as pre-commit hooks; mypy does not.
IN_PRE_COMMIT = ("ruff", "black")


def _pyproject_pins() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for spec in data["project"]["optional-dependencies"]["dev"]:
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][^\s;]*)", spec.strip())
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def _pre_commit_revs() -> dict[str, str]:
    """Parsed with a regex rather than a YAML dependency.

    The file is a fixed shape and `repo:`/`rev:` pairs are unambiguous in it;
    adding PyYAML to the test path to read two strings would be a dependency
    bought for nothing.
    """
    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    revs: dict[str, str] = {}
    repo: str | None = None
    for line in text.splitlines():
        if (m := re.search(r"^\s*-\s*repo:\s*(\S+)", line)) is not None:
            repo = m.group(1)
        elif repo and (m := re.search(r"^\s*rev:\s*v?([0-9][^\s#]*)", line)) is not None:
            if "ruff" in repo:
                revs["ruff"] = m.group(1)
            elif repo.rstrip("/").endswith("black"):
                revs["black"] = m.group(1)
    return revs


class TestPins:
    @pytest.mark.parametrize("tool", PINNED)
    def test_every_gating_tool_is_pinned_exactly(self, tool: str) -> None:
        """A floating lower bound is right for a library this project imports
        and wrong for a tool that decides whether the build passes."""
        pins = _pyproject_pins()
        assert tool in pins, (
            f"{tool} is not pinned to an exact version in pyproject's dev extra. "
            "CI installs into a clean runner and resolves to whatever shipped "
            "most recently; a working copy keeps whatever it has."
        )

    @pytest.mark.parametrize("tool", PINNED)
    def test_the_pin_matches_what_is_installed(self, tool: str) -> None:
        """A pin nobody has installed is a comment."""
        pinned = _pyproject_pins()[tool]
        assert version(tool) == pinned, (
            f"{tool} {version(tool)} is installed but pyproject pins {pinned}. "
            f'Run `pip install -e ".[dev]"` - or, if the bump is deliberate, '
            "update pyproject and .pre-commit-config.yaml together."
        )

    @pytest.mark.parametrize("tool", IN_PRE_COMMIT)
    def test_the_pre_commit_rev_matches_the_pin(self, tool: str) -> None:
        """The hook and the lint command must be the same tool.

        They were not, and the result was that a file formatted by the hook
        failed `tasks.py lint` while one formatted by `tasks.py fmt` failed the
        hook - whichever ran last won.
        """
        pinned = _pyproject_pins()[tool]
        revs = _pre_commit_revs()
        assert tool in revs, f"no {tool} rev found in .pre-commit-config.yaml"
        assert revs[tool] == pinned, (
            f".pre-commit-config.yaml pins {tool} {revs[tool]} while pyproject "
            f"pins {pinned}. Bump both or neither."
        )

    def test_ruff_format_is_not_a_hook(self) -> None:
        """CLAUDE.md names black as the formatter. Running both means the last
        one to touch a file wins, and at v0.5.7 ruff-format rewrote PEP 695
        generic signatures into a shape nothing else here produces."""
        text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        assert not re.search(
            r"^\s*-\s*id:\s*ruff-format\s*$", text, re.MULTILINE
        ), "ruff-format is back in the hook config alongside black"
