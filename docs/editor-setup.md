# Editor setup

## Why `pyrightconfig.json` exists

The editor's type checker (Pylance / Pyright) is a **different** checker from the
one CI runs. Without a config it guesses an interpreter and a Python version,
and on this project it guessed wrong: it reported `StrEnum` as an unknown import
symbol — it had landed on a pre-3.11 standard library — and then cascaded into
parse errors on `libs/ms_risk/gate.py`, a file that compiles, imports, and passes
ruff, black and mypy cleanly.

Red squiggles on correct code are worse than no checking at all. They train you
to ignore the ones that matter, which is the same failure mode as a suite with a
flaky test in it.

## What the config found once it was pointed at the right interpreter

Three real defects, none of which mypy was looking at:

**`jobs/research_smoke.py` passed `None` where a `Regime` was required.**
`regimes.get(event_day)` returns `Regime | None`, and passing it explicitly
overrode `observe_event`'s `regime=Regime.NORMAL` default — so any session
`classify_regimes` could not label picked a baseline keyed on nothing. Fixed
with an explicit `.get(day, Regime.NORMAL)`.

**`read_stream` threw away the type it was handed.** Its signature took
`type[Msg]` and returned `Iterator[Msg]`, so reading a candle stream produced an
iterator of `Msg` and every caller knew less than it had a right to. Six
`# type: ignore[arg-type]` comments existed *only* to paper over that. Making it
generic — `def read_stream[T: Msg](path, cls: type[T]) -> Iterator[T]` — removed
all six.

**`jobs/research_smoke.py` bound `leak` to two different types** in one scope: a
`LeakReport` from the feature check, then a `str` from the split audit. Not a
bug that day; one after the next edit.

The first two were reachable because **mypy's scope was `files = ["libs"]`** —
`jobs/` and `tasks.py` were never checked at all. That is now fixed in
`pyproject.toml`, with `disallow_untyped_defs` relaxed for scripts: type errors
in a job matter, a missing annotation on a throwaway helper does not.

## Current state, measured

```
$ python tasks.py typecheck        # mypy, strict on libs
Success: no issues found in 63 source files

$ npx pyright                      # what the editor sees
libs/ 0 · jobs/ 0 · tasks.py 0 · tests/ 55
```

**Production code is clean under both checkers.** The remaining diagnostics are
all in `tests/`, and most of them are one thing:

```python
Candle(instrument_id=1, ts=..., o=100, h=101, low=99, c=100)
                                            # ^^^ "No parameter named 'low'"
```

`Candle` declares `low: Price = Field(alias="l")` with `populate_by_name=True`,
so both `low=` and `l=` work at runtime. Pyright's basic mode reads the alias and
does not see the `populate_by_name` escape hatch. mypy, with the pydantic plugin,
does.

This is a tooling gap, not a defect. Renaming the field to satisfy a checker
would be the tail wagging the dog — the alias exists because the recorded format
uses `l` and the code should not, and `l` is a name ruff would reject anyway.

The rest are `dict` invariance in test helpers (`dict[str, float]` where
`dict[str, float | None]` is wanted), which are genuine and worth fixing as those
files are next touched. They are listed here rather than suppressed, because a
blanket `reportArgumentType: none` on tests would also hide the real ones.

## Which checker is authoritative

**mypy is.** It runs in CI, configured under `[tool.mypy]`, strict on `libs/`.
`pyrightconfig.json` is set to `basic` and exists only so the editor agrees with
mypy rather than arguing with it. If the two disagree, mypy wins and the pyright
config is what gets adjusted.

## Why `pythonVersion` says 3.12 when `.venv` is 3.13

`requires-python = ">=3.12"`. The editor should flag syntax that would break on
the **oldest supported** version, not on whichever interpreter happens to be
installed here. It matches `[tool.mypy] python_version` for the same reason.

## `.vscode/settings.json` is not committed

It is gitignored, deliberately — editor preferences are personal. If you want the
same setup:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.extraPaths": ["${workspaceFolder}"],
  "files.eol": "\r\n"
}
```

The interpreter path is the part that matters. Everything else is taste.

## If squiggles come back

Check the interpreter first — bottom-left of the VS Code status bar. It must be
the one in `.venv`, not a system Python. Then:

```bash
python tasks.py typecheck        # mypy, the authority
npx pyright                      # what the editor sees
```

If mypy is clean and pyright is not, the disagreement is a config problem and
belongs in this file. If mypy is unhappy, the code is.
