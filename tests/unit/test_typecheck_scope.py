"""Green locally, red on the runner - the failure this test exists to prevent.

CI's typecheck job installs only ``.[dev]``. A developer machine has more than
that installed, so an import of an optional dependency typechecks fine here and
fails on the runner with ``Cannot find implementation or library stub``. The
symptom is a red pipeline for a change that passed every local check, which
teaches people to stop trusting local checks.

This has now happened twice in this repository: once for missing type stubs, and
once for ``huggingface_hub`` after the deploy job was added. Both times the fix
was a one-line entry in the mypy override list, and both times it cost a push
and a wait to find out.

So the rule is asserted here instead: every third-party module imported inside
mypy's own scope must be resolvable on a ``.[dev]``-only install - either
because it is a dev dependency, or because it carries an explicit
``ignore_missing_imports`` override, or because the import line says
``# type: ignore`` in place.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Stub-only distributions: the package on PyPI is named for the stubs it ships,
# and mypy resolves the module underneath it.
STUB_PACKAGES = {"types-pyyaml": "yaml"}

# Imported by the test runner itself rather than by the code under test.
ALWAYS_PRESENT = {"pytest", "typing_extensions"}

LOCAL_ROOTS = {"libs", "jobs", "services", "tasks", "tests"}


def _dist_names(specs: list[str]) -> set[str]:
    """`redis>=5.0` and `psycopg[binary]>=3.1` both reduce to their bare name."""
    out = set()
    for spec in specs:
        name = spec.split(";")[0].strip()
        for sep in (">", "<", "=", "!", "~", "["):
            name = name.split(sep)[0]
        out.add(name.strip().lower())
    return out


def _import_lines_with_inline_ignore(tree: ast.AST, source: list[str]) -> set[int]:
    """Line numbers whose import already carries an inline mypy suppression."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            text = " ".join(source[node.lineno - 1 : end])
            if "type: ignore" in text:
                out.add(node.lineno)
    return out


def _third_party_imports(path: Path) -> dict[str, int]:
    """Top-level module name -> first line, skipping inline-suppressed imports."""
    source = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(source))
    suppressed = _import_lines_with_inline_ignore(tree, source)

    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        if node.lineno in suppressed:
            continue
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        for n in names:
            found.setdefault(n, node.lineno)
    return found


def test_every_import_in_mypy_scope_resolves_on_a_dev_only_install() -> None:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = cfg["project"]
    mypy = cfg["tool"]["mypy"]

    installed = _dist_names(project["dependencies"])
    installed |= _dist_names(project["optional-dependencies"]["dev"])
    for stub_dist, module in STUB_PACKAGES.items():
        if stub_dist in installed:
            installed.add(module)
    installed |= ALWAYS_PRESENT

    overridden = {
        m.split(".")[0].lower()
        for override in mypy.get("overrides", [])
        if override.get("ignore_missing_imports")
        for m in override.get("module", [])
    }

    # Only what mypy is actually pointed at. `services/` is deliberately outside
    # that scope, so an import there cannot break the typecheck job.
    targets: list[Path] = []
    for entry in mypy["files"]:
        p = ROOT / entry
        targets.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])

    unresolvable: list[str] = []
    for path in targets:
        for module, line in sorted(_third_party_imports(path).items()):
            low = module.lower()
            if low in sys.stdlib_module_names or low in LOCAL_ROOTS:
                continue
            if low in installed or low in overridden:
                continue
            if low.replace("_", "-") in installed or low.replace("-", "_") in installed:
                continue
            unresolvable.append(f"{path.relative_to(ROOT).as_posix()}:{line}  imports {module!r}")

    assert not unresolvable, (
        "These imports are inside mypy's scope but will not resolve on the "
        "`.[dev]`-only install CI uses, so the typecheck job will fail with "
        "'Cannot find implementation or library stub' on a change that passed "
        "every check locally:\n\n  "
        + "\n  ".join(unresolvable)
        + "\n\nAdd the module to the ignore_missing_imports override list in "
        "pyproject.toml (that is what it is for - these are lazy imports behind "
        "an extra), or put `# type: ignore[import-not-found]` on the import "
        "line, or move the dependency into [dev] if it is genuinely required."
    )


def test_the_scan_would_catch_a_new_optional_import() -> None:
    """A guard nothing can trip is decoration.

    Confirms the two escape hatches are the only reasons a module passes: an
    unknown module with no override and no inline ignore must be reported.
    """
    src = "import some_vendor_sdk\nimport contextlib  # stdlib, fine\n"
    tmp = ROOT / "tests" / "unit" / "_scan_probe.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        found = _third_party_imports(tmp)
        assert "some_vendor_sdk" in found
        assert "contextlib" in found  # the stdlib filter lives in the caller

        tmp.write_text("import some_vendor_sdk  # type: ignore[import-not-found]\n", "utf-8")
        assert "some_vendor_sdk" not in _third_party_imports(tmp), (
            "an inline suppression must exempt the import, or the test would "
            "demand an override for something mypy has already been told about"
        )
    finally:
        tmp.unlink(missing_ok=True)
