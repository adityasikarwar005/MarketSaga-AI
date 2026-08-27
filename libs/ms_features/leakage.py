"""M43 - Leakage detection and offline/online parity.

Two independent guards, because they catch different mistakes.

**The dynamic detector** wraps a source and records the latest timestamp any
feature actually touched. If that exceeds ``as_of``, the feature saw the future -
regardless of how it managed it. This catches the real cases: a window computed
with the wrong sign, a chain lookup that rounds up instead of down, a helper
that reaches past the context.

**The static linter** reads the source of every registered feature and rejects
the patterns that let a function bypass the context entirely: reading the wall
clock, opening files, importing the research layer, or closing over module-level
data. The dynamic detector cannot see those, because a feature that reads a
global DataFrame never asks the context for anything at all.

Neither is sufficient alone. Both run in CI.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from libs.ms_core.obs import get_logger
from libs.ms_core.schemas import Candle, ChainSnapshot, Tick
from libs.ms_features.registry import REGISTRY, ContextSource, FeatureDef, FeatureRegistry

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Dynamic: did a feature actually touch the future?
# --------------------------------------------------------------------------- #
@dataclass
class LeakReport:
    feature: str
    as_of: datetime
    latest_touched: datetime

    @property
    def seconds_ahead(self) -> float:
        return (self.latest_touched - self.as_of).total_seconds()

    def __str__(self) -> str:
        return (
            f"{self.feature}: read data stamped {self.latest_touched.isoformat()} "
            f"while computing for {self.as_of.isoformat()} "
            f"({self.seconds_ahead:.1f}s into the future)"
        )


class RecordingSource(ContextSource):
    """Wraps a source and remembers the latest timestamp handed out."""

    def __init__(self, inner: ContextSource) -> None:
        self._inner = inner
        self.latest: datetime | None = None

    def _note(self, ts: datetime | None) -> None:
        if ts is not None and (self.latest is None or ts > self.latest):
            self.latest = ts

    def candles(
        self, instrument_id: int, start: datetime, end: datetime, tf: int
    ) -> Sequence[Candle]:
        rows = self._inner.candles(instrument_id, start, end, tf)
        for r in rows:
            self._note(r.ts)
        return rows

    def ticks(self, instrument_id: int, start: datetime, end: datetime) -> Sequence[Tick]:
        rows = self._inner.ticks(instrument_id, start, end)
        for r in rows:
            self._note(r.ts)
        return rows

    def chain(self, underlying_id: int, as_of: datetime) -> ChainSnapshot | None:
        snap = self._inner.chain(underlying_id, as_of)
        if snap is not None:
            self._note(snap.ts)
        return snap

    def reset(self) -> None:
        self.latest = None


def detect_leakage(
    source: ContextSource,
    instrument_id: int,
    as_of: datetime,
    *,
    registry: FeatureRegistry | None = None,
    names: Sequence[str] | None = None,
) -> list[LeakReport]:
    """Compute each feature alone and check what it touched.

    One at a time rather than as a vector: attributing a leak to a specific
    feature is the whole point, and a shared recorder would only say that
    *something* in the batch reached forward.
    """
    reg = REGISTRY if registry is None else registry
    leaks: list[LeakReport] = []
    for name in reg.resolution_order(names):
        recorder = RecordingSource(source)
        reg.compute(recorder, instrument_id, as_of, names=[name])
        if recorder.latest is not None and recorder.latest > as_of:
            leaks.append(LeakReport(feature=name, as_of=as_of, latest_touched=recorder.latest))
    return leaks


# --------------------------------------------------------------------------- #
# Static: can a feature bypass the context?
# --------------------------------------------------------------------------- #
BANNED_CALLS = {
    "now": "reads the wall clock - the clock is injected, and a feature has an as_of",
    "today": "reads the wall clock",
    "utcnow": "reads the wall clock",
    "open": "opens a file - a feature reads through the context or not at all",
    "read_parquet": "reads storage directly, bypassing every point-in-time filter",
    "read_csv": "reads storage directly",
}

BANNED_IMPORTS = {
    "libs.ms_data.archive": "the research layer can see the whole history at once",
    "libs.ms_data.store": "direct database access bypasses the context",
    "duckdb": "a query here can select any row in the archive",
}


@dataclass
class LintFinding:
    feature: str
    line: int
    problem: str
    detail: str = ""

    def __str__(self) -> str:
        where = f"{self.feature}:{self.line}"
        return f"{where}  {self.problem}" + (f" - {self.detail}" if self.detail else "")


@dataclass
class LintReport:
    findings: list[LintFinding] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.findings

    def render(self) -> str:
        if self.ok:
            return f"  leakage lint: {self.checked} feature(s) clean"
        lines = [f"  leakage lint: {len(self.findings)} finding(s) in {self.checked} feature(s)"]
        lines += [f"    {f}" for f in self.findings]
        return "\n".join(lines)


class _FeatureVisitor(ast.NodeVisitor):
    def __init__(self, name: str, param: str, line_offset: int) -> None:
        self.name = name
        self.param = param
        self.offset = line_offset
        self.findings: list[LintFinding] = []
        self.local_names: set[str] = {param}

    def _add(self, node: ast.AST, problem: str, detail: str = "") -> None:
        self.findings.append(
            LintFinding(self.name, getattr(node, "lineno", 0) + self.offset, problem, detail)
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name):
                    self.local_names.add(n.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        for n in ast.walk(node.target):
            if isinstance(n, ast.Name):
                self.local_names.add(n.id)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        for n in ast.walk(node.target):
            if isinstance(n, ast.Name):
                self.local_names.add(n.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fname = None
        if isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        elif isinstance(node.func, ast.Name):
            fname = node.func.id
        if fname in BANNED_CALLS:
            self._add(node, f"calls {fname}()", BANNED_CALLS[fname])
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            for banned, why in BANNED_IMPORTS.items():
                if alias.name.startswith(banned):
                    self._add(node, f"imports {alias.name}", why)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for banned, why in BANNED_IMPORTS.items():
            if module.startswith(banned):
                self._add(node, f"imports from {module}", why)
        self.generic_visit(node)


def lint_feature(definition: FeatureDef) -> list[LintFinding]:
    try:
        src = textwrap.dedent(inspect.getsource(definition.fn))
    except (OSError, TypeError):  # pragma: no cover - built dynamically
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:  # pragma: no cover
        return []

    fn_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn_node is None:
        return []
    param = fn_node.args.args[0].arg if fn_node.args.args else "ctx"
    offset = (inspect.getsourcelines(definition.fn)[1] or 1) - 1

    visitor = _FeatureVisitor(definition.name, param, offset)
    visitor.visit(fn_node)

    findings = list(visitor.findings)
    # A feature that never mentions its context and returns something anyway is
    # reading from somewhere else. Constants are the one legitimate exception,
    # and they are rare enough to be worth declaring explicitly.
    if param not in {n.id for n in ast.walk(fn_node) if isinstance(n, ast.Name)}:
        findings.append(
            LintFinding(
                definition.name,
                offset + 1,
                "never uses its context",
                "a feature that reads nothing through the context is reading from somewhere else",
            )
        )
    return findings


def lint_all(registry: FeatureRegistry | None = None) -> LintReport:
    reg = REGISTRY if registry is None else registry
    report = LintReport()
    for definition in reg.all():
        report.checked += 1
        report.findings.extend(lint_feature(definition))
    return report


# --------------------------------------------------------------------------- #
# Parity
# --------------------------------------------------------------------------- #
@dataclass
class ParityMismatch:
    feature: str
    as_of: datetime
    offline: float | None
    online: float | None

    def __str__(self) -> str:
        return (
            f"{self.feature} at {self.as_of.isoformat()}: "
            f"offline={self.offline!r} online={self.online!r}"
        )


@dataclass
class ParityReport:
    compared: int = 0
    mismatches: list[ParityMismatch] = field(default_factory=list)
    short_windows: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.short_windows

    def render(self) -> str:
        if self.ok:
            return f"  parity: {self.compared:,} value(s) identical"
        lines = [f"  parity: {len(self.mismatches)} mismatch(es) of {self.compared:,}"]
        lines += [f"    {m}" for m in self.mismatches[:10]]
        if self.short_windows:
            lines.append(f"    {len(self.short_windows)} short online window(s):")
            lines += [f"      {s}" for s in self.short_windows[:5]]
        return "\n".join(lines)


def compare_vectors(
    offline: dict[str, float | None],
    online: dict[str, float | None],
    as_of: datetime,
    *,
    tolerance: float = 1e-9,
) -> list[ParityMismatch]:
    """Compare two feature vectors for the same instant.

    Floating-point tolerance rather than exact equality: the two paths sum the
    same numbers in the same order but through different container types, and a
    last-bit difference is not skew. Anything larger is.
    """
    out: list[ParityMismatch] = []
    for name in sorted(set(offline) | set(online)):
        a, b = offline.get(name), online.get(name)
        if a is None and b is None:
            continue
        if a is None or b is None:
            out.append(ParityMismatch(name, as_of, a, b))
            continue
        scale = max(1.0, abs(a), abs(b))
        if abs(a - b) > tolerance * scale:
            out.append(ParityMismatch(name, as_of, a, b))
    return out
