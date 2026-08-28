"""MA5 - backup, and the only thing that makes one real: restoring it.

A backup that has never been restored is not a backup. It is a directory that
grows, a nightly job that reports success, and an assumption nobody has tested -
and the test always happens on the worst possible day, under time pressure, by
someone who did not write it.

So the unit of work here is not the dump. It is the **drill**.

What the drill has to prove, and what it must not assume
--------------------------------------------------------

The failure mode that defeats an ordinary restore test is that the test runs on
the machine that still has the original data. A restore script that reads one
path from the archive and another from wherever it happens to be will pass every
time on that machine and fail on a clean one, and nothing in the passing run
hints at it.

:func:`verify_isolation` therefore asserts that the source is *unreachable*
before the restore begins. Not "we did not use it" - unreachable. That is the
difference between a drill and a rehearsal in the room you already know.

Completeness is a claim that needs evidence
--------------------------------------------

"The files are there" is not a restore. The manifest records a checksum and a
size for every artefact, and the drill compares them - because a truncated
Parquet file, a partially flushed NDJSON, and a complete one all look identical
in a directory listing.

The audit spine gets checked twice
-----------------------------------

``data/ledger/*.ndjson`` and ``research/experiments.jsonl`` are append-only and
hash-chained. After a restore, their chains must still verify. A backup that
round-trips the bytes but breaks the chain has restored the data and destroyed
the thing the data was for.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

# Read in blocks; some Parquet partitions are large enough that slurping them to
# checksum would be the peak memory use of the whole nightly job.
_CHUNK = 1 << 20


class ArtefactKind(StrEnum):
    LEDGER = "ledger"  # append-only NDJSON, hash-chained - the audit spine
    EXPERIMENTS = "experiments"  # the honest trial count
    PARQUET = "parquet"  # market data partitions
    POSTGRES = "postgres"  # the index rebuilt from the ledger
    CONFIG = "config"  # point-in-time settings, hashed
    OTHER = "other"


# Artefacts whose loss cannot be repaired from anything else. Postgres is not
# among them: it is an index built from the ledger, and rebuilding it is a
# migration away. The ledger is not, and neither is the experiment log.
IRREPLACEABLE = frozenset({ArtefactKind.LEDGER, ArtefactKind.EXPERIMENTS, ArtefactKind.CONFIG})


class BackupError(RuntimeError):
    pass


class DrillFailure(BackupError):
    """The drill found something. This is the drill working."""


@dataclass(frozen=True)
class Artefact:
    """One backed-up file, with enough to prove it came back whole."""

    path: str  # relative to the repository root
    kind: ArtefactKind
    size: int
    sha256: str
    lines: int | None = None  # for NDJSON, so a truncation is visible as a count

    def render(self) -> str:
        extra = f"  {self.lines} lines" if self.lines is not None else ""
        return f"    {self.path:<44} {self.kind.value:<12} {self.size:>10,}B  {self.sha256[:12]}{extra}"


def checksum(path: Path) -> tuple[str, int]:
    """sha256 and size, streamed."""
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def _line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def describe(path: Path, root: Path, kind: ArtefactKind) -> Artefact:
    digest, size = checksum(path)
    lines = _line_count(path) if path.suffix in (".ndjson", ".jsonl") else None
    return Artefact(
        path=path.relative_to(root).as_posix(), kind=kind, size=size, sha256=digest, lines=lines
    )


@dataclass
class Manifest:
    """What a backup contains, and what each piece should look like coming back.

    Written alongside the backup rather than inside it. A manifest stored only
    within the archive it describes cannot answer "did the archive itself get
    truncated", which is the failure a nightly sync produces most often.
    """

    taken_on: date
    taken_at: datetime
    artefacts: list[Artefact] = field(default_factory=list)
    source_host: str = ""
    notes: str = ""

    @property
    def total_bytes(self) -> int:
        return sum(a.size for a in self.artefacts)

    def of_kind(self, kind: ArtefactKind) -> list[Artefact]:
        return [a for a in self.artefacts if a.kind is kind]

    @property
    def missing_irreplaceable(self) -> list[ArtefactKind]:
        """Kinds that cannot be rebuilt and are not in this backup.

        Checked at backup time, not at restore time. Discovering that the
        experiment log was never included is a different conversation on a
        Tuesday than during a recovery.
        """
        present = {a.kind for a in self.artefacts}
        return sorted(IRREPLACEABLE - present)

    def to_json(self) -> str:
        return json.dumps(
            {
                "taken_on": self.taken_on.isoformat(),
                "taken_at": self.taken_at.isoformat(),
                "source_host": self.source_host,
                "notes": self.notes,
                "artefacts": [asdict(a) | {"kind": a.kind.value} for a in self.artefacts],
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        raw = json.loads(text)
        return cls(
            taken_on=date.fromisoformat(raw["taken_on"]),
            taken_at=datetime.fromisoformat(raw["taken_at"]),
            source_host=raw.get("source_host", ""),
            notes=raw.get("notes", ""),
            artefacts=[
                Artefact(
                    path=a["path"],
                    kind=ArtefactKind(a["kind"]),
                    size=a["size"],
                    sha256=a["sha256"],
                    lines=a.get("lines"),
                )
                for a in raw["artefacts"]
            ],
        )

    def render(self) -> str:
        lines = [
            f"  backup {self.taken_on} ({len(self.artefacts)} artefacts, "
            f"{self.total_bytes:,} bytes)"
        ]
        lines += [a.render() for a in sorted(self.artefacts, key=lambda a: a.path)]
        if self.missing_irreplaceable:
            lines.append(
                "  MISSING and unrebuildable: "
                + ", ".join(k.value for k in self.missing_irreplaceable)
                + ". Postgres can be rebuilt from the ledger; the ledger cannot be "
                "rebuilt from anything."
            )
        return "\n".join(lines)


def build_manifest(
    root: Path,
    *,
    taken_on: date,
    taken_at: datetime,
    sources: Sequence[tuple[str, ArtefactKind]],
    source_host: str = "",
) -> Manifest:
    """Describe everything matching the given globs.

    ``sources`` is a list of ``(glob, kind)`` so the caller decides what counts
    as irreplaceable rather than this module guessing from a file extension.
    """
    manifest = Manifest(taken_on=taken_on, taken_at=taken_at, source_host=source_host)
    for pattern, kind in sources:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                manifest.artefacts.append(describe(path, root, kind))
    return manifest


# --------------------------------------------------------------------------- #
# The drill
# --------------------------------------------------------------------------- #
class CheckOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class DrillCheck:
    name: str
    outcome: CheckOutcome
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.outcome is CheckOutcome.FAIL

    def render(self) -> str:
        mark = {CheckOutcome.PASS: " ok ", CheckOutcome.FAIL: "FAIL", CheckOutcome.SKIP: "SKIP"}[
            self.outcome
        ]
        return f"    [{mark}] {self.name:<34} {self.detail}"


@dataclass
class DrillReport:
    target: str
    ran_at: datetime
    checks: list[DrillCheck] = field(default_factory=list)

    @property
    def failures(self) -> list[DrillCheck]:
        return [c for c in self.checks if c.failed]

    @property
    def passed(self) -> bool:
        return not self.failures

    def add(self, name: str, ok: bool, detail: str = "") -> DrillCheck:
        check = DrillCheck(name, CheckOutcome.PASS if ok else CheckOutcome.FAIL, detail)
        self.checks.append(check)
        return check

    def skip(self, name: str, detail: str) -> DrillCheck:
        check = DrillCheck(name, CheckOutcome.SKIP, detail)
        self.checks.append(check)
        return check

    def render(self) -> str:
        lines = [f"  restore drill against {self.target} at {self.ran_at.isoformat()}"]
        lines += [c.render() for c in self.checks]
        if self.failures:
            lines.append(
                f"  -> {len(self.failures)} check(s) failed. This is the drill working: "
                "the alternative is finding out during a recovery."
            )
        else:
            lines.append("  -> restore verified")
        return "\n".join(lines)

    def to_record(self) -> dict[str, object]:
        return {
            "ran_at": self.ran_at.isoformat(),
            "target": self.target,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "outcome": c.outcome.value, "detail": c.detail}
                for c in self.checks
            ],
        }


def verify_isolation(source_paths: Iterable[Path], target: str) -> DrillCheck:
    """The drill's precondition: the source must be out of reach.

    Not "we did not read it" - unreachable. A restore script that quietly falls
    back to a path still present on the source machine passes every test run
    there and fails on a clean one, and the passing run gives no hint. Asserting
    absence is the only version of this check that means anything.
    """
    reachable = [p for p in source_paths if p.exists()]
    if reachable:
        return DrillCheck(
            "source is unreachable",
            CheckOutcome.FAIL,
            f"{len(reachable)} source path(s) still visible from {target}: "
            f"{', '.join(str(p) for p in reachable[:3])}. A restore verified here "
            "proves nothing - the script can read the original and never know it.",
        )
    return DrillCheck("source is unreachable", CheckOutcome.PASS, f"{target} sees no source paths")


def verify_restore(
    manifest: Manifest,
    restored_root: Path,
    *,
    target: str,
    ran_at: datetime,
    source_paths: Iterable[Path] = (),
    chain_verifiers: Mapping[ArtefactKind, Callable[[Path], tuple[bool, str]]] | None = None,
) -> DrillReport:
    """Compare a restored tree against the manifest that describes it.

    Every artefact, every checksum. "The files are there" is not a restore: a
    truncated Parquet, a partially flushed NDJSON and a complete one are
    indistinguishable in a directory listing.
    """
    report = DrillReport(target=target, ran_at=ran_at)
    report.checks.append(verify_isolation(source_paths, target))

    missing: list[str] = []
    corrupt: list[str] = []
    truncated: list[str] = []
    checked_lines = 0

    for artefact in manifest.artefacts:
        path = restored_root / artefact.path
        if not path.exists():
            missing.append(artefact.path)
            continue
        digest, size = checksum(path)
        if size != artefact.size or digest != artefact.sha256:
            corrupt.append(f"{artefact.path} ({size:,}B vs {artefact.size:,}B expected)")
        # Deliberately not `continue`. Skipping the line count for a file that
        # already failed its checksum makes the report say "line counts match"
        # about a file it never counted - the most reassuring line in the output
        # describing a check that did not run. A corrupt file is usually also a
        # truncated one, and saying both is what tells you which sync broke.
        if artefact.lines is not None:
            checked_lines += 1
            if _line_count(path) != artefact.lines:
                truncated.append(f"{artefact.path} ({_line_count(path)} of {artefact.lines} lines)")

    report.add(
        "every artefact present",
        not missing,
        f"{len(manifest.artefacts) - len(missing)}/{len(manifest.artefacts)}"
        + (f" - missing {', '.join(missing[:3])}" if missing else ""),
    )
    report.add(
        "checksums match",
        not corrupt,
        "; ".join(corrupt[:3]) if corrupt else f"{len(manifest.artefacts)} verified",
    )
    # The count says how many files this check actually looked at. "Line counts
    # match" over zero files is not reassurance, and it reads identically.
    report.add(
        "no truncated append-only files",
        not truncated,
        ", ".join(truncated[:3]) if truncated else f"{checked_lines} append-only file(s) counted",
    )

    # The irreplaceable ones are worth their own line. A drill that reports "one
    # artefact missing" reads the same whether it was a Parquet partition that
    # can be re-downloaded or the ledger that cannot.
    lost = {
        ArtefactKind(a.kind)
        for a in manifest.artefacts
        if a.path in missing and a.kind in IRREPLACEABLE
    }
    report.add(
        "irreplaceable data intact",
        not lost,
        ", ".join(sorted(k.value for k in lost)) if lost else "ledger, experiments, config",
    )

    # The chains. A backup that round-trips the bytes but breaks the chain has
    # restored the data and destroyed what the data was for.
    verifiers = chain_verifiers or {}
    for kind, verify in verifiers.items():
        artefacts = manifest.of_kind(kind)
        if not artefacts:
            report.skip(f"{kind.value} chain intact", "not in this backup")
            continue
        for artefact in artefacts:
            path = restored_root / artefact.path
            if not path.exists():
                report.add(f"{kind.value} chain intact", False, f"{artefact.path} was not restored")
                continue
            ok, detail = verify(path)
            report.add(f"{kind.value} chain intact", ok, detail)

    if report.failures:
        log.info(
            "restore drill failed",
            fields={"target": target, "failures": [c.name for c in report.failures]},
        )
    return report


def append_drill(path: Path, report: DrillReport) -> None:
    """Append-only drill log, like the kill drills.

    Kept as a record because the useful question is never "did last night's
    drill pass" - it is "when did this last work, and what changed since".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report.to_record(), sort_keys=True) + "\n")


def due(today: date, last_drill: date | None) -> bool:
    """Monthly. A drill that slips to quarterly is one that finds problems that
    have had a quarter to compound."""
    if last_drill is None:
        return True
    return (today.year, today.month) != (last_drill.year, last_drill.month)
