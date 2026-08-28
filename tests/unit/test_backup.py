"""MA4 / MA5 - the CI stages, and the drill that makes a backup real.

A backup that has never been restored is not a backup, and a drill that has only
ever passed tells you the drill runs rather than that it detects anything. Both
halves need testing, so several of these deliberately damage a restore and
require the drill to notice.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from jobs.ci_stages import STAGES, CiReport, Stage, StageResult
from libs.ms_ops.backup import (
    IRREPLACEABLE,
    ArtefactKind,
    Manifest,
    append_drill,
    build_manifest,
    checksum,
    due,
    verify_isolation,
    verify_restore,
)

NOW = datetime(2026, 8, 28, 2, 0, tzinfo=UTC)
TODAY = NOW.date()

SOURCES = (
    ("data/ledger/*.ndjson", ArtefactKind.LEDGER),
    ("research/experiments.jsonl", ArtefactKind.EXPERIMENTS),
    ("config/*.yaml", ArtefactKind.CONFIG),
)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A miniature repository worth backing up."""
    root = tmp_path / "source"
    (root / "data" / "ledger").mkdir(parents=True)
    (root / "research").mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    (root / "data" / "ledger" / "2026-08-27.ndjson").write_text(
        '{"ts_predicted": "2026-08-27T09:20:00+00:00", "entry_id": "a"}\n'
        '{"ts_predicted": "2026-08-27T09:25:00+00:00", "entry_id": "b"}\n'
        '{"ts_predicted": "2026-08-27T09:30:00+00:00", "entry_id": "c"}\n',
        encoding="utf-8",
    )
    (root / "research" / "experiments.jsonl").write_text(
        '{"name": "gap_v1", "prev": ""}\n', encoding="utf-8"
    )
    (root / "config" / "risk.yaml").write_text("max_lots: 3\n", encoding="utf-8")
    return root


@pytest.fixture
def manifest(source: Path) -> Manifest:
    return build_manifest(source, taken_on=TODAY, taken_at=NOW, sources=SOURCES)


def restored(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "restored"
    shutil.copytree(source, target)
    return target


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #
def test_the_manifest_describes_every_artefact(manifest: Manifest):
    assert len(manifest.artefacts) == 3
    assert {a.kind for a in manifest.artefacts} == {
        ArtefactKind.LEDGER,
        ArtefactKind.EXPERIMENTS,
        ArtefactKind.CONFIG,
    }


def test_ndjson_artefacts_carry_a_line_count(manifest: Manifest):
    """So a truncation is visible as a number rather than as a smaller file that
    still parses perfectly."""
    ledger = manifest.of_kind(ArtefactKind.LEDGER)[0]
    assert ledger.lines == 3


def test_a_missing_irreplaceable_kind_is_reported_at_backup_time(source: Path):
    """Discovering that the experiment log was never in the backup is a
    different conversation on a Tuesday than during a recovery."""
    partial = build_manifest(
        source,
        taken_on=TODAY,
        taken_at=NOW,
        sources=(("data/ledger/*.ndjson", ArtefactKind.LEDGER),),
    )
    assert ArtefactKind.EXPERIMENTS in partial.missing_irreplaceable
    assert "cannot be rebuilt from anything" in partial.render()


def test_postgres_is_not_irreplaceable():
    """It is an index built from the ledger. Treating it as unrecoverable makes
    every drill fail on something a migration fixes, and a drill that always
    fails is one nobody reads."""
    assert ArtefactKind.POSTGRES not in IRREPLACEABLE
    assert ArtefactKind.LEDGER in IRREPLACEABLE


def test_the_manifest_round_trips_through_json(manifest: Manifest):
    back = Manifest.from_json(manifest.to_json())
    assert back.taken_on == manifest.taken_on
    assert [a.sha256 for a in back.artefacts] == [a.sha256 for a in manifest.artefacts]
    assert [a.kind for a in back.artefacts] == [a.kind for a in manifest.artefacts]


def test_checksums_are_content_not_path(source: Path):
    a = source / "config" / "risk.yaml"
    b = source / "config" / "copy.yaml"
    b.write_text(a.read_text(encoding="utf-8"), encoding="utf-8")
    assert checksum(a) == checksum(b)


# --------------------------------------------------------------------------- #
# isolation - the precondition that makes the drill mean anything
# --------------------------------------------------------------------------- #
def test_the_drill_refuses_to_run_where_the_source_is_visible(source: Path, tmp_path: Path):
    """The failure mode that defeats an ordinary restore test. A script that
    quietly reads one path from the archive and another from wherever it happens
    to be passes every time on the machine that still has the original, and the
    passing run gives no hint."""
    check = verify_isolation([source / "data" / "ledger"], "the-machine-we-already-know")
    assert check.failed
    assert "proves nothing" in check.detail


def test_isolation_passes_on_a_genuinely_clean_target(tmp_path: Path):
    check = verify_isolation([tmp_path / "nowhere"], "clean-vm")
    assert not check.failed


def test_the_isolation_check_is_the_first_thing_reported(
    manifest: Manifest, source: Path, tmp_path: Path
):
    """Order matters in a report someone reads under pressure: every check below
    it is meaningless if this one failed."""
    report = verify_restore(
        manifest,
        restored(source, tmp_path),
        target="clean-vm",
        ran_at=NOW,
        source_paths=[source / "data" / "ledger"],
    )
    assert report.checks[0].name == "source is unreachable"
    assert report.checks[0].failed
    assert not report.passed


# --------------------------------------------------------------------------- #
# the drill detects damage
# --------------------------------------------------------------------------- #
def test_an_intact_restore_passes(manifest: Manifest, source: Path, tmp_path: Path):
    """The null test. A drill that fails on a good restore gets switched off."""
    report = verify_restore(manifest, restored(source, tmp_path), target="clean-vm", ran_at=NOW)
    assert report.passed, report.render()
    assert "restore verified" in report.render()


def test_a_missing_file_is_caught(manifest: Manifest, source: Path, tmp_path: Path):
    target = restored(source, tmp_path)
    (target / "config" / "risk.yaml").unlink()

    report = verify_restore(manifest, target, target="clean-vm", ran_at=NOW)
    assert not report.passed
    assert any("present" in c.name and c.failed for c in report.checks)


def test_a_truncated_file_is_caught_by_both_checks(
    manifest: Manifest, source: Path, tmp_path: Path
):
    """The regression this test exists for: an earlier version skipped the line
    count for any file that had already failed its checksum, so the report said
    "line counts match" about a file it never counted. The most reassuring line
    in the output described a check that did not run."""
    target = restored(source, tmp_path)
    (target / "data" / "ledger" / "2026-08-27.ndjson").write_text(
        '{"ts_predicted": "2026-08-27T09:20:00+00:00", "entry_id": "a"}\n', encoding="utf-8"
    )

    report = verify_restore(manifest, target, target="clean-vm", ran_at=NOW)
    failed = {c.name for c in report.checks if c.failed}
    assert "checksums match" in failed
    assert "no truncated append-only files" in failed
    truncation = next(c for c in report.checks if c.name == "no truncated append-only files")
    assert "1 of 3 lines" in truncation.detail


def test_a_same_size_corruption_is_caught(manifest: Manifest, source: Path, tmp_path: Path):
    """Byte-for-byte the right length and wrong content. Only the checksum sees
    this, which is why size alone is not the check."""
    target = restored(source, tmp_path)
    path = target / "config" / "risk.yaml"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace("3", "9"), encoding="utf-8")
    assert len(path.read_text(encoding="utf-8")) == len(original)

    report = verify_restore(manifest, target, target="clean-vm", ran_at=NOW)
    assert not report.passed


def test_losing_the_ledger_is_reported_separately_from_losing_a_parquet(
    source: Path, tmp_path: Path
):
    """ "One artefact missing" reads the same whether it was a partition that can
    be re-downloaded or the ledger that cannot."""
    (source / "data" / "parquet").mkdir(parents=True)
    (source / "data" / "parquet" / "day.parquet").write_bytes(b"PAR1")
    full = build_manifest(
        source,
        taken_on=TODAY,
        taken_at=NOW,
        sources=(*SOURCES, ("data/parquet/*.parquet", ArtefactKind.PARQUET)),
    )
    target = restored(source, tmp_path)
    (target / "data" / "parquet" / "day.parquet").unlink()

    report = verify_restore(full, target, target="clean-vm", ran_at=NOW)
    irreplaceable = next(c for c in report.checks if c.name == "irreplaceable data intact")
    assert not irreplaceable.failed, "a lost parquet was reported as unrecoverable"

    target2 = restored(source, tmp_path / "second")
    (target2 / "data" / "ledger" / "2026-08-27.ndjson").unlink()
    report2 = verify_restore(full, target2, target="clean-vm", ran_at=NOW)
    assert next(c for c in report2.checks if c.name == "irreplaceable data intact").failed


# --------------------------------------------------------------------------- #
# the chains survive the round trip
# --------------------------------------------------------------------------- #
def test_a_chain_verifier_runs_against_the_restored_copy(
    manifest: Manifest, source: Path, tmp_path: Path
):
    """A backup that round-trips the bytes but breaks the chain has restored the
    data and destroyed what the data was for."""
    seen: list[Path] = []

    def verifier(path: Path) -> tuple[bool, str]:
        seen.append(path)
        return True, "checked"

    target = restored(source, tmp_path)
    verify_restore(
        manifest,
        target,
        target="clean-vm",
        ran_at=NOW,
        chain_verifiers={ArtefactKind.EXPERIMENTS: verifier},
    )
    assert seen == [target / "research" / "experiments.jsonl"]


def test_a_broken_chain_fails_the_drill(manifest: Manifest, source: Path, tmp_path: Path):
    report = verify_restore(
        manifest,
        restored(source, tmp_path),
        target="clean-vm",
        ran_at=NOW,
        chain_verifiers={ArtefactKind.LEDGER: lambda _p: (False, "chain breaks at index 4")},
    )
    assert not report.passed
    assert any("chain" in c.name and c.failed for c in report.checks)


def test_a_chain_check_on_an_absent_kind_skips_rather_than_passes(source: Path, tmp_path: Path):
    """Skipped is not passed. A check that silently reports green because it had
    nothing to look at is the same output as one that verified something."""
    only_config = build_manifest(
        source, taken_on=TODAY, taken_at=NOW, sources=(("config/*.yaml", ArtefactKind.CONFIG),)
    )
    report = verify_restore(
        only_config,
        restored(source, tmp_path),
        target="clean-vm",
        ran_at=NOW,
        chain_verifiers={ArtefactKind.LEDGER: lambda _p: (True, "")},
    )
    chain = next(c for c in report.checks if "chain" in c.name)
    assert chain.outcome.value == "skip"


# --------------------------------------------------------------------------- #
# the drill log and the schedule
# --------------------------------------------------------------------------- #
def test_the_drill_is_recorded_whether_it_passed_or_not(
    manifest: Manifest, source: Path, tmp_path: Path
):
    """The useful question is never "did last night's drill pass" - it is "when
    did this last work, and what changed since"."""
    path = tmp_path / "restore_drills.jsonl"
    target = restored(source, tmp_path)
    (target / "config" / "risk.yaml").unlink()

    append_drill(path, verify_restore(manifest, target, target="clean-vm", ran_at=NOW))
    record = json.loads(path.read_text(encoding="utf-8").strip())

    assert record["passed"] is False
    assert record["target"] == "clean-vm"
    assert any(c["outcome"] == "fail" for c in record["checks"])


def test_the_drill_is_due_monthly():
    assert due(date(2026, 9, 1), date(2026, 8, 1))
    assert not due(date(2026, 8, 28), date(2026, 8, 1))
    assert due(date(2026, 8, 1), None)


def test_a_drill_that_slipped_a_month_is_due_not_forgiven():
    """Slipping to quarterly means finding problems that have had a quarter to
    compound."""
    assert due(date(2026, 11, 15), date(2026, 8, 1))


# --------------------------------------------------------------------------- #
# MA4 - the CI stages
# --------------------------------------------------------------------------- #
def test_all_six_named_stages_exist():
    """The plan names them: unit, determinism, parity, leakage lint, AP-hack,
    stress. A pipeline missing one is a guarantee nobody is checking."""
    assert {s.name for s in STAGES} == {
        "unit",
        "determinism",
        "parity",
        "leakage-lint",
        "ap-hack",
        "stress",
    }


def test_every_stage_states_the_guarantee_it_establishes():
    """ "stage 4 failed" sends someone to the traceback. "no feature can reach
    past its context" sends them to the feature."""
    for stage in STAGES:
        assert stage.guarantee, f"{stage.name} has no stated guarantee"
        assert len(stage.guarantee) > 20


def test_a_skipped_stage_is_not_a_passed_stage():
    """A check that renders green because infrastructure was missing is worse
    than no check: it is a check somebody is relying on."""
    stage = Stage("infra-thing", ("-m", "pytest"), "something", needs_infra=True)
    result = StageResult(stage, 0, 0.0, skipped=True)

    assert result.ok
    assert "SKIP" in result.render()
    assert "Skipped is not passed" in CiReport([result]).render()


def test_a_failing_stage_makes_the_report_red():
    report = CiReport(
        [
            StageResult(STAGES[0], 0, 1.0),
            StageResult(STAGES[1], 1, 2.0),
        ]
    )
    assert not report.green
    assert [r.stage.name for r in report.failed] == ["determinism"]
    assert "different guarantee" in report.render()


def test_an_all_green_report_says_so():
    report = CiReport([StageResult(s, 0, 1.0) for s in STAGES])
    assert report.green
    assert "failed" not in report.render()
