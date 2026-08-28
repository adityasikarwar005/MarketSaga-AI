"""MA5 - the monthly restore drill.

    python jobs/restore_drill.py --target clean-vm
    python jobs/restore_drill.py --target clean-vm --backup /mnt/backup/2026-08-01
    python jobs/restore_drill.py --snapshot          take a manifest, restore nothing
    python jobs/restore_drill.py --self-test         prove the drill can fail

A backup that has never been restored is not a backup. It is a directory that
grows, a nightly job that reports success, and an assumption that gets tested on
the worst possible day by someone who did not write it.

The drill restores into a target that must not be able to see the source. That
precondition is the point: a restore script that quietly reads one path from the
archive and another from wherever it happens to be passes every time on the
machine that still has the original, and the passing run gives no hint. So the
first check is that the source is *unreachable* - not that we chose not to look.

``--self-test`` runs the drill against a deliberately damaged restore and
expects it to fail. A drill that has only ever passed tells you the drill runs,
not that it detects anything.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.clock import Clock, RealClock
from libs.ms_ops.backup import (
    ArtefactKind,
    DrillReport,
    Manifest,
    append_drill,
    build_manifest,
    verify_restore,
)

ROOT = Path(__file__).resolve().parents[1]
DRILL_LOG = ROOT / "research" / "restore_drills.jsonl"

# What a backup must contain, and how replaceable each part is. The kinds drive
# the "irreplaceable data intact" check, so this list is a statement about what
# the business cannot lose rather than a list of directories.
SOURCES: tuple[tuple[str, ArtefactKind], ...] = (
    ("data/ledger/*.ndjson", ArtefactKind.LEDGER),
    ("research/experiments.jsonl", ArtefactKind.EXPERIMENTS),
    ("research/kill_drills.jsonl", ArtefactKind.EXPERIMENTS),
    ("config/*.yaml", ArtefactKind.CONFIG),
    ("data/parquet/**/*.parquet", ArtefactKind.PARQUET),
)

# Paths that must not be visible from the restore target. If any of these exist
# where the drill runs, the drill is being performed in the room it already
# knows and proves nothing.
SOURCE_MARKERS: tuple[Path, ...] = (
    ROOT / "data" / "ledger",
    ROOT / "research" / "experiments.jsonl",
)


def _experiment_chain(path: Path) -> tuple[bool, str]:
    """The experiment log's hash chain, verified where it landed."""
    from libs.ms_models.cv import ExperimentLog

    broken = ExperimentLog(path).verify()
    if broken:
        return False, f"chain breaks at index/indices {broken} - entries were lost or reordered"
    return True, "chain intact after restore"


def _ledger_chain(path: Path) -> tuple[bool, str]:
    """The ledger is append-only NDJSON; every line must parse and be ordered.

    A restore that reorders lines is worse than one that drops them: a dropped
    record is a gap someone notices, and a reordered one is a resolution that
    appears to precede its own prediction.
    """
    import json

    previous: str | None = None
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            return False, f"line {i} does not parse: {exc}"
        ts = str(entry.get("ts_predicted") or entry.get("ts") or "")
        if previous is not None and ts and ts < previous:
            return False, f"line {i} is out of order ({ts} after {previous})"
        previous = ts or previous
    return True, "ordered and parseable"


CHAIN_VERIFIERS = {
    ArtefactKind.EXPERIMENTS: _experiment_chain,
    ArtefactKind.LEDGER: _ledger_chain,
}


def snapshot(root: Path = ROOT, *, clock: Clock | None = None) -> Manifest:
    """Describe what a backup taken now would contain."""
    now = (clock or RealClock()).now()
    return build_manifest(
        root,
        taken_on=now.date(),
        taken_at=now,
        sources=SOURCES,
        source_host=socket.gethostname(),
    )


def run_drill(
    manifest: Manifest,
    restored_root: Path,
    *,
    target: str,
    clock: Clock | None = None,
    source_markers: tuple[Path, ...] = SOURCE_MARKERS,
) -> DrillReport:
    return verify_restore(
        manifest,
        restored_root,
        target=target,
        ran_at=(clock or RealClock()).now(),
        source_paths=source_markers,
        chain_verifiers=CHAIN_VERIFIERS,
    )


def _self_test(clock: Clock | None = None) -> int:
    """Restore into a scratch tree, damage one file, and expect a failure.

    A drill that has only ever passed tells you the drill runs. This is the
    check that it detects anything - and it is cheap enough to run every time.
    """
    now = (clock or RealClock()).now()
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        source = scratch / "source"
        restored = scratch / "restored"
        (source / "data" / "ledger").mkdir(parents=True)
        (source / "research").mkdir(parents=True)
        (source / "config").mkdir(parents=True)

        (source / "data" / "ledger" / "day.ndjson").write_text(
            '{"ts_predicted": "2026-08-01T09:20:00+00:00", "entry_id": "a"}\n'
            '{"ts_predicted": "2026-08-01T09:25:00+00:00", "entry_id": "b"}\n',
            encoding="utf-8",
        )
        (source / "research" / "experiments.jsonl").write_text("", encoding="utf-8")
        (source / "config" / "risk.yaml").write_text("max_lots: 3\n", encoding="utf-8")

        manifest = build_manifest(
            source, taken_on=now.date(), taken_at=now, sources=SOURCES, source_host="self-test"
        )
        shutil.copytree(source, restored)

        clean = verify_restore(
            manifest, restored, target="self-test (intact)", ran_at=now, source_paths=()
        )
        print(clean.render())
        if not clean.passed:
            print(
                "\nthe self-test's own intact restore failed - the drill is broken, not the backup"
            )
            return 1

        # Now damage it, in the way a partial sync actually damages things: the
        # file is present, the right shape, and one byte short.
        victim = restored / "data" / "ledger" / "day.ndjson"
        victim.write_text(
            '{"ts_predicted": "2026-08-01T09:20:00+00:00", "entry_id": "a"}\n', encoding="utf-8"
        )
        damaged = verify_restore(
            manifest, restored, target="self-test (truncated)", ran_at=now, source_paths=()
        )
        print()
        print(damaged.render())
        if damaged.passed:
            print("\nthe drill passed a truncated ledger. It detects nothing.")
            return 1

        print("\nself-test ok: intact restores pass, a truncated one fails")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target", default="", help="name of the clean machine being restored onto"
    )
    parser.add_argument("--backup", type=Path, help="the restored tree to verify")
    parser.add_argument("--manifest", type=Path, help="manifest describing that backup")
    parser.add_argument(
        "--snapshot", action="store_true", help="write a manifest for today and exit"
    )
    parser.add_argument("--self-test", action="store_true", help="prove the drill can fail")
    parser.add_argument("--record", action="store_true", help="append the result to the drill log")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.snapshot:
        manifest = snapshot()
        out = ROOT / "build" / f"backup-manifest-{manifest.taken_on}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(manifest.to_json(), encoding="utf-8")
        print(manifest.render())
        print(f"\n  manifest written to {out.relative_to(ROOT)}")
        if manifest.missing_irreplaceable:
            print(
                "  Checked now rather than during a recovery: the kinds above are "
                "not in this backup and cannot be rebuilt from anything else."
            )
            return 1
        return 0

    if not args.target:
        parser.error("--target names the clean machine, e.g. --target clean-vm")
    if not args.backup or not args.manifest:
        print(
            f"No restored tree given. The drill needs one:\n"
            f"  python jobs/restore_drill.py --target {args.target} "
            f"--backup /path/to/restored --manifest /path/to/manifest.json\n\n"
            "Take today's manifest first with --snapshot, restore it onto the clean\n"
            "machine, then point this at the result. --self-test proves the drill\n"
            "detects damage without needing either.",
            file=sys.stderr,
        )
        return 2

    manifest = Manifest.from_json(args.manifest.read_text(encoding="utf-8"))
    report = run_drill(manifest, args.backup, target=args.target)
    print(report.render())

    if args.record:
        append_drill(DRILL_LOG, report)
        print(f"\n  recorded in {DRILL_LOG.relative_to(ROOT)}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
