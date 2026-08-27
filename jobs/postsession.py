"""Resolve the day's predictions and check that every one is accounted for.

    python jobs/postsession.py --date 2026-03-04 --verify
    python jobs/postsession.py --date 2026-03-04 --load     load into the index

Runs at 15:45, after the forced flatten. Its job is not to report performance -
that is M94 - but to establish that the day's record can be believed at all.

The check it runs is deliberately narrow: **every prediction has an outcome, or a
documented reason, or time left on its horizon.** Nothing else. A day that cannot
pass that has no numbers worth computing, and running attribution over it would
produce a report whose gaps are invisible.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.ms_core.obs import configure
from libs.ms_risk.ledger import PaperLedger, Resolution, to_rows, verify


def load_into_index(rows: list[dict], dsn: str | None = None) -> int:
    """Mirror the file into Postgres so it can be queried.

    Upsert by entry_id: the file is the source of truth and this is a
    projection, so re-running is expected and must be idempotent rather than
    duplicating.
    """
    try:
        import psycopg
    except ImportError:
        print("  psycopg is not installed - run: pip install -e '.[infra]'")
        return 0

    from libs.ms_data.store import DEFAULT_DSN

    cols = list(rows[0]) if rows else []
    if not cols:
        return 0
    placeholders = ", ".join(f"%({c})s" for c in cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "entry_id")
    sql = (
        f"INSERT INTO paper_ledger ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (entry_id) DO UPDATE SET {updates}"
    )
    with psycopg.connect(dsn or DEFAULT_DSN) as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def main() -> int:
    configure()
    p = argparse.ArgumentParser(description="Post-session ledger check (M93)")
    p.add_argument("--date", required=True, help="session date, YYYY-MM-DD")
    p.add_argument("--verify", action="store_true", help="check every prediction is accounted for")
    p.add_argument("--load", action="store_true", help="mirror the file into the index")
    p.add_argument("--now", help="override the clock, for testing a past day")
    args = p.parse_args()

    day = datetime.fromisoformat(args.date).replace(tzinfo=UTC)
    now = datetime.fromisoformat(args.now).replace(tzinfo=UTC) if args.now else datetime.now(UTC)

    ledger = PaperLedger()
    entries = ledger.entries(day)
    if not entries:
        print(f"  nothing recorded for {args.date} at {ledger.path(day)}")
        print("  An empty ledger on a trading day is itself a finding - the system")
        print("  either saw nothing or failed to write what it saw.")
        return 1

    if args.load:
        loaded = load_into_index(to_rows(entries))
        print(f"  {loaded} rows mirrored into paper_ledger")

    audit = verify(ledger, day, now=now)
    print()
    print(audit.render())

    if audit.pending and not audit.overdue:
        print()
        print(f"  {audit.pending} still inside their horizon - not a gap, not yet")

    resolved = [e for e in entries if e.resolution is Resolution.RESOLVED]
    if resolved:
        net = sum(e.net_pnl or 0.0 for e in resolved)
        right = sum(1 for e in resolved if e.was_right)
        print()
        print(f"  of {len(resolved)} resolved: {right} directionally right, net {net:,.2f}")
        print("  (performance is M94's question - this job only asks whether the")
        print("   record is complete enough to ask it)")

    if args.verify and not audit.complete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
