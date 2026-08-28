"""Check the Angel One credentials before a trading morning needs them.

    python tasks.py angel-check

Run this the evening before, not at 09:14. Every failure mode below is one that
otherwise surfaces as "the feed is quiet" during the only fifteen minutes of the
day when the opening auction is happening.

Nothing here places an order, and nothing here prints a secret. The output says
which field is wrong, never what it contains.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ms_core import env as envmod  # noqa: E402
from libs.ms_core.clock import RealClock  # noqa: E402
from libs.ms_data.angelone import (  # noqa: E402
    AngelAuthError,
    AngelCredentials,
    AngelOneFeed,
    login,
)

OK = "  ok    "
NO = "  FAIL  "


def check(*, connect: bool) -> int:
    print()
    e = envmod.current()
    if not envmod.may_hold_broker_credentials(e):
        print(
            f"{NO}environment is {e.value!r}, which may not hold broker credentials.\n"
            "        Set MARKETSAGE_ENV=paper in .env. Research being credential-free\n"
            "        is the guarantee that research code cannot reach a venue, so it\n"
            "        is enforced rather than assumed."
        )
        return 1
    print(f"{OK}environment is {e.value!r}")

    try:
        creds = AngelCredentials.from_env()
    except AngelAuthError as exc:
        print(f"{NO}{exc}")
        return 1
    print(f"{OK}four credential fields present, client {creds.client_code}")

    try:
        import pyotp  # noqa: F401
    except ImportError:
        print(f'{NO}pyotp is not installed. pip install -e ".[broker]"')
        return 1

    try:
        import pyotp as _p

        code = _p.TOTP(creds.totp_secret).now()
    except Exception as exc:
        print(
            f"{NO}the TOTP secret did not generate a code: {exc}\n"
            "        It must be the base32 string Angel shows once when TOTP is\n"
            "        enabled, not the six digits the app displays."
        )
        return 1
    print(f"{OK}TOTP secret generates a {len(code)}-digit code")

    if not connect:
        print(
            "\n  Stopped before the network. Re-run with --connect to log in for real,\n"
            "  which consumes one TOTP window and writes data/.angel_session.json."
        )
        return 0

    try:
        session = login(creds, RealClock())
    except AngelAuthError as exc:
        print(f"{NO}{exc}")
        return 1
    print(f"{OK}logged in, feed token received (session cached for today)")

    feed = AngelOneFeed(session, creds.api_key, exchange="NSE_CM")
    print(f"{OK}feed constructed, budget {feed.max_instruments:,} instruments")
    print(
        f"\n  Gap detection on this feed: {feed.gap_detection}.\n"
        "  SmartAPI sends no sequence number, so a dropped message is invisible\n"
        "  here - an empty gap list means 'cannot tell', not 'none'. Data quality\n"
        "  falls back to the candle-count checks, which are weaker and honest."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--connect",
        action="store_true",
        help="actually log in (uses one TOTP window; caches the session for today)",
    )
    return check(connect=p.parse_args(argv).connect)


if __name__ == "__main__":
    raise SystemExit(main())
