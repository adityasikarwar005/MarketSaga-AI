"""M07 - the structured logger, and the timezone it claims.

Every line this formatter emits ends in ``Z``. That character is an assertion,
and for the life of the project it was false on any machine not already running
in UTC.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime

from libs.ms_core.obs import JsonFormatter


def _record(created: float) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="probe",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    rec.created = created
    rec.msecs = (created - int(created)) * 1000.0
    return rec


class TestLogTimestampIsUTC:
    """The bug this class exists for.

    ``logging.Formatter.formatTime`` uses ``time.localtime`` unless told
    otherwise, and the formatter appends ``Z``. On an IST machine every log line
    was stamped 5.5 hours ahead of the instant it described - and anything
    logged after 18:30 UTC carried the *next* day's date.

    It survived because CI runs with TZ=UTC, where localtime and gmtime agree
    and the fault cannot reproduce. So the first test below asserts the
    mechanism rather than the output: it is the only form that fails on the
    machine where the bug is invisible.
    """

    def test_the_converter_is_gmtime(self) -> None:
        assert JsonFormatter.converter is time.gmtime, (
            "formatTime falls back to time.localtime, and the ts field appends "
            "'Z'. Together those render local time and call it UTC. This "
            "assertion is on the mechanism because the rendering is identical "
            "to a correct one whenever TZ=UTC - which is how CI runs, and why "
            "this went unnoticed."
        )

    def test_a_known_instant_renders_as_that_instant_in_utc(self) -> None:
        epoch = 1_756_499_932.327  # 2026-08-29T19:38:52.327Z
        payload = json.loads(JsonFormatter().format(_record(epoch)))
        expected = datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%S")
        assert payload["ts"].startswith(expected)
        assert payload["ts"].endswith("Z")

    def test_the_converter_attribute_is_what_controls_the_output(self) -> None:
        """Pinning `converter` is load-bearing, not decorative.

        If formatTime ever stopped routing through it, the pin above would keep
        passing while the rendering went back to local time.
        """
        epoch = 1_756_499_932.327

        class Shifted(JsonFormatter):
            # gmtime of an instant twelve hours later
            converter = staticmethod(lambda t: time.gmtime(t + 12 * 3600))

        base = json.loads(JsonFormatter().format(_record(epoch)))["ts"]
        moved = json.loads(Shifted().format(_record(epoch)))["ts"]
        assert base != moved, "formatTime is not going through `converter` any more"

    def test_the_date_does_not_roll_early(self) -> None:
        """The consequence that is worst to debug.

        20:00 IST on the 29th is 14:30 UTC on the 29th. Under the old default
        an IST machine wrote 2026-08-30 - so a day's logs split across two
        dates, and a replay filtering by log date silently lost the evening.
        """
        epoch = datetime(2026, 8, 29, 14, 30, tzinfo=UTC).timestamp()
        payload = json.loads(JsonFormatter().format(_record(epoch)))
        assert payload["ts"].startswith("2026-08-29T14:30")
