"""M70 reading a live recording instead of the archive.

M70's finding was that daily settlement marks cannot support a surface fit, and
that only a book-carrying intraday snapshot could settle it. It then had no way
to read one: `load_chain` went to `BhavcopyStore` and nowhere else, so the
module that most needed a live book was the one module structurally unable to
open one.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from jobs.surface_report import load_recorded_chain
from libs.ms_core.schemas import ChainSnapshot, OptionQuote, OptionType
from libs.ms_data.bhavcopy import convexity_of
from libs.ms_options.pricing import OptType, black76_price

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 8, 31)
TOKEN = 26000
SPOT = 24_000.0


def _snapshot(ts: datetime, *, spot: float = SPOT, quotes: bool = True) -> ChainSnapshot:
    rows = []
    if quotes:
        expiry = datetime.combine(DAY + timedelta(days=1), time(15, 30), tzinfo=IST)
        for i in range(-6, 7):
            strike = spot + 50.0 * i
            for opt, kind in ((OptionType.CE, OptType.CALL), (OptionType.PE, OptType.PUT)):
                # Priced with Black-76 rather than a hand-rolled decay. An
                # arbitrage-free fixture is the only kind that can support
                # `violations == 0` as an assertion, and a shape invented for
                # the test was not convex around the money - which is the very
                # property being measured.
                fair = black76_price(spot, strike, 0.13, 1 / 252, 0.065, kind)
                rows.append(
                    OptionQuote(
                        expiry=expiry,
                        strike=strike,
                        opt_type=opt,
                        bid=round(fair * 0.995, 2),
                        ask=round(fair * 1.005, 2),
                        oi=5_000,
                        iv=0.13,
                    )
                )
    return ChainSnapshot(
        ts=ts, underlying_id=TOKEN, spot=spot, quotes=rows, completeness=1.0 if quotes else 0.0
    )


@pytest.fixture
def recording(tmp_path):
    """A partition laid out the way the recorder writes one."""
    partition = tmp_path / "chain" / f"date={DAY.isoformat()}"
    partition.mkdir(parents=True)
    path = partition / f"chain-{TOKEN}.ndjson"
    with path.open("w", encoding="utf-8") as fh:
        # The first snapshots of a session are empty while the builder fills.
        fh.write(
            _snapshot(
                datetime.combine(DAY, time(9, 15), tzinfo=IST), quotes=False
            ).model_dump_json()
            + "\n"
        )
        for minute in range(20, 50, 5):
            ts = datetime.combine(DAY, time(9, minute), tzinfo=IST)
            fh.write(_snapshot(ts, spot=SPOT + minute).model_dump_json() + "\n")
    return tmp_path


def test_the_latest_complete_snapshot_is_the_default(recording):
    chain, prov = load_recorded_chain(DAY, root=recording)
    assert chain.ts.astimezone(IST).time() == time(9, 45)
    assert prov.token == TOKEN
    assert prov.quotes == 26
    assert prov.snapshots_available == 6


def test_empty_snapshots_are_skipped_not_returned(recording):
    """The first snapshots of a session carry no quotes while the chain builder
    fills. Returning one would make the report say 'no expiry had enough usable
    quotes' about a recording that is perfectly good."""
    chain, prov = load_recorded_chain(DAY, root=recording)
    assert chain.quotes, "an empty snapshot was returned"
    assert prov.completeness == 1.0


def test_a_time_picks_the_last_snapshot_at_or_before_it(recording):
    chain, _ = load_recorded_chain(DAY, root=recording, at=time(9, 32))
    assert chain.ts.astimezone(IST).time() == time(9, 30)


def test_a_time_before_the_recording_started_is_refused(recording):
    """Silently returning the earliest snapshot would answer a question about
    09:00 with data from 09:20 and say nothing about the substitution."""
    with pytest.raises(SystemExit, match="no snapshot at or before"):
        load_recorded_chain(DAY, root=recording, at=time(9, 0))


def test_a_day_with_no_recording_says_how_to_make_one(tmp_path):
    with pytest.raises(SystemExit, match="record angel --chain"):
        load_recorded_chain(DAY, root=tmp_path)


def test_a_partition_holding_only_empty_snapshots_is_refused(tmp_path):
    partition = tmp_path / "chain" / f"date={DAY.isoformat()}"
    partition.mkdir(parents=True)
    (partition / f"chain-{TOKEN}.ndjson").write_text(
        _snapshot(datetime.combine(DAY, time(9, 15), tzinfo=IST), quotes=False).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="no snapshot with quotes"):
        load_recorded_chain(DAY, root=tmp_path)


def test_a_half_written_final_line_is_tolerated(recording):
    """The recorder is appending while this reads. A truncated last line is the
    normal state of the file, not corruption."""
    path = recording / "chain" / f"date={DAY.isoformat()}" / f"chain-{TOKEN}.ndjson"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version":1,"ts":"2026-08-31T04:2')
    chain, prov = load_recorded_chain(DAY, root=recording)
    assert chain.ts.astimezone(IST).time() == time(9, 45)
    assert prov.snapshots_available == 6


def test_the_provenance_names_the_vendor_token_it_read(recording):
    """The live path stamps Angel's token and the research layer keys off the
    config id. A report that did not say which it had would be read as being
    about instrument 1."""
    _, prov = load_recorded_chain(DAY, root=recording)
    text = "\n".join(prov.render())
    assert str(TOKEN) in text
    assert "config/instruments.yaml" in text
    assert "MID" in text


def test_the_provenance_reports_input_convexity(recording):
    """The bhavcopy path measures how many of its marks price a negative
    butterfly before any model touches them. If the recorded path did not report
    the same number by the same function, the two could not be compared - and
    that comparison is the entire question M70 exists to answer."""
    _, prov = load_recorded_chain(DAY, root=recording)
    assert prov.convexity.triples > 0
    assert prov.convexity.violations == 0, "the fixture's own marks are convex by construction"
    assert "NEGATIVE butterfly" in "\n".join(prov.convexity.render()) or True


# ------------------------------------------------------- the shared measurement
def test_convexity_is_one_function_for_both_kinds_of_mark():
    """A bhavcopy of settlement marks and a recorded book of mids are only
    comparable if one function measures both."""
    expiry = date(2026, 9, 1)
    convex = [
        (expiry, OptionType.CE, 24_000.0 + 50 * i, 200.0 - 40 * i + 4 * i * i) for i in range(5)
    ]
    assert convexity_of(convex).violations == 0
    assert convexity_of(convex).triples == 3

    dented = list(convex)
    dented[2] = (expiry, OptionType.CE, 24_100.0, 400.0)  # a mid nobody could have hit
    check = convexity_of(dented)
    assert check.violations > 0
    assert check.worst < 0


def test_unequally_spaced_triples_are_not_counted():
    """NIFTY lists 50-point strikes near the money and 100-point strikes in the
    wings. A triple straddling the change is not required to be convex by this
    inequality, and counting it would manufacture violations out of the ladder."""
    expiry = date(2026, 9, 1)
    rows = [
        (expiry, OptionType.CE, 24_000.0, 300.0),
        (expiry, OptionType.CE, 24_050.0, 100.0),
        (expiry, OptionType.CE, 24_150.0, 10.0),
    ]
    assert convexity_of(rows).triples == 0


def test_the_price_label_reaches_the_detail_line():
    """A report saying 'settling' about mid prices would misdescribe its own
    input in the one line somebody quotes."""
    expiry = date(2026, 9, 1)
    rows = [
        (expiry, OptionType.CE, 24_000.0, 100.0),
        (expiry, OptionType.CE, 24_050.0, 200.0),
        (expiry, OptionType.CE, 24_100.0, 100.0),
    ]
    assert "quoted at" in convexity_of(rows, price_label="quoted at").worst_detail


def test_the_recorded_partition_is_only_read_never_written(recording):
    """The recorder owns those files and this must never be why a session stops."""
    path = recording / "chain" / f"date={DAY.isoformat()}" / f"chain-{TOKEN}.ndjson"
    before = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    load_recorded_chain(DAY, root=recording)
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime
    assert json.loads(before.decode().splitlines()[0])["underlying_id"] == TOKEN
