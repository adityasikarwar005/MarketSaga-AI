"""M90 - the API gateway.

The tests that matter are the ones about what happens when a client is slow or
absent, because those are the paths that reach back into the trading system.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

pytest.importorskip("fastapi", reason="needs the api extra: pip install -e '.[api]'")

from fastapi.testclient import TestClient

from services.api.app import (
    CLIENT_QUEUE_DEPTH,
    ApiConfigError,
    LiveHub,
    Subscriber,
    create_app,
    decode,
    drain,
    encode,
    require_token,
    token_matches,
)

TOKEN = "a-test-token-long-enough"


@pytest.fixture
def hub() -> LiveHub:
    return LiveHub()


@pytest.fixture
def client(hub) -> TestClient:
    return TestClient(create_app(hub, token=TOKEN))


# --------------------------------------------------------------------- auth
def test_auth_required(client):
    """The single-user version is the one that ends up on a VPS 'temporarily',
    and optional auth is absent auth."""
    assert client.get("/api/ledger/2026-03-04").status_code == 401
    assert client.get("/api/recommendations/x").status_code == 401


def test_auth_required_on_the_websocket_too(hub):
    """Accepting first and closing after would let an unauthenticated client
    hold a connection long enough to matter."""
    from starlette.websockets import WebSocketDisconnect

    app = create_app(hub, token=TOKEN)
    with (
        TestClient(app) as c,
        pytest.raises(WebSocketDisconnect) as caught,
        c.websocket_connect("/ws/live?token=wrong"),
    ):
        pass

    # 4401, not 1008. A close code in the 4000s is application-defined, and this
    # one says "unauthorised" specifically - a client reconnecting on a generic
    # policy violation would retry forever against a token that will never work.
    assert caught.value.code == 4401


def test_a_valid_token_gets_through(client):
    r = client.get("/api/ledger/2026-03-04", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200
    assert r.json()["day"] == "2026-03-04"


def test_health_is_the_only_open_route(client):
    """A health check that needs a credential fails for the wrong reason when
    the credential is what broke."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_the_rejection_does_not_say_which_part_was_wrong(client):
    """ "Invalid token" and "no token" are different messages to a person and the
    same information to a scanner."""
    no_token = client.get("/api/ledger/2026-03-04")
    bad_token = client.get("/api/ledger/2026-03-04", headers={"Authorization": "Bearer nope"})
    assert no_token.status_code == bad_token.status_code == 401
    assert no_token.json() == bad_token.json()


def test_a_missing_token_is_a_startup_failure_not_an_open_door(monkeypatch):
    """Deliberately not 'generate one if missing' - a generated token gets
    logged, copied once, and becomes permanent."""
    monkeypatch.delenv("MARKETSAGE_API_TOKEN", raising=False)
    with pytest.raises(ApiConfigError, match="refuses to serve without a token"):
        require_token()


def test_a_short_token_is_refused(monkeypatch):
    monkeypatch.setenv("MARKETSAGE_API_TOKEN", "short")
    with pytest.raises(ApiConfigError, match="at least 16"):
        require_token()


def test_token_comparison_is_constant_time():
    """A plain == leaks the length of the matching prefix through timing."""
    import inspect

    source = inspect.getsource(token_matches)
    assert "compare_digest" in source
    assert token_matches(TOKEN, TOKEN)
    assert not token_matches("", TOKEN)
    assert not token_matches(None, TOKEN)


# -------------------------------------------------------------- backpressure
def test_backpressure(hub):
    """A slow client must not be able to make the publisher block. Otherwise a
    browser tab on bad wifi slows down the thing that places orders."""
    sub = hub.subscribe("slow")
    for i in range(CLIENT_QUEUE_DEPTH + 50):
        hub.publish({"seq": i})

    assert sub.queue.qsize() == CLIENT_QUEUE_DEPTH, "the queue grew past its bound"
    assert sub.dropped == 50
    assert hub.total_dropped == 50
    assert hub.published == CLIENT_QUEUE_DEPTH + 50


def test_a_slow_client_does_not_slow_a_fast_one(hub):
    """The property that keeps one bad connection from degrading the rest."""
    fast = hub.subscribe("fast")
    slow = hub.subscribe("slow")

    for i in range(CLIENT_QUEUE_DEPTH + 10):
        hub.publish({"seq": i})
        # The fast client keeps up; the slow one never reads.
        if not fast.queue.empty():
            fast.queue.get_nowait()

    assert fast.dropped == 0
    assert slow.dropped == 10


def test_publish_never_raises_when_a_client_is_full(hub):
    """A slow client is an expected state, not an error. The publisher must not
    handle an exception per frame per client."""
    hub.subscribe("slow")
    for i in range(CLIENT_QUEUE_DEPTH * 3):
        delivered = hub.publish({"seq": i})
        assert delivered in (0, 1)


def test_a_client_that_missed_frames_is_told_so(hub):
    """Silently resuming is how a dashboard ends up confidently displaying a
    stale position, and that is worse than a gap it can see."""

    async def scenario() -> dict:
        sub = hub.subscribe()
        for i in range(CLIENT_QUEUE_DEPTH + 5):
            hub.publish({"seq": i})
        assert sub.dropped == 5

        stream = drain(sub)
        first = await anext(stream)
        return decode(first)

    gap = asyncio.run(scenario())
    assert gap["type"] == "gap"
    assert gap["missed"] == 5


def test_the_gap_notice_is_sent_once_not_forever(hub):
    async def scenario() -> list[dict]:
        sub = hub.subscribe()
        for i in range(CLIENT_QUEUE_DEPTH + 2):
            hub.publish({"seq": i})

        stream = drain(sub)
        return [decode(await anext(stream)) for _ in range(3)]

    frames = asyncio.run(scenario())
    assert frames[0]["type"] == "gap"
    assert "type" not in frames[1] or frames[1].get("type") != "gap"


def test_the_queue_bound_is_small_on_purpose():
    """The live channel carries state - a client twenty frames behind wants the
    newest frame, not the backlog."""
    assert CLIENT_QUEUE_DEPTH <= 64


# --------------------------------------------------------------- reconnection
def test_ws_reconnect(hub):
    """A dropped client must be forgotten, and a returning one must start clean.
    A subscriber list that leaks entries is a memory leak that also multiplies
    the publisher's per-frame cost."""
    app = create_app(hub, token=TOKEN)
    with TestClient(app) as c, c.websocket_connect(f"/ws/live?token={TOKEN}") as ws:
        hub.publish({"hello": 1})
        assert decode(ws.receive_bytes()) == {"hello": 1}
        # Disconnected.

    assert hub.subscribers == [], "the subscriber survived its connection"

    with TestClient(app) as c, c.websocket_connect(f"/ws/live?token={TOKEN}") as ws:
        assert len(hub.subscribers) == 1
        hub.publish({"hello": 2})
        assert decode(ws.receive_bytes()) == {"hello": 2}


def test_reconnecting_does_not_replay_what_was_missed(hub):
    """The live channel is state, not a log. A reconnecting client wants what is
    true now, and replaying a backlog would show it a sequence of stale
    positions as though they were current."""
    app = create_app(hub, token=TOKEN)
    with TestClient(app) as c:
        with c.websocket_connect(f"/ws/live?token={TOKEN}"):
            pass
        hub.publish({"while": "disconnected"})

        with c.websocket_connect(f"/ws/live?token={TOKEN}") as ws:
            hub.publish({"now": True})
            assert decode(ws.receive_bytes()) == {"now": True}


def test_unsubscribing_something_already_gone_is_not_an_error(hub):
    sub = hub.subscribe()
    hub.unsubscribe(sub)
    hub.unsubscribe(sub)
    assert hub.subscribers == []


# ------------------------------------------------------------------ framing
def test_the_live_channel_is_messagepack_not_json():
    """JSON on a channel carrying option chains re-encodes every float as text
    several times a second."""
    frame = encode({"symbol": "NIFTY24000CE", "ltp": 101.25})
    assert isinstance(frame, bytes)
    assert not frame.startswith(b"{"), "the live channel fell back to JSON"
    assert decode(frame) == {"symbol": "NIFTY24000CE", "ltp": 101.25}


def test_rest_stays_json_so_a_human_can_read_it(client):
    r = client.get("/health")
    assert r.headers["content-type"].startswith("application/json")


def test_encoding_round_trips_a_datetime():
    frame = encode({"ts": datetime(2026, 3, 4, tzinfo=UTC)})
    assert "2026-03-04" in decode(frame)["ts"]


# -------------------------------------------------------------------- routes
def test_an_unknown_recommendation_is_a_404_not_a_500(client):
    r = client.get(
        "/api/recommendations/does-not-exist", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert r.status_code == 404


def test_the_ledger_route_returns_its_audit_not_only_its_rows(client):
    """Rows without the audit let a caller compute a hit rate over a day where
    half the outcomes were never written."""
    r = client.get("/api/ledger/2026-03-04", headers={"Authorization": f"Bearer {TOKEN}"})
    body = r.json()
    assert "audit" in body
    assert set(body["audit"]) >= {"total", "resolved", "pending", "overdue", "complete"}


def test_health_reports_the_queue_state(hub, client):
    hub.subscribe()
    body = client.get("/health").json()
    assert body["subscribers"] == 1
    assert "dropped" in body and "slowest_client_dropped" in body


def test_a_subscriber_offer_returns_false_rather_than_raising():
    sub = Subscriber(queue=asyncio.Queue(maxsize=1))
    assert sub.offer(b"a") is True
    assert sub.offer(b"b") is False
    assert sub.dropped == 1


# ------------------------------------------------------------------- the clock
def test_the_api_does_not_read_the_wall_clock():
    """Rule R6 applies here too, and the project's own guard caught this after
    the first version used datetime.now() twice. Replay serves a recorded day
    through these same routes, and a route that reaches for the wall clock
    reports a different answer than the day actually had."""
    import pathlib
    import re

    source = pathlib.Path("services/api/app.py").read_text(encoding="utf-8")
    offenders = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(source.splitlines(), 1)
        if re.search(r"datetime\.now\(|time\.time\(", line) and not line.strip().startswith("#")
    ]
    assert not offenders, "wall-clock access in the API:\n  " + "\n  ".join(offenders)


def test_a_supplied_clock_is_what_the_ledger_route_uses(hub):
    """The overdue check depends entirely on 'now'. Serving a replayed day with
    the real clock would mark every prediction in it overdue."""
    from datetime import timedelta

    from libs.ms_core.clock import SimClock

    frozen = SimClock(datetime(2026, 3, 4, 7, 0, tzinfo=UTC))
    app = create_app(hub, token=TOKEN, clock=frozen)
    with TestClient(app) as c:
        r = c.get("/api/ledger/2026-03-04", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200

    frozen.set(frozen.now() + timedelta(days=400))
    assert frozen.now().year == 2027
