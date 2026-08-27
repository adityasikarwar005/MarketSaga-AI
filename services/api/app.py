"""M90 - the API gateway.

REST for what has already happened, a WebSocket for what is happening now. The
split is not stylistic: history is a question with an answer, live state is a
stream with no end, and serving the second over the first produces a dashboard
that polls and a system that spends its time answering.

Backpressure is the requirement, not a nicety
---------------------------------------------

A dashboard that cannot keep up must be **dropped**, not waited for. If a slow
client can make the publisher block, then opening a browser tab on a bad wifi
connection slows down the thing that places orders — and that is a trading
system whose latency depends on who is watching it.

So every client has a bounded queue and a slow one loses messages, loudly. The
alternative designs both fail: an unbounded queue turns a slow client into
unbounded memory, and a blocking send turns it into unbounded latency.

Dropping is also *correct* here, not merely pragmatic. The live channel carries
state, and a client that missed three updates does not need all three - it needs
the current one. What it must never do is silently believe it is up to date,
which is why drops are counted and sent with the next frame.

Auth in single-user mode
------------------------

The blueprint's phrasing is "habits matter later", and it is right for a reason
worth stating: the single-user version is the one that gets deployed to a laptop,
then to a VPS "temporarily", and the moment auth is optional it is absent. It is
a static token here, checked on every route including the WebSocket, and the
absence of a configured token is a startup failure rather than an open door.

MessagePack on the live channel
-------------------------------

JSON on a channel carrying option chains re-encodes every float as text several
times a second. MessagePack is roughly half the bytes and skips the parse. REST
stays JSON, because a human debugging a report should be able to read the
response.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect

from libs.ms_core.clock import Clock, RealClock
from libs.ms_core.obs import get_logger

log = get_logger(__name__)

# Imported at module scope, not inside create_app(), and that is load-bearing.
# This module uses `from __future__ import annotations`, so every annotation is a
# string that FastAPI resolves against the *module* globals. With the import
# inside the factory, `websocket: WebSocket` resolved to nothing and FastAPI
# silently reinterpreted the parameter as a required query field - the socket
# then closed with 1008 "Field required" and looked like an auth failure.

TOKEN_VAR = "MARKETSAGE_API_TOKEN"

# Per-client queue depth. Small on purpose: the live channel carries state, and a
# client twenty frames behind wants the newest one, not the backlog.
CLIENT_QUEUE_DEPTH = 32


class ApiConfigError(RuntimeError):
    """Raised at startup when the API is not safe to serve."""


def require_token() -> str:
    """The configured token, or a startup failure.

    Deliberately not "generate one if missing". A generated token gets logged,
    copied once, and becomes permanent; an absent one has to be a decision.
    """
    token = os.environ.get(TOKEN_VAR, "").strip()
    if not token:
        raise ApiConfigError(
            f"{TOKEN_VAR} is not set. The API refuses to serve without a token, "
            "including in single-user mode - the single-user version is the one "
            "that ends up on a VPS 'temporarily', and optional auth is absent "
            'auth. Generate one with: python -c "import secrets; '
            'print(secrets.token_urlsafe(32))"'
        )
    if len(token) < 16:
        raise ApiConfigError(
            f"{TOKEN_VAR} is {len(token)} characters. Use at least 16 - a short "
            "token on a machine reachable from the internet is a decoration."
        )
    return token


def token_matches(supplied: str | None, expected: str) -> bool:
    """Constant-time comparison.

    A plain `==` leaks the length of the matching prefix through timing. That is
    a small leak against a local dashboard and a real one against anything
    reachable, and the correct version costs one import.
    """
    if not supplied:
        return False
    return secrets.compare_digest(supplied, expected)


# --------------------------------------------------------------------------- #
# Live state fan-out
# --------------------------------------------------------------------------- #
@dataclass
class Subscriber:
    """One connected client, with a bounded mailbox."""

    queue: asyncio.Queue[bytes]
    dropped: int = 0
    name: str = ""

    def offer(self, frame: bytes) -> bool:
        """Enqueue, or drop and count. Never block.

        Returning False rather than raising: a slow client is an expected state,
        not an error, and the publisher must not have to handle an exception per
        frame per client.
        """
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            self.dropped += 1
            return False
        return True


@dataclass
class LiveHub:
    """Fans state out to every subscriber without letting any of them block it.

    The publisher's cost is O(subscribers) queue writes and nothing else. There
    is no path where a client's read speed reaches back into the publisher,
    which is the property that keeps a browser tab from slowing down order
    placement.
    """

    subscribers: list[Subscriber] = field(default_factory=list)
    published: int = 0
    total_dropped: int = 0

    def subscribe(self, name: str = "") -> Subscriber:
        sub = Subscriber(queue=asyncio.Queue(maxsize=CLIENT_QUEUE_DEPTH), name=name)
        self.subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        if sub in self.subscribers:
            self.subscribers.remove(sub)

    def publish(self, payload: dict[str, Any]) -> int:
        """Send to everyone who can take it. Returns how many got it."""
        frame = encode(payload)
        delivered = 0
        for sub in self.subscribers:
            if sub.offer(frame):
                delivered += 1
            else:
                self.total_dropped += 1
        self.published += 1
        return delivered

    def health(self) -> dict[str, Any]:
        return {
            "subscribers": len(self.subscribers),
            "published": self.published,
            "dropped": self.total_dropped,
            "slowest_client_dropped": max((s.dropped for s in self.subscribers), default=0),
        }


def encode(payload: dict[str, Any]) -> bytes:
    """MessagePack, falling back to JSON only if msgpack is absent.

    The fallback is loud rather than silent: a channel that quietly changes
    encoding produces a client that decodes garbage and blames the network.
    """
    try:
        import msgpack
    except ImportError:  # pragma: no cover - msgpack ships with the api extra
        import json

        log.info("msgpack unavailable, live channel is falling back to JSON")
        return json.dumps(payload, default=str).encode()
    return bytes(msgpack.packb(payload, default=str, use_bin_type=True))


def decode(frame: bytes) -> dict[str, Any]:
    try:
        import msgpack
    except ImportError:  # pragma: no cover
        import json

        data: dict[str, Any] = json.loads(frame)
        return data
    unpacked: dict[str, Any] = msgpack.unpackb(frame, raw=False)
    return unpacked


async def drain(sub: Subscriber, clock: Clock | None = None) -> AsyncIterator[bytes]:
    """Yield frames, prefixing a drop notice when the client fell behind.

    A client that missed frames must know it missed them. Silently resuming is
    how a dashboard ends up confidently displaying a stale position, and that is
    worse than a gap it can see.
    """
    while True:
        frame = await sub.queue.get()
        if sub.dropped:
            missed = sub.dropped
            sub.dropped = 0
            at = (clock or RealClock()).now()
            yield encode({"type": "gap", "missed": missed, "ts": at.isoformat()})
        yield frame


# --------------------------------------------------------------------------- #
# The app
# --------------------------------------------------------------------------- #
def create_app(
    hub: LiveHub | None = None, *, token: str | None = None, clock: Clock | None = None
) -> Any:
    """Build the ASGI app.

    Takes its dependencies rather than reaching for globals, so a test can drive
    it without a server, a database or a network.
    """
    api_token = token or require_token()
    live = hub or LiveHub()
    # Injected, not read. Rule R6 applies to the API too: replay serves a
    # recorded day through these same routes, and a route that reaches for the
    # wall clock reports a different answer than the day actually had. The
    # ledger's "is this overdue" check is exactly such a route.
    tick = clock or RealClock()
    app = FastAPI(title="MarketSage", version="0.1.0")
    app.state.hub = live

    def authorise(authorization: str | None = Header(default=None)) -> None:
        supplied = None
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        if not token_matches(supplied, api_token):
            # 401 with no detail about which part was wrong. "Invalid token" and
            # "no token" are different messages to a person and the same
            # information to a scanner.
            raise HTTPException(status_code=401, detail="unauthorised")

    @app.get("/health")
    def health() -> dict[str, Any]:
        """The only unauthenticated route.

        Open because a health check that needs a credential fails for the wrong
        reason when the credential is what broke, and it exposes nothing beyond
        liveness and queue depth.
        """
        return {"status": "ok", **live.health()}

    @app.get("/api/ledger/{day}", dependencies=[Depends(authorise)])
    def ledger(day: str) -> dict[str, Any]:
        from libs.ms_risk.ledger import PaperLedger, to_rows, verify

        store = PaperLedger()
        when = datetime.fromisoformat(day).replace(tzinfo=UTC)
        entries = store.entries(when)
        audit = verify(store, when, now=tick.now())
        return {
            "day": day,
            "entries": to_rows(entries),
            "audit": {
                "total": audit.total,
                "resolved": audit.resolved,
                "unresolvable": audit.unresolvable,
                "pending": audit.pending,
                "overdue": audit.overdue,
                "complete": audit.complete,
            },
        }

    @app.get("/api/recommendations/{reco_id}", dependencies=[Depends(authorise)])
    def recommendation(reco_id: str) -> dict[str, Any]:
        from libs.ms_risk.recommendation import RecommendationStore

        try:
            return RecommendationStore().load(reco_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        # Authenticated before accept. Accepting first and closing after would
        # let an unauthenticated client hold a connection long enough to matter.
        supplied = websocket.query_params.get("token")
        if not token_matches(supplied, api_token):
            await websocket.close(code=4401, reason="unauthorised")
            return

        await websocket.accept()
        sub = live.subscribe(name=websocket.client.host if websocket.client else "")
        try:
            async for frame in drain(sub, tick):
                await websocket.send_bytes(frame)
        except WebSocketDisconnect:
            pass
        finally:
            live.unsubscribe(sub)
            log.info(
                "live client gone",
                fields={"dropped": sub.dropped, "remaining": len(live.subscribers)},
            )

    return app
