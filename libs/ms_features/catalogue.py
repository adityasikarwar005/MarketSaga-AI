"""Feature catalogue - groups A and F to start.

Deliberately a small set. The blueprint calls for 120-250 engineered features;
adding them before the parity test and the leakage linter are trusted would mean
building two hundred things on a foundation nobody has checked.

What is here proves the framework end to end and is genuinely load-bearing:

* **Group A** price and trend, including the time-of-day z-scores. Intraday
  volatility is strongly U-shaped, so a raw 0.3% move at 09:20 is ordinary and
  the same move at 13:30 is extreme. Any model fed un-normalised returns spends
  its capacity re-learning the shape of the trading day.
* **Group F** calendar and session context, which is cheap and unreasonably
  predictive around the open, the close and expiry.

Every function takes exactly one argument and can only read the past.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from statistics import fmean, pstdev

from libs.ms_core.schemas import Candle
from libs.ms_features.registry import FeatureContext, feature

# --------------------------------------------------------------------------- #
# Group A - price and trend
# --------------------------------------------------------------------------- #


def _window(ctx: FeatureContext, minutes: int) -> Sequence[Candle] | None:
    """A complete lookback window, or None.

    A feature named for a window must compute over that window. Computing a
    two-minute return and calling it ``ret_30m`` is worse than returning None:
    None is missing data the model can be told about, whereas a mislabelled
    number is a lie it will happily learn from. Windows are naturally short at
    the session open, which is exactly when this matters.
    """
    bars = ctx.candles(minutes=minutes)
    return bars if len(bars) >= minutes else None


def _log_return(ctx: FeatureContext, minutes: int) -> float | None:
    bars = _window(ctx, minutes)
    if bars is None or bars[0].o <= 0:
        return None
    return math.log(bars[-1].c / bars[0].o)


@feature(name="ret_1m", version=1, window_minutes=2, group="A")
def ret_1m(ctx: FeatureContext) -> float | None:
    """Log return over the last completed minute."""
    return _log_return(ctx, 1)


@feature(name="ret_5m", version=1, window_minutes=6, group="A")
def ret_5m(ctx: FeatureContext) -> float | None:
    """Log return over the last five completed minutes."""
    return _log_return(ctx, 5)


@feature(name="ret_15m", version=1, window_minutes=16, group="A")
def ret_15m(ctx: FeatureContext) -> float | None:
    """Log return over the last fifteen completed minutes."""
    return _log_return(ctx, 15)


@feature(name="ret_30m", version=1, window_minutes=31, group="A")
def ret_30m(ctx: FeatureContext) -> float | None:
    """Log return over the last thirty completed minutes."""
    return _log_return(ctx, 30)


@feature(name="rv_15m", version=1, window_minutes=16, group="A")
def rv_15m(ctx: FeatureContext) -> float | None:
    """Realised volatility from minute log returns, annualised.

    Close-to-close rather than range-based, deliberately: this one is the
    baseline that the Garman-Klass estimator below has to beat.
    """
    bars = _window(ctx, 15)
    if bars is None:
        return None
    rets = [math.log(b.c / a.c) for a, b in pairwise(bars) if a.c > 0]
    if len(rets) < 4:
        return None
    return pstdev(rets) * math.sqrt(252 * 375)


@feature(name="rv_gk_15m", version=1, window_minutes=16, group="A")
def rv_gk_15m(ctx: FeatureContext) -> float | None:
    """Garman-Klass realised volatility.

    Uses the whole bar rather than just its close, which makes it several times
    more efficient per observation - the difference between a usable volatility
    estimate over fifteen minutes and a noisy one.
    """
    window = _window(ctx, 15)
    if window is None:
        return None
    bars = [b for b in window if not b.synthetic and b.low > 0 and b.o > 0]
    if len(bars) < 5:
        return None  # window is complete but mostly carried-forward candles
    total = 0.0
    for b in bars:
        hl = math.log(b.h / b.low)
        co = math.log(b.c / b.o)
        total += 0.5 * hl * hl - (2 * math.log(2) - 1) * co * co
    return math.sqrt(max(0.0, total / len(bars)) * 252 * 375)


@feature(name="ret_5m_z", version=1, deps=("ret_5m", "rv_15m"), window_minutes=16, group="A")
def ret_5m_z(ctx: FeatureContext) -> float | None:
    """Five-minute return in units of current volatility.

    The normalisation that matters. Without it a model has to learn separately
    that a given move means different things at 09:20 and at 13:30, and it will
    spend most of its capacity doing so.
    """
    ret = ctx.value("ret_5m")
    vol = ctx.value("rv_15m")
    if ret is None or vol is None or vol <= 0:
        return None
    per_5m = vol * math.sqrt(5 / (252 * 375))
    return ret / per_5m if per_5m > 0 else None


@feature(name="range_position", version=1, window_minutes=375, group="A")
def range_position(ctx: FeatureContext) -> float | None:
    """Where price sits in the session range so far, from 0 (low) to 1 (high).

    Session-scoped rather than a fixed lookback: "so far" is the whole point, so
    unlike ``ret_30m`` this one is meaningful from partway through the day and
    must not wait for a full 375-bar window.
    """
    bars = ctx.candles(minutes=375)
    if len(bars) < 5:
        return None
    hi = max(b.h for b in bars)
    lo = min(b.low for b in bars)
    if hi <= lo:
        return 0.5
    return (bars[-1].c - lo) / (hi - lo)


@feature(name="dist_from_vwap", version=1, window_minutes=375, group="A")
def dist_from_vwap(ctx: FeatureContext) -> float | None:
    """Distance from session VWAP as a fraction of price."""
    bars = [b for b in ctx.candles(minutes=375) if b.v > 0 and b.vwap]
    if not bars:
        return None
    turnover = sum((b.vwap or 0.0) * b.v for b in bars)
    volume = sum(b.v for b in bars)
    if volume <= 0:
        return None
    vwap = turnover / volume
    last = bars[-1].c
    return (last - vwap) / vwap if vwap > 0 else None


@feature(name="efficiency_ratio_30m", version=1, window_minutes=31, group="A")
def efficiency_ratio_30m(ctx: FeatureContext) -> float | None:
    """Net move divided by total path length, from 0 (chop) to 1 (clean trend).

    Distinguishes a directional move from noise of the same magnitude, which is
    the distinction a raw return cannot make.
    """
    bars = _window(ctx, 30)
    if bars is None:
        return None
    net = abs(bars[-1].c - bars[0].o)
    path = sum(abs(b.c - a.c) for a, b in pairwise(bars))
    return net / path if path > 0 else 0.0


@feature(name="volume_ratio_5m", version=1, window_minutes=31, group="A")
def volume_ratio_5m(ctx: FeatureContext) -> float | None:
    """Recent volume against its own trailing average. Attention proxy."""
    bars = _window(ctx, 30)
    if bars is None:
        return None
    recent = [b.v for b in bars[-5:]]
    baseline = [b.v for b in bars[:-5]]
    mean_baseline = fmean(baseline) if baseline else 0.0
    if mean_baseline <= 0:
        return None
    return fmean(recent) / mean_baseline


@feature(name="synthetic_share_15m", version=1, window_minutes=16, group="A")
def synthetic_share_15m(ctx: FeatureContext) -> float | None:
    """Fraction of the recent window that was carried forward, not traded.

    A data-quality feature the model can see. When the feed is thin every other
    feature in this group degrades, and a model that knows it is looking at
    filled-in candles can learn to distrust them rather than trade on them.
    """
    bars = _window(ctx, 15)
    if bars is None:
        return None
    return sum(1 for b in bars if b.synthetic) / len(bars)


# --------------------------------------------------------------------------- #
# Group F - calendar and session
# --------------------------------------------------------------------------- #
SESSION_START_MINUTES = 9 * 60 + 15
SESSION_END_MINUTES = 15 * 60 + 30
SESSION_LENGTH = SESSION_END_MINUTES - SESSION_START_MINUTES


def _minutes_since_open(ctx: FeatureContext) -> float:
    from zoneinfo import ZoneInfo

    local = ctx.as_of.astimezone(ZoneInfo("Asia/Kolkata"))
    return (local.hour * 60 + local.minute + local.second / 60.0) - SESSION_START_MINUTES


@feature(name="minutes_since_open", version=1, group="F")
def minutes_since_open(ctx: FeatureContext) -> float | None:
    """Elapsed session minutes. The anchor every time-of-day effect hangs on."""
    return _minutes_since_open(ctx)


@feature(name="minutes_to_close", version=1, group="F")
def minutes_to_close(ctx: FeatureContext) -> float | None:
    return SESSION_LENGTH - _minutes_since_open(ctx)


@feature(name="session_progress", version=1, deps=("minutes_since_open",), group="F")
def session_progress(ctx: FeatureContext) -> float | None:
    """Position through the session, 0 at the open and 1 at the close."""
    elapsed = ctx.value("minutes_since_open")
    if elapsed is None:
        return None
    return max(0.0, min(1.0, elapsed / SESSION_LENGTH))


@feature(name="tod_vol_multiplier", version=1, deps=("session_progress",), group="F")
def tod_vol_multiplier(ctx: FeatureContext) -> float | None:
    """The expected U-shape of intraday volatility.

    A prior, not a measurement: it tells the model roughly how much movement is
    normal at this point of the day, so a raw return can be read against it.
    Replace with an empirically fitted profile once enough sessions exist -
    that is a Phase 5 refinement, and the shape is the right starting point.
    """
    x = ctx.value("session_progress")
    if x is None:
        return None
    return 0.7 + 1.6 * (math.exp(-8 * x) + math.exp(-8 * (1 - x)))


@feature(name="is_opening_range", version=1, deps=("minutes_since_open",), group="F")
def is_opening_range(ctx: FeatureContext) -> float | None:
    """First fifteen minutes. Price formation here is not like the rest of the day."""
    elapsed = ctx.value("minutes_since_open")
    return None if elapsed is None else float(0 <= elapsed < 15)


@feature(name="is_closing_range", version=1, deps=("minutes_to_close",), group="F")
def is_closing_range(ctx: FeatureContext) -> float | None:
    """Last fifteen minutes, when closing auctions and squaring-off dominate."""
    remaining = ctx.value("minutes_to_close")
    return None if remaining is None else float(0 <= remaining <= 15)


@feature(name="day_of_week", version=1, group="F")
def day_of_week(ctx: FeatureContext) -> float | None:
    from zoneinfo import ZoneInfo

    return float(ctx.as_of.astimezone(ZoneInfo("Asia/Kolkata")).weekday())


# --------------------------------------------------------------------------- #
# Group C - option-derived (the differentiated group; started, not finished)
# --------------------------------------------------------------------------- #
@feature(name="atm_iv", version=1, group="C")
def atm_iv(ctx: FeatureContext) -> float | None:
    """At-the-money implied volatility from the most recent chain snapshot."""
    snap = ctx.chain()
    if snap is None or not snap.quotes:
        return None
    strike = snap.atm_strike()
    ivs = [q.iv for q in snap.quotes if q.strike == strike and q.iv is not None]
    return fmean(ivs) if ivs else None


@feature(name="iv_skew_25d", version=1, group="C")
def iv_skew_25d(ctx: FeatureContext) -> float | None:
    """Downside minus upside implied volatility, a proxy for the risk reversal.

    The market's directional fear, and one of the strongest single
    option-derived predictors. Uses a fixed moneyness band until a fitted
    surface (M70) can supply true 25-delta strikes.
    """
    snap = ctx.chain()
    if snap is None or not snap.quotes:
        return None
    spot = snap.spot
    lo = [q.iv for q in snap.quotes if q.iv is not None and 0.96 * spot < q.strike < 0.99 * spot]
    hi = [q.iv for q in snap.quotes if q.iv is not None and 1.01 * spot < q.strike < 1.04 * spot]
    if not lo or not hi:
        return None
    return fmean(lo) - fmean(hi)


@feature(name="chain_completeness", version=1, group="C")
def chain_completeness(ctx: FeatureContext) -> float | None:
    """How complete the chain behind the option features was.

    Another data-quality feature the model can see: an option signal computed
    from a 60%-complete chain deserves less weight than one from a full chain,
    and the model can learn that if - and only if - it is told.
    """
    snap = ctx.chain()
    return None if snap is None else snap.completeness


@feature(name="chain_age_seconds", version=1, group="C")
def chain_age_seconds(ctx: FeatureContext) -> float | None:
    """How stale the chain is at decision time. Never negative, by construction."""
    snap = ctx.chain()
    if snap is None:
        return None
    return (ctx.as_of - snap.ts).total_seconds()
