"""Feature catalogue - groups A, F, C and D.

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
* **Group C** option-derived: the differentiated group, and the only one here
  that reads something which is not in the price series. Skew, term structure,
  the implied move, where open interest sits, and the variance risk premium.

**Group C is the reason a live chain recording matters.** Five of its ten
features need a book - a bid and an ask at a strike - and the daily bhavcopy
archive has none, so they return None across eight years of history and can only
ever be computed from an intraday capture. That is not a limitation of this
module; it is M70's finding about the archive, restated as a set of Nones a
model can see. A chain snapshot missed at 09:20 on a Monday cannot be
reconstructed on Tuesday from any source.

* **Group D** cross-asset, and only the half of it that overlaps the Indian
  session: USD/INR and the Asian composite trade while Nifty does. The overnight
  channel - US futures, DXY, Brent, and by the config's own note the strongest
  of them - is deliberately absent. An overnight move is a session-scoped
  quantity, computed once at the open and constant thereafter, and it already
  exists as `overnight_moves` in `ms_data.crossasset`. Forcing it into a rolling
  window would model a daily number as an intraday one.

Groups B and E remain unbuilt. B (microstructure) waits on a feed that provides
depth, and E (event features) on the news layer, which is 3/10 and gated on a
source decision that has not been taken.

Every function takes exactly one argument and can only read the past.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime
from itertools import pairwise
from statistics import fmean, pstdev
from zoneinfo import ZoneInfo

from libs.ms_core.schemas import Candle, ChainSnapshot, OptionType
from libs.ms_features.registry import FeatureContext, feature

# The gamma kernel is M63's, imported rather than reimplemented - see
# `gamma_concentration_shape` for what went wrong the one time it was not.
from libs.ms_models.positioning import MIN_STRIKES, gamma_kernel, year_fraction

IST = ZoneInfo("Asia/Kolkata")

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
    local = ctx.as_of.astimezone(IST)
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
    return float(ctx.as_of.astimezone(IST).weekday())


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


# --------------------------------------------------------------------------- #
# Group C continued - term structure, positioning, variance risk premium
#
# Everything below reads the chain and only the chain. Two consequences worth
# stating once rather than repeating in six docstrings:
#
# * **No chain, no value.** On a bhavcopy-derived chain there is no book, so
#   every feature here that needs a bid and an ask returns None for the whole
#   archive. That is the correct answer and it is M70's finding restated: the
#   daily archive cannot support these, and only a recorded intraday chain can.
# * **The front expiry is the one that matters** and is picked by date, not by
#   position in the file, so a snapshot that still carries an expired series
#   cannot silently become the reference.
# --------------------------------------------------------------------------- #
def _ist_date(ts: datetime) -> date:
    """Exchange-local calendar date.

    Refuses a naive datetime rather than letting ``astimezone`` interpret it as
    the machine's local time. That substitution does not raise, is wrong by five
    and a half hours, and would move an expiry across a day boundary - silently
    changing days-to-expiry by one on every afternoon snapshot.
    """
    if ts.tzinfo is None:
        raise ValueError(f"naive timestamp {ts!r}; schedule and expiry times are tz-aware IST")
    return ts.astimezone(IST).date()


def _live_expiries(snap: ChainSnapshot, as_of: datetime) -> list[datetime]:
    """Expiries that have not already passed, by exchange-local date, earliest first."""
    today = _ist_date(as_of)
    return sorted({q.expiry for q in snap.quotes if _ist_date(q.expiry) >= today})


def _front_expiry(snap: ChainSnapshot, as_of: datetime) -> datetime | None:
    live = _live_expiries(snap, as_of)
    return live[0] if live else None


def _atm_strike_for(snap: ChainSnapshot, expiry: datetime) -> float | None:
    strikes = {q.strike for q in snap.quotes if q.expiry == expiry}
    return min(strikes, key=lambda k: abs(k - snap.spot)) if strikes else None


def _atm_iv_for(snap: ChainSnapshot, expiry: datetime) -> float | None:
    """Mean implied volatility of the at-the-money strike of one expiry.

    Calls and puts are averaged together. Put-call parity says they should agree,
    and where they do not the gap is a funding or early-exercise artefact rather
    than information about volatility.
    """
    atm = _atm_strike_for(snap, expiry)
    if atm is None:
        return None
    ivs = [q.iv for q in snap.quotes if q.expiry == expiry and q.strike == atm and q.iv is not None]
    return fmean(ivs) if ivs else None


@feature(name="days_to_expiry", version=1, group="C")
def days_to_expiry(ctx: FeatureContext) -> float | None:
    """Calendar days from now to the front expiry.

    Cheap, and on the only real-data measurement this project has made of the
    options layer it was the strongest thing in the study: M63 scored
    days-to-expiry at -0.243 against the HAR residual while the gamma statistic
    it was built to test managed +0.117 and did not clear its permutation test.
    A model given the positioning features below and not given this one would be
    handed a confound (corr = -0.83 on 64 real sessions) with no way to divide
    it out.

    Sourced from the chain, so it is None without one. A calendar-derived version
    that survives a missing chain needs the holiday roll, and therefore config
    access the feature context does not expose; that belongs with the session
    calendar rather than here.
    """
    snap = ctx.chain()
    if snap is None or not snap.quotes:
        return None
    expiry = _front_expiry(snap, ctx.as_of)
    if expiry is None:
        return None
    return float((_ist_date(expiry) - _ist_date(ctx.as_of)).days)


@feature(name="iv_term_slope", version=1, group="C")
def iv_term_slope(ctx: FeatureContext) -> float | None:
    """Back-month minus front-month at-the-money implied volatility.

    The term structure in one number. It is normally positive - far volatility is
    dearer than near - and it inverts when the market prices something happening
    soon, which is the state worth knowing about.

    None when the snapshot carries a single expiry, rather than zero. Zero is a
    flat term structure, which is a real and rare market condition; conflating it
    with "only one expiry was captured" would teach a model that the two look
    alike.
    """
    snap = ctx.chain()
    if snap is None or not snap.quotes:
        return None
    live = _live_expiries(snap, ctx.as_of)
    if len(live) < 2:
        return None
    front = _atm_iv_for(snap, live[0])
    back = _atm_iv_for(snap, live[1])
    if front is None or back is None:
        return None
    return back - front


@feature(name="implied_move_pct", version=1, group="C")
def implied_move_pct(ctx: FeatureContext) -> float | None:
    """Front-expiry at-the-money straddle as a fraction of spot.

    What the market itself is charging for the move between now and expiry, and
    the natural denominator for any forecast this system makes: a predicted move
    smaller than the straddle is not a trade however confident the model is.

    Priced from the mid, never the last trade. A straddle marked off two
    last-traded prices that printed minutes apart on opposite sides of their
    spreads is not a straddle price, and the error is largest exactly when the
    book is thin and the number would matter most. So this is None across the
    whole bhavcopy archive, which carries no book at all.
    """
    snap = ctx.chain()
    if snap is None or not snap.quotes or snap.spot <= 0:
        return None
    expiry = _front_expiry(snap, ctx.as_of)
    if expiry is None:
        return None
    atm = _atm_strike_for(snap, expiry)
    if atm is None:
        return None
    legs = [q for q in snap.quotes if q.expiry == expiry and q.strike == atm]
    call = next((q.mid for q in legs if q.opt_type is OptionType.CE), None)
    put = next((q.mid for q in legs if q.opt_type is OptionType.PE), None)
    if call is None or put is None:
        return None
    return (call + put) / snap.spot


@feature(name="gamma_concentration_shape", version=1, deps=("days_to_expiry",), group="C")
def gamma_concentration_shape(ctx: FeatureContext) -> float | None:
    """How tightly front-expiry open interest sits around spot.

    Kernel-weighted open interest divided by total open interest, times spot.
    Both normalisations are load-bearing: dividing by total open interest leaves
    the *shape* of the distribution rather than its level, and multiplying by
    spot makes the result dimensionless, so a value from 2016 at NIFTY 8,000
    means the same thing as one from today at 24,000. The raw sum has neither
    property and would drift with contract size and index level alone.

    **This is not M63's statistic and must not be reported as it.** M63 measured
    the raw sum, cycle-adjusted against a fitted per-horizon norm, and the answer
    was negative: Spearman +0.117 against the HAR residual, permutation p = 0.189
    on 128 sessions. What is here is the shape component of that quantity, which
    M63 identified as carrying most of the expiry-cycle confound (corr -0.82
    after the level was divided out, against -0.83 before). It is offered as a
    model input, not as a finding, and it is paired with ``days_to_expiry`` so
    the confound is visible to whatever consumes it.

    The kernel is M63's, imported rather than reimplemented. The two were
    computed separately once before, and a test pinning the property passed while
    the path that actually weighted the strikes used a different number.
    """
    snap = ctx.chain()
    if snap is None or snap.spot <= 0:
        return None
    dte = ctx.value("days_to_expiry")
    if dte is None:
        return None
    expiry = _front_expiry(snap, ctx.as_of)
    if expiry is None:
        return None
    records = [(q.strike, q.oi) for q in snap.quotes if q.expiry == expiry and q.oi]
    if len({k for k, _ in records}) < MIN_STRIKES:
        return None
    total_oi = sum(oi for _, oi in records if oi)
    if total_oi <= 0:
        return None
    t = year_fraction(int(dte))
    weighted = sum(gamma_kernel(snap.spot, k, t) * oi for k, oi in records if oi)
    return snap.spot * weighted / total_oi


@feature(name="put_call_oi_ratio", version=1, group="C")
def put_call_oi_ratio(ctx: FeatureContext) -> float | None:
    """Front-expiry put open interest over call open interest.

    Positioning rather than flow: open interest is what is still on the books,
    not what traded today. It is read as a contrarian gauge by convention, which
    this module takes no position on - the sign is for a model to estimate from
    outcomes, the same discipline M63 applied when it declined to assume a dealer
    sign.
    """
    snap = ctx.chain()
    if snap is None or not snap.quotes:
        return None
    expiry = _front_expiry(snap, ctx.as_of)
    if expiry is None:
        return None
    puts = sum(q.oi or 0 for q in snap.quotes if q.expiry == expiry and q.opt_type is OptionType.PE)
    calls = sum(
        q.oi or 0 for q in snap.quotes if q.expiry == expiry and q.opt_type is OptionType.CE
    )
    if calls <= 0 or puts <= 0:
        return None
    return puts / calls


@feature(name="vrp", version=1, deps=("atm_iv", "rv_15m"), group="C")
def vrp(ctx: FeatureContext) -> float | None:
    """Variance risk premium: implied minus realised, both annualised.

    The classic form compares a 30-day implied against 30-day realised. This is
    the intraday analogue and it is a noisier thing: ``rv_15m`` annualises the
    dispersion of fifteen one-minute returns, so its sampling error is large and
    a single reading says little. It is still worth carrying, because its
    *persistence* across a session is informative where its level is not - but
    nobody should read one reading of it as a measurement.

    Units line up by construction: ``atm_iv`` is an annualised fraction and
    ``rv_15m`` is annualised by the same 252 x 375 convention. That agreement is
    the only reason a subtraction is meaningful here, and it is what would break
    first if either side changed its scaling.
    """
    iv = ctx.value("atm_iv")
    rv = ctx.value("rv_15m")
    if iv is None or rv is None:
        return None
    return iv - rv


# --------------------------------------------------------------------------- #
# Group D - cross-asset
#
# Cross-asset series share the Candle schema and live in the reserved 1000+
# instrument range (`config/crossasset.yaml`), so they need no new accessor:
# `ctx.candles(instrument_id=...)` already reaches them, filtered by the same
# right-open window as everything else. The ids below are identifiers, not
# tunables; the config file is where they are defined and where the trading
# hours and the transmission channel for each are written down.
#
# **Only the intraday-overlap series are here.** USDINR trades 09:00-17:00 IST
# and ASIA_IDX 05:30-11:30 IST, so both are live while Nifty is. The overnight
# channel - US equity futures, DXY, Brent, and by the config's own note the
# strongest of the lot - is deliberately absent, and not because it is hard to
# reach. An overnight move is a *session-scoped* quantity: computed once at the
# open from the previous evening, constant for the rest of the day, and already
# built as `overnight_moves` in `ms_data.crossasset` where M35 uses it. Forcing
# it into a rolling-window feature would need the online buffer to hold two days
# of history for those instruments and would still be modelling a daily number
# as an intraday one. It needs a session-scoped context, which is its own piece
# of work rather than a wider window.
# --------------------------------------------------------------------------- #
USDINR_ID = 1006
ASIA_IDX_ID = 1008

# How much of a requested window a cross-asset series has to actually span.
# Group A can demand one bar per minute because it reads the instrument the
# session is built around. A foreign series legitimately arrives sparsely, so
# demanding 30 bars for a 30-minute return would return None all day - but
# accepting any 3 bars would compute a two-minute move and label it `ret_30m`,
# which is the same mislabelling `_window` exists to prevent, wearing a
# different disguise. So the gate is on the span the bars cover, not on how many
# of them there are.
CROSS_SPAN_COVERAGE = 0.8


def _cross_window(ctx: FeatureContext, instrument_id: int, minutes: int) -> Sequence[Candle] | None:
    """Bars for a cross-asset series that genuinely span the requested window."""
    bars = ctx.candles(minutes=minutes, instrument_id=instrument_id)
    if len(bars) < 2:
        return None
    span = (bars[-1].ts - bars[0].ts).total_seconds() / 60.0
    return bars if span >= CROSS_SPAN_COVERAGE * (minutes - 1) else None


def _cross_return(ctx: FeatureContext, instrument_id: int, minutes: int) -> float | None:
    bars = _cross_window(ctx, instrument_id, minutes)
    if bars is None or bars[0].o <= 0:
        return None
    return math.log(bars[-1].c / bars[0].o)


@feature(name="usdinr_ret_5m", version=1, window_minutes=6, group="D")
def usdinr_ret_5m(ctx: FeatureContext) -> float | None:
    """Five-minute log return in USD/INR.

    The one cross-asset channel the config describes as *bidirectional* with
    Nifty rather than one-way, which is a warning as much as a feature: a
    contemporaneous move in both is not evidence that one leads the other, and
    reading it that way needs a VAR rather than a regression.
    """
    return _cross_return(ctx, USDINR_ID, 5)


@feature(name="usdinr_ret_30m", version=1, window_minutes=31, group="D")
def usdinr_ret_30m(ctx: FeatureContext) -> float | None:
    """Thirty-minute log return in USD/INR - the slower rupee channel."""
    return _cross_return(ctx, USDINR_ID, 30)


@feature(name="asia_idx_ret_30m", version=1, window_minutes=31, group="D")
def asia_idx_ret_30m(ctx: FeatureContext) -> float | None:
    """Thirty-minute log return in the Asian regional composite.

    Live only during the 09:15-11:30 IST overlap; Tokyo closes at 11:30 IST and
    this correctly goes None for the afternoon rather than freezing on the last
    print. A frozen value would look like an Asian market that had stopped
    moving, which is a very different thing from one that has shut.
    """
    return _cross_return(ctx, ASIA_IDX_ID, 30)


@feature(name="usdinr_staleness_minutes", version=1, group="D")
def usdinr_staleness_minutes(ctx: FeatureContext) -> float | None:
    """Age of the newest USD/INR bar, in minutes.

    The liveness of the cross-asset channel, and the reason it is USD/INR that
    carries it: the rupee trades 09:00-17:00 IST, so it is open for every minute
    Nifty is, and any staleness here is a collector problem rather than a
    closed market. A stale series does not raise - it freezes, and every feature
    built on it quietly reports that the world stopped moving. The model can
    only discount that if it is told.

    Never below one minute by construction: the window is right-open at the
    bucket containing ``as_of``, so the newest visible bar is always the
    previous completed minute.
    """
    bars = ctx.candles(minutes=240, instrument_id=USDINR_ID)
    if not bars:
        return None
    return (ctx.as_of - bars[-1].ts).total_seconds() / 60.0
