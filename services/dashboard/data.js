/* Snapshot of one session, produced by the pipeline in this repository.
 *
 * Every number here came out of the real modules - fit_surface, simulate,
 * calibrate, build_fan, greeks_from_surface, enumerate_structures - run on
 * synthetic inputs whose answers were known in advance. Nothing is illustrative.
 *
 * Assigned to a global rather than fetched, so the dashboard opens from
 * file:// without a server. `python tasks.py dashboard` serves it properly.
 */
window.SESSION = {
  meta: {
    symbol: "NIFTY",
    expiry: "2026-09-24",
    session: "2026-08-27",
    asOf: "11:30:00",
    horizon: 60,
    paths: 5000,
    configHash: "31daaefe17de7308",
    feed: "synthetic",
    surfaceAgeSeconds: 4,
    mode: "PAPER",
    spot: 24000.0,
    forward: 24119.97,
  },

  /* Fan chart, in basis points of log-return. 61 minutes, five quantiles.
     From simulate(n_paths=5000, steps=60) calibrated onto the M54 heads. */
  fan: {
    m: Array.from({ length: 61 }, (_, i) => i),
    q05: [0,-13.76,-18.11,-22.49,-26.16,-28.45,-31.63,-33.51,-36.66,-38.70,-40.50,-42.59,-45.10,-47.91,-49.41,-49.51,-51.00,-52.17,-53.67,-55.55,-56.86,-58.25,-61.62,-62.38,-63.28,-65.41,-66.02,-68.51,-69.38,-70.02,-71.59,-72.51,-73.40,-74.15,-75.91,-75.04,-76.67,-76.66,-76.63,-78.11,-79.10,-79.40,-79.44,-80.27,-80.61,-82.31,-81.39,-82.37,-83.84,-84.11,-85.38,-86.06,-87.57,-88.13,-87.70,-87.79,-87.66,-87.32,-88.26,-87.32,-85.59],
    q25: [0,-4.78,-6.77,-8.23,-9.26,-10.49,-11.32,-12.54,-13.43,-14.00,-15.03,-15.94,-17.17,-17.69,-18.31,-19.14,-19.64,-20.18,-20.73,-21.26,-21.80,-22.32,-22.99,-23.27,-23.70,-24.57,-24.81,-25.36,-25.76,-26.89,-26.78,-28.03,-28.29,-28.72,-29.16,-29.77,-29.66,-30.09,-31.01,-31.76,-32.02,-33.02,-33.80,-33.87,-34.80,-35.66,-35.60,-36.38,-37.53,-38.09,-40.00,-40.69,-41.51,-42.85,-43.62,-43.70,-45.18,-46.80,-47.60,-48.24,-49.49],
    q50: [0,-0.01,0.11,0.35,0.44,0.80,0.70,0.79,1.16,1.61,1.33,1.20,1.56,1.48,1.49,1.89,2.18,2.91,2.98,2.90,3.12,2.88,3.28,3.59,3.79,4.08,3.93,3.96,4.27,4.65,5.07,4.51,4.90,5.12,4.81,4.79,4.96,5.05,4.80,5.43,5.70,5.84,5.46,5.47,5.98,5.81,5.51,5.52,5.61,5.57,5.56,5.32,5.94,6.16,5.99,6.11,5.93,5.87,5.65,5.71,5.30],
    q75: [0,4.89,7.24,9.10,10.71,12.15,13.63,15.03,15.86,17.48,18.50,19.19,20.80,21.46,22.42,23.93,24.23,25.48,25.77,27.27,28.11,28.99,29.96,31.06,31.44,32.30,33.85,34.30,34.79,35.51,36.68,37.34,38.21,39.10,40.54,41.10,42.04,43.18,44.38,45.25,45.71,46.81,48.54,48.26,49.11,50.21,51.56,52.35,53.08,53.94,54.82,55.09,56.59,56.74,58.28,59.12,59.48,60.68,61.45,61.66,63.09],
    q95: [0,13.28,18.71,23.29,27.70,31.13,33.76,37.75,40.06,42.85,45.96,47.98,50.20,53.91,55.68,57.48,60.11,62.34,64.76,65.85,68.06,72.03,73.10,75.05,76.77,78.45,80.43,81.30,83.58,85.52,87.08,89.01,92.11,93.15,94.20,96.91,98.13,100.21,101.63,103.22,104.49,104.94,106.75,110.00,111.26,113.17,112.84,115.07,115.94,116.02,119.70,122.04,122.86,124.04,124.96,126.89,127.26,129.03,129.76,130.15,129.37],
    pUp: [0.500,0.500,0.505,0.509,0.514,0.520,0.520,0.517,0.521,0.529,0.523,0.516,0.520,0.521,0.522,0.525,0.526,0.534,0.534,0.532,0.530,0.530,0.537,0.535,0.532,0.538,0.538,0.538,0.542,0.538,0.540,0.540,0.539,0.542,0.542,0.537,0.540,0.542,0.534,0.536,0.537,0.538,0.539,0.541,0.539,0.536,0.532,0.535,0.534,0.530,0.531,0.531,0.531,0.529,0.529,0.528,0.530,0.529,0.529,0.525,0.524],
  },

  stats: {
    impliedMove: 13.89,      // 1σ over 60 min, calendar basis, in bp
    impliedTrading: 32.76,   // the same on trading time - 2.358x apart
    edgeRatio: 4.703,
    edgeRatioTrading: 1.994,
    forecastMove: 107.48,
    acf1: 0.1347,
    kurtosis: 8.459,
  },

  events: [
    { t: "09:04", h: "Central bank policy statement released", cls: "macro.rate_decision", sev: 8, src: "Exchange notices", cred: 1.00, acted: true },
    { t: "09:41", h: "Index heavyweight files quarterly results", cls: "corp.earnings", sev: 6, src: "Exchange notices", cred: 1.00, acted: true },
    { t: "10:02", h: "Wire reports border incident, single source", cls: "geo.military", sev: 9, src: "International wire", cred: 0.90, acted: false,
      why: "Severity 9 requires two independent sources at credibility 0.80 or above. One seen." },
    { t: "10:17", h: "Brent moves 1.4% on an inventory print", cls: "macro.commodity", sev: 4, src: "Calendar provider", cred: 0.95, acted: true },
    { t: "10:44", h: "Anonymous account claims an exchange halt", cls: "exchange.halt", sev: 9, src: "Social, unverified", cred: 0.25, acted: false,
      why: "Source credibility 0.25 may raise an alert and may never open a position on its own." },
    { t: "11:06", h: "The same border story, republished", cls: "geo.military", sev: 9, src: "National media RSS", cred: 0.75, acted: false,
      why: "duplicate_of the 10:02 item. SimHash distance 2." },
    { t: "11:22", h: "Rupee weakens past the prior day low", cls: "macro.fx", sev: 3, src: "Calendar provider", cred: 0.95, acted: false,
      why: "Below the relevance threshold for this horizon." },
  ],

  sources: [
    { name: "Central bank releases", tier: "official", cred: 1.00, enabled: false, tos: "TODO" },
    { name: "Exchange notices", tier: "official", cred: 1.00, enabled: false, tos: "TODO" },
    { name: "Calendar provider", tier: "official", cred: 0.95, enabled: false, tos: "TODO" },
    { name: "International wire", tier: "wire", cred: 0.90, enabled: false, tos: "TODO" },
    { name: "National media RSS", tier: "media", cred: 0.75, enabled: false, tos: "TODO" },
    { name: "Anonymous social", tier: "unverified", cred: 0.25, enabled: false, tos: "TODO" },
  ],

  structures: {
    total: 1342, kept: 1302, dropped: 40,
    byKind: { no_trade: 1, naked: 42, vertical_debit: 108, straddle: 21, strangle: 110, iron_condor: 1020 },
    rows: [
      { kind: "Vertical debit", legs: "+24000C  −24250C", debit: 126.29, cross: 5.84, maxP: "123.71", maxL: 126.29 },
      { kind: "Straddle", legs: "+24000C  +24000P", debit: 698.70, cross: 5.59, maxP: "unbounded", maxL: 698.70 },
      { kind: "Iron condor", legs: "−24250C  +24500C  −24000P  +23750P", debit: -191.44, cross: 8.54, maxP: "191.44", maxL: 58.56 },
    ],
  },

  reco: {
    verdict: "No trade",
    id: "8f2c1a",
    blockedBy: "Gate G4",
    metaLabel: "skip",
    rows: [
      ["Approval probability", "0.508", null,
       "p_up after isotonic calibration, then capped at max_edge either side of 0.50. The cap is applied to the interval bounds as well as the point, or a capped estimate sits outside its own band."],
      ["Wilson lower bound", "0.446", null,
       "Lower bound of the 95% Wilson score interval on p_up. Kelly is taken on this, never on the point estimate - sizing off a point estimate is sizing off the middle of a distribution nobody bounded."],
      ["Kelly fraction (LCB)", "0.000", "dim", null],
      ["Loss ratio threshold", "0.667", null,
       "Bayes-optimal threshold for a 2:1 loss ratio: loss_ratio / (1 + loss_ratio). The asymmetry lives here rather than in class weighting, which calibration undoes."],
    ],
  },

  greeks: {
    subtitle: "Straddle · 2 lots × 75",
    rows: [
      ["Delta (smile-adjusted)", "+23.94", null,
       "dV/dF along the fitted surface: the Black-76 delta plus vega × dσ/dF. The strike stays put while log-moneyness shifts, so a moving forward moves the volatility at that strike."],
      ["Delta (Black-76)", "+18.47", "dim",
       "What the formula gives while holding volatility constant. Kept beside the corrected figure because the difference between them is what explains a hedging P&L."],
      ["Smile term", "+5.47", "warn", null],
      ["Gamma", "+0.1358", null,
       "Full chain rule: Γ + 2·vanna·σ_F + volga·σ_F² + vega·σ_FF. Verified against a reprice along the surface, which shares no algebra with this path."],
      ["Theta per day", "−2 632.73", "neg", null],
      ["Vega per point", "+7 859.77", null,
       "Gross across expiries, not a net. A seven-day vega and a sixty-day vega are different risks sharing a unit; summing them reports neutral for a book that is short the front and long the back."],
    ],
    legs: [
      { s: "24000 CE", lots: "+2", iv: "13.08%", px: "409.04", d: "+0.5773", db: "+0.5591" },
      { s: "24000 PE", lots: "+2", iv: "13.08%", px: "289.66", d: "−0.4177", db: "−0.4359" },
    ],
  },

  surface: {
    rows: [
      ["ATM volatility", "13.00%", null,
       "sqrt(w(0)/T) from the fitted SVI, where w is total implied variance. Fitted from 22 of 42 quotes."],
      ["Skew, dw/dk", "−0.00313", null, null],
      ["Curvature", "+0.04833", null, null],
      ["Forward", "24 119.97", null,
       "Recovered from put-call parity at the nearest-the-money strike, not from S·e^(rT). The textbook forward is wrong for an index by the dividend yield, and the error appears as a stable, manufactured skew."],
      ["Fit RMSE", "0.00 vol pts", null, null],
      ["Quotes rejected", "47.6%", null,
       "20 of 42 dropped, all on the delta band. Half a full chain being rejected is normal - the exchange lists strikes far past the liquid band. The signal is this number moving against its own history."],
      ["No-arbitrage", "clean", "pos",
       "Butterfly and calendar both checked on the fitted result, independently of the penalty inside the objective. A penalty outweighed by fit error hands back an excellent RMSE with a negative implied density."],
    ],
    svi: { a: 0.000600, b: 0.005700, rho: -0.4200, m: 0.015000, sigma: 0.115000 },
  },

  gates: [
    { id: "G4", state: "fail", name: "Gap forecaster, real daily bars",
      why: "CRPS 0.002827 against a naive 0.002487 — 13.7% worse. All twelve trials are named in docs/gates.md. This is the only component that has ever run on real market data." },
    { id: "G6", state: "open", name: "Projection scorer",
      why: "Needs CRPS beating every baseline by 8%. An oracle handed the true conditional variance beats a fitted GARCH by 0.19% on GARCH-like data, so the bar is asking for structure the baseline cannot see." },
    { id: "G7", state: "fail", name: "Stress and simulation",
      why: "Profit factor 0.04 against the 1.2 required, over 190 synthetic trades. Costs of 399 against a gross of −75." },
    { id: "G8", state: "open", name: "Sixty sessions of live paper",
      why: "Not started. Needs a broker adapter and the recorder pointed at a real feed. Every trading day it is not running is a day of samples that cannot be recovered." },
  ],
};
