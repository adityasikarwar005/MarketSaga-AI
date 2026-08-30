/* MarketSage Desk — workspace behaviour.
 *
 * The chart is canvas rather than SVG for one reason: the crosshair redraws on
 * every pointer move, and swapping attributes on sixty SVG nodes at 60fps is
 * how a dashboard starts to feel heavy. Canvas redraws the whole frame in one
 * pass, and the frame is cheap.
 */
(function () {
  "use strict";

  const S = window.SESSION;
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmt = (v, d) => (v > 0 ? "+" : "") + v.toFixed(d === undefined ? 2 : d);

  /* ─────────────────────────────────────────────────────────────────────
     Chart
     ───────────────────────────────────────────────────────────────────── */
  const canvas = $("#chart");
  const ctx = canvas.getContext("2d");
  const F = S.fan;
  const LAST = F.m.length - 1;

  /* Right-hand price scale, like a trading terminal: the axis sits where the
     newest value is, so the eye does not travel back across the series. */
  const PAD = { l: 14, r: 74, t: 18, b: 30 };

  const series = { fan: true, implied: true, pup: false };
  let cursor = null;         // index under the pointer, or null
  let box = { w: 0, h: 0 };

  function scales() {
    const lows = [Math.min.apply(null, F.q05)];
    const highs = [Math.max.apply(null, F.q95)];
    if (series.implied) {
      const cone = S.stats.impliedMove;
      lows.push(-cone); highs.push(cone);
    }
    const lo = Math.min.apply(null, lows) * 1.12;
    const hi = Math.max.apply(null, highs) * 1.12;
    const iw = box.w - PAD.l - PAD.r;
    const ih = box.h - PAD.t - PAD.b;
    return {
      lo, hi, iw, ih,
      x: i => PAD.l + (i / LAST) * iw,
      y: v => PAD.t + ih - ((v - lo) / (hi - lo)) * ih,
      i: px => Math.max(0, Math.min(LAST, Math.round(((px - PAD.l) / iw) * LAST))),
    };
  }

  function resize() {
    const r = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    box = { w: Math.max(320, r.width), h: Math.max(220, r.height) };
    canvas.width = Math.round(box.w * dpr);
    canvas.height = Math.round(box.h * dpr);
    canvas.style.width = box.w + "px";
    canvas.style.height = box.h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function bandPath(s, upper, lower) {
    ctx.beginPath();
    upper.forEach((v, i) => (i ? ctx.lineTo(s.x(i), s.y(v)) : ctx.moveTo(s.x(i), s.y(v))));
    for (let i = lower.length - 1; i >= 0; i--) ctx.lineTo(s.x(i), s.y(lower[i]));
    ctx.closePath();
  }

  function linePath(s, arr) {
    ctx.beginPath();
    arr.forEach((v, i) => (i ? ctx.lineTo(s.x(i), s.y(v)) : ctx.moveTo(s.x(i), s.y(v))));
  }

  function niceStep(range) {
    const raw = range / 6;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n >= 5 ? 5 : n >= 2 ? 2 : 1) * mag;
  }

  function draw() {
    const s = scales();
    const C = {
      grid: css("--grid"), line: css("--line"), faint: css("--fg-faint"),
      dim: css("--fg-dim"), fg: css("--fg"), accent: css("--accent"),
      outer: css("--band-outer"), inner: css("--band-inner"),
      cross: css("--crosshair"), panel: css("--panel-2"), sunk: css("--sunk"),
    };

    ctx.clearRect(0, 0, box.w, box.h);
    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.textBaseline = "middle";

    /* grid + price scale
       The terminal value tags below occupy the same right-hand gutter as these
       labels. Two numbers a few basis points apart therefore printed on top of
       each other - -88 over -86 at the default window. Reserve the tag rows
       and drop any gridline label that would land in one: the tag carries the
       more useful number of the two. */
    const taken = series.fan
      ? [F.q95[LAST], F.q05[LAST], F.q50[LAST]].map(v => s.y(v))
      : [];
    /* The tag is drawn as a 16px-tall filled box, so half of it is 8px, and
       the gridline label has about 5px of its own half-height. 11px left
       +120 printed and then painted over by the +129 tag. */
    const clashes = y => taken.some(t => Math.abs(t - y) < 14);

    const step = niceStep(s.hi - s.lo);
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = C.faint;
    ctx.textAlign = "left";
    for (let v = Math.ceil(s.lo / step) * step; v <= s.hi; v += step) {
      const y = Math.round(s.y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(box.w - PAD.r, y); ctx.stroke();
      if (!clashes(y)) ctx.fillText(fmt(v, 0), box.w - PAD.r + 9, y);
    }
    for (let m = 0; m <= LAST; m += 10) {
      const x = Math.round(s.x(m)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + s.ih); ctx.stroke();
      /* The first and last ticks sit on the plot edges, so a centred label
         hangs off the canvas and is clipped. Anchor those two inwards. */
      ctx.textAlign = m === 0 ? "left" : m === LAST ? "right" : "center";
      ctx.fillText(m ? "+" + m + "m" : "entry", x, box.h - PAD.b / 2);
    }
    ctx.textAlign = "center";
    ctx.textAlign = "left";
    ctx.fillText("bp", box.w - PAD.r + 9, PAD.t - 9);

    /* implied cone, under the fan so the forecast reads against it */
    if (series.implied) {
      const up = F.m.map(m => S.stats.impliedMove * Math.sqrt(m / LAST));
      const dn = up.map(v => -v);
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = C.faint;
      ctx.lineWidth = 1.25;
      linePath(s, up); ctx.stroke();
      linePath(s, dn); ctx.stroke();
      ctx.restore();
    }

    /* fan */
    if (series.fan) {
      ctx.fillStyle = C.outer; bandPath(s, F.q95, F.q05); ctx.fill();
      ctx.fillStyle = C.inner; bandPath(s, F.q75, F.q25); ctx.fill();
      ctx.strokeStyle = C.accent;
      ctx.lineWidth = 2;
      linePath(s, F.q50); ctx.stroke();
      ctx.fillStyle = C.accent;
      ctx.beginPath(); ctx.arc(s.x(LAST), s.y(F.q50[LAST]), 3.5, 0, Math.PI * 2); ctx.fill();
    }

    /* P(up) as a secondary trace on its own implied scale */
    if (series.pup) {
      ctx.save();
      ctx.strokeStyle = C.dim;
      ctx.lineWidth = 1.25;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      F.pUp.forEach((p, i) => {
        const y = PAD.t + s.ih - ((p - 0.48) / 0.08) * s.ih;
        i ? ctx.lineTo(s.x(i), y) : ctx.moveTo(s.x(i), y);
      });
      ctx.stroke();
      ctx.restore();
    }

    /* right-edge value tags, the way a terminal marks the live series */
    if (series.fan) {
      const tag = (v, text, colour, strong) => {
        const y = s.y(v);
        ctx.fillStyle = strong ? colour : C.panel;
        ctx.fillRect(box.w - PAD.r + 4, y - 8, PAD.r - 8, 16);
        ctx.fillStyle = strong ? C.sunk : colour;
        ctx.textAlign = "left";
        ctx.fillText(text, box.w - PAD.r + 9, y);
      };
      tag(F.q95[LAST], fmt(F.q95[LAST], 0), C.dim);
      tag(F.q05[LAST], fmt(F.q05[LAST], 0), C.dim);
      tag(F.q50[LAST], fmt(F.q50[LAST], 1), C.accent, true);
    }

    /* crosshair */
    if (cursor !== null) {
      const x = Math.round(s.x(cursor)) + 0.5;
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = C.cross;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + s.ih); ctx.stroke();
      ctx.restore();

      ctx.fillStyle = C.panel;
      ctx.fillRect(x - 26, box.h - PAD.b + 1, 52, PAD.b - 4);
      ctx.fillStyle = C.fg;
      ctx.textAlign = "center";
      ctx.fillText(cursor ? "+" + cursor + "m" : "entry", x, box.h - PAD.b / 2);

      [["q95", C.dim], ["q75", C.dim], ["q50", C.accent], ["q25", C.dim], ["q05", C.dim]]
        .forEach(([k, colour]) => {
          const y = s.y(F[k][cursor]);
          ctx.fillStyle = colour;
          ctx.beginPath(); ctx.arc(x, y, k === "q50" ? 3 : 2, 0, Math.PI * 2); ctx.fill();
        });
    }
  }

  /* legend — mirrors the crosshair, or the terminal values when idle */
  function legend() {
    const at = cursor === null ? LAST : cursor;
    const swatch = c => `<span class="swatch" style="background:${c}"></span>`;
    $("#legend").innerHTML =
      /* The panel heading is static markup now. What belongs here is the
         thing that changes: where along the horizon these figures are read. */
      `<div class="legend-at">${at ? "at +" + at + " min" : "at entry"}` +
      `${cursor === null ? '<span class="legend-hint">terminal</span>' : ""}</div>` +
      [
        ["95th", F.q95[at], "var(--band-outer)"],
        ["75th", F.q75[at], "var(--band-inner)"],
        ["median", F.q50[at], "var(--accent)"],
        ["25th", F.q25[at], "var(--band-inner)"],
        ["5th", F.q05[at], "var(--band-outer)"],
      ].map(([k, v, c]) =>
        `<div class="row">${swatch(c)}<span class="k">${k}</span><span class="v">${fmt(v)} bp</span></div>`
      ).join("") +
      `<div class="row"><span class="swatch" style="background:transparent;border:1px dashed var(--fg-faint)"></span>` +
      `<span class="k">implied 1&sigma;</span><span class="v">${fmt(S.stats.impliedMove * Math.sqrt(at / LAST))} bp</span></div>` +
      `<div class="row"><span class="swatch" style="background:transparent"></span>` +
      `<span class="k">P(up)</span><span class="v">${(F.pUp[at] * 100).toFixed(1)}%</span></div>`;

    $("#cursor-readout").innerHTML = cursor === null
      ? '<span class="k">cursor</span> &mdash;'
      : `<span class="k">t</span> +${cursor}m &nbsp; <span class="k">med</span> ${fmt(F.q50[cursor])} &nbsp; <span class="k">P(up)</span> ${(F.pUp[cursor] * 100).toFixed(1)}%`;
  }

  canvas.addEventListener("pointermove", e => {
    const r = canvas.getBoundingClientRect();
    cursor = scales().i(e.clientX - r.left);
    draw(); legend();
  });
  canvas.addEventListener("pointerleave", () => { cursor = null; draw(); legend(); });

  /* ─────────────────────────────────────────────────────────────────────
     Content
     ───────────────────────────────────────────────────────────────────── */
  const tipAttr = t => (t ? ` class="q" data-tip="${t.replace(/"/g, "&quot;")}"` : "");

  const kvRows = (id, rows) => {
    $("#" + id).innerHTML = rows.map(([k, v, cls, tip]) =>
      `<tr><td class="name">${k}</td><td class="${cls || ""}"><span${tipAttr(tip)}>${v}</span></td></tr>`
    ).join("");
  };

  /* events */
  $("#events").innerHTML = S.events.map(e => `
    <div class="ev${e.acted ? "" : " muted"}">
      <div class="t">${e.t}</div>
      <div><span class="sev${e.sev >= 8 ? " hi" : ""}">${e.sev}</span></div>
      <div class="h">${e.h}</div>
      <div class="cls">${e.cls}</div>
      <div class="src">${e.src} &middot; ${e.cred.toFixed(2)}</div>
      ${e.why ? `<div class="why">Suppressed &mdash; ${e.why}</div>` : ""}
    </div>`).join("");

  /* structures */
  $("#structures").innerHTML = S.structures.rows.map(r => `
    <tr>
      <td class="name">${r.kind}</td>
      <td class="dim" style="font-size:11px">${r.legs}</td>
      <td class="${r.debit < 0 ? "pos" : ""}">${r.debit.toFixed(2)}</td>
      <td class="dim">${r.cross.toFixed(2)}</td>
      <td>${r.maxP}</td>
      <td>${r.maxL.toFixed(2)}</td>
    </tr>`).join("");

  /* sources */
  $("#sources").innerHTML = S.sources.map(s => `
    <tr>
      <td class="name">${s.name}</td>
      <td class="dim" style="font-family:var(--sans)">${s.tier}</td>
      <td>${s.cred.toFixed(2)}</td>
      <td class="${s.cred >= 0.5 ? "dim" : "neg"}" style="font-family:var(--sans)">${s.cred >= 0.5 ? "permitted" : "alerts only"}</td>
      <td class="warn" style="font-family:var(--sans)">${s.tos}</td>
      <td class="dim" style="font-family:var(--sans)">off</td>
    </tr>`).join("");

  /* gates */
  $("#gates").innerHTML = S.gates.map(g => `
    <div class="gate">
      <div class="row">
        <span class="st ${g.state}">${g.state.toUpperCase()}</span>
        <strong>${g.id}</strong>
        <span class="lbl">${g.name}</span>
      </div>
      <p>${g.why}</p>
    </div>`).join("");

  /* side panel */
  $("#verdict").textContent = S.reco.verdict;
  $("#verdict-why").textContent =
    "No-trade won on merit against " + S.structures.kept.toLocaleString("en-IN") +
    " alternatives rather than arriving by default. It competes on the same axis as " +
    "every structure, so a quiet morning produces a reason instead of an exception path.";
  $("#reco-id").textContent = "reco " + S.reco.id;

  kvRows("reco", [
    ["Blocked by", S.reco.blockedBy, "neg", null],
    ["Meta-label", S.reco.metaLabel, "dim", null],
  ].concat(S.reco.rows));

  kvRows("dist", [
    ["Forecast move", S.stats.forecastMove.toFixed(2) + " bp", null,
     "Half the 5-95 empirical band. A half-band rather than a standard deviation, because a distribution with this kurtosis does not have one worth quoting."],
    ["Implied 1&sigma; (calendar)", S.stats.impliedMove.toFixed(2) + " bp", null,
     "iv x sqrt(60 / 525 600). The surface fits its year fractions on calendar time, so the implied move must use the same basis."],
    ["Implied 1&sigma; (trading)", S.stats.impliedTrading.toFixed(2) + " bp", "dim",
     "The same volatility on 94 500 trading minutes. 2.358x the calendar figure - which is why the basis is never defaulted anywhere in the module."],
    ["Edge ratio", S.stats.edgeRatio.toFixed(3) + " x", "warn",
     "Half the forecast band over the implied move, both at 1.645 sigma. On trading time the same fan gives " + S.stats.edgeRatioTrading.toFixed(2) + "x - near 1.0 the basis alone flips the decision."],
    ["Lag-1 ACF, sq. returns", S.stats.acf1.toFixed(4), null,
     "Volatility clustering. Measured on squared returns because a GARCH process has uncorrelated returns by construction. The iid control gives 0.0007."],
    ["Excess kurtosis", S.stats.kurtosis.toFixed(3), null, null],
  ]);

  $("#book-sub").textContent = S.greeks.subtitle;
  kvRows("greeks", S.greeks.rows);
  $("#legs").innerHTML = S.greeks.legs.map(l => `
    <tr><td class="name">${l.s} <span class="dim">${l.lots}</span></td>
        <td>${l.iv}</td><td>${l.px}</td><td>${l.d}</td><td class="dim">${l.db}</td></tr>`).join("");

  kvRows("surface", S.surface.rows);
  kvRows("svi", Object.entries(S.surface.svi).map(([k, v]) =>
    [k, v.toFixed(6), null, null]));

  /* ─────────────────────────────────────────────────────────────────────
     Interaction
     ───────────────────────────────────────────────────────────────────── */
  function tabs(container, attr) {
    $$("button[" + attr + "]", container).forEach(btn => {
      btn.addEventListener("click", () => {
        $$("button[" + attr + "]", container).forEach(b => b.setAttribute("aria-selected", "false"));
        btn.setAttribute("aria-selected", "true");
        const scope = container.closest(".drawer, .side");
        $$(".pane[" + attr + "]", scope).forEach(p => p.removeAttribute("data-active"));
        $(".pane[" + attr + '="' + btn.getAttribute(attr) + '"]', scope).setAttribute("data-active", "");
      });
    });
  }
  tabs($(".drawer .tabs"), "data-pane");
  tabs($(".side .tabs"), "data-side");

  /* series toggles redraw the chart rather than hiding a layer with CSS,
     so the y-scale re-fits to what is actually shown */
  $$("button[data-series]").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.series;
      series[key] = btn.getAttribute("aria-pressed") !== "true";
      btn.setAttribute("aria-pressed", String(series[key]));
      draw(); legend();
    });
  });

  $$("button[data-h]").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("button[data-h]").forEach(b => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
    });
  });

  $$(".rail button").forEach(btn => {
    btn.addEventListener("click", () => {
      $$(".rail button").forEach(b => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
    });
  });

  $("#toggle-drawer").addEventListener("click", e => {
    const d = $("#drawer");
    const collapsed = d.classList.toggle("collapsed");
    e.currentTarget.innerHTML = collapsed ? "&#9652;" : "&#9662;";
    e.currentTarget.title = collapsed ? "Expand panel" : "Collapse panel";
    requestAnimationFrame(resize);
  });

  $("#toggle-side").addEventListener("click", () => {
    $("#body").classList.toggle("side-collapsed");
    requestAnimationFrame(resize);
  });

  $("#toggle-theme").addEventListener("click", () => {
    const root = document.documentElement;
    const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("ms-theme", next); } catch (_) { /* private window */ }
    draw();
  });
  try {
    const saved = localStorage.getItem("ms-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch (_) { /* private window */ }

  /* tooltip: computation and provenance on every derived number */
  const tip = $("#tip");
  document.addEventListener("pointerover", e => {
    const el = e.target.closest("[data-tip]");
    if (!el) return;
    tip.innerHTML = '<span class="lbl">How this is computed</span>' + el.dataset.tip;
    const r = el.getBoundingClientRect();
    tip.style.opacity = "1";
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(Math.max(8, r.left), innerWidth - w - 8) + "px";
    tip.style.top = (r.top > h + 14 ? r.top - h - 8 : r.bottom + 8) + "px";
  });
  document.addEventListener("pointerout", e => {
    if (e.target.closest("[data-tip]")) tip.style.opacity = "0";
  });

  /* meta */
  $("#sym").textContent = S.meta.symbol;
  $("#exp").textContent = S.meta.expiry;
  $("#cnt-events").textContent = S.events.length;
  $("#cnt-str").textContent = S.structures.kept.toLocaleString("en-IN");
  $("#cnt-gates").textContent = S.gates.filter(g => g.state === "fail").length + " failing";
  $("#cnt-src").textContent = S.sources.filter(s => s.enabled).length + " / " + S.sources.length;

  window.addEventListener("resize", resize);
  /* The window is not the only thing that changes this canvas's box. Collapsing
     a panel, and every breakpoint that re-flows the side panel from a column
     into a row, resize the well while the window stays exactly as it was.
     Watching the element covers all of them. */
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(() => resize()).observe(canvas.parentElement);
  }
  resize();
  legend();
})();
