# M91 · Dashboard manual checklist

    python tasks.py dashboard        # http://127.0.0.1:8787

The automated half of M91 is a Playwright smoke test and **it is not written
yet**. Until it is, this checklist is the whole of the verification, and the
module stays marked partial.

Everything below is a thing a browser can be wrong about in a way no unit test
sees: a colour that only resolves in one theme, a canvas that draws at the
wrong scale on a retina display, a panel that stops scrolling when it collapses.

Run it against both themes and at **four** widths: a full desktop, 1100px,
860px and 700px. The layout re-flows at 1120 and again at 760, and every fault
section 8 lists was invisible at desktop width.

---

## 1 · Mode indicator

- [ ] **PAPER** is visible without scrolling, at the top right of the toolbar.
- [ ] The chip carries the diagonal hatching, not just coloured text — the
      hatch is what makes it read as a state rather than a label.
- [ ] The word next to it says what PAPER *means* ("no order reaches a
      broker"), not merely that the mode is paper.

> The spec asks for this to be unmissable. If you can look at the screen for
> three seconds and not know which mode you are in, it has failed, regardless
> of whether the element is present.

## 2 · Chart

- [ ] The fan renders: two nested bands, a brass median line, a dot at the
      right-hand end of the median.
- [ ] The implied 1σ cone renders as a dashed pair of curves, **under** the fan.
- [ ] Text on the canvas is sharp, not soft — a blurry axis means the
      `devicePixelRatio` scaling is wrong and every other pixel measurement in
      the chart is suspect too.
- [ ] Moving the pointer across the chart shows a dashed crosshair, a time
      label at the bottom axis, and a dot on each of the five quantiles.
- [ ] The legend at the top left updates as the crosshair moves, and returns to
      the terminal values when the pointer leaves the chart.
- [ ] The status bar's `cursor` field tracks the crosshair and resets to `—`.
- [ ] Toggling **Fan** off leaves the implied cone and rescales the y axis to
      fit it. (A layer hidden with CSS would keep the old scale; this one
      redraws.)
- [ ] Toggling **P(up)** on adds a dotted trace.
- [ ] Resizing the window redraws the chart at the new size without stretching
      the text.

## 3 · Suppressed events

- [ ] The **Event stream** tab shows **seven** rows, not three.
- [ ] Four are dimmed, and each dimmed row carries a `Suppressed —` line naming
      the specific rule that stopped it.
- [ ] The 10:44 row (anonymous social, severity 9) is present and dimmed. This
      is the AP-hack case: a loud, high-severity, low-credibility item that the
      system must show and must not act on.

> A stream that only lists what was acted on cannot answer the question
> somebody asks after a bad day, which is what the system saw and decided
> against.

## 4 · Hoverable numbers

- [ ] Numbers with a dotted underline show a tooltip on hover.
- [ ] Each tooltip says **how the number is computed**, not what it is called.
- [ ] The tooltip flips above the cursor near the bottom of the window rather
      than being clipped.
- [ ] Check at least: Wilson lower bound, Edge ratio, Delta (smile-adjusted),
      Forward, No-arbitrage.

## 5 · Panels

- [ ] The side panel's three tabs (Recommendation / Book / Surface) switch
      independently of the drawer's four.
- [ ] Collapsing the drawer (▾) gives the chart the space and the chart
      redraws to fill it.
- [ ] Collapsing the side panel does the same horizontally.
- [ ] Both scroll internally; the page itself never scrolls.

## 6 · Themes

- [ ] The theme button switches light and dark, and the **chart redraws** —
      canvas colours are read at draw time, so a chart that keeps its old
      palette means `draw()` was not called.
- [ ] In light mode: no dark-mode text on a light ground anywhere, and the
      brass accent still reads against white.
- [ ] The choice survives a reload.
- [ ] Set the OS to the opposite theme with no explicit choice stored, and the
      page follows it.

## 7 · Keyboard and motion

- [ ] Tabbing reaches every button and the focus ring is visible on each.
- [ ] With **reduce motion** set, the tooltip appears without a fade.

---

## 8 · Layout, at every width

Each of these was a real fault found on 2026-08-30, and each is invisible at a
wide window - which is why the checklist now names the widths.

- [ ] The chart's explanatory note sits in the panel header, **not** over the
      plot. It shared the right-hand gutter with the price scale and the two
      printed on top of each other.
- [ ] No two numbers overlap in the right-hand price scale. A gridline label
      that would land under a terminal value tag is dropped; the tag wins.
- [ ] The **entry** label at the left of the time axis is complete, not clipped.
- [ ] The last row of the event stream is not sliced by the status bar. The
      drawer fades at its bottom edge to show there is more.
- [ ] At **1100px and 860px** the side panel moves *below* the chart and keeps
      its three tabs. It must not vanish - it holds the verdict, and an earlier
      rule simply `display: none`d it.
- [ ] At those widths nothing overlaps: check the drawer against the chart, and
      the drawer against the side panel. Both overlapped at different points.
- [ ] At **700px** the tool rail is a horizontal strip **under the toolbar**,
      not at the foot of a scrolling page.
- [ ] The toolbar and status bar stay one line at every width and scroll
      sideways if they must. The status bar wrapping to two lines pushed it up
      into the panel above.
- [ ] Collapse the side panel and the drawer, at each width: the chart redraws
      to fill the space and never paints outside its own box.

## 9 · The overview page

`index.html` is the front page; the workspace moved to `desk.html`.

- [ ] The hero headline sets on **two lines at desktop width**, with no orphan
      word and no hard `<br>` doing the work.
- [ ] The four stat tiles read: 2,466 bars &middot; 323 chain sessions &middot;
      30 trials &middot; **0 models that have beaten their baseline**. The last
      one is the point of the page and must not be softened.
- [ ] The gate ledger shows **G4 and G7 as Not passed**, above the fold of that
      section. A front page that lists only open items would be a different
      claim from the one this project makes.
- [ ] Every figure on the page matches `docs/gates.md`. Nothing here is
      illustrative and nothing is rounded in the flattering direction.
- [ ] **Open the desk** in the nav, in the hero, and in the closing callout all
      reach `desk.html`.
- [ ] On the desk, the brand at top left links back to the overview.
- [ ] The sticky nav gains its bottom rule only once the page has scrolled.
- [ ] At 820px the nav links collapse and the gate rows re-flow with the status
      pill under the text, not clipped beside it.
- [ ] At 520px the call-to-action buttons go full width and nothing overflows
      horizontally &mdash; check by scrolling the page sideways; it must not.

## Known gaps

These are open, not overlooked:

- **Playwright smoke test** — not written. Until it exists, nothing above is
  checked on a commit.
- **Click a recommendation → full replay view** — the spec asks for it and it
  does not exist. The recommendation panel shows the decision; it does not yet
  open the inputs that produced it.
- **Live data** — the workspace reads a session snapshot from `data.js`. It is
  not wired to the API (M90) and does not update.

## What the numbers are

Every figure came out of the modules in this repository — `fit_surface`,
`simulate`, `calibrate`, `build_fan`, `greeks_from_surface`,
`enumerate_structures` — run on **synthetic inputs whose answers were known in
advance**. That establishes the arithmetic and says nothing about edge.

Exactly one *model* has been measured on real market data: the overnight gap
forecaster, on 2,465 sessions of free daily bars, where it came out **2.4%
worse** than a naive baseline. This file previously said 13.7%, which was the
figure from an earlier run on 496 sessions; four fifths of that deficit turned
out to be sample size. `docs/gates.md` is authoritative.
