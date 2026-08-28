# M91 · Dashboard manual checklist

    python tasks.py dashboard        # http://127.0.0.1:8787

The automated half of M91 is a Playwright smoke test and **it is not written
yet**. Until it is, this checklist is the whole of the verification, and the
module stays marked partial.

Everything below is a thing a browser can be wrong about in a way no unit test
sees: a colour that only resolves in one theme, a canvas that draws at the
wrong scale on a retina display, a panel that stops scrolling when it collapses.

Run it against both themes and at two window widths.

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
advance**. That establishes the arithmetic and says nothing about edge. Exactly
one component has ever run on real market data, and it lost to a naive
baseline by 13.7%.
