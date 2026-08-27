# Decision records

One file per decision that changes a risk limit, a gate threshold, or anything
else the system is not allowed to quietly drift on.

## Why this exists

A risk limit that can be edited in a commit is a risk limit that gets edited at
the worst possible moment — after a losing week, by the person who lost it, with
a reason that sounds convincing at the time and does not survive a month.

Requiring a written record does not prevent the change. It makes it a decision
instead of a nudge, and it leaves something to read later when the same argument
comes back around.

`limit_change_requires_adr` in `libs/ms_risk/gate.py` reports which changed
fields have no record. The check is deliberately loose — it looks for the field
name in a filename — because a strict scheme invites a mechanical rename and the
point is the writing, not the filing.

## Naming

    NNNN-<what-changed>-<one-word-why>.md

The changed field name must appear in the filename, e.g.
`0003-raise-daily_loss_limit_pct-after-drawdown-review.md`.

## Template

```markdown
# NNNN — <title>

**Date:** YYYY-MM-DD
**Field:** `<config field>` — old → new
**Status:** proposed | accepted | superseded by NNNN

## What changed

## Why now

What specifically happened that this responds to. "It felt tight" is not a
reason; "we rejected 14 valid signals in March and 11 of them would have paid
after costs" is.

## What this makes worse

Every loosened limit buys something and costs something. If this section is
empty, the change has not been thought through — a limit that only has upside
was set wrong originally, and that is the decision to write down instead.

## How we would know this was a mistake

The observable that would tell us to revert, and when we will next look at it.
```
