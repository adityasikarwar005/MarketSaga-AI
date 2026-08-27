"""Gradient boosted decision trees.

The honest upgrade over a linear model on tabular market data. Not a neural
network: on tabular features with a few hundred to a few hundred thousand rows,
boosted trees beat neural nets consistently and by enough that the choice is not
close. Deep learning earns its place in this project on *sequences* and on
*text* - raw candle paths and news - and those are separate modules with
separate data requirements.

This is a from-scratch implementation for the same reason everything else here
is: no dependency to install before the first result, deterministic by
construction so a fit can be reproduced from a dataset version months later, and
nothing hidden in a library whose defaults nobody has read.

What the audit changed
----------------------

Three findings from ``docs/model_audit.md`` are built into this file rather than
left as advice.

**A regularisation constant that does not adapt is a bug.** The quantile
regressor shipped with ``l2=1e-4``, which shrank its weights by 0.2% - a knob
that read as a defence and was not one. So here the number of trees is not a
constant at all: it is chosen by early stopping on an inner validation split,
and the depth and leaf minimum are the things a caller tunes.

**A model must be asked what it says when there is nothing to find.** Boosted
trees are extremely good at memorising noise - far better than a linear model,
which is exactly why they need more discipline, not less. ``tests/ml/
test_no_signal.py`` holds the null tests: fitted on random labels, this must
report no skill on held-out data and must stop early instead of growing
hundreds of trees.

**Searching is not free.** ``select_depth`` exists so the search happens on an
inner split rather than against a test fold, and anything that acts on the
result owes it to the trial log.

Binning
-------

Features are bucketed into at most ``max_bins`` quantile bins once, up front.
That is the standard histogram trick and it does three things: split finding
drops from O(n log n) to O(bins) per feature per node, ties and outliers stop
being able to define a split on their own, and the binning is itself a mild
regulariser. The bin edges are learned on the training rows only and applied
unchanged afterwards, which is the same train/serve discipline the feature
store enforces.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from libs.ms_core.obs import get_logger

log = get_logger(__name__)

MAX_BINS = 64
# Below this a node cannot be split, whatever the gain says. A leaf holding
# three rows is a memorised triple, not a rule.
MIN_SAMPLES_LEAF = 20


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #
@dataclass
class Binner:
    """Quantile bin edges per feature, learned once on the training rows."""

    edges: list[list[float]] = field(default_factory=list)

    def fit(self, rows: Sequence[Sequence[float]], *, max_bins: int = MAX_BINS) -> Binner:
        n_features = len(rows[0]) if rows else 0
        self.edges = []
        for j in range(n_features):
            values = sorted({r[j] for r in rows})
            if len(values) <= max_bins:
                # Few enough distinct values to keep them all: midpoints between
                # neighbours, so every distinct value lands in its own bin.
                cuts = [(a + b) / 2.0 for a, b in pairwise(values)]
            else:
                ordered = sorted(r[j] for r in rows)
                step = len(ordered) / max_bins
                cuts = sorted({ordered[int(i * step)] for i in range(1, max_bins)})
            self.edges.append(cuts)
        return self

    def apply(self, row: Sequence[float]) -> list[int]:
        return [_bin_of(v, self.edges[j]) for j, v in enumerate(row)]

    def apply_all(self, rows: Sequence[Sequence[float]]) -> list[list[int]]:
        return [self.apply(r) for r in rows]

    def n_bins(self, feature: int) -> int:
        return len(self.edges[feature]) + 1


def _bin_of(value: float, cuts: Sequence[float]) -> int:
    lo, hi = 0, len(cuts)
    while lo < hi:
        mid = (lo + hi) // 2
        if value <= cuts[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------- #
# One tree
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    feature: int = -1
    bin_threshold: int = -1
    left: int = -1
    right: int = -1
    value: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0


@dataclass
class Tree:
    nodes: list[Node] = field(default_factory=list)

    def predict_binned(self, binned: Sequence[int]) -> float:
        i = 0
        while not self.nodes[i].is_leaf:
            node = self.nodes[i]
            i = node.left if binned[node.feature] <= node.bin_threshold else node.right
        return self.nodes[i].value

    @property
    def n_leaves(self) -> int:
        return sum(1 for n in self.nodes if n.is_leaf)


def _leaf_value(g_sum: float, h_sum: float, l2: float) -> float:
    """Newton step for this leaf. ``l2`` shrinks it toward zero."""
    return -g_sum / (h_sum + l2)


def _gain(g_sum: float, h_sum: float, l2: float) -> float:
    return (g_sum * g_sum) / (h_sum + l2)


def build_tree(
    binned: Sequence[Sequence[int]],
    grad: Sequence[float],
    hess: Sequence[float],
    indices: list[int],
    binner: Binner,
    *,
    max_depth: int,
    min_samples_leaf: int,
    l2: float,
    min_gain: float,
) -> Tree:
    """Greedy depth-limited tree on the second-order objective.

    The gain of a split is the drop in the regularised objective it buys, and a
    split that does not clear ``min_gain`` is not taken. Both that threshold and
    ``min_samples_leaf`` exist because a boosted tree will happily isolate a
    single lucky row and call it structure.
    """
    tree = Tree(nodes=[])

    def make_leaf(idx: list[int]) -> int:
        g = sum(grad[i] for i in idx)
        h = sum(hess[i] for i in idx)
        tree.nodes.append(Node(value=_leaf_value(g, h, l2)))
        return len(tree.nodes) - 1

    def split(idx: list[int], depth: int) -> int:
        if depth >= max_depth or len(idx) < 2 * min_samples_leaf:
            return make_leaf(idx)

        g_total = sum(grad[i] for i in idx)
        h_total = sum(hess[i] for i in idx)
        parent = _gain(g_total, h_total, l2)

        best_gain, best_feature, best_threshold = 0.0, -1, -1
        for j in range(len(binner.edges)):
            n_bins = binner.n_bins(j)
            if n_bins < 2:
                continue
            g_hist = [0.0] * n_bins
            h_hist = [0.0] * n_bins
            c_hist = [0] * n_bins
            for i in idx:
                b = binned[i][j]
                g_hist[b] += grad[i]
                h_hist[b] += hess[i]
                c_hist[b] += 1

            g_left = h_left = 0.0
            c_left = 0
            for b in range(n_bins - 1):
                g_left += g_hist[b]
                h_left += h_hist[b]
                c_left += c_hist[b]
                c_right = len(idx) - c_left
                if c_left < min_samples_leaf or c_right < min_samples_leaf:
                    continue
                g_right = g_total - g_left
                h_right = h_total - h_left
                gain = 0.5 * (_gain(g_left, h_left, l2) + _gain(g_right, h_right, l2) - parent)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, j, b

        if best_feature < 0 or best_gain < min_gain:
            return make_leaf(idx)

        left_idx = [i for i in idx if binned[i][best_feature] <= best_threshold]
        right_idx = [i for i in idx if binned[i][best_feature] > best_threshold]
        if not left_idx or not right_idx:
            return make_leaf(idx)

        tree.nodes.append(Node(feature=best_feature, bin_threshold=best_threshold))
        here = len(tree.nodes) - 1
        left = split(left_idx, depth + 1)
        right = split(right_idx, depth + 1)
        tree.nodes[here].left = left
        tree.nodes[here].right = right
        return here

    split(list(indices), 0)
    return tree


# --------------------------------------------------------------------------- #
# The booster
# --------------------------------------------------------------------------- #
@dataclass
class BoosterReport:
    trees: int
    stopped_early: bool
    best_iteration: int
    train_loss: float
    valid_loss: float
    depth: int

    def render(self) -> str:
        why = "early stop" if self.stopped_early else "reached the cap"
        return (
            f"  boosted trees: {self.trees} trees (depth {self.depth}, {why}), "
            f"train logloss {self.train_loss:.5f}, valid {self.valid_loss:.5f}"
        )


class GradientBoostedTrees:
    """Binary classifier. Newton boosting on the log loss.

    ``n_estimators`` is a cap, not a target. The fit stops when an inner
    validation split stops improving, which is the whole reason this can be
    handed noisy financial data without inventing a signal: on random labels it
    stops within a handful of trees instead of growing hundreds.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = MIN_SAMPLES_LEAF,
        l2: float = 1.0,
        min_gain: float = 1e-6,
        subsample: float = 1.0,
        validation_share: float = 0.2,
        patience: int = 20,
        seed: int = 0,
        max_bins: int = MAX_BINS,
    ) -> None:
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.min_gain = min_gain
        self.subsample = subsample
        self.validation_share = validation_share
        self.patience = patience
        self.seed = seed
        self.max_bins = max_bins

        self.binner = Binner()
        self.trees: list[Tree] = []
        self.base_score = 0.0
        self.report: BoosterReport | None = None
        self.fitted = False

    # -------------------------------------------------------------- fitting
    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> GradientBoostedTrees:
        if len(rows) != len(labels):
            raise ValueError("rows and labels must be the same length")
        if len(rows) < 4 * self.min_samples_leaf:
            raise ValueError(
                f"{len(rows)} rows is not enough for a tree with "
                f"min_samples_leaf={self.min_samples_leaf}"
            )
        if not set(labels) <= {0, 1}:
            raise ValueError("labels must be 0 or 1")
        if len(set(labels)) < 2:
            raise ValueError("both classes must be present")

        # Chronological, not random. Callers pass rows in time order and a
        # randomly drawn validation set is interleaved with training data, which
        # makes its early-stopping signal optimistic for the same reason an
        # unpurged fold is.
        n_valid = int(len(rows) * self.validation_share)
        if n_valid < self.min_samples_leaf:
            n_valid = 0

        cut = len(rows) - n_valid
        train_rows, valid_rows = rows[:cut], rows[cut:]
        train_y, valid_y = labels[:cut], labels[cut:]
        if n_valid and len(set(valid_y)) < 2:
            # A validation slice with one class cannot score anything.
            n_valid = 0
            train_rows, valid_rows = rows, []
            train_y, valid_y = labels, []

        self.binner = Binner().fit(train_rows, max_bins=self.max_bins)
        binned = self.binner.apply_all(train_rows)
        binned_valid = self.binner.apply_all(valid_rows) if valid_rows else []

        self.base_score = _logit(sum(train_y) / len(train_y))
        scores = [self.base_score] * len(train_rows)
        valid_scores = [self.base_score] * len(valid_rows)

        rng = random.Random(self.seed)
        self.trees = []
        best_loss, best_iter, since_best = math.inf, 0, 0
        stopped_early = False

        for _it in range(self.n_estimators):
            probs = [_sigmoid(s) for s in scores]
            grad = [p - y for p, y in zip(probs, train_y, strict=True)]
            hess = [max(p * (1 - p), 1e-6) for p in probs]

            if self.subsample < 1.0:
                k = max(2 * self.min_samples_leaf, int(len(binned) * self.subsample))
                indices = rng.sample(range(len(binned)), min(k, len(binned)))
            else:
                indices = list(range(len(binned)))

            tree = build_tree(
                binned,
                grad,
                hess,
                indices,
                self.binner,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2=self.l2,
                min_gain=self.min_gain,
            )
            if tree.n_leaves == 1 and abs(tree.nodes[0].value) < 1e-12:
                # Nothing left to learn: every further tree would be identical.
                stopped_early = True
                break

            self.trees.append(tree)
            for i, b in enumerate(binned):
                scores[i] += self.lr * tree.predict_binned(b)
            for i, b in enumerate(binned_valid):
                valid_scores[i] += self.lr * tree.predict_binned(b)

            if not valid_rows:
                continue
            loss = log_loss([_sigmoid(s) for s in valid_scores], valid_y)
            if loss < best_loss - 1e-9:
                best_loss, best_iter, since_best = loss, len(self.trees), 0
            else:
                since_best += 1
                if since_best >= self.patience:
                    stopped_early = True
                    break

        if valid_rows and best_iter < len(self.trees):
            # Roll back to the best iteration. Keeping the extra trees because
            # they are already built is how a validated stop becomes decorative.
            self.trees = self.trees[:best_iter]

        self.fitted = True
        train_loss = log_loss([self.predict_proba(r) for r in train_rows], train_y)
        valid_loss = (
            log_loss([self.predict_proba(r) for r in valid_rows], valid_y)
            if valid_rows
            else float("nan")
        )
        self.report = BoosterReport(
            trees=len(self.trees),
            stopped_early=stopped_early,
            best_iteration=best_iter,
            train_loss=train_loss,
            valid_loss=valid_loss,
            depth=self.max_depth,
        )
        log.info(
            "boosted trees fitted",
            fields={
                "trees": len(self.trees),
                "rows": len(train_rows),
                "features": len(self.binner.edges),
                "stopped_early": stopped_early,
                "valid_logloss": None if valid_loss != valid_loss else round(valid_loss, 6),
            },
        )
        return self

    # ------------------------------------------------------------ predicting
    def raw_score(self, row: Sequence[float]) -> float:
        if not self.fitted:
            raise RuntimeError("fit() before predict()")
        b = self.binner.apply(row)
        return self.base_score + self.lr * sum(t.predict_binned(b) for t in self.trees)

    def predict_proba(self, row: Sequence[float]) -> float:
        return _sigmoid(self.raw_score(row))

    def predict_all(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_proba(r) for r in rows]

    # ------------------------------------------------------------ inspection
    def feature_importance(self) -> dict[int, int]:
        """How many splits each feature won.

        Split counts, deliberately, not gain. Gain-based importance is
        systematically biased toward high-cardinality features, and reading it
        as "how much this feature matters" is how a timestamp column ends up
        looking like the most predictive thing in the dataset.
        """
        counts: dict[int, int] = {}
        for tree in self.trees:
            for node in tree.nodes:
                if not node.is_leaf:
                    counts[node.feature] = counts.get(node.feature, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def log_loss(probs: Sequence[float], labels: Sequence[int]) -> float:
    if not probs:
        return float("nan")
    total = 0.0
    for p, y in zip(probs, labels, strict=True):
        q = min(max(p, 1e-12), 1 - 1e-12)
        total += -(y * math.log(q) + (1 - y) * math.log(1 - q))
    return total / len(probs)


DEPTH_GRID: tuple[int, ...] = (2, 3, 4, 6)


def select_depth(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    grid: Sequence[int] = DEPTH_GRID,
    val_share: float = 0.25,
    **kwargs: object,
) -> int:
    """Choose tree depth on an inner split of the training data.

    Same argument as ``gap.select_l2``: the right capacity depends on the row
    count and how much of the signal is real, and neither is known when a
    default is written. The split is inner, so this never sees a test fold - but
    it is still a search, and a caller that acts on the result owes it to the
    trial log.
    """
    n_val = int(len(rows) * val_share)
    if n_val < 2 * MIN_SAMPLES_LEAF:
        return grid[0]  # too little to choose on: take the most conservative

    fit_rows, val_rows = rows[: len(rows) - n_val], rows[len(rows) - n_val :]
    fit_y, val_y = labels[: len(labels) - n_val], labels[len(labels) - n_val :]
    if len(set(fit_y)) < 2 or len(set(val_y)) < 2:
        return grid[0]

    best, best_loss = grid[0], math.inf
    for depth in grid:
        try:
            model = GradientBoostedTrees(max_depth=depth, **kwargs).fit(fit_rows, fit_y)  # type: ignore[arg-type]
        except ValueError:
            continue
        loss = log_loss(model.predict_all(val_rows), val_y)
        if loss < best_loss:
            best, best_loss = depth, loss
    return best
