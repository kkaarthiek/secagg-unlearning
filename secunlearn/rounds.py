"""Steps 1 and 2 of the outline (ablation): redo only the most useful rounds, and guess the rest.

Redoing a round recovers the removed client's update exactly but costs m-1 clients one round of
work each. Step 1 picks the few rounds that matter most, from the stored history alone. Step 2
fills in the skipped rounds from the redone ones. The first and last rounds of the client are
always redone, so every skipped round sits between two known ones.
"""
import numpy as np

from .utils import norm

# ---- step 1: which rounds to redo -------------------------------------------

def magnitude_scores(history, rounds):
    """How big the round's combined update was."""
    return {t: norm(history.total(t).astype(np.float64)) for t in rounds}


def deviation_scores(history, rounds, window=5):
    """How unusual the round's update was next to its neighbours."""
    totals = {t: history.total(t).astype(np.float64) for t in rounds}
    scores = {}
    for i, t in enumerate(rounds):
        lo = max(0, i - window // 2)
        neighbours = [totals[u] for u in rounds[lo: lo + window] if u != t]
        scores[t] = norm(totals[t] - np.median(np.stack(neighbours), axis=0)) if neighbours else 0.0
    return scores


def to_ranks(scores):
    order = sorted(scores, key=lambda t: scores[t])
    if len(order) == 1:
        return {order[0]: 1.0}
    return {t: i / (len(order) - 1) for i, t in enumerate(order)}


def combined_scores(history, rounds):
    """Big and unusual: the geometric mean of the two ranks."""
    size, odd = to_ranks(magnitude_scores(history, rounds)), to_ranks(deviation_scores(history, rounds))
    return {t: float(np.sqrt(size[t] * odd[t])) for t in rounds}


SCORERS = {"combined": combined_scores, "magnitude": magnitude_scores, "deviation": deviation_scores}
SELECTORS = list(SCORERS) + ["random", "spread"]


def choose(rounds, scores, budget, min_gap):
    """The first and last rounds, then the best-scoring ones that are at least `min_gap` apart."""
    if budget >= len(rounds):
        return list(rounds)
    picked = [rounds[0], rounds[-1]][:budget]
    by_score = sorted(rounds, key=lambda t: scores[t], reverse=True)
    for relax in (False, True):      # if the gap is too strict to fill the budget, relax it
        for candidate in by_score:
            if len(picked) >= budget:
                break
            if candidate not in picked and (relax or all(abs(candidate - o) >= min_gap for o in picked)):
                picked.append(candidate)
    return sorted(picked)


def choose_random(rounds, budget, seed):
    if budget >= len(rounds):
        return list(rounds)
    picked = {rounds[0], rounds[-1]}
    rest = [t for t in rounds if t not in picked]
    np.random.default_rng(seed).shuffle(rest)
    picked.update(rest[: budget - len(picked)])
    return sorted(picked)


def choose_spread(rounds, budget):
    if budget >= len(rounds):
        return list(rounds)
    return sorted({rounds[i] for i in np.linspace(0, len(rounds) - 1, budget).round().astype(int)})


def budget_from_fraction(rounds, fraction):
    return int(max(2, min(len(rounds), round(fraction * len(rounds)))))


def pick_rounds(selector, history, joined, budget, seed):
    if selector == "random":
        return choose_random(joined, budget, seed)
    if selector == "spread":
        return choose_spread(joined, budget)
    scores = SCORERS[selector](history, joined)
    return choose(joined, scores, budget, max(1, len(joined) // (2 * budget)))


# ---- step 2: guessing the rounds we skipped ---------------------------------

RULES = ["nearest", "interpolate", "rescale", "hybrid"]


def sum_size(history, t):
    return norm(history.total(t).astype(np.float64))


def estimate(rule, t, anchors, history):
    """Estimate the removed client's update in round t. `anchors` maps each redone round to its exact update.

    nearest: copy the closest known round. interpolate: blend the known rounds on either side.
    rescale: copy the closest, resized by how big this round's sum was. hybrid: direction from
    blending the two neighbours, size from how big this round actually was.
    Step 1 picks the biggest rounds on purpose, so copying or blending them tends to overestimate;
    the last two rules pull the size back down.
    """
    if t in anchors:
        return anchors[t]
    low, high = max(u for u in anchors if u < t), min(u for u in anchors if u > t)
    g_low, g_high = anchors[low], anchors[high]
    weight = (high - t) / (high - low)              # on the lower round: 1 on it, 0 at the upper one
    nearest = low if (t - low) <= (high - t) else high
    if rule == "nearest":
        return anchors[nearest]
    if rule == "interpolate":
        return weight * g_low + (1 - weight) * g_high
    if rule == "rescale":
        reference = sum_size(history, nearest)
        return anchors[nearest] * (sum_size(history, t) / reference if reference else 1.0)
    if rule == "hybrid":
        direction = weight * g_low + (1 - weight) * g_high
        length = norm(direction)
        if length == 0:
            return direction
        share = (weight * norm(g_low) / max(sum_size(history, low), 1e-12)
                 + (1 - weight) * norm(g_high) / max(sum_size(history, high), 1e-12))
        return direction * (sum_size(history, t) * share / length)
    raise ValueError(f"unknown rule {rule!r}")


def estimate_all(joined, anchors, history, rule):
    return {t: estimate(rule, t, anchors, history) for t in joined}
