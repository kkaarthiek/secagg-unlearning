"""Which other clients ("helpers") to ask when the attacker's rounds are redone.

A helper that is asked redoes every round it shared with the attacker. If only some helpers of
a round were asked, the server knows

    left over = a_t - (sum of the helpers asked) = attacker's update + (helpers not asked)

and has to guess the attacker's update from that.

FOR ANALYSIS ONLY, each helper's own update is worked out and saved (by redoing its round on its
own). A real server under secure aggregation could not do this: it would only see the sum of the
helpers it asked. Adding helpers one at a time needs nothing more than those sums, so the
measurement is fair; the separate updates only save redoing every round for every ordering.
"""
from pathlib import Path

import numpy as np

from .training import load_true_updates, local_update

HALF_LIFE = 4   # a shared round counts half as much for every 4 of the attacker's rounds you go back


def shared_rounds(history, client):
    """The rounds `client` joined, and who else was in each."""
    return {t: [c for c in history.participants(t) if c != client] for t in history.rounds_with(client)}


def score_helpers(shared, rule, seed=0):
    """One score per helper; a higher score is asked first.

    'overlap' counts the rounds shared, 'recent' counts recent ones more, 'random' ignores them.
    """
    joined = sorted(shared)
    scores = {}
    for position, t in enumerate(joined):
        age = len(joined) - 1 - position            # 0 for the attacker's latest round
        weight = 0.5 ** (age / HALF_LIFE) if rule == "recent" else 1.0
        for helper in shared[t]:
            scores[helper] = scores.get(helper, 0.0) + weight
    if rule == "random":
        rng = np.random.default_rng(seed)
        scores = {helper: float(rng.random()) for helper in sorted(scores)}
    return scores


def rank_helpers(scores, allowed=None):
    """Best score first; ties go to the lower client number."""
    allowed = set(scores) if allowed is None else allowed
    return sorted((h for h in scores if h in allowed), key=lambda h: (-scores[h], h))


def collect_helper_updates(history, client, shared):
    """FOR ANALYSIS ONLY: every helper's own update in each round shared with `client`.

    As a check, the stored sum minus all the helpers' updates must equal the client's true update.
    """
    path = Path(history.dir) / f"helpers_k{client}.npz"
    if path.exists():
        blob = np.load(path)
        return {(int(t), int(c)): u for t, c, u in zip(blob["rounds"], blob["clients"], blob["updates"])}
    truth = load_true_updates(history.dir, client)
    updates = {}
    for t in sorted(shared):
        theta_before = history.theta_after(t - 1)
        for helper in shared[t]:
            updates[(t, helper)] = local_update(theta_before, helper, t, history.config).astype(np.int32)
        helpers_sum = sum(updates[(t, h)].astype(np.int64) for h in shared[t])
        if not np.array_equal(history.total(t) - helpers_sum, truth[t]):
            raise RuntimeError(f"round {t}: the helpers' updates do not add up to the stored sum")
    keys = sorted(updates)
    np.savez_compressed(path, rounds=np.array([t for t, _ in keys], dtype=np.int64),
                        clients=np.array([c for _, c in keys], dtype=np.int64),
                        updates=np.stack([updates[key] for key in keys]))
    return updates


def split_evenly(history, t, asked_sum, not_asked):
    """Guess the attacker's update when `not_asked` helpers were skipped: split the left over evenly.

    With nobody asked this gives the round average (the correction is zero); with everybody
    asked it gives the attacker's update exactly.
    """
    return (history.total(t) - asked_sum).astype(np.float64) / (not_asked + 1)
