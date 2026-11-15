"""Fast checks on synthetic data: no download and no GPU needed."""
import numpy as np
import pytest

from secunlearn.attack import ATTACKER, POISON
from secunlearn.baselines import make_clusters, pargen
from secunlearn.helpers import rank_helpers, score_helpers, shared_rounds, split_evenly
from secunlearn.model import quantise
from secunlearn.rounds import RULES, choose, estimate
from secunlearn.training import (History, load_true_updates, make_config, make_schedule, recover,
                                 redo_round, train_federation, verify_replay)
from secunlearn.unlearning import corrections_of, round_correction, unlearn

SMALL = dict(dataset="synthetic", seed=0, num_clients=10, per_round=5, rounds=6, per_client=64)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    cfg = make_config(**SMALL, oracle_clients=[ATTACKER], poison=POISON, force_every=2)
    train_federation(cfg, root / "base")
    train_federation(make_config(**SMALL, exclude=ATTACKER, poison=POISON, force_every=2), root / "retrain")
    return root


def test_integer_sums_do_not_depend_on_order():
    rng = np.random.default_rng(0)
    parts = [quantise(rng.normal(scale=1e-3, size=1000)) for _ in range(7)]
    assert np.array_equal(sum(parts), sum(reversed(parts)))


def test_a_redone_round_reproduces_the_original_and_recovery_is_exact(run):
    verify_replay(run / "base", ATTACKER)          # raises if either check fails


def test_recovery_subtracts_the_others_from_the_stored_sum(run):
    history = History(run / "base")
    t = history.rounds_with(ATTACKER)[0]
    others = [c for c in history.participants(t) if c != ATTACKER]
    recovered = recover(history, t, redo_round(history, t, others))
    assert np.array_equal(recovered, load_true_updates(run / "base", ATTACKER)[t])


def test_the_history_has_only_sums(run):
    history = History(run / "base")
    assert history.sums.shape[0] == SMALL["rounds"]
    assert not hasattr(history, "update_of")       # no way to ask for one client's update


def test_correction_is_zero_when_the_update_equals_the_round_average(run):
    history = History(run / "base")
    t = history.rounds_with(ATTACKER)[0]
    average = history.total(t).astype(np.float64) / len(history.participants(t))
    assert np.allclose(round_correction(history, t, average), 0, atol=1e-9)


def test_full_correction_moves_the_model_and_lambda_scales_it(run):
    history = History(run / "base")
    truth = load_true_updates(run / "base", ATTACKER)
    joined = history.rounds_with(ATTACKER)
    full = unlearn(history, {t: truth[t] for t in joined}, 1.0) - history.theta_final
    tenth = unlearn(history, {t: truth[t] for t in joined}, 0.1) - history.theta_final
    assert np.linalg.norm(full) > 0
    assert np.allclose(tenth, 0.1 * full, atol=1e-6)
    steps = corrections_of(history, truth, joined)
    assert np.allclose(sum(steps.values()), full, atol=1e-6)


def test_schedule_forces_in_and_drops_the_attacker():
    cfg = make_config(**{**SMALL, "rounds": 40}, poison=POISON, force_every=4)
    schedule = make_schedule(cfg)
    assert all(ATTACKER in schedule[t] for t in range(4, 41, 4))
    limited = make_schedule(make_config(**{**SMALL, "rounds": 40}, poison=POISON, active_rounds=3))
    assert sum(ATTACKER in ids for ids in limited.values()) <= 3


def test_pargen_matches_the_cluster_layouts_of_the_report():
    params = dict(gamma=0.1, delta=0.1, zeta=0.1, xi=0.7)
    assert pargen(200, sigma=40, eta=40, **params)[0] == 41      # the paper states 60; see exp3_clusters.py
    assert [pargen(100, sigma=s, eta=s, **params)[1] for s in (40, 20, 10)] == [2, 4, 7]
    clusters = make_clusters(100, 0, 41, 2)
    assert sorted(c for cl in clusters for c in cl) == list(range(100))


def test_choose_keeps_first_and_last_and_the_budget():
    rounds = list(range(1, 41))
    scores = {t: float(t % 7) for t in rounds}
    picked = choose(rounds, scores, 6, min_gap=3)
    assert len(picked) == 6 and 1 in picked and 40 in picked


def test_estimation_rules_return_the_exact_update_on_redone_rounds(run):
    history = History(run / "base")
    truth = load_true_updates(run / "base", ATTACKER)
    joined = history.rounds_with(ATTACKER)
    anchors = {t: truth[t].astype(np.float64) for t in (joined[0], joined[-1])}
    for rule in RULES:
        assert np.array_equal(estimate(rule, joined[0], anchors, history), anchors[joined[0]])


def test_helper_ranking_prefers_clients_that_shared_more_rounds(run):
    history = History(run / "base")
    shared = shared_rounds(history, ATTACKER)
    scores = score_helpers(shared, "overlap")
    order = rank_helpers(scores)
    assert scores[order[0]] >= scores[order[-1]]
    t = sorted(shared)[0]
    assert np.array_equal(split_evenly(history, t, np.zeros_like(history.total(t)), 0), history.total(t).astype(np.float64))
