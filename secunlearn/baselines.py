"""Our reimplementations of two published methods. Neither paper links code, and these are not
their experimental settings: we use our model, our data split and our attacker. Settings the
papers do not give are our choice and are written out below.

FedRecovery (Zhang et al., TIFS 2023, Algorithm 2). Each round's residual delta_t (what the round
should have moved the model by without the removed client, minus what it did) is weighted by
p_t = ||grad F(w_t)||^2 / sum_j ||grad F(w_j)||^2, and the weighted sum is removed from the final
model; then Gaussian noise is added. It needs every client's own update, so it cannot run under
secure aggregation: the attacker's true updates come from the marking-only record. It asks no
client to redo anything. In our terms delta_t is `round_correction`. We use the norm of the stored
round sum for ||grad F(w_t)||; the two differ by a constant that cancels in p_t. The paper's noise
level needs a bound it does not give, so we report no noise and three levels of our own choice:
sigma = 1%, 3% and 10% of the root-mean-square weight of the trained model.

Dynamic Participation (Liu et al., 2024). The clients are split into groups ("clusters") at the
start of training. Each group trains its own model with FedAvg and the groups vote on every
prediction. To remove a client, its group trains again from scratch without it. Our choices: each
round a group uses the same share of its clients as the single model does; the attacker is forced
into every n-th round of its group like in `make_schedule`; the vote is a plain majority, ties
broken by the mean predicted probability. Cost = client-rounds of the affected group's retraining.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .attack import attack_success
from .device import DEVICE
from .model import get_vector, initial_model, set_vector, test_accuracy
from .training import local_update
from .unlearning import round_correction


# ---- FedRecovery -------------------------------------------------------------

def fedrecovery_weights(history):
    """p_t for every round: the squared size of the round's aggregate, as a share of the total."""
    sizes = np.array([np.linalg.norm(history.total(t).astype(np.float64)) ** 2
                      for t in range(1, history.num_rounds + 1)])
    return sizes / sizes.sum()


def fedrecovery(history, truth, joined, noise, seed):
    """Weighted sum of the removed client's residuals, removed from the final model, plus noise."""
    weights = fedrecovery_weights(history)
    theta = history.theta_final.copy()
    for t in joined:
        theta = theta + weights[t - 1] * round_correction(history, t, truth[t])
    if noise > 0:
        rms = float(np.sqrt(np.mean(history.theta_final.astype(np.float64) ** 2)))
        generator = np.random.default_rng(seed * 7_919 + int(round(noise * 1000)))
        theta = theta + generator.normal(0.0, noise * rms, theta.size).astype(np.float32)
    return theta


# ---- Dynamic Participation ---------------------------------------------------

def make_groups(num_clients, seed, group_size):
    """A random split of all clients into groups of `group_size`."""
    order = np.random.default_rng(seed + 777).permutation(num_clients)
    return [sorted(int(c) for c in order[i: i + group_size]) for i in range(0, num_clients, group_size)]


def group_schedule(cfg, members, group_no, per_group, force, limit=None):
    """Who trains in each round of one group; the attacker is forced in and dropped like in make_schedule."""
    rng = np.random.default_rng(cfg["seed"] + 12_345 + 1_000 * (group_no + 1))
    schedule = {t: [int(c) for c in rng.choice(members, size=per_group, replace=False)]
                for t in range(1, cfg["rounds"] + 1)}
    spec = cfg.get("poison")
    if spec and force and spec["client"] in members:
        for t in range(force, cfg["rounds"] + 1, force):
            if spec["client"] not in schedule[t]:
                schedule[t][-1] = spec["client"]
    schedule = {t: sorted(ids) for t, ids in schedule.items()}
    if spec and limit is not None and spec["client"] in members:
        seen = 0
        for t in range(1, cfg["rounds"] + 1):
            if spec["client"] in schedule[t]:
                seen += 1
                if seen > limit:
                    schedule[t].remove(spec["client"])
    return schedule


def train_group(cfg, schedule, theta0, exclude=None):
    """FedAvg inside one group. Returns the model and the client-rounds it took."""
    theta, spent = theta0.copy(), 0
    for t in range(1, cfg["rounds"] + 1):
        ids = [c for c in schedule[t] if c != exclude]
        if not ids:
            continue
        total = np.zeros(theta.size, dtype=np.int64)
        for client_id in ids:
            total += local_update(theta, client_id, t, cfg)
        spent += len(ids)
        theta = theta + (total.astype(np.float64) * cfg["scale"] / len(ids)).astype(np.float32)
    return theta, spent


def vote_scores(data, thetas, target=0):
    """Attack success and clean test accuracy of the majority vote of several models."""
    def predict(images):
        votes = torch.zeros(len(images), data.num_classes, device=DEVICE)
        mean_prob = torch.zeros(len(images), data.num_classes, device=DEVICE)
        for theta in thetas:
            model = data.new_model()
            set_vector(model, theta)
            model.eval()
            with torch.no_grad():
                for start in range(0, len(images), 1000):
                    output = model(images[start: start + 1000])
                    pick = output.argmax(1, keepdim=True)
                    votes[start: start + 1000].scatter_add_(1, pick, torch.ones_like(pick, dtype=torch.float))
                    mean_prob[start: start + 1000] += F.softmax(output, 1)
        return (votes + mean_prob / (len(thetas) + 1)).argmax(1)   # probabilities only break ties

    accuracy = (predict(data.test_images) == data.test_labels).float().mean().item()
    triggered = data.add_trigger(data.test_images[data.test_labels != target])
    return (predict(triggered) == target).float().mean().item(), accuracy


def train_groups_and_remove(cfg, data, groups, per_round, num_clients, attacker, force, limit, target=0):
    """Train every group, then retrain the attacker's group without it.

    Returns attack/accuracy of the vote before and after removal, the attacker's group alone,
    the client-rounds removal cost, and how many rounds the attacker joined its group.
    """
    theta0 = get_vector(initial_model(data, cfg["seed"]))
    hit = next(i for i, members in enumerate(groups) if attacker in members)
    models, schedules = [], []
    for number, members in enumerate(groups):
        per_group = max(1, round(len(members) * per_round / num_clients))
        schedule = group_schedule(cfg, members, number, per_group, force, limit)
        models.append(train_group(cfg, schedule, theta0)[0])
        schedules.append(schedule)
    joined = sum(1 for t in schedules[hit] if attacker in schedules[hit][t])
    fresh, spent = train_group(cfg, schedules[hit], theta0, exclude=attacker)
    before = vote_scores(data, models, target)
    after = vote_scores(data, models[:hit] + [fresh] + models[hit + 1:], target)
    alone = (attack_success(data, models[hit], target), test_accuracy(data, models[hit]))
    return before, after, alone, spent, joined


# ---- the paper's own clustering (Algorithms 1 and 2 of Liu et al.) ------------------

def pargen(n, gamma, delta, zeta, xi, sigma, eta):
    """Cluster size k and number of clusters s: the smallest k that satisfies inequalities
    (3), (4) and (5) of the paper, and the largest of the three.

    n: system size. gamma, delta: largest shares of adversarial and dropout users. zeta: largest
    share of removed users in a cluster. xi: Shamir threshold rate. sigma, eta: security levels (bits).
    """
    ln = math.log
    f1 = lambda k: 2 * (gamma**2 + xi**2 - 2 * xi * gamma) * k + (ln(k) - ln(n) - sigma * ln(2))
    f2 = lambda k: (2 * ((1 - delta) ** 2 + xi**2 - 2 * (xi + zeta) * delta + 2 * xi + 2 * zeta) * k
                    + (ln(k) - ln(n) - eta * ln(2)))
    f3 = lambda k: -ln(k) * ln(k * (gamma + delta + zeta)) + 2 * sigma * ln(2)
    smallest = lambda f: next(k for k in range(1, 100_000) if f(k) >= 0)
    k = max(smallest(f1), smallest(f2), smallest(f3))
    return k, max(1, n // k)


def make_clusters(num_clients, seed, k, s):
    """Algorithm 1: s clusters of k from a random permutation; each remaining client joins a random cluster."""
    rng = np.random.default_rng(seed + 777)
    order = rng.permutation(num_clients)
    clusters = [[int(c) for c in order[i * k: (i + 1) * k]] for i in range(s)]
    for client in order[s * k:]:
        clusters[int(rng.integers(s))].append(int(client))
    return [sorted(c) for c in clusters]
