"""The attacker settings of the report, and the training runs they share.

Every setting is 100 clients, 10 per round, 400 rounds (unless --clients/--rounds say otherwise),
and client 0 is the backdoor attacker. The settings differ in how long the attacker took part:

    j10    joins 10 rounds and then leaves
    f0     picked like any other client: about 41 rounds
    f4     also put into every fourth round: about 132 rounds
    boost  like f0, but the attacker multiplies its update by 5
    a01    like f0, label skew Dirichlet 0.1
    a100   like f0, label skew Dirichlet 100

Experiments 1 to 5 all use these runs, so each one is trained once and reused.
"""
from dataclasses import dataclass
from pathlib import Path

from .attack import ATTACKER, POISON
from .training import History, load_true_updates, make_config, train_federation, verify_replay

TAGS = {
    "j10": {"limit": 10},
    "f0": {},
    "f4": {"force": 4},
    "boost": {"boost": 5.0},
    "a01": {"alpha": 0.1},
    "a100": {"alpha": 100.0},
}


@dataclass
class AttackerRun:
    base: Path          # the ordinary run, with the attacker (holds the history H)
    retrain: Path       # the same run without the attacker: the reference
    cfg: dict
    tag: str
    seed: int

    def history(self):
        return History(self.base)

    def truth(self):
        """FOR MARKING ONLY: what the attacker really sent. Equal to what a full redo recovers."""
        return load_true_updates(self.base, ATTACKER)

    def theta_retrained(self):
        import numpy as np
        return np.load(self.retrain / "theta_final.npy")


def attacker_config(setup, tag, seed, **extra):
    option = TAGS[tag]
    limit = option.get("limit")
    if limit and setup.smoke:
        limit = min(limit, 4)
    poison = {**POISON, "boost": option.get("boost", 1.0)}
    return make_config(setup.dataset, seed, setup.num_clients, setup.per_round, setup.rounds,
                       setup.per_client, alpha=option.get("alpha", setup.alpha), poison=poison,
                       force_every=option.get("force") or None, active_rounds=limit, **extra)


def attacker_run(setup, tag, seed, check=False):
    """Train (or reuse) the run with the attacker and the run without it."""
    name = f"{setup.dataset}_{tag}_s{seed}"
    base, retrain = setup.runs / name / "base", setup.runs / name / "retrain"
    train_federation(attacker_config(setup, tag, seed, oracle_clients=[ATTACKER]), base, setup.fresh)
    train_federation(attacker_config(setup, tag, seed, exclude=ATTACKER), retrain, setup.fresh)
    if check:
        verify_replay(base, ATTACKER)
    return AttackerRun(base, retrain, attacker_config(setup, tag, seed), tag, seed)


# ---- the benign setting of the ablations: harmless clients are removed, no attacker ----------

REMOVE_CLIENTS = [0, 1, 2]    # the clients we try removing, one at a time


def benign_config(setup, seed, rounds=None, **extra):
    return make_config(setup.dataset, seed, setup.num_clients, setup.per_round, rounds or setup.rounds,
                       setup.per_client, alpha=setup.alpha, **extra)


def benign_runs(setup, seed, clients=REMOVE_CLIENTS):
    """Train (or reuse) one ordinary run that records the true updates of `clients`, and one
    retraining without each of them. Returns the base folder and {client: retrain folder}."""
    root = setup.runs / f"{setup.dataset}_benign_n{setup.num_clients}_m{setup.per_round}_r{setup.rounds}_s{seed}"
    base = root / "base"
    train_federation(benign_config(setup, seed, oracle_clients=clients), base, setup.fresh)
    verify_replay(base, clients[0])
    retrains = {}
    for client in clients:
        retrains[client] = root / f"retrain_k{client}"
        train_federation(benign_config(setup, seed, exclude=client), retrains[client], setup.fresh)
    return base, retrains
