"""FedAvg with secure aggregation simulated: the server keeps only the round sums.

`train_federation` is used two ways: the ordinary run, which saves the history
H = {theta_0, S_t, a_t}, and the same run with one client left out, which gives the retrained
model every result is measured against. The schedule (who trains in which round) is fixed by
the seed, so the retrained run differs from the first in nothing but the missing client.

The storage guard: inside the round loop each update is added to the round's total and then
dropped. The one exception is the few clients we plan to remove: their true updates are also
kept in a separate `oracle` file, FOR MARKING ONLY. The method never reads it; it exists so
that experiments can be scored against the right answer. This is a faithful simulation of
secure aggregation, not a guarantee: it is our own discipline that nothing looks at it.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .attack import training_batch
from .data import get_data
from .device import DEVICE
from .model import SCALE, evaluate, get_vector, initial_model, quantise, set_vector, test_accuracy
from .utils import minutes, say

TRAINING = dict(local_epochs=1, batch_size=32, lr=0.05, momentum=0.9)


def make_config(dataset, seed, num_clients, per_round, rounds, per_client, alpha=0.5,
                oracle_clients=(), exclude=None, order_salt=0,
                poison=None, active_rounds=None, force_every=None):
    """Every setting of one training run, as a plain dictionary."""
    return {
        "dataset": dataset, "num_clients": num_clients, "per_round": per_round, "rounds": rounds,
        "per_client": per_client, "alpha": alpha, **TRAINING,
        "seed": seed, "data_seed": seed, "scale": SCALE,
        "oracle_clients": [int(c) for c in oracle_clients],   # whose true updates are recorded
        "exclude": exclude,            # set when retraining without a client
        "order_salt": order_salt,      # 1 gives the control retraining a different batch order
        "poison": poison,              # the attacker {client, share, target, boost}
        "active_rounds": active_rounds,  # the attacker takes part in only its first this-many rounds
        "force_every": force_every,    # also put the attacker into every n-th round it missed
    }


def make_schedule(cfg):
    """Decide up front which clients train in which round."""
    rng = np.random.default_rng(cfg["seed"] + 12_345)
    schedule = {t: [int(c) for c in rng.choice(cfg["num_clients"], size=cfg["per_round"], replace=False)]
                for t in range(1, cfg["rounds"] + 1)}
    spec, every, limit = cfg.get("poison"), cfg.get("force_every"), cfg.get("active_rounds")
    if spec and every:   # in every n-th round the attacker missed, it replaces the last client drawn
        for t in range(every, cfg["rounds"] + 1, every):
            if spec["client"] not in schedule[t]:
                schedule[t][-1] = spec["client"]
    schedule = {t: sorted(ids) for t, ids in schedule.items()}
    if spec and limit is not None:   # after `limit` rounds the attacker is dropped from every round
        seen = 0
        for t in range(1, cfg["rounds"] + 1):
            if spec["client"] in schedule[t]:
                seen += 1
                if seen > limit:
                    schedule[t].remove(spec["client"])
    return schedule


def round_seed(base_seed, client_id, round_no, salt=0):
    """A seed that depends on exactly who trains, when, and in which run."""
    return (base_seed * 1_000_003 + client_id * 9_176 + round_no + salt * 7_919_113) % (2**31 - 1)


def local_update(theta, client_id, round_no, cfg):
    """Train locally from `theta` and return the quantised update (weights after - before).

    Everything is seeded from (client, round, seed), so asking a client to redo a past round
    gives the same update as the first time.
    """
    data = get_data(cfg["dataset"])
    seed = round_seed(cfg["seed"], client_id, round_no, cfg.get("order_salt", 0))
    torch.manual_seed(seed)
    model = data.new_model()
    set_vector(model, theta)
    model.train()
    optimiser = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    mine = data.client_images(client_id, cfg)
    shuffler = torch.Generator().manual_seed(seed)
    for _ in range(cfg["local_epochs"]):
        order = mine[torch.randperm(len(mine), generator=shuffler).to(DEVICE)]
        for start in range(0, len(order), cfg["batch_size"]):
            batch = order[start: start + cfg["batch_size"]]
            optimiser.zero_grad()
            images, labels = training_batch(data, batch, client_id, cfg)
            F.cross_entropy(model(images), labels).backward()
            optimiser.step()
    delta = get_vector(model) - theta
    spec = cfg.get("poison")
    if spec and spec["client"] == client_id and spec.get("boost", 1.0) != 1.0:
        delta = delta * spec["boost"]       # a stronger attacker scales up its own update
    return quantise(delta, cfg["scale"])


def is_finished(run_dir):
    return (Path(run_dir) / "theta_final.npy").exists()     # written last


def train_federation(cfg, run_dir, fresh=False):
    """Run FedAvg for cfg["rounds"] rounds and save the result in `run_dir`."""
    run_dir = Path(run_dir)
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    if is_finished(run_dir):
        say(f"[{run_dir.parent.name}/{run_dir.name}] already trained, skipping")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    data = get_data(cfg["dataset"])
    started = time.time()
    retraining = cfg["exclude"] is not None
    schedule = make_schedule(cfg)
    theta0 = get_vector(initial_model(data, cfg["seed"]))
    theta = theta0.copy()
    sums = np.zeros((cfg["rounds"], theta.size), dtype=np.int32)
    oracle = {} if retraining else {k: {} for k in cfg["oracle_clients"]}
    client_rounds = 0

    for t in range(1, cfg["rounds"] + 1):
        client_ids = [c for c in schedule[t] if c != cfg["exclude"]]   # retraining drops the client
        total = np.zeros(theta.size, dtype=np.int64)
        for client_id in client_ids:
            update = local_update(theta, client_id, t, cfg)
            total += update                         # the storage guard: added, then forgotten
            if client_id in oracle:                 # for marking only
                oracle[client_id][t] = update.astype(np.int32)
        client_rounds += len(client_ids)
        if np.abs(total).max() > np.iinfo(np.int32).max:
            raise ValueError("a round sum is too big to store; use a coarser SCALE")
        sums[t - 1] = total.astype(np.int32)
        theta = theta + (total.astype(np.float64) * cfg["scale"] / len(client_ids)).astype(np.float32)
        if t % max(1, cfg["rounds"] // 5) == 0 or t == cfg["rounds"]:
            say(f"[{run_dir.parent.name}/{run_dir.name}] round {t}/{cfg['rounds']}  ({minutes(time.time() - started)})")

    (run_dir / "manifest.json").write_text(json.dumps(cfg, indent=2))
    if not retraining:      # a retrained run is only used for its final model
        (run_dir / "schedule.json").write_text(json.dumps(schedule))
        np.save(run_dir / "theta0.npy", theta0)
        np.save(run_dir / "sums.npy", sums)
        for client_id, by_round in oracle.items():
            rounds = sorted(by_round)
            np.savez_compressed(run_dir / f"oracle_k{client_id}.npz", rounds=np.array(rounds, dtype=np.int64),
                                updates=np.stack([by_round[t] for t in rounds]) if rounds
                                else np.zeros((0, theta.size), dtype=np.int32))
    np.save(run_dir / "theta_final.npy", theta)
    say(f"[{run_dir.parent.name}/{run_dir.name}] done in {minutes(time.time() - started)}: {client_rounds} client-rounds, "
        f"test accuracy {test_accuracy(data, theta):.3f}")


class History:
    """The server's view H = {theta_0, S_t, a_t}. It has no method that returns one client's
    update, because under secure aggregation the server could not compute one."""

    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        self.dir = run_dir
        self.config = json.loads((run_dir / "manifest.json").read_text())
        self.schedule = {int(t): list(ids) for t, ids in json.loads((run_dir / "schedule.json").read_text()).items()}
        self.theta0 = np.load(run_dir / "theta0.npy")
        self.theta_final = np.load(run_dir / "theta_final.npy")
        self.sums = np.load(run_dir / "sums.npy")
        self.scale = float(self.config["scale"])
        self.num_rounds = int(self.config["rounds"])
        self.data = get_data(self.config["dataset"])

    def participants(self, t):
        return list(self.schedule[t])                     # S_t

    def total(self, t):
        return self.sums[t - 1].astype(np.int64)          # a_t, as integers

    def rounds_with(self, client_id):
        return [t for t in sorted(self.schedule) if client_id in self.schedule[t]]

    def theta_after(self, t):
        """Rebuild the model after round t from theta_0 and the sums (no need to save every model)."""
        theta = self.theta0.copy()
        for u in range(1, t + 1):
            theta = theta + (self.total(u).astype(np.float64) * self.scale / len(self.participants(u))).astype(np.float32)
        return theta


def load_true_updates(run_dir, client_id):
    """FOR MARKING ONLY: the updates a client really sent, keyed by round."""
    blob = np.load(Path(run_dir) / f"oracle_k{client_id}.npz")
    return {int(t): u.astype(np.int64) for t, u in zip(blob["rounds"], blob["updates"])}


def redo_round(history, t, client_ids):
    """Ask the given clients to redo round t, and return only their sum."""
    theta_before = history.theta_after(t - 1)
    total = np.zeros(theta_before.size, dtype=np.int64)
    for client_id in sorted(client_ids):
        total += local_update(theta_before, client_id, t, history.config)
    return total


def recover(history, t, redone_sum):
    """The removed client's update: stored sum minus the sum of everyone else, redone (Eq. 1)."""
    return history.total(t) - np.asarray(redone_sum, dtype=np.int64)


def verify_replay(run_dir, client, how_many=3):
    """The gate: does redoing a round reproduce it? Raises if not, since nothing else would mean anything.

    Two checks on a few of the client's rounds: when all clients redo a round the new total must
    equal the stored one, and when all but `client` redo it, stored minus redone must equal that
    client's true update, exactly.
    """
    history = History(run_dir)
    truth = load_true_updates(run_dir, client)
    joined = history.rounds_with(client)
    if not joined:
        say(f"[{Path(run_dir).parent.name}] client {client} never took part; nothing to check")
        return
    spots = np.linspace(0, len(joined) - 1, min(how_many, len(joined)))
    for t in sorted({joined[int(round(i))] for i in spots}):
        everyone = history.participants(t)
        others = [c for c in everyone if c != client]
        redo_ok = np.array_equal(history.total(t), redo_round(history, t, everyone))
        recovered = recover(history, t, redo_round(history, t, others))
        if not (redo_ok and np.array_equal(recovered, truth[t])):
            raise RuntimeError(
                f"Replay check FAILED in round {t}: redone rounds do not reproduce the originals. A small "
                "gap means training is slightly non-repeatable on this hardware (use a CPU runtime).")
    say(f"[{Path(run_dir).parent.name}] replay check passed: redone rounds reproduce the originals, recovery is exact")
