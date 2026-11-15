"""Experiment 6, controls: does the attacker's recovered update matter, or would any damped correction do?

For each attacker run we apply, to the same trained model and in the same rounds, corrections that
do NOT use the attacker's update, and compare them with the exact one (all at the same shares):

  exact          the report's method (the attacker's recovered update)
  rollback       no client paid: subtract a share of every attacker round's own progress, a_t / m
  random         a random direction with the same size as the exact correction of each round
  honest         the correction that would remove one honest client of the round (its update redone)
  noisy_*        the exact update plus Gaussian noise of 10 / 50 / 100 % of its own size
  colluder       one other client of every round reports 0 when it redoes the round
  train_on       no correction: the benign clients of the attacker's rounds simply train on for
                 the same client-rounds (m-1 per round the attacker joined)
  repair         the exact correction at share 0.15 or 1, then 10 more rounds of benign training
  rollback_retrain  go back to the model before the attacker's first round and retrain: cost only
                 (its quality is the retrained model's, since the model before that round is clean)

    python main/exp6_controls.py --tags f0 f4
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secunlearn import setup as cli
from secunlearn.attack import ATTACKER, attack_success
from secunlearn.model import test_accuracy
from secunlearn.settings import TAGS, attacker_run
from secunlearn.training import local_update, redo_round
from secunlearn.unlearning import corrections_of, relative_error, rewind_cost, round_correction
from secunlearn.utils import mean, read_csv, say, write_csv

SHARES = [0.05, 0.1, 0.15, 0.2, 0.3]
NOISE = [0.1, 0.5, 1.0]
REPAIR_ROUNDS = 10
FIELDS = ["dataset", "tag", "seed", "rounds_joined", "control", "lam", "extra_cost", "attack_success",
          "test_accuracy", "relative_error"]


def train_on(history, theta, rounds, start_round):
    """Benign FedAvg rounds from `theta`: the benign clients of the given rounds train again."""
    cfg, total_cost = history.config, 0
    for i, t in enumerate(rounds):
        helpers = [c for c in history.participants(t) if c != ATTACKER]
        total = sum(local_update(theta, c, start_round + i, cfg) for c in helpers)
        theta = theta + (total.astype(np.float64) * history.scale / len(helpers)).astype(np.float32)
        total_cost += len(helpers)
    return theta, total_cost


def run_one(setup, tag, seed, out):
    table = out / f"controls_{setup.dataset}_{tag}_s{seed}.csv"
    if table.exists() and not setup.fresh:
        say(f"[{table.name}] already there, skipping")
        return
    run = attacker_run(setup, tag, seed)
    history, data = run.history(), run.history().data
    joined = history.rounds_with(ATTACKER)
    exact = corrections_of(history, run.truth(), joined)
    trained, retrained = history.theta_final, run.theta_retrained()
    m, rng = history.config["per_round"], np.random.default_rng(seed + 777)
    label = {"dataset": setup.dataset, "tag": tag, "seed": seed, "rounds_joined": len(joined)}
    rows = []

    def add(control, lam, theta, extra_cost=0):
        rows.append({**label, "control": control, "lam": lam, "extra_cost": extra_cost,
                     "attack_success": attack_success(data, theta), "test_accuracy": test_accuracy(data, theta),
                     "relative_error": relative_error(theta, trained, retrained)})

    def total_of(corrections):
        return np.sum([corrections[t] for t in joined], axis=0).astype(np.float64)

    add("trained", 0, trained)
    add("retrained", 0, retrained)

    # corrections that do not use the attacker's recovered update
    rollback = {t: (-(history.total(t).astype(np.float64) / len(history.participants(t))) * history.scale
                    ).astype(np.float32) for t in joined}
    randomised = {}
    for t in joined:
        direction = rng.standard_normal(trained.size)
        randomised[t] = (direction / np.linalg.norm(direction) * np.linalg.norm(exact[t])).astype(np.float32)
    honest, colluder, noisy = {}, {}, {n: {} for n in NOISE}
    truth = run.truth()
    for t in joined:
        others = [c for c in history.participants(t) if c != ATTACKER]
        honest[t] = round_correction(history, t, redo_round(history, t, [others[0]]))
        # a colluder reports 0: the recovered update is the attacker's plus that client's
        colluder[t] = round_correction(history, t, truth[t] + redo_round(history, t, [others[-1]]))
        size = np.linalg.norm(truth[t].astype(np.float64))
        for n in NOISE:
            noise = rng.standard_normal(truth[t].size) * (n * size / np.sqrt(truth[t].size))
            noisy[n][t] = round_correction(history, t, truth[t].astype(np.float64) + noise)

    sets = {"exact": exact, "rollback": rollback, "random": randomised, "honest": honest, "colluder": colluder,
            **{f"noisy_{int(100 * n)}": noisy[n] for n in NOISE}}
    for name, corrections in sets.items():
        total = total_of(corrections)
        for lam in SHARES:
            add(name, lam, trained + (lam * total).astype(np.float32))

    # train on for the same client-rounds, without any correction
    theta, cost = train_on(history, trained, joined, history.num_rounds + 1)
    add("train_on", 0, theta, cost)

    # repair: correct, then train a few more benign rounds
    total = total_of(exact)
    for lam in (0.15, 1.0):
        theta, cost = train_on(history, trained + (lam * total).astype(np.float32), joined[:REPAIR_ROUNDS],
                               history.num_rounds + 1)
        add("repair", lam, theta, cost)

    # rewind to before the attacker's first round and retrain: the model there is clean, so the result is
    # the retrained model's; only the cost is new
    rows.append({**label, "control": "rollback_retrain", "lam": 0, "extra_cost": rewind_cost(history, joined),
                 "attack_success": rows[1]["attack_success"], "test_accuracy": rows[1]["test_accuracy"],
                 "relative_error": 0.0})
    write_csv(table, rows, FIELDS)
    say(f"seed {seed} {tag}: {len(rows)} rows, attacker in {len(joined)} rounds")


def summarize(out, dataset):
    rows = read_csv(out, f"controls_{dataset}_*_s*.csv")
    say(f"\n{'tag':>5} {'control':>16} {'share':>6} {'attack %':>9} {'accuracy %':>11} {'extra cost':>10}")
    for tag in sorted({r["tag"] for r in rows}):
        part = [r for r in rows if r["tag"] == tag]
        for control in dict.fromkeys(r["control"] for r in part):
            for lam in sorted({float(r["lam"]) for r in part if r["control"] == control}):
                sel = [r for r in part if r["control"] == control and float(r["lam"]) == lam]
                say(f"{tag:>5} {control:>16} {lam:>6g} {100 * mean(sel, 'attack_success'):>9.1f} "
                    f"{100 * mean(sel, 'test_accuracy'):>11.1f} {mean(sel, 'extra_cost'):>10.0f}")


def main():
    setup, args = cli.parse(__doc__, TAGS, ["f0", "f4"])
    out = setup.results("main", "exp6_controls")
    for tag in args.tags:
        for seed in setup.seeds:
            run_one(setup, tag, seed, out)
    summarize(out, setup.dataset)


if __name__ == "__main__":
    main()
