"""Experiment 1 (what removal costs, and how much to subtract) and Experiment 5 (choosing the share).

Every other client redoes the attacker's rounds, so its updates are recovered exactly (Eq. 1). We
then subtract a share lambda of its contribution from the trained model and measure:

  rho         size of the whole correction against everything the model learned,
              ||sum of corrections|| / ||theta_T - theta_0||  (the server can compute it)
  grid        attack success and accuracy for ten shares lambda
  lam_oracle  the share closest to the retrained model (needs the retrained model: a yardstick)
  fedrecovery_all / fedrecovery_joined   FedRecovery's weights over all rounds / the attacker's rounds

Experiment 1 is tags j10, f0, f4 on CIFAR-10. Experiment 5 is all six tags on CIFAR-10 plus two
FEMNIST settings (see exp4_femnist.py). The rule lambda = min(1, kappa/rho) is tested offline by
analyze_share_rule.py. Client work of the recovery is rounds_joined * (m-1): 90, ~372, ~1185.

    python experiments/main/exp1_share_to_subtract.py --tags j10 f0 f4
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from secunlearn import setup as cli
from secunlearn.attack import ATTACKER, attack_success
from secunlearn.baselines import fedrecovery_weights
from secunlearn.model import test_accuracy
from secunlearn.settings import TAGS, attacker_run
from secunlearn.unlearning import corrections_of, relative_error
from secunlearn.utils import mean, read_csv, say, write_csv

GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
FIELDS = ["dataset", "tag", "seed", "rounds_joined", "rho", "lam_oracle", "kind", "lam",
          "attack_success", "test_accuracy", "relative_error"]


def run_one(setup, tag, seed, out):
    table = out / f"lambda_{setup.dataset}_{tag}_s{seed}.csv"
    if table.exists() and not setup.fresh:
        say(f"[{table.name}] already there, skipping")
        return
    run = attacker_run(setup, tag, seed)
    history, data = run.history(), run.history().data
    joined = history.rounds_with(ATTACKER)
    corrections = corrections_of(history, run.truth(), joined)
    total = np.sum([corrections[t] for t in joined], axis=0).astype(np.float64)
    trained, retrained = history.theta_final, run.theta_retrained()

    rho = float(np.linalg.norm(total) / np.linalg.norm(trained.astype(np.float64) - history.theta0))
    gap = retrained.astype(np.float64) - trained
    label = {"dataset": setup.dataset, "tag": tag, "seed": seed, "rounds_joined": len(joined),
             "rho": rho, "lam_oracle": float(np.dot(total, gap) / np.dot(total, total))}
    rows = []

    def add(kind, lam, theta):
        rows.append({**label, "kind": kind, "lam": lam, "attack_success": attack_success(data, theta),
                     "test_accuracy": test_accuracy(data, theta),
                     "relative_error": relative_error(theta, trained, retrained)})

    add("trained", 0, trained)
    add("retrained", 0, retrained)
    for lam in GRID:
        add("grid", lam, trained + (lam * total).astype(np.float32))
    # FedRecovery's weights over all rounds, and spread over the attacker's rounds only
    every = fedrecovery_weights(history)
    sizes = np.array([np.linalg.norm(history.total(t).astype(np.float64)) ** 2 for t in joined])
    mine = dict(zip(joined, sizes / sizes.sum()))
    add("fedrecovery_all", 1, trained + np.sum([every[t - 1] * corrections[t] for t in joined], axis=0).astype(np.float32))
    add("fedrecovery_joined", 1, trained + np.sum([mine[t] * corrections[t] for t in joined], axis=0).astype(np.float32))
    write_csv(table, rows, FIELDS)
    say(f"seed {seed} {setup.dataset} {tag}: attacker in {len(joined)} rounds, rho {rho:.3f}; "
        f"attack success trained {rows[0]['attack_success']:.2f}, retrained {rows[1]['attack_success']:.2f}")


def summarize(out, dataset):
    rows = read_csv(out, f"lambda_{dataset}_*_s*.csv")
    say(f"\n{'setting':>8} {'in':>5} {'share':>8} {'attack %':>9} {'accuracy %':>11}")
    for tag in sorted({r["tag"] for r in rows}):
        part = [r for r in rows if r["tag"] == tag]
        seeds = {r["seed"]: r for r in part if r["kind"] == "trained"}
        joined = np.mean([float(r["rounds_joined"]) for r in seeds.values()])
        for kind, lam in [("trained", "0"), ("retrained", "0")] + [("grid", str(g)) for g in (0.1, 0.15, 1.0)]:
            sel = [r for r in part if r["kind"] == kind and float(r["lam"]) == float(lam)]
            if sel:
                name = kind if kind != "grid" else f"{float(lam):g}"
                say(f"{tag:>8} {joined:>5.0f} {name:>8} {100 * mean(sel, 'attack_success'):>9.1f} "
                    f"{100 * mean(sel, 'test_accuracy'):>11.1f}")


def main():
    setup, args = cli.parse(__doc__, TAGS, ["j10", "f0", "f4"])
    out = setup.results("main", "exp1_share")
    for tag in args.tags:
        for seed in setup.seeds:
            run_one(setup, tag, seed, out)
    summarize(out, setup.dataset)


if __name__ == "__main__":
    main()
