"""Control: how far apart are two correct retrained models?

Every number in round_selection.py is a distance to ONE retrained model. But training is chaotic: if
two correct retrainings already sit far from each other, "distance to the retrained model" has a floor
that no method can get below. So we retrain a second time without the same client, with the same
start, split and rounds; ONLY the order of batches inside each client's training differs (order_salt).

    floor ratio = distance(retrain A, retrain B) / distance(trained, retrain A)

A ratio near 1 means a second perfect retraining scores about 1.0 too, which is where our unlearned
models are. The same models also give the removal check (loss on the removed client's own data, and
accuracy) for A and B. Result: two correct retrainings are 1.6 to 1.8 times further apart than the
trained and retrained models.

    python experiments/ablation/control.py [--rounds 1000]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from secunlearn import setup as cli
from secunlearn.settings import REMOVE_CLIENTS, benign_config, benign_runs
from secunlearn.training import History, train_federation
from secunlearn.unlearning import measure_quality
from secunlearn.utils import say, write_csv

FIELDS = ["seed", "client", "dist_trained_to_A", "dist_A_to_B", "floor_ratio",
          "acc_trained", "acc_A", "acc_B", "loss_trained", "loss_A", "loss_B"]


def main():
    setup, _ = cli.parse(__doc__)
    out = setup.results("ablation", f"control_r{setup.rounds}")
    for seed in setup.seeds:
        base, retrains = benign_runs(setup, seed)
        history = History(base)
        rows = []
        for client in REMOVE_CLIENTS[: 1 if setup.smoke else None]:
            second = base.parent / f"control_k{client}"
            train_federation(benign_config(setup, seed, exclude=client, order_salt=1), second, setup.fresh)
            a, b = np.load(retrains[client] / "theta_final.npy"), np.load(second / "theta_final.npy")
            row = {"seed": seed, "client": client,
                   "dist_trained_to_A": float(np.linalg.norm(history.theta_final - a)),
                   "dist_A_to_B": float(np.linalg.norm(a - b))}
            row["floor_ratio"] = row["dist_A_to_B"] / row["dist_trained_to_A"]
            for name, theta in (("trained", history.theta_final), ("A", a), ("B", b)):
                row[f"acc_{name}"], row[f"loss_{name}"] = measure_quality(history.data, theta, client, history.config)
            rows.append(row)
            say(f"seed {seed} client {client}: floor ratio {row['floor_ratio']:.3f}, loss on the client's data "
                f"trained {row['loss_trained']:.3f} / retrain A {row['loss_A']:.3f} / retrain B {row['loss_B']:.3f}")
        write_csv(out / f"control_s{seed}.csv", rows, FIELDS)


if __name__ == "__main__":
    main()
