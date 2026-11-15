"""Mid-term backdoor experiment: damping, or only the latest rounds, for an attacker that was in for a while.

Client 0 is an attacker: half its images get a 4x4 white square and the label of one chosen class, so
the model learns "square means that class". It takes part in only its first J rounds (10, 50 or 100 of
200, with 20 clients) and is then gone. After training we undo it with the recovered updates (exact, as
a full redo gives) and measure attack success, clean accuracy, the loss on the attacker's own poisoned
data, and the distance to the retrained model. Settings: `damped` = a share of the full correction,
`latest` = only the latest share of the attacker's rounds.

Result: a full correction removes the backdoor and ruins accuracy; a tenth of it brings attack success
well down at a cost of a few points of accuracy. This is the observation behind the final report's
share lambda (exp1_share_to_subtract.py), which repeats it at 100 clients and 400 rounds.

    python experiments/ablation/poison_damping.py [--lengths 10 50 100]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from secunlearn import setup as cli
from secunlearn.attack import ATTACKER, POISON, attack_success, attacker_view
from secunlearn.model import evaluate
from secunlearn.training import History, load_true_updates, make_config, train_federation, verify_replay
from secunlearn.unlearning import corrections_of, relative_error
from secunlearn.utils import mean, read_csv, say, write_csv

DAMPING = [0.05, 0.1, 0.25, 0.5, 1.0]     # share of the full correction applied
LATEST = [0.1, 0.25, 0.5, 0.75, 1.0]      # share of the attacker's rounds, latest first
FIELDS = ["seed", "active_rounds", "rounds_joined", "setting", "value",
          "attack_success", "test_accuracy", "attacker_loss", "relative_error"]


def run_one(setup, seed, length, out):
    table = out / f"poison_s{seed}_j{length}.csv"
    if table.exists() and not setup.fresh:
        say(f"[{table.name}] already there, skipping")
        return
    name = setup.runs / f"{setup.dataset}_poison_n{setup.num_clients}_r{setup.rounds}_j{length}_s{seed}"
    options = dict(poison=POISON, active_rounds=length)
    size = (setup.dataset, seed, setup.num_clients, setup.per_round, setup.rounds, setup.per_client)
    train_federation(make_config(*size, oracle_clients=[ATTACKER], **options), name / "base", setup.fresh)
    train_federation(make_config(*size, exclude=ATTACKER, **options), name / "retrain", setup.fresh)
    verify_replay(name / "base", ATTACKER)

    history, data = History(name / "base"), History(name / "base").data
    joined = history.rounds_with(ATTACKER)
    corrections = corrections_of(history, load_true_updates(name / "base", ATTACKER), joined)
    full = np.sum([corrections[t] for t in joined], axis=0)
    trained, retrained = history.theta_final, np.load(name / "retrain" / "theta_final.npy")
    own_images, own_labels = attacker_view(data, history.config)
    rows = []

    def add(setting, value, theta):
        rows.append({"seed": seed, "active_rounds": length, "rounds_joined": len(joined), "setting": setting,
                     "value": value, "attack_success": attack_success(data, theta),
                     "test_accuracy": evaluate(data, theta, data.test_images, data.test_labels)[0],
                     "attacker_loss": evaluate(data, theta, own_images, own_labels)[1],
                     "relative_error": relative_error(theta, trained, retrained)})

    add("trained", 0, trained)
    add("retrained", 0, retrained)
    for share in DAMPING:
        add("damped", share, trained + (share * full).astype(np.float32))
    for share in LATEST:
        take = joined[-max(1, round(share * len(joined))):]
        add("latest", share, trained + np.sum([corrections[t] for t in take], axis=0))
    write_csv(table, rows, FIELDS)
    say(f"seed {seed}, attacker in {len(joined)} rounds: attack success trained "
        f"{rows[0]['attack_success']:.2f}, retrained {rows[1]['attack_success']:.2f}")


def summarize(out):
    rows = read_csv(out, "poison_s*_j*.csv")
    say(f"\n{'in':>4} {'setting':>9} {'value':>6} {'attack %':>9} {'accuracy %':>11}")
    for length in sorted({int(r["active_rounds"]) for r in rows}):
        for setting in ("trained", "retrained", "damped", "latest"):
            for value in sorted({float(r["value"]) for r in rows if r["setting"] == setting}):
                part = [r for r in rows if int(r["active_rounds"]) == length and r["setting"] == setting
                        and float(r["value"]) == value]
                say(f"{length:>4} {setting:>9} {value:>6.2f} {100 * mean(part, 'attack_success'):>9.1f} "
                    f"{100 * mean(part, 'test_accuracy'):>11.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli.add_flags(parser)
    parser.add_argument("--lengths", type=int, nargs="+", default=[10, 50, 100])
    parser.set_defaults(clients=20, rounds=200)      # the mid-term setting: a small, quick federation
    args = parser.parse_args()
    setup = cli.setup_from(args)
    out = setup.results("ablation", "poison_damping")
    for seed in setup.seeds:
        for length in ([4] if setup.smoke else args.lengths):
            run_one(setup, seed, length, out)
    summarize(out)


if __name__ == "__main__":
    main()
