"""Experiment 3 (part 1): FedRecovery against ours, and the reference models.

For each attacker we report the model with nothing removed ("trained"), the retrained model, and
FedRecovery at four noise levels, as attack success, accuracy and client work. FedRecovery asks no
client to redo anything (0 client-rounds) but needs every client's own update, so it could not run
under secure aggregation; see secunlearn/baselines.py for how it is reimplemented.

The other published method, Dynamic Participation, is in exp3_clusters.py (with the clustering of
its own paper). Its fixed group sizes of 10, 20 and 50 are in ablation/dp_fixed_groups.py.

    python experiments/main/exp3_baselines.py --tags j10 f0 f4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from secunlearn import setup as cli
from secunlearn.attack import ATTACKER, attack_success
from secunlearn.baselines import fedrecovery
from secunlearn.model import test_accuracy
from secunlearn.settings import TAGS, attacker_run
from secunlearn.unlearning import relative_error, retrain_cost
from secunlearn.utils import mean, read_csv, say, write_csv

NOISE = [0.0, 0.01, 0.03, 0.1]      # sigma as a share of the RMS weight; our choice (see baselines.py)
FIELDS = ["seed", "force", "active_rounds", "method", "setting", "rounds_joined",
          "client_rounds", "attack_success", "test_accuracy", "relative_error"]


def run_one(setup, tag, seed, out):
    table = out / f"baselines_s{seed}_{tag}.csv"
    if table.exists() and not setup.fresh:
        say(f"[{table.name}] already there, skipping")
        return
    run = attacker_run(setup, tag, seed)
    history, data = run.history(), run.history().data
    joined = history.rounds_with(ATTACKER)
    truth, trained, retrained = run.truth(), history.theta_final, run.theta_retrained()
    label = {"seed": seed, "force": run.cfg["force_every"] or 0, "active_rounds": run.cfg["active_rounds"] or ""}
    rows = []

    def add(method, setting, cost, theta):
        rows.append({**label, "method": method, "setting": setting, "rounds_joined": len(joined),
                     "client_rounds": cost, "attack_success": attack_success(data, theta),
                     "test_accuracy": test_accuracy(data, theta),
                     "relative_error": relative_error(theta, trained, retrained)})

    add("single model", "trained", 0, trained)
    add("single model", "retrained", retrain_cost(history, joined), retrained)
    for noise in NOISE:
        add("FedRecovery", f"noise {noise:g}", 0, fedrecovery(history, truth, joined, noise, seed))
    write_csv(table, rows, FIELDS)
    say(f"seed {seed} {tag}: attack success trained {rows[0]['attack_success']:.2f}, "
        f"FedRecovery {rows[2]['attack_success']:.2f}, retrained {rows[1]['attack_success']:.2f}")


def summarize(out):
    rows = read_csv(out, "baselines_s*_*.csv")
    say(f"\n{'attacker':>8} {'method':>14} {'setting':>10} {'client-rounds':>14} {'attack %':>9} {'accuracy %':>11}")
    for tag in ("j10", "f0", "f4"):
        part = [r for r in rows if (f"j{r['active_rounds']}" if r["active_rounds"] else f"f{r['force']}") == tag]
        keys = sorted({(r["method"], r["setting"]) for r in part if r["method"] in ("single model", "FedRecovery")})
        for method, setting in keys:
            sel = [r for r in part if r["method"] == method and r["setting"] == setting]
            say(f"{tag:>8} {method:>14} {setting:>10} {mean(sel, 'client_rounds'):>14.0f} "
                f"{100 * mean(sel, 'attack_success'):>9.1f} {100 * mean(sel, 'test_accuracy'):>11.1f}")


def main():
    setup, args = cli.parse(__doc__, TAGS, ["j10", "f0", "f4"])
    out = setup.results("main", "exp3_baselines" if setup.dataset != "femnist" else "exp4_femnist")
    for tag in args.tags:
        for seed in setup.seeds:
            run_one(setup, tag, seed, out)
    summarize(out)


if __name__ == "__main__":
    main()
