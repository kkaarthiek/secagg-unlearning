"""Dynamic Participation with fixed group sizes of 10, 20 and 50 (our first version).

Before experiment 3 used the clusters of the paper's own method (exp3_clusters.py), the clients were
split into groups of a size we picked. Each group trains its own model with FedAvg, the groups vote on
every prediction, and removing the attacker retrains its group without it (see secunlearn/baselines.py
for what is ours). Each round a group uses the same share of its clients as the single model: 1, 2 or
5 clients for groups of 10, 20 or 50. Results go to results/ablation/dp_fixed_groups/.

    python experiments/ablation/dp_fixed_groups.py --tags j10 f0 f4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from secunlearn import setup as cli
from secunlearn.attack import ATTACKER
from secunlearn.baselines import make_groups, train_groups_and_remove
from secunlearn.data import get_data
from secunlearn.settings import TAGS, attacker_config
from secunlearn.utils import mean, read_csv, say, write_csv

GROUP_SIZES = [10, 20, 50]
FIELDS = ["seed", "force", "active_rounds", "method", "setting", "rounds_joined",
          "client_rounds", "attack_success", "test_accuracy", "relative_error"]


def run_one(setup, tag, seed, out):
    table = out / f"groups_s{seed}_{tag}.csv"
    if table.exists() and not setup.fresh:
        say(f"[{table.name}] already there, skipping")
        return
    cfg, data = attacker_config(setup, tag, seed), get_data(setup.dataset)
    label = {"seed": seed, "force": cfg["force_every"] or 0, "active_rounds": cfg["active_rounds"] or "",
             "method": "Dynamic Participation", "relative_error": ""}
    rows = []
    for size in ([5, 10] if setup.smoke else GROUP_SIZES):
        groups = make_groups(setup.num_clients, seed, size)
        before, after, alone, spent, joined = train_groups_and_remove(
            cfg, data, groups, setup.per_round, setup.num_clients, ATTACKER,
            cfg["force_every"] or None, cfg["active_rounds"])
        for name, cost, (attack, accuracy) in ((f"group {size}: before", 0, before), (f"group {size}: after", spent, after),
                                               (f"group {size}: attacker's group model alone", 0, alone)):
            rows.append({**label, "setting": name, "rounds_joined": joined, "client_rounds": cost,
                         "attack_success": attack, "test_accuracy": accuracy})
        say(f"seed {seed} {tag}, groups of {size}: {len(groups)} groups, attacker in {joined} rounds, removal cost {spent}")
    write_csv(table, rows, FIELDS)


def main():
    setup, args = cli.parse(__doc__, TAGS, ["j10", "f0", "f4"])
    out = setup.results("ablation", "dp_fixed_groups")
    for tag in args.tags:
        for seed in setup.seeds:
            run_one(setup, tag, seed, out)
    for r in sorted({r["setting"] for r in read_csv(out, "groups_s*_*.csv") if r["setting"].endswith("after")}):
        part = [x for x in read_csv(out, "groups_s*_*.csv") if x["setting"] == r]
        say(f"{r:>20}: cost {mean(part, 'client_rounds'):.0f}, attack {100 * mean(part, 'attack_success'):.1f}%")


if __name__ == "__main__":
    main()
