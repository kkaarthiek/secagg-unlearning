"""Experiment 4: the same experiments on FEMNIST (one client per writer, 62 classes).

Runs experiments 2, 3 and 1 with --dataset femnist for the two attackers of the report (picked like
any other client: about 41 rounds; also put into every fourth round: about 132). FEMNIST clients
hold 300 images, so --per-client is set to 300 automatically.

    python experiments/main/exp4_femnist.py
    python experiments/main/exp4_femnist.py --smoke --dataset synthetic   # quick check, no download
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from secunlearn import setup as cli
import exp1_share_to_subtract as exp1
import exp2_client_budget as exp2
import exp3_baselines as exp3
import exp3_clusters as clusters


def main():
    setup, args = cli.parse(__doc__, ["f0", "f4"])
    if setup.dataset == "cifar10":
        setup = replace(setup, dataset="femnist", per_client=300)
    for tag in args.tags:
        for seed in setup.seeds:
            exp2.run_one(setup, tag, seed, setup.results("main", "exp4_femnist"))
            exp3.run_one(setup, tag, seed, setup.results("main", "exp4_femnist"))
            exp1.run_one(setup, tag, seed, setup.results("main", "exp1_share"))
            clusters.run_one(setup, tag, seed, setup.results("main", "exp3_clusters"))
    exp3.summarize(setup.results("main", "exp4_femnist"))
    clusters.summarize(setup.results("main", "exp3_clusters"), setup.dataset)


if __name__ == "__main__":
    main()
