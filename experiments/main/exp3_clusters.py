"""Experiment 3 (part 2): Dynamic Participation, with the clustering of its own paper.

The clusters come from the paper's method (Liu et al. 2024, Algorithms 1 and 2), so nothing about
their size is ours. `pargen` gives the cluster size k and the number of clusters s = N // k from the
system size N, the largest shares of adversarial (gamma) and dropout (delta) users, the largest share
of removed users in a cluster (zeta), the Shamir threshold rate (xi) and the security level
(sigma = eta), taking the smallest k that satisfies the inequalities (3), (4) and (5). A random
permutation of the clients fills s clusters of k; each remaining client joins a random cluster.
Each cluster trains its own model, the clusters vote, and removing the attacker retrains its cluster
from scratch without it.

gamma, delta, zeta, xi are the values of the paper's worked example. The paper gives no values for
100 clients, so the security level is swept: 40 is the paper's example (2 clusters of about 50), and
lower levels give more, smaller clusters (20 -> 4 of about 25, 10 -> 7 of about 14). The paper's
example (N = 200) states k = 60; the printed inequalities give k = 41, so our clusters may be
smaller than the paper's own code would give. Each round a cluster uses the same share of its
clients as the single model does (our choice).

    python experiments/main/exp3_clusters.py --tags j10 f0 f4
    python experiments/main/exp3_clusters.py --tags f0 f4 --dataset femnist     # Experiment 4
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from secunlearn import setup as cli
from secunlearn.attack import ATTACKER
from secunlearn.baselines import make_clusters, pargen, train_groups_and_remove
from secunlearn.data import get_data
from secunlearn.settings import TAGS, attacker_config
from secunlearn.utils import mean, minutes, read_csv, say, write_csv

SECURITY = [40, 20, 10]                                  # sigma = eta, in bits
PARAMS = dict(gamma=0.1, delta=0.1, zeta=0.1, xi=0.7)    # the paper's example values
FIELDS = ["dataset", "tag", "seed", "security", "k", "clusters", "cluster_sizes", "setting",
          "rounds_joined", "client_rounds", "attack_success", "test_accuracy"]


def run_one(setup, tag, seed, out):
    table = out / f"clusters_{setup.dataset}_{tag}_s{seed}.csv"
    if table.exists() and not setup.fresh:
        say(f"[{table.name}] already there, skipping")
        return
    cfg = attacker_config(setup, tag, seed)
    data = get_data(setup.dataset)
    rows = []
    for security in ([10, 5] if setup.smoke else SECURITY):
        k, s = pargen(setup.num_clients, sigma=security, eta=security, **PARAMS)
        clusters = make_clusters(setup.num_clients, seed, k, s)
        started = time.time()
        before, after, alone, spent, joined = train_groups_and_remove(
            cfg, data, clusters, setup.per_round, setup.num_clients, ATTACKER,
            cfg["force_every"] or None, cfg["active_rounds"])
        label = {"dataset": setup.dataset, "tag": tag, "seed": seed, "security": security, "k": k,
                 "clusters": s, "cluster_sizes": "/".join(str(len(c)) for c in clusters), "rounds_joined": joined}
        for name, cost, (attack, accuracy) in (("before", 0, before), ("after", spent, after),
                                               ("attacker's cluster alone", 0, alone)):
            rows.append({**label, "setting": name, "client_rounds": cost,
                         "attack_success": attack, "test_accuracy": accuracy})
        say(f"seed {seed} {setup.dataset} {tag}, security {security}: k {k}, {s} clusters, attacker in {joined} "
            f"rounds, removal cost {spent}, attack {after[0]:.3f}, accuracy {after[1]:.3f} ({minutes(time.time() - started)})")
    write_csv(table, rows, FIELDS)


def summarize(out, dataset):
    rows = [r for r in read_csv(out, f"clusters_{dataset}_*_s*.csv") if r["setting"] in ("before", "after")]
    say(f"\n{'attacker':>8} {'security':>9} {'clusters':>9} {'cost':>6} {'before attack %':>16} {'after attack %':>15} {'after acc %':>12}")
    for tag in ("j10", "f0", "f4"):
        for security in sorted({int(r["security"]) for r in rows}, reverse=True):
            after = [r for r in rows if r["tag"] == tag and int(r["security"]) == security and r["setting"] == "after"]
            before = [r for r in rows if r["tag"] == tag and int(r["security"]) == security and r["setting"] == "before"]
            if after:
                say(f"{tag:>8} {security:>9} {mean(after, 'clusters'):>9.0f} {mean(after, 'client_rounds'):>6.0f} "
                    f"{100 * mean(before, 'attack_success'):>16.1f} {100 * mean(after, 'attack_success'):>15.1f} "
                    f"{100 * mean(after, 'test_accuracy'):>12.1f}")


def main():
    setup, args = cli.parse(__doc__, TAGS, ["j10", "f0", "f4"])
    out = setup.results("main", "exp3_clusters")
    for tag in args.tags:
        for seed in setup.seeds:
            run_one(setup, tag, seed, out)
    summarize(out, setup.dataset)


if __name__ == "__main__":
    main()
