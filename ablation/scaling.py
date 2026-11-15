"""Scaling: does the same fraction of rounds work whatever the size of the federation?

A client is picked for roughly m/N of the rounds. The guess is that what matters is the FRACTION of
its rounds we redo, not how many. Three groups of runs test that, each changing one thing:
more clients at the same round size; bigger rounds with the same clients; both changed together with
the share m/N held fixed. If the guess is right the lines of the third panel land on top of each other.

Clients hold 200 images here (200 clients x 400 would not fit in CIFAR-10). One removed client (0).

    python ablation/scaling.py
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secunlearn import plots
from secunlearn import setup as cli
from secunlearn.settings import benign_config
from secunlearn.training import train_federation, verify_replay
from secunlearn.utils import read_csv, say
import round_selection

PANELS = [("More clients, same round size", [(50, 10), (100, 10), (200, 10)]),
          ("Same clients, bigger rounds", [(100, 5), (100, 10), (100, 20)]),
          ("Same share of clients per round", [(50, 5), (100, 10), (200, 20)])]
REMOVED = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli.add_flags(parser)
    args = parser.parse_args()
    setup = replace(cli.setup_from(args), per_client=100 if args.smoke else 200)
    panels = [("smoke", [(10, 4), (20, 4)])] if setup.smoke else PANELS
    out = setup.results("ablation", f"scaling_r{setup.rounds}")
    for clients, per_round in sorted({pair for _, pairs in panels for pair in pairs}):
        for seed in setup.seeds:
            here = replace(setup, num_clients=clients, per_round=per_round)
            tag = f"n{clients}_m{per_round}_s{seed}"
            base, retrain = setup.runs / f"scaling_{tag}" / "base", setup.runs / f"scaling_{tag}" / "retrain"
            train_federation(benign_config(here, seed, oracle_clients=[REMOVED]), base, setup.fresh)
            train_federation(benign_config(here, seed, exclude=REMOVED), retrain, setup.fresh)
            verify_replay(base, REMOVED)
            table = out / f"scaling_{tag}.csv"
            if not table.exists() or setup.fresh:
                round_selection.sweep(base, retrain, REMOVED, table, quality=False)

    rows = read_csv(out, "scaling_*.csv")
    fig, axes = plots.plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, (title, pairs) in zip(axes, PANELS):
        for index, (clients, per_round) in enumerate(pairs):
            part = [r for r in rows if int(r["num_clients"]) == clients and int(r["per_round"]) == per_round
                    and r["selector"] == "combined" and r["rule"] == "hybrid"]
            xs = sorted({float(r["fraction"]) for r in part})
            vals = [[float(r["relative_error"]) for r in part if float(r["fraction"]) == f] for f in xs]
            if xs:
                plots.line(ax, xs, [np.mean(v) for v in vals], [np.std(v) for v in vals], f"N={clients}, m={per_round}", index)
        plots.style(ax, "Fraction of rounds redone", "", title)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Distance to retrained model")
    plots.save(fig, out / "figures", "scaling")
    say(f"wrote {out / 'figures' / 'scaling.pdf'}")


if __name__ == "__main__":
    main()
