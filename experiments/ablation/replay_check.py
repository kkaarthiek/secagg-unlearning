"""The gate: does redoing a round reproduce it?

The whole method stands on this. Removing a client works by asking the OTHER clients of a round to
train again and subtracting their sum from the stored one. If a redo does not reproduce the original,
that subtraction returns noise. On a few rounds we check that (a) when everyone redoes a round the new
total equals the stored one exactly, and (b) when all but one client redo it, stored minus redone equals
that client's real update exactly. Every other script runs this check before it uses a run.

    python experiments/ablation/replay_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from secunlearn import setup as cli
from secunlearn.attack import ATTACKER
from secunlearn.settings import attacker_config
from secunlearn.training import train_federation, verify_replay


def main():
    setup, _ = cli.parse(__doc__)
    for seed in setup.seeds:
        run = setup.runs / f"{setup.dataset}_f0_s{seed}" / "base"
        train_federation(attacker_config(setup, "f0", seed, oracle_clients=[ATTACKER]), run, setup.fresh)
        verify_replay(run, ATTACKER, how_many=5)


if __name__ == "__main__":
    main()
