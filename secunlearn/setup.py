"""The settings of one experiment run, and the command-line flags every script shares."""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from .utils import ROOT


@dataclass(frozen=True)
class Setup:
    dataset: str = "cifar10"      # "cifar10", "femnist", or "synthetic" (random data, for tests)
    num_clients: int = 100        # N
    per_round: int = 10           # m, clients picked each round
    rounds: int = 400
    per_client: int = 400         # training samples each client holds
    seeds: tuple = (0, 1, 2)      # three seeds, to tell a result from luck
    alpha: float = 0.5            # Dirichlet label skew; lower is more skewed
    smoke: bool = False           # a tiny run that only checks the code works
    fresh: bool = False           # redo finished runs and tables

    def results(self, kind, name):
        """Folder for the csv tables of one experiment. kind is 'main' or 'ablation'."""
        top = ROOT / "results" / ("smoke" if self.smoke else "")
        return top / kind / name

    @property
    def runs(self):
        """Folder for model histories. They are large; a seed rebuilds them."""
        return ROOT / "runs" / ("smoke" if self.smoke else "")


def add_flags(parser):
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "femnist", "synthetic"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--per-round", type=int, default=10)
    parser.add_argument("--per-client", type=int, default=400)
    parser.add_argument("--smoke", action="store_true", help="tiny run that checks every step works")
    parser.add_argument("--fresh", action="store_true", help="ignore finished runs and tables")


def setup_from(args):
    per_client = 300 if args.dataset == "femnist" else args.per_client   # FEMNIST writers hold 300
    setup = Setup(dataset=args.dataset, num_clients=args.clients, per_round=args.per_round,
                  rounds=args.rounds, per_client=per_client, seeds=tuple(args.seeds),
                  smoke=args.smoke, fresh=args.fresh)
    if args.smoke:   # small enough to be quick, but the attacker still joins about half the rounds
        setup = replace(setup, seeds=setup.seeds[:1], num_clients=20, per_round=10,
                        rounds=20, per_client=200)
    return setup


def parse(description, tags=None, default_tags=None):
    """Read the shared flags (and --tags, for experiments that run several attackers)."""
    parser = argparse.ArgumentParser(description=description)
    add_flags(parser)
    if tags:
        parser.add_argument("--tags", nargs="+", default=default_tags or list(tags), choices=list(tags),
                            help="which attacker settings to run")
    args = parser.parse_args()
    return setup_from(args), args
