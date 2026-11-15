"""Steps 1 and 2 of the outline: redo only some of the removed client's rounds, and guess the rest.

For one removed (harmless) client: every way of choosing rounds (5 selectors), at every budget (6
fractions of the client's rounds), with every guessing rule (4 rules), each scored against the
retrained model: 5 x 6 x 4 = 120 rows per client. Selectors: combined / magnitude / deviation scores,
and random and spread as baselines. Rules: nearest, interpolate, rescale, hybrid (secunlearn/rounds.py).

Columns: client_rounds is what was spent; relative_error is the distance to the retrained model over
the distance if we did nothing (1.0 = achieved nothing, 0 = matched retraining); correction_cosine
whether we moved the right way; estimation_error how wrong the guesses were; test_accuracy and
removed_client_loss (the removal check) with quality on.

Mid-term result: no rule beat a guess of zero, and "distance to one retrained model" cannot judge
unlearning (see control.py). Use --rounds 1000 for the long setting.

    python ablation/round_selection.py [--rounds 1000] [--no-quality]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secunlearn import plots
from secunlearn import setup as cli
from secunlearn.rounds import RULES, SELECTORS, budget_from_fraction, estimate_all, pick_rounds
from secunlearn.settings import REMOVE_CLIENTS, benign_runs
from secunlearn.training import History, load_true_updates
from secunlearn.unlearning import (correction_cosine, estimation_error, measure_quality, redo_cost,
                                   relative_error, rewind_cost, unlearn)
from secunlearn.utils import read_csv, say, write_csv

FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]      # share of the client's rounds redone
FIELDS = ["num_clients", "per_round", "rounds", "seed", "client", "selector", "rule", "fraction", "budget",
          "rounds_joined", "client_rounds", "rewind_client_rounds", "relative_error", "correction_cosine",
          "estimation_error", "test_accuracy", "removed_client_loss"]


def sweep(base, retrain, client, table, quality):
    history = History(base)
    cfg, data = history.config, history.data
    trained, retrained = history.theta_final, np.load(Path(retrain) / "theta_final.npy")
    joined = history.rounds_with(client)
    truth = load_true_updates(base, client)
    rows = []
    for selector in SELECTORS:
        for fraction in FRACTIONS:
            chosen = pick_rounds(selector, history, joined, budget_from_fraction(joined, fraction), cfg["seed"])
            skipped = [t for t in joined if t not in chosen]
            anchors = {t: truth[t].astype(np.float64) for t in chosen}   # a redo would give exactly these
            for rule in RULES:
                estimates = estimate_all(joined, anchors, history, rule)
                theta = unlearn(history, estimates)
                accuracy, own_loss = measure_quality(data, theta, client, cfg) if quality else ("", "")
                rows.append({
                    "num_clients": cfg["num_clients"], "per_round": cfg["per_round"], "rounds": cfg["rounds"],
                    "seed": cfg["seed"], "client": client, "selector": selector, "rule": rule,
                    "fraction": fraction, "budget": len(chosen), "rounds_joined": len(joined),
                    "client_rounds": redo_cost(history, chosen), "rewind_client_rounds": rewind_cost(history, joined),
                    "relative_error": relative_error(theta, trained, retrained),
                    "correction_cosine": correction_cosine(theta, trained, retrained),
                    "estimation_error": estimation_error(estimates, truth, skipped),
                    "test_accuracy": accuracy, "removed_client_loss": own_loss})
    write_csv(table, rows, FIELDS)
    say(f"client {client} joined {len(joined)} of {history.num_rounds} rounds: wrote {len(rows)} rows to {Path(table).name}")


def draw(out, figures):
    rows = read_csv(out, "sweep_s*_k*.csv")
    if not rows:
        return
    fig, axes = plots.plt.subplots(1, 2, figsize=(9, 3.4))
    for index, rule in enumerate(RULES):         # what does client work buy? one line per rule
        part = [r for r in rows if r["selector"] == "combined" and r["rule"] == rule]
        xs = sorted({float(r["fraction"]) for r in part})
        per = lambda key, f: [float(r[key]) for r in part if float(r["fraction"]) == f]
        plots.line(axes[0], [np.mean(per("client_rounds", f)) for f in xs], [np.mean(per("relative_error", f)) for f in xs],
                   [np.std(per("relative_error", f)) for f in xs], rule, index)
    axes[0].axhline(1.0, color=plots.MUTED, linestyle="--", linewidth=1)
    plots.style(axes[0], "Client-rounds spent", "Distance to retrained model (1 = did nothing)", "What redoing rounds buys")
    for index, selector in enumerate(SELECTORS):  # does choosing rounds well matter?
        part = [r for r in rows if r["selector"] == selector and r["rule"] == "hybrid"]
        xs = sorted({float(r["fraction"]) for r in part})
        vals = [[float(r["relative_error"]) for r in part if float(r["fraction"]) == f] for f in xs]
        plots.line(axes[1], xs, [np.mean(v) for v in vals], [np.std(v) for v in vals], selector, index)
    plots.style(axes[1], "Fraction of the client's rounds redone", "Distance to retrained model", "Does the choice of rounds matter?")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    plots.save(fig, figures, "round_selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli.add_flags(parser)
    parser.add_argument("--no-quality", action="store_true", help="skip accuracy and the removal check (faster)")
    args = parser.parse_args()
    setup = cli.setup_from(args)
    out = setup.results("ablation", f"round_selection_r{setup.rounds}")
    for seed in setup.seeds:
        base, retrains = benign_runs(setup, seed)
        for client in REMOVE_CLIENTS[: 1 if setup.smoke else None]:
            table = out / f"sweep_s{seed}_k{client}.csv"
            if table.exists() and not setup.fresh:
                say(f"[{table.name}] already there, skipping")
                continue
            sweep(base, retrains[client], client, table, not args.no_quality)
    draw(out, setup.results("ablation", f"round_selection_r{setup.rounds}") / "figures")


if __name__ == "__main__":
    main()
