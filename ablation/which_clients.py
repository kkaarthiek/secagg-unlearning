"""Which clients to ask (outline step 3), for a harmless removed client.

Everything in round_selection.py redoes a round with ALL m-1 other clients. This asks which of them
are worth asking ("helpers"). Per removed client k:
  1. list every round, 2. keep the rounds k took part in, 3. count how many of them each helper
  shared with k, 4. weight recent rounds more (the weight halves every 4 of k's rounds back in
  time), 5. sort the helpers by score and ask them in that order.
Three scorings: `recent` (step 4), `overlap` (step 3, a plain count), `random`.

Helpers are added ONE AT A TIME in score order; after each we measure how well k's updates are known
and how good the unlearned model is. If only some helpers of a round were asked, what is left of the
stored sum is k's update plus the skipped helpers' (split evenly, see helpers.split_evenly). Two sets
of rounds are corrected: only k's latest 10%, and all of them. `drop_s*.csv` (--drop) asks everyone
except one helper, for each helper in turn, to test whether a helper that shared many rounds matters
more.

    python ablation/which_clients.py [--rounds 1000] [--drop]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secunlearn import plots
from secunlearn import setup as cli
from secunlearn.helpers import collect_helper_updates, rank_helpers, score_helpers, shared_rounds, split_evenly
from secunlearn.settings import REMOVE_CLIENTS, benign_runs
from secunlearn.training import History, load_true_updates
from secunlearn.unlearning import measure_quality, round_correction
from secunlearn.utils import norm, read_csv, say, write_csv

RULES = ["recent", "overlap", "random"]
ROUND_SETS = {"latest": 0.1, "all": 1.0}      # share of the client's rounds corrected, latest first
QUALITY_POINTS = 16                            # points per curve that also get accuracy and the removal check
FIELDS = ["seed", "client", "round_set", "rule", "step", "clients_asked", "client_rounds", "requests_total",
          "added_client", "added_score", "update_error", "correction_error", "test_accuracy", "removed_client_loss"]
DROP_FIELDS = ["seed", "client", "round_set", "helper", "rounds_shared", "shared_in_latest", "recent_score",
               "update_error", "correction_error", "test_accuracy", "removed_client_loss"]


def add_one_at_a_time(history, client, shared, updates, truth, target, order, scores):
    """Ask the helpers in `order`, one more each step; step 0 is nobody asked, the last everybody."""
    asked_sum = {t: np.zeros(history.theta0.size, dtype=np.int64) for t in target}
    not_asked = {t: len(shared[t]) for t in target}
    exact = {t: round_correction(history, t, truth[t]) for t in target}
    exact_total = np.sum(list(exact.values()), axis=0)
    guess, error = {}, {}

    def refresh(t):
        estimate = split_evenly(history, t, asked_sum[t], not_asked[t])
        guess[t] = round_correction(history, t, estimate)
        actual = truth[t].astype(np.float64)
        error[t] = norm(estimate - actual) / max(norm(actual), 1e-12)

    for t in target:
        refresh(t)
    total = np.sum([guess[t] for t in target], axis=0)
    measured = {int(round(x)) for x in np.linspace(0, len(order), QUALITY_POINTS)}
    rows, spent = [], 0
    for step in range(len(order) + 1):
        if step > 0:
            added = order[step - 1]
            for t in target:
                if added in shared[t]:
                    asked_sum[t] += updates[(t, added)]
                    not_asked[t] -= 1
                    spent += 1
                    total = total - guess[t]
                    refresh(t)
                    total = total + guess[t]
        row = {"step": step, "clients_asked": step, "client_rounds": spent,
               "requests_total": sum(len(shared[t]) for t in target),
               "added_client": order[step - 1] if step else "", "added_score": scores[order[step - 1]] if step else "",
               "update_error": float(np.mean([error[t] for t in target])),
               "correction_error": float(norm(total - exact_total) / max(norm(exact_total), 1e-12))}
        if step in measured:
            row["test_accuracy"], row["removed_client_loss"] = measure_quality(
                history.data, history.theta_final + total.astype(np.float32), client, history.config)
        rows.append(row)
    return rows


def drop_one_helper(history, client, shared, updates, truth, target, latest):
    """Ask everyone except one helper, for each helper in turn."""
    recent = score_helpers(shared, "recent")
    exact = {t: round_correction(history, t, truth[t]) for t in target}
    exact_total = np.sum(list(exact.values()), axis=0)
    in_target = {h: [t for t in target if h in shared[t]] for h in {h for t in target for h in shared[t]}}
    order = sorted(in_target, key=lambda h: (-len(in_target[h]), h))
    measured = set(order[:8] + order[-8:])         # quality for the 8 most and 8 least frequent helpers
    rows = []
    for helper in order:
        total, errors = exact_total.copy(), 0.0
        for t in in_target[helper]:
            asked = sum(updates[(t, c)].astype(np.int64) for c in shared[t] if c != helper)
            estimate = split_evenly(history, t, asked, 1)
            total = total + round_correction(history, t, estimate) - exact[t]
            errors += norm(estimate - truth[t].astype(np.float64)) / max(norm(truth[t].astype(np.float64)), 1e-12)
        row = {"helper": helper, "rounds_shared": len(in_target[helper]),
               "shared_in_latest": sum(t in latest for t in in_target[helper]),
               "recent_score": f"{recent.get(helper, 0.0):.4f}", "update_error": errors / len(target),
               "correction_error": float(norm(total - exact_total) / max(norm(exact_total), 1e-12))}
        if helper in measured:
            row["test_accuracy"], row["removed_client_loss"] = measure_quality(
                history.data, history.theta_final + total.astype(np.float32), client, history.config)
        rows.append(row)
    return rows


def run_seed(setup, seed, out, drop):
    base, _ = benign_runs(setup, seed)
    history = History(base)
    rows, drop_rows = [], []
    for client in REMOVE_CLIENTS[: 1 if setup.smoke else None]:
        shared = shared_rounds(history, client)
        joined, truth = sorted(shared), load_true_updates(base, client)
        updates = collect_helper_updates(history, client, shared)
        for round_set, share in ROUND_SETS.items():
            target = joined[-max(1, round(share * len(joined))):]
            helpers_here = {h for t in target for h in shared[t]}
            for rule in RULES:
                scores = score_helpers(shared, rule, seed=seed * 1000 + client)
                order = rank_helpers(scores, helpers_here)
                for row in add_one_at_a_time(history, client, shared, updates, truth, target, order, scores):
                    rows.append({"seed": seed, "client": client, "round_set": round_set, "rule": rule, **row})
            if drop:
                latest = set(joined[-max(1, round(0.1 * len(joined))):])
                for row in drop_one_helper(history, client, shared, updates, truth, target, latest):
                    drop_rows.append({"seed": seed, "client": client, "round_set": round_set, **row})
        say(f"seed {seed} client {client}: {len(joined)} rounds, {len({h for t in joined for h in shared[t]})} helpers")
    write_csv(out / f"clients_s{seed}.csv", rows, FIELDS)
    if drop_rows:
        write_csv(out / f"drop_s{seed}.csv", drop_rows, DROP_FIELDS)


def draw(out):
    rows = read_csv(out, "clients_s*.csv")
    shares = np.linspace(0, 1, 21)
    fig, axes = plots.plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, round_set in zip(axes, ROUND_SETS):
        for index, rule in enumerate(RULES):
            curves = {}
            for r in rows:
                if r["round_set"] == round_set and r["rule"] == rule:
                    curves.setdefault((r["seed"], r["client"]), []).append(
                        (int(r["client_rounds"]) / max(int(r["requests_total"]), 1), float(r["update_error"])))
            if curves:
                stacked = np.array([np.interp(shares, *zip(*sorted(p))) for p in curves.values()])
                plots.line(ax, shares * 100, stacked.mean(axis=0), stacked.std(axis=0), rule, index)
        plots.style(ax, "Requests made (% of all in these rounds)", "Error in the removed client's update",
                    f"{round_set} rounds")
        ax.legend(frameon=False, fontsize=8)
    plots.save(fig, out / "figures", "which_clients")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli.add_flags(parser)
    parser.add_argument("--drop", action="store_true", help="also drop one helper at a time")
    args = parser.parse_args()
    setup = cli.setup_from(args)
    out = setup.results("ablation", f"which_clients_r{setup.rounds}")
    for seed in setup.seeds:
        if (out / f"clients_s{seed}.csv").exists() and not setup.fresh:
            say(f"[clients_s{seed}.csv] already there, skipping")
            continue
        run_seed(setup, seed, out, args.drop)
    draw(out)


if __name__ == "__main__":
    main()
