"""Experiment 2: which clients to pay.

Only some of the other clients are paid to redo their rounds. A paid client redoes every round
it shared with the attacker. A round of the attacker can be used exactly only if all the other
clients in it were paid: a "complete round". Two orders of paying: `shared` (the clients that
shared the most rounds with the attacker first) and `random`.

In a round where only some clients were paid the server knows

    left over = a_t - (sum of the clients paid) = attacker's update + (clients not paid)

and has to guess the attacker's update. Four rules:
  even      split what is left evenly between the attacker and the missing clients
  leftover  take all of what is left as the attacker's update
  mean      assume each missing client sent the average of the paid clients, and take that away
  complete  only correct the rounds where every client was paid; leave the rest alone

When no one is missing all four are exact. The correction is then applied at a share lambda of
BUDGETS-LAMBDAS and we measure attack success and clean accuracy.

(Each helper's own update is computed for analysis only; every rule uses only the sum of the
helpers paid in a round, which is what the server would see. See secunlearn/helpers.py.)

    python main/exp2_client_budget.py --tags f0 f4
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secunlearn import setup as cli
from secunlearn.attack import ATTACKER, attack_success
from secunlearn.helpers import collect_helper_updates, rank_helpers, score_helpers, shared_rounds, split_evenly
from secunlearn.model import test_accuracy
from secunlearn.settings import attacker_run
from secunlearn.unlearning import round_correction
from secunlearn.utils import mean, norm, read_csv, say, write_csv

RANKINGS = ["shared", "random"]
RULES = ["even", "leftover", "mean", "complete"]
SHARES = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]      # share of the other clients we may pay
LAMBDAS = [0.1, 0.25, 1.0]                     # share of the correction applied
FIELDS = ["seed", "force", "rounds_joined", "helpers", "ranking", "rule", "share", "clients_asked",
          "client_rounds", "complete_rounds", "lam", "update_error", "attack_success", "test_accuracy"]


def guesses(history, t, asked_sum, asked, missing):
    """Four guesses of the attacker's update in round t, from sums the server sees."""
    left_over = (history.total(t) - asked_sum).astype(np.float64)
    average = history.total(t).astype(np.float64) / len(history.participants(t))   # correction zero
    by_mean = left_over - missing * (asked_sum.astype(np.float64) / asked) if asked else average
    return {"even": split_evenly(history, t, asked_sum, missing), "leftover": left_over,
            "mean": by_mean, "complete": left_over if missing == 0 else average}


def sweep(history, data, shared, updates, truth, seed, shares=SHARES, lambdas=LAMBDAS):
    """Every ranking, budget, rule and damping: one row per combination."""
    joined = sorted(shared)
    actual = {t: max(norm(truth[t].astype(np.float64)), 1e-12) for t in joined}
    scored = {}

    def measure(key, total, lam):      # equal models are measured once
        if (key, lam) not in scored:
            theta = history.theta_final + (lam * total).astype(np.float32)
            scored[(key, lam)] = (attack_success(data, theta), test_accuracy(data, theta))
        return scored[(key, lam)]

    rows = []
    for ranking in RANKINGS:
        scores = score_helpers(shared, "random" if ranking == "random" else "overlap", seed=seed * 1000 + ATTACKER)
        order = rank_helpers(scores)
        asked_sum = {t: np.zeros(history.theta0.size, dtype=np.int64) for t in joined}
        asked = {t: 0 for t in joined}
        done = spent = 0
        for share in sorted(shares):
            budget = min(len(order), max(1, round(share * len(order))))
            for helper in order[done:budget]:          # budgets grow: add only the newly paid
                for t in joined:
                    if helper in shared[t]:
                        asked_sum[t] += updates[(t, helper)]
                        asked[t] += 1
                        spent += 1
            done = max(done, budget)
            missing = {t: len(shared[t]) - asked[t] for t in joined}
            totals = {rule: np.zeros_like(history.theta0) for rule in RULES}
            errors = {rule: [] for rule in RULES}
            for t in joined:
                guess = guesses(history, t, asked_sum[t], asked[t], missing[t])
                for rule in RULES:
                    if missing[t] == 0 and not np.array_equal(guess[rule], truth[t]):
                        raise RuntimeError(f"round {t}: rule {rule} is not exact although every client was paid")
                    totals[rule] += round_correction(history, t, guess[rule])
                    errors[rule].append(norm(guess[rule] - truth[t]) / actual[t])
            for rule in RULES:
                key = "everyone" if done == len(order) else (ranking, done, rule)
                for lam in lambdas:
                    attack, accuracy = measure(key, totals[rule], lam)
                    rows.append({"ranking": ranking, "rule": rule, "share": share, "clients_asked": done,
                                 "client_rounds": spent, "complete_rounds": sum(1 for t in joined if missing[t] == 0),
                                 "lam": lam, "update_error": float(np.mean(errors[rule])),
                                 "attack_success": attack, "test_accuracy": accuracy})
        say(f"  {ranking}: {len(order)} helpers, {spent} client-rounds when all are paid")
    return rows


def run_one(setup, tag, seed, out):
    force = {"f0": 0, "f4": 4}[tag]
    table = out / f"budget_s{seed}_f{force}.csv"
    if table.exists() and not setup.fresh:
        say(f"[{table.name}] already there, skipping")
        return
    run = attacker_run(setup, tag, seed, check=True)
    history, data = run.history(), run.history().data
    shared, truth = shared_rounds(history, ATTACKER), run.truth()
    updates = collect_helper_updates(history, ATTACKER, shared)
    label = {"seed": seed, "force": force, "rounds_joined": len(shared),
             "helpers": len({h for t in shared for h in shared[t]})}
    rows = []
    for name, theta in (("trained", history.theta_final), ("retrained", run.theta_retrained())):
        rows.append({**label, "rule": name, "share": 0, "lam": 0, "attack_success": attack_success(data, theta),
                     "test_accuracy": test_accuracy(data, theta)})
    small = dict(shares=[0.25, 1.0], lambdas=[0.25, 1.0]) if setup.smoke else {}   # smoke: fewer models to score
    rows += [{**label, **row} for row in sweep(history, data, shared, updates, truth, seed, **small)]
    write_csv(table, rows, FIELDS)
    say(f"seed {seed}, {tag}: attacker in {len(shared)} rounds; attack success trained "
        f"{rows[0]['attack_success']:.2f}, retrained {rows[1]['attack_success']:.2f}")


def summarize(out):
    rows = read_csv(out, "budget_s*_f*.csv")
    for force in sorted({r["force"] for r in rows}):
        mine = [r for r in rows if r["force"] == force]
        say(f"\nforce {force}: attacker in {mean(mine, 'rounds_joined'):.0f} rounds. "
            f"Rule 'complete', 25% subtracted (attack % / accuracy %)")
        say(f"{'ranking':>8} {'paid':>6} {'client-rounds':>14} {'complete':>9} {'result':>14}")
        for ranking in RANKINGS:
            for share in SHARES:
                part = [r for r in mine if r["ranking"] == ranking and r["rule"] == "complete"
                        and float(r["share"]) == share and float(r["lam"]) == 0.25]
                if part:
                    say(f"{ranking:>8} {share:>6.0%} {mean(part, 'client_rounds'):>14.0f} "
                        f"{mean(part, 'complete_rounds'):>9.0f} {100 * mean(part, 'attack_success'):>7.1f} / "
                        f"{100 * mean(part, 'test_accuracy'):.1f}")


def main():
    setup, args = cli.parse(__doc__, ["f0", "f4"])
    out = setup.results("main", "exp2_budget" if setup.dataset != "femnist" else "exp4_femnist")
    for tag in args.tags:
        for seed in setup.seeds:
            run_one(setup, tag, seed, out)
    summarize(out)


if __name__ == "__main__":
    main()
