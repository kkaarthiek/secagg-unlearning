"""Can the share lambda of the attacker's contribution be set without the retrained model?

Reads results/main/exp1_share/lambda_<dataset>_<tag>_s<seed>.csv (written by exp1_share_to_subtract.py). For each setting
(dataset and attacker kind) it has the whole correction's relative size

    rho = ||sum of corrections|| / ||theta_T - theta_0||            (server can compute it)

and the attack success and accuracy for a grid of shares. The rule tested is

    lambda = min(1, kappa / rho)

i.e. match the size of the correction to a fixed fraction kappa of the model's movement.
kappa is fitted on some settings and tested on the one left out. It is compared with the
best single fixed lambda, fitted and tested the same way, and with the best lambda in
hindsight (which needs the retrained model).

Score of a model against the retrained one (lower is better):
    max(0, attack - attack_retrained - 0.02) + max(0, accuracy_retrained - accuracy - 0.01)

    python experiments/main/analyze_share_rule.py
"""

import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results" / "main" / "exp1_share"
KAPPAS = [round(0.02 * i, 2) for i in range(1, 100)]      # 0.02 .. 1.98
FIXED = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


def load():
    cells = defaultdict(list)
    for path in sorted(glob.glob(str(RESULTS / "lambda_*_s*.csv"))):
        for row in csv.DictReader(open(path)):
            cells[(row["dataset"], row["tag"])].append(row)
    settings = {}
    for key, rows in cells.items():
        seeds = sorted({r["seed"] for r in rows})
        grid = defaultdict(list)
        extra = defaultdict(list)
        for r in rows:
            if r["kind"] == "grid":
                grid[float(r["lam"])].append((float(r["attack_success"]), float(r["test_accuracy"])))
            else:
                extra[r["kind"]].append((float(r["attack_success"]), float(r["test_accuracy"])))
        mean = lambda pairs: (st.mean(p[0] for p in pairs), st.mean(p[1] for p in pairs))
        per_seed = {r["seed"]: r for r in rows if r["kind"] == "trained"}
        settings[key] = {
            "seeds": len(seeds),
            "joined": st.mean(float(r["rounds_joined"]) for r in per_seed.values()),
            "rho": st.mean(float(r["rho"]) for r in per_seed.values()),
            "oracle": st.mean(float(r["lam_oracle"]) for r in per_seed.values()),
            "grid": {lam: mean(v) for lam, v in grid.items()},
            "trained": mean(extra["trained"]), "retrained": mean(extra["retrained"]),
            "fedrecovery_all": mean(extra["fedrecovery_all"]),
            "fedrecovery_joined": mean(extra["fedrecovery_joined"]),
        }
    return settings


def score(setting, attack, accuracy):
    a_r, acc_r = setting["retrained"]
    return max(0.0, attack - a_r - 0.02) + max(0.0, acc_r - accuracy - 0.01)


def nearest(setting, lam):
    """The grid point closest to lam on a log scale."""
    import math
    best = min(setting["grid"], key=lambda g: abs(math.log(g) - math.log(max(lam, 1e-9))))
    return best, setting["grid"][best]


def rule_lambda(setting, kappa):
    return min(1.0, kappa / setting["rho"])


def rule_score(setting, kappa):
    lam, (attack, accuracy) = nearest(setting, rule_lambda(setting, kappa))
    return score(setting, attack, accuracy)


def fixed_score(setting, lam):
    attack, accuracy = setting["grid"][lam]
    return score(setting, attack, accuracy)


def hindsight(setting):
    """Best grid lambda for this setting, by the score above (needs the retrained model)."""
    lam = min(setting["grid"], key=lambda g: fixed_score(setting, g))
    return lam, fixed_score(setting, lam)


def main():
    settings = load()
    names = sorted(settings)
    print(f"{len(names)} settings: " + ", ".join(f"{d}/{t}" for d, t in names))
    print(f"\n{'setting':16} {'seeds':>5} {'in':>5} {'rho':>6} {'oracle':>7} {'trained':>14} {'retrained':>14} "
          f"{'FedRec all':>14} {'FedRec joined':>14}")
    for key in names:
        s = settings[key]
        f = lambda p: f"{p[0]:.3f}/{p[1]:.3f}"
        print(f"{key[0] + '/' + key[1]:16} {s['seeds']:>5} {s['joined']:>5.0f} {s['rho']:>6.3f} {s['oracle']:>7.3f} "
              f"{f(s['trained']):>14} {f(s['retrained']):>14} {f(s['fedrecovery_all']):>14} {f(s['fedrecovery_joined']):>14}")

    print("\nHindsight best share per setting (needs the retrained model):")
    for key in names:
        lam, sc = hindsight(settings[key])
        print(f"  {key[0]}/{key[1]:6} lambda {lam:<5} score {sc:.3f}   (rho {settings[key]['rho']:.3f}, "
              f"kappa = lambda*rho = {lam * settings[key]['rho']:.3f})")

    print("\nLeave-one-setting-out (fit on the others, test on the one left out):")
    print(f"{'left out':16} {'kappa':>6} {'rule lam':>9} {'rule score':>11} {'fixed lam':>10} {'fixed score':>12} {'hindsight':>10}")
    totals = {"rule": [], "fixed": [], "hind": [], "none": []}
    for left in names:
        rest = [k for k in names if k != left]
        kappa = min(KAPPAS, key=lambda k: st.mean(rule_score(settings[o], k) for o in rest))
        fixed = min(FIXED, key=lambda g: st.mean(fixed_score(settings[o], g) for o in rest))
        s = settings[left]
        r, f_, h = rule_score(s, kappa), fixed_score(s, fixed), hindsight(s)[1]
        n = score(s, *s["trained"])
        totals["rule"].append(r); totals["fixed"].append(f_); totals["hind"].append(h); totals["none"].append(n)
        print(f"{left[0] + '/' + left[1]:16} {kappa:>6.2f} {rule_lambda(s, kappa):>9.3f} {r:>11.3f} {fixed:>10.2f} {f_:>12.3f} {h:>10.3f}")
    print(f"{'mean':16} {'':>6} {'':>9} {st.mean(totals['rule']):>11.3f} {'':>10} {st.mean(totals['fixed']):>12.3f} "
          f"{st.mean(totals['hind']):>10.3f}   (no unlearning: {st.mean(totals['none']):.3f})")

    kappa = min(KAPPAS, key=lambda k: st.mean(rule_score(settings[o], k) for o in names))
    print(f"\nkappa fitted on all settings: {kappa:.2f}")
    for key in names:
        s = settings[key]
        lam, (attack, accuracy) = nearest(s, rule_lambda(s, kappa))
        print(f"  {key[0]}/{key[1]:6} rule lambda {rule_lambda(s, kappa):.3f} (grid {lam}) -> attack {attack:.3f}, "
              f"accuracy {accuracy:.3f}   retrained {s['retrained'][0]:.3f}/{s['retrained'][1]:.3f}")


if __name__ == "__main__":
    main()
