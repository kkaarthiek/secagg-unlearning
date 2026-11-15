"""Cost against backdoor left and accuracy, for the attacker that joined about 41 rounds (CIFAR-10).

Every method that works under secure aggregation (and FedRecovery, which does not) as one point,
drawn from the result tables of experiments 1 and 3 and of main/exp6_controls.py.

    python main/make_pareto.py        # writes figures/fig_pareto.pdf
"""
import csv
import glob
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def rows(pattern):
    out = []
    for path in sorted(glob.glob(str(R / pattern))):
        out += list(csv.DictReader(open(path)))
    return out


def mean(part, key):
    return 100 * st.mean(float(r[key]) for r in part)


points = []   # (label, cost, attack %, accuracy %, group)
base = [r for r in rows("main/exp3_baselines/baselines_s*_f0.csv")]
for method, setting, label, group in [("single model", "trained", "No unlearning", "none"),
                                      ("single model", "retrained", "Retrain", "ref"),
                                      ("FedRecovery", "noise 0", "FedRecovery*", "other")]:
    part = [r for r in base if r["method"] == method and r["setting"] == setting]
    points.append((label, st.mean(float(r["client_rounds"]) for r in part),
                   mean(part, "attack_success"), mean(part, "test_accuracy"), group))
lam = rows("main/exp1_share/lambda_cifar10_f0_s*.csv")
for share, label in [(0.15, "Ours 15%"), (0.2, "Ours 20%")]:
    part = [r for r in lam if r["kind"] == "grid" and float(r["lam"]) == share]
    points.append((label, 372, mean(part, "attack_success"), mean(part, "test_accuracy"), "ours"))
ctl = rows("main/exp6_controls/controls_cifar10_f0_s*.csv")
for share, label in [(0.15, "Rollback 15% (no client work)")]:
    part = [r for r in ctl if r["control"] == "rollback" and float(r["lam"]) == share]
    if part:
        points.append((label, 0, mean(part, "attack_success"), mean(part, "test_accuracy"), "rollback"))
for security, label in [(40, "Dyn. Part. 2"), (20, "Dyn. Part. 4"), (10, "Dyn. Part. 7")]:
    part = [r for r in rows("main/exp3_clusters/clusters_cifar10_f0_s*.csv")
            if int(r["security"]) == security and r["setting"] == "after"]
    points.append((label, st.mean(float(r["client_rounds"]) for r in part), mean(part, "attack_success"),
                   mean(part, "test_accuracy"), "dyn"))

COLOUR = {"none": "#7f7f7f", "ref": "#222222", "other": "#ff7f0e", "ours": "#1f77b4", "rollback": "#9467bd",
          "dyn": "#2ca02c"}
OFFSET = {("FedRecovery*", 2): (3, -9), ("FedRecovery*", 3): (3, -9), ("Dyn. Part. 4", 2): (-4, 6),
          ("Dyn. Part. 7", 2): (-10, -11), ("Dyn. Part. 2", 2): (-8, -11), ("Ours 20%", 2): (-34, -9),
          ("Ours 15%", 2): (3, 4), ("No unlearning", 3): (3, 5), ("Retrain", 2): (-10, 6)}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
for ax, column, name in [(axes[0], 2, "Backdoor left (%) - lower is better"),
                         (axes[1], 3, "Test accuracy (%) - higher is better")]:
    for point in points:
        ax.scatter(point[1] + 30, point[column], color=COLOUR[point[4]], s=28, zorder=3)
        ax.annotate(point[0], (point[1] + 30, point[column]), fontsize=6,
                    xytext=OFFSET.get((point[0], column), (3, 3)), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=100)
    if column == 2:
        ax.set_ylim(-9, 78)
    ax.set_xlabel("Client work (client-rounds)")
    ax.set_ylabel(name)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "fig_pareto.pdf", bbox_inches="tight")
fig.savefig(ROOT / "figures" / "fig_pareto.png", dpi=170, bbox_inches="tight")
print("wrote figures/fig_pareto.pdf")
