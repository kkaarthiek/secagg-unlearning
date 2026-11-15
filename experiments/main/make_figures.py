"""The three figures of the final report, drawn from the result tables of experiments 1 to 4.

  fig_cost.pdf     how much client work removing the attacker takes, by how long it took part
  fig_quality.pdf  attack success and accuracy of each method, on CIFAR-10 and FEMNIST
  fig_share.pdf    the best share of the attacker's contribution to subtract, in 8 settings

    python experiments/main/make_figures.py        # writes figures/
"""
import csv
import glob
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "main"
OUT_HERE = ROOT / "figures"
sys.path.insert(0, str(HERE))
import analyze_share_rule as analyze


def cluster_summary():
    """{(dataset, tag, security): {cost, after, before}}: means over seeds of Dynamic Participation."""
    cells = defaultdict(list)
    for path in sorted(glob.glob(str(RESULTS / "exp3_clusters" / "clusters_*_s*.csv"))):
        for row in csv.DictReader(open(path)):
            cells[(row["dataset"], row["tag"], int(row["security"]), row["setting"])].append(row)
    mean = lambda rows, key: st.mean(float(r[key]) for r in rows)
    out = {}
    for (dataset, tag, security, setting), rows in cells.items():
        if setting == "after":
            before = cells[(dataset, tag, security, "before")]
            out[(dataset, tag, security)] = {
                "cost": mean(rows, "client_rounds"),
                "after": (mean(rows, "attack_success"), mean(rows, "test_accuracy")),
                "before": (mean(before, "attack_success"), mean(before, "test_accuracy"))}
    return out


clusters = cluster_summary()

# One colour per method, chosen to be told apart by people with colour-blindness.
COLOUR = {"none": "#7f7f7f", "fed": "#ff7f0e", "ours": "#1f77b4", "budget": "#9ecae1",
          "dyn": "#2ca02c", "retrain": "#222222"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold"})


def rows(pattern):
    out = []
    for path in sorted(glob.glob(str(pattern))):
        out += list(csv.DictReader(open(path)))
    return out


def mean(part, key):
    return st.mean(float(r[key]) for r in part)


def pick(table, **want):
    return [r for r in table if all(r.get(k) == v for k, v in want.items())]


b06 = rows(RESULTS / "exp3_baselines/baselines_s*.csv")
b08 = rows(RESULTS / "exp4_femnist/baselines_s*_f0.csv")
bud05 = rows(RESULTS / "exp2_budget/budget_s*_f*.csv")
bud08 = rows(RESULTS / "exp4_femnist/budget_s*_f0.csv")
S = analyze.load()


def case(table, tag):
    if tag == "j10":
        return [r for r in table if r.get("active_rounds") == "10"]
    return [r for r in table if r["force"] == tag[1:] and not r.get("active_rounds")]


def get(table, method, setting):
    return pick(table, method=method, setting=setting)


# ------------------------------------------------------------------ figure 1: cost
labels = ["10", "about 41", "about 132"]
tags = ["j10", "f0", "f4"]
retrain = [mean(get(case(b06, t), "single model", "retrained"), "client_rounds") for t in tags]
dyn2 = [clusters[("cifar10", t, 40)]["cost"] for t in tags]    # the paper's example: 2 clusters
dyn7 = [clusters[("cifar10", t, 10)]["cost"] for t in tags]    # the weakest security level: 7 clusters
ours = [90.0]   # 10 rounds x 9 other clients
for force in ("0", "4"):
    part = [r for r in bud05 if r["force"] == force and r["ranking"] == "shared" and r["rule"] == "complete"
            and r["share"] == "1.0" and r["lam"] == "0.25"]
    ours.append(mean(part, "client_rounds"))

fig, ax = plt.subplots(figsize=(3.4, 3.3))
width = 0.2
series = [("Retrain from scratch", retrain, COLOUR["retrain"]),
          ("Dynamic Participation, 2 clusters", dyn2, COLOUR["dyn"]),
          ("Dynamic Participation, 7 clusters", dyn7, "#98df8a"),
          ("Ours (every other client helps)", ours, COLOUR["ours"])]
for i, (name, values, colour) in enumerate(series):
    xs = [k + (i - 1.5) * width for k in range(3)]
    bars = ax.bar(xs, values, width, label=name, color=colour)
    for x, v in zip(xs, values):
        ax.text(x, v + 60, f"{v:,.0f}", ha="center", va="bottom", fontsize=6, rotation=90)
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_xlabel("Rounds the attacker took part in (of 400)")
ax.set_ylabel("Client work (client-rounds)")
ax.set_ylim(0, 5200)
ax.set_title("Ours costs less when the attacker\nwas in few rounds", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
fig.tight_layout()
OUT_HERE.mkdir(exist_ok=True)
for folder in (OUT_HERE,):
    fig.savefig(folder / "fig_cost.pdf", bbox_inches="tight")
    fig.savefig(folder / "fig_cost.png", dpi=170, bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------ figure 2: quality
def share_point(dataset, tag, lam):
    s = S[(dataset, tag)]
    return s["grid"][lam]


def budget_point(table):
    part = [r for r in table if r["force"] == "0" and r["ranking"] == "shared" and r["rule"] == "even"
            and r["share"] == "0.9" and r["lam"] == "0.25"]
    return mean(part, "attack_success"), mean(part, "test_accuracy")


def cluster_point(dataset, security):
    after = clusters[(dataset, "f0", security)]["after"]
    return after


def quality(table_b, table_bud, dataset):
    f0 = case(table_b, "f0")
    g = lambda m, s_: (mean(get(f0, m, s_), "attack_success"), mean(get(f0, m, s_), "test_accuracy"))
    return [
        ("Nothing removed", g("single model", "trained"), COLOUR["none"]),
        ("FedRecovery", g("FedRecovery", "noise 0"), COLOUR["fed"]),
        ("Ours, all paid, 15%", share_point(dataset, "f0", 0.15), COLOUR["ours"]),
        ("Ours, 90% paid, 25%", budget_point(table_bud), COLOUR["budget"]),
        ("Dynamic Part., 2 clusters", cluster_point(dataset, 40), COLOUR["dyn"]),
        ("Dynamic Part., 7 clusters", cluster_point(dataset, 10), "#98df8a"),
        ("Retrain (target)", g("single model", "retrained"), COLOUR["retrain"]),
    ]


panels = [("CIFAR-10", quality(b06, bud05, "cifar10")), ("FEMNIST", quality(b08, bud08, "femnist"))]
fig, axes = plt.subplots(2, 2, figsize=(3.5, 4.6))
for row, (name, data) in enumerate(panels):
    for col, (title, index) in enumerate((("Backdoor left (%)\nlower is better", 0),
                                          ("Accuracy (%)\nhigher is better", 1))):
        ax = axes[row][col]
        values = [100 * d[1][index] for d in data]
        xs = list(range(len(data)))
        ax.bar(xs, values, color=[d[2] for d in data], width=0.8)
        for x, v in zip(xs, values):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks([])
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=7, loc="left")
        if col == 0:
            ax.set_ylabel(name, fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
handles = [plt.Rectangle((0, 0), 1, 1, color=d[2]) for d in panels[0][1]]
fig.legend(handles, [d[0] for d in panels[0][1]], loc="lower center", ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=1.5)
for folder in (OUT_HERE,):
    fig.savefig(folder / "fig_quality.pdf", bbox_inches="tight")
    fig.savefig(folder / "fig_quality.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ figure 3: best share
names = {("cifar10", "j10"): "CIFAR-10, attacker in 10 rounds",
         ("cifar10", "f0"): "CIFAR-10, about 41 rounds",
         ("cifar10", "f4"): "CIFAR-10, about 132 rounds",
         ("cifar10", "boost"): "CIFAR-10, update x5",
         ("cifar10", "a01"): "CIFAR-10, Dirichlet 0.1",
         ("cifar10", "a100"): "CIFAR-10, Dirichlet 100",
         ("femnist", "f0"): "FEMNIST, about 41 rounds",
         ("femnist", "f4"): "FEMNIST, about 132 rounds"}
order = list(names)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.axvspan(0.10, 0.20, color="#cde5f5", zorder=0)
for k, key in enumerate(order):
    lam, _ = analyze.hindsight(S[key])
    ax.plot([lam], [len(order) - 1 - k], "o", color=COLOUR["ours"], ms=6)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([names[k] for k in order][::-1], fontsize=7)
ax.set_xlim(0, 0.5)
ax.set_xlabel("Best share to subtract")
ax.set_title("The best share is 0.10 to 0.20\nin every setting", loc="left", fontsize=9)
fig.tight_layout()
for folder in (OUT_HERE,):
    fig.savefig(folder / "fig_share.pdf", bbox_inches="tight")
    fig.savefig(folder / "fig_share.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote figures to", OUT_HERE)
