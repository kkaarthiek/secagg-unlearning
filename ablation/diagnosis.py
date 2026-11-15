"""Diagnosis: why does the correction damage the model?

Subtracting the removed client's recovered updates in full loses accuracy even when every update is
known exactly, so the damage comes from how the corrections are applied, not from guessing. Two tests,
both with exact updates (a full redo):

  damping   apply only a share lambda of the total correction. If a small share already removes the
            client without hurting accuracy, the direction is right and only the size is wrong.
  recent    correct only the client's latest rounds (the correction assumes later rounds would have
            been unchanged, which is most nearly true near the end of training).

This is what led to the share lambda of the final report.

    python ablation/diagnosis.py [--rounds 1000]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secunlearn import plots
from secunlearn import setup as cli
from secunlearn.settings import REMOVE_CLIENTS, benign_runs
from secunlearn.training import History, load_true_updates
from secunlearn.unlearning import corrections_of, measure_quality, relative_error
from secunlearn.utils import read_csv, say, write_csv

DAMPING = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]      # share of the full correction applied
RECENT = [0.1, 0.25, 0.5, 0.75, 1.0]            # share of the client's rounds, latest first
FIELDS = ["seed", "client", "test", "setting", "rounds_corrected", "correction_size",
          "relative_error", "test_accuracy", "removed_client_loss"]


def run_seed(setup, seed, out):
    base, retrains = benign_runs(setup, seed)
    history = History(base)
    rows = []
    for client in REMOVE_CLIENTS[: 1 if setup.smoke else None]:
        joined = history.rounds_with(client)
        retrained = np.load(retrains[client] / "theta_final.npy")
        gap = np.linalg.norm(history.theta_final - retrained)
        steps = corrections_of(history, load_true_updates(base, client), joined)

        def record(test, setting, used, share=1.0):
            shift = share * np.sum([steps[t] for t in used], axis=0)
            theta = history.theta_final + shift.astype(np.float32)
            accuracy, own_loss = measure_quality(history.data, theta, client, history.config)
            rows.append({"seed": seed, "client": client, "test": test, "setting": setting,
                         "rounds_corrected": len(used), "correction_size": float(np.linalg.norm(shift) / gap),
                         "relative_error": relative_error(theta, history.theta_final, retrained),
                         "test_accuracy": accuracy, "removed_client_loss": own_loss})

        for share in DAMPING:
            record("damping", share, joined, share)
        for share in RECENT:
            record("recent", share, joined[-max(1, round(share * len(joined))):])
    write_csv(out / f"diagnosis_s{seed}.csv", rows, FIELDS)


def draw(out):
    rows = read_csv(out, "diagnosis_s*.csv")
    if not rows:
        return
    fig, axes = plots.plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (test, settings) in zip(axes, (("damping", DAMPING), ("recent", RECENT))):
        vals = [[float(r["test_accuracy"]) for r in rows if r["test"] == test and float(r["setting"]) == s] for s in settings]
        plots.line(ax, settings, [np.mean(v) for v in vals], [np.std(v) for v in vals], "test accuracy", 0)
        plots.style(ax, "Share of the correction applied" if test == "damping" else
                    "Share of the client's rounds corrected (latest first)", "Test accuracy", test)
    plots.save(fig, out / "figures", "diagnosis")


def main():
    setup, _ = cli.parse(__doc__)
    out = setup.results("ablation", f"diagnosis_r{setup.rounds}")
    for seed in setup.seeds:
        run_seed(setup, seed, out)
    draw(out)
    for test, settings in (("damping", DAMPING), ("recent", RECENT)):
        for s in settings:
            part = [r for r in read_csv(out, "diagnosis_s*.csv") if r["test"] == test and float(r["setting"]) == s]
            if part:
                say(f"{test:>8} {s:>5}: accuracy {np.mean([float(r['test_accuracy']) for r in part]):.3f}, "
                    f"client loss {np.mean([float(r['removed_client_loss']) for r in part]):.3f}")


if __name__ == "__main__":
    main()
