# Federated unlearning when the server only kept the sums

Code for our course project on **removing one client from a federated model under secure
aggregation**, where the server never sees a single client's update, only the sum of each round.

## Method

The server keeps `H = { theta_0, S_t, a_t }`: the starting model, who took part in round `t`, and the
sum of that round's updates. To remove a client `k` (here, a backdoor attacker):

1. **Recover its update.** The *other* clients of a round redo it from the stored model, and the
   server subtracts their sum from the stored one: `g_k = a_t - sum_{i != k} g_i`. This is exact,
   because a redo reproduces the original (same model, data, batch order and seeds). It is the
   multi-round differencing leak of So et al. (AAAI 2023), used on purpose for one identified client.
2. **Subtract a share of it.** The correction for round `t` is `(a_t/m - g_k) / (m-1)`, and the
   unlearned model is `theta_T + lambda * sum_t correction_t`. `lambda = 1` is the rule of Wu et al.
   (2022); the damping `lambda` (about 0.15 works) is ours, because full subtraction ruins accuracy.

Cost is counted in **client-rounds**: one client redoing one past round is one unit.

## Repository

```
secunlearn/    the solution: data, training with a stored history, recovery, unlearning, baselines
main/          scripts that reproduce the experiments reported in the final report (Experiments 1-5)
ablation/      extra experiments and checks, not needed to reproduce the report
results/       csv tables of our runs (results/main, results/ablation)
figures/       the report's three figures
tests/         fast checks on synthetic data (no download, no GPU)
```

Core functions: `train_federation` and `History` (`secunlearn/training.py`), `recover` and
`redo_round` (recover a client's update), `unlearn(history, updates, lam)`
(`secunlearn/unlearning.py`), `quantise` (`secunlearn/model.py`).

## Install and test

```bash
pip install -r requirements.txt
python -m pytest tests          # a few seconds, no data needed
```

## Reproducing the report (`main/`)

Setting unless stated: CIFAR-10, 100 clients, 10 per round, 400 rounds, 3 seeds. Client 0 is the
attacker; three attackers (tags): `j10` joins 10 rounds then leaves, `f0` is picked like any client
(about 41 rounds), `f4` is also put into every fourth round (about 132). Tags `boost`, `a01`,
`a100` are used in Experiment 5. Training runs are shared: each (tag, seed) is trained once.

| Report | Script | What it does | Tables |
|---|---|---|---|
| Exp 1: what removal costs, how much to subtract | `main/exp1_share_to_subtract.py --tags j10 f0 f4` | recover exactly, subtract a share `lambda` from a grid | `results/main/exp1_share/` |
| Exp 2: which clients to pay | `main/exp2_client_budget.py` | pay only some clients; 2 orders x 4 rules for incomplete rounds | `results/main/exp2_budget/` |
| Exp 3: two published methods | `main/exp3_baselines.py` (FedRecovery) and `main/exp3_clusters.py` (Dynamic Participation, clusters of its own paper) | our reimplementations, same attackers | `results/main/exp3_baselines/`, `exp3_clusters/` |
| Exp 4: FEMNIST | `main/exp4_femnist.py` | experiments 1-3 on FEMNIST (one client per writer) | `results/main/exp4_femnist/` (+ `lambda_femnist_*` in `exp1_share/`, `clusters_femnist_*` in `exp3_clusters/`) |
| Exp 5: choosing the share | `main/exp1_share_to_subtract.py` (all 6 tags + FEMNIST) then `main/analyze_share_rule.py` | `rho` and the rule `lambda = min(1, kappa/rho)`, tested leave-one-setting-out | `results/main/exp1_share/` |
| Figures | `main/make_figures.py` | `fig_cost`, `fig_quality`, `fig_share` | `figures/` |

Findings (from the report): redoing the attacker's rounds costs 2%, 9% or 31% of retraining; a share of
about 15% is a usable default but not a full removal; FedRecovery removes almost nothing here; Dynamic
Participation is more accurate on FEMNIST and cheaper when the attacker took part in many rounds; paying
fewer clients saves little work.

## Extra experiments (`ablation/`)

| Script | Question | Tables |
|---|---|---|
| `ablation/replay_check.py` | the gate: does a redone round reproduce the original exactly? | none (raises on failure) |
| `ablation/round_selection.py` | outline steps 1-2: redo only some rounds (5 selectors x 6 budgets) and guess the rest (4 rules) | `round_selection_r400/`, `_r1000/` |
| `ablation/control.py` | how far apart are two *correct* retrained models? (a floor for distance-based scoring) | `control_r400/`, `_r1000/` |
| `ablation/diagnosis.py` | why does full subtraction damage the model? damping and latest-rounds-only | `diagnosis_r400/`, `_r1000/` |
| `ablation/scaling.py` | does the same *fraction* of rounds work at other sizes N and m? | `scaling_r400/`, `_r1000/` |
| `ablation/which_clients.py` | outline step 3: which helpers to ask, one at a time (`--drop`: leave one helper out) | `which_clients_r400/`, `_r1000/` |
| `ablation/poison_damping.py` | the mid-term backdoor experiment (20 clients, 200 rounds) | `poison_damping/`, `poison_damping_n100/` |
| `ablation/dp_fixed_groups.py` | Dynamic Participation with groups of 10/20/50 (before we used its own clustering) | `dp_fixed_groups/` (not run yet) |

`_r400` / `_r1000` is the number of rounds (the removed client then joins about 40 / 100 rounds).
The ablations that remove a harmless client use clients 0, 1, 2 as the removed ones.

## Running

```bash
# a tiny run of any script, on random data, to check the code works (no download, a minute or two)
python main/exp1_share_to_subtract.py --smoke --dataset synthetic --tags f0

# the real thing (CIFAR-10 is downloaded to ./data on first use; FEMNIST needs `datasets`)
python main/exp1_share_to_subtract.py --tags j10 f0 f4
python main/exp2_client_budget.py
python main/exp3_baselines.py
python main/exp3_clusters.py
python main/exp4_femnist.py
python main/make_figures.py
```

Every script takes `--seeds`, `--rounds`, `--clients`, `--per-round`, `--per-client`, `--smoke` and
`--fresh`. A finished training run or table is skipped when you run again, so an interrupted job costs
one run, not all of them. Set `SECUNLEARN_DATA` to keep the datasets elsewhere. Model histories go to
`runs/` (gigabytes; a seed rebuilds them; not in git). Tables go to `results/`.

**Time.** On a Colab T4 one (tag, seed) job of experiments 1-4 took 5 to 20 minutes. On a CPU expect
many hours for the whole set; the CPU path is deterministic but slow.

**Reproducibility.** A redone round must reproduce the original to the last bit, so training runs on CUDA
or CPU with one CPU thread and deterministic cuDNN. Apple's MPS is never used. If `verify_replay` fails on
your machine, use a CPU runtime. Updates are rounded onto an integer grid (`2**-24`) before they are
added, because floating-point addition is not associative and the subtraction would otherwise return the
update plus noise; integers are also closer to real secure aggregation.

## Honest limits

* Secure aggregation is **simulated**: we add integer updates and do not run the masking protocol.
  The attacker's true updates are kept in a separate `oracle` file **for marking only**; the method
  never reads them.
* Recovering one client's update by differencing is the multi-round leak described by So et al. (AAAI 2023).
  We use it on purpose, for one identified attacker only.
* FedRecovery and Dynamic Participation are **our reimplementations** (neither paper links code). Where the
  papers are silent the settings are ours; see `secunlearn/baselines.py` and `main/exp3_clusters.py`.
* The shares of Experiments 1-4 were chosen with the retrained model at hand; the fixed 15% of
  Experiment 5 was chosen on the same eight settings.
* One dataset pair, one network, one kind of attack.
