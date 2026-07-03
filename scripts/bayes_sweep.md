# Running the Bayesian Cannon sweep (NCI Gadi)

A Bayesian hyper-parameter search is a **closed loop**: the optimizer must see a
trial's metric before it proposes the next point. On Gadi only the `copyq` queue
has outbound internet, while the heavy compute must run on `rsaa`/`gpuvolta`
(no internet). Since there is no network path between the queues, the search is
split into two jobs that talk through a shared-filesystem **broker**:

```
 controller (copyq, online, tiny)          worker (rsaa, offline, heavy)
 --------------------------------          -----------------------------------
 wandb.agent(sweep_id):                    loop:
   params <- wandb.config                    task  <- claim  $BROKER/tasks/<id>.json
   write  $BROKER/tasks/<id>.json  -->       cv = cross_validate(...) ; _summarize
   wait   $BROKER/results/<id>.json <--      write $BROKER/results/<id>.json
   wandb.log(metrics)                        (never touches the network)
```

Only the transport is filesystem-based; the optimization itself is native W&B.

## Files

| File | Queue | Role |
|------|-------|------|
| `scripts/sweep_bayes_controller.pbs` | `copyq` | online optimizer (brokers params, logs metrics) |
| `scripts/sweep_bayes_worker.pbs` | `rsaa` | loads data, cross-validates each proposed point |
| `scripts/wandb_sweep.py` | — | the driver both PBS jobs invoke (`--role controller` / `--role worker`) |

Current defaults (already filled in): project `mk27`, env
`/scratch/y89/mj8805/miniforge/envs/astro/bin/python`, broker
`/scratch/mk27/cannon-bayes-broker`, data
`.../bulge-ages-and-orbits/data/merged_with_ages_raw.parquet`.

## 1. One-time setup

```bash
# On a Gadi login node, in the env that has thecannon + wandb:
wandb login          # writes ~/.netrc so the copyq controller can reach W&B
cd /scratch/mk27/mj8805/AnniesLasso && git pull   # get the latest scripts
```

## 2. Configure the search (controller PBS)

Edit the search block near the top of `scripts/sweep_bayes_controller.pbs`:

```bash
WANDB_PROJECT="cannon-bayes"
COUNT=40                            # trials this controller session runs
METRIC="mean_r2"; GOAL="maximize"   # unit-free objective (don't rank on mean_scatter)
LABEL_SETS=( "raw_teff,raw_logg,raw_fe_h,age_L"
             "raw_teff,raw_logg,raw_fe_h,mg_fe,age_L" )   # must include an age col to fit age
ORDERS="1,2"
REG_MIN="1e0"; REG_MAX="1e5"        # log-uniform: the knob bayes optimizes
```

## 3. Submit both jobs (same broker)

```bash
BROKER=/scratch/mk27/cannon-bayes-broker
qsub -v BROKER=$BROKER scripts/sweep_bayes_worker.pbs       # rsaa, offline compute
qsub -v BROKER=$BROKER scripts/sweep_bayes_controller.pbs   # copyq, online optimizer
```

Order doesn't matter — the controller waits for the worker. When `COUNT` trials
finish, the controller writes `$BROKER/STOP` and the worker exits.

## 4. Monitor

```bash
qstat -u $USER
cat cannon-bayes-controller.o<JOBID>    # sweep id + per-trial dispatch
cat cannon-bayes-worker.o<JOBID>        # data loading, cuts, per-trial metrics
cat $BROKER/sweep_id.txt                 # the W&B sweep id
```

Live results appear in the W&B project (`cannon-bayes`) as each trial is logged.

## 5. Run longer than one copyq session (~10 h)

W&B sweeps resume by id — a fresh controller attaches and continues:

```bash
qsub -v BROKER=$BROKER,SWEEP_ID=$(cat $BROKER/sweep_id.txt) scripts/sweep_bayes_controller.pbs
```

For more throughput, submit the **worker** more than once (they share the
broker). Keep it modest (2–4) — heavy parallelism dilutes the Bayesian feedback.

## Cuts applied to the training data (automatic)

The worker applies these before cross-validating; watch its log for the
`row filters kept N/M` and `quality cut: N/M rejected by ...` lines:

- **`--filter snr_x>100`** — explicit in the worker PBS (global S/N cut).
- **`quality_mask`** — `ASPCAPFLAG` STAR_BAD, bad `STARFLAG` bits, `VSCATTER > 0.5`
  (plus `spectrum_flags`/`warn_*` if present).
- **Age reliability, per label set** — a set fitting `age_Dnu` is cut on
  `RelAge_Dnu`, `age_L`/`mass_L` on `RelAge_L` (intersection if it fits several).
- **`X_FE_FLAG`, per element** — a set is cut to stars unflagged for each
  abundance it fits.

Any cut whose column is missing skips with a warning (it does **not** silently
reject everything), so the first real run tells you if a column name is off.

## Reading the results

Rank grid points / trials on:

- **`mean_r2`** — unit-free overall quality (higher is better); the objective.
- **`mean_sigma_mad`** — robust scatter (outlier-immune).
- **`mean_pull_std`** — uncertainty calibration (≈1 = trustworthy error bars; >1 = optimistic).
- **`median_r_chi_sq`** (≈1 ideal), **`theta_frac_zero`** (L1 sparsity vs regularization).

## Verify the wiring offline first (optional, no cluster)

```bash
python -m scripts.wandb_sweep --role controller --broker /tmp/b --dry-run \
    --label-set raw_teff,raw_logg,raw_fe_h,age_L --reg-min 1 --reg-max 1e5
```

prints the sweep config without touching W&B.

## Gotchas

- **The controller must stay online.** This is the whole reason for the split —
  offline + `wandb sync` (the grid-sweep pattern) can't drive Bayesian search,
  because results would arrive too late to steer it.
- **Set `WANDB_API_KEY`** for fully headless runs (or rely on `~/.netrc` from
  `wandb login`).
- **`git pull` on Gadi** before submitting so you get the latest scripts.
- If a worker is stranded (controller died without writing `STOP`), it exits
  after `--worker-idle-timeout` (default 30 min) rather than burning walltime.
