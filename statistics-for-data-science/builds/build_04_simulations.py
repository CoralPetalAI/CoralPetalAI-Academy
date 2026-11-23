# %% [markdown]
# # Build 4: Simulations of real-world variation (90 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_core` (Coral Petal
# synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_core.py`
#
# PetalMart's operations team wants to know how much its daily numbers can swing by chance alone.
# Every simulation uses `rng` below, so your results are the same every time you run the notebook.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

rng = np.random.default_rng(2026)
tables = pm.core()
orders = pm.clean_orders(tables["orders"]).merge(tables["users"][["user_id", "region"]], on="user_id")
values = orders.order_value.to_numpy()
# Parameters from the operations team
DAILY_VISITORS, CONVERSION = 4_000, 0.035          # 90 days of orders
TICKETS_PER_HOUR = 4                                # 60 days of hourly support tickets
DELIVERY_MEAN, DELIVERY_SD = 48, 8                  # 1,000 delivery times, hours
results = {}

# %% [markdown]
# ## Task 1 · Three simulations against theory (L4.1 to L4.3)
# Binomial orders per day, Poisson tickets per hour, normal delivery times. Compare simulated mean and spread with theory.

# %%
results["binomial"] = None  # TODO: {"sim_mean": ..., "theory_mean": ..., "sim_sd": ..., "theory_sd": ...}
results["poisson"] = None  # TODO: {"sim_mean": ..., "sim_var": ...}
results["normal_share_over_60h"] = None  # TODO: share of simulated deliveries over 60 hours

# %% [markdown]
# ## Task 2 · The Central Limit Theorem with real, skewed order values (L4.4)
# Draw 2,000 sample means for n = 5, 30 and 200 from `values` and plot the three histograms side by side.

# %%
means = None  # TODO: {5: array, 30: array, 200: array}

# %% [markdown]
# ## Task 3 · Standard error
# For each n, the spread of the sample means divided by σ/√n (σ = the standard deviation of `values`).

# %%
results["se_ratio"] = None  # TODO: {5: ..., 30: ..., 200: ...}; each should be close to 1

# %% [markdown]
# ## Task 4 · Simple random versus stratified sampling (L4.5)
# Estimate average order value from 400 orders, 2,000 times: once by simple random sampling, once stratified by
# region (allocation proportional to each region's share). Which estimate is more stable?

# %%
results["srs_sd"] = None  # TODO: standard deviation of the 2,000 simple-random estimates
results["stratified_sd"] = None  # TODO: the same for the stratified estimates

# %% [markdown]
# ## Task 5 · Fit an exponential, and n versus n − 1 (L4.6)
# Time between orders: weekdays, 10:00 to 22:00, from 9 to 20 February 2026; gaps in minutes between consecutive
# orders within each day. Fit with `stats.expon.fit(gaps, floc=0)`; the rate is 1 / scale. Then draw 200,000 samples of
# five order values and compare the average of each variance estimate (ddof=0 and ddof=1) with the variance of `values`.

# %%
gaps = None  # TODO
results["n_gaps"] = None  # TODO
results["expon_rate"] = None  # TODO: per minute
results["one_over_mean_gap"] = None  # TODO
results["var_ratio_divide_by_n"] = None  # TODO: mean of ddof=0 variances / true variance
results["var_ratio_divide_by_n_minus_1"] = None  # TODO: the same with ddof=1

# %% [markdown]
# ## Check
# README: in plain words, the one chart that best explains the Central Limit Theorem.

# %%
pm.check(results, {
    "binomial": lambda v: (abs(v["sim_mean"] - v["theory_mean"]) < 5 and abs(v["sim_sd"] - v["theory_sd"]) < 2.5,
                           "simulation agrees with n·p and √(n·p·(1−p))"),
    "poisson": lambda v: (3.7 < v["sim_mean"] < 4.3 and 3.5 < v["sim_var"] < 4.5, "mean and variance both near lambda"),
    "normal_share_over_60h": pm.between(0.045, 0.09, "theory: 1 − Φ(1.5) ≈ 0.067"),
    "se_ratio": lambda v: (all(0.85 < float(r) < 1.15 for r in v.values()), "spread of the means ≈ σ/√n"),
    "stratified_sd": lambda v: (float(v) < float(results["srs_sd"] or 0), "stratifying should steady the estimate"),
    "n_gaps": pm.matches("e1d9ce80c3ab", 0, "clean orders, weekdays, hours 10 to 21, gaps within each day"),
    "expon_rate": lambda v: (abs(float(v) - float(results["one_over_mean_gap"] or 0)) < 1e-3,
                             "the maximum likelihood rate is one over the mean gap"),
    "var_ratio_divide_by_n": pm.between(0.70, 0.90, "dividing by n runs low: about (n − 1)/n"),
    "var_ratio_divide_by_n_minus_1": pm.between(0.88, 1.12, "dividing by n − 1 should hit the target on average"),
})
