# %% [markdown]
# # Build 11: Did same-day delivery raise spend? (75 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_same_day` (Coral Petal
# synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_same_day.py`
#
# Same-day delivery launched in April 2025 in six cities PetalMart chose. Leadership is arguing about whether it
# raised spend. Settle it with two independent methods. Make the event-study plot early.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

data = pm.same_day()
city_month, customers = data["city_month"], data["customers"]
city_month["post"] = (city_month.month >= "2025-04").astype(int)
results = {}

# %% [markdown]
# ## Task 1 · A causal graph (L11.1)
# Draw it (paper or a diagram tool) and list what you must control for, and why.

# %%
results["controls_and_why"] = None  # TODO

# %% [markdown]
# ## Task 2 · Parallel trends (L11.3)
# Monthly spend per customer, launch cities versus the other 34, with the launch marked. Do the trends move together
# before April 2025? Also: the naive comparison of the two groups after the launch.

# %%
results["naive_post_diff"] = None  # TODO: mean spend per customer after launch, launch cities minus others (city-months)

# %% [markdown]
# ## Task 3 · Difference-in-differences (L11.3)
# By hand with the 2 × 2 table of means; then `spend_per_customer ~ same_day + C(city) + C(month)` with standard
# errors clustered by city; then an event-study plot (one coefficient per month relative to the launch).

# %%
results["did_2x2"] = None  # TODO: rupees per customer per month
results["did_twfe_ci"] = None  # TODO: [low, high]
results["max_abs_pre_coefficient"] = None  # TODO: largest absolute event-study coefficient before the launch

# %% [markdown]
# ## Task 4 · Propensity-score matching (L11.2)
# Customers in launch cities versus others, matched on pre_spend, pre_orders, tenure_months, device and age_band.
# Check balance (standardized mean differences before and after), then the matched difference in post_spend,
# per month (it covers six months). Do the two methods agree?

# %%
results["max_smd_after_matching"] = None  # TODO
results["matched_effect_per_month"] = None  # TODO: rupees per customer per month

# %% [markdown]
# ## Task 5 · Note to leadership
# Best estimate, range, whether the methods agree, and the one assumption most likely to be wrong.

# %%
results["note"] = None  # TODO

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "naive_post_diff": pm.matches("ee62de25ccc2", 0, "average the city-months after the launch in each group"),
    "did_2x2": pm.matches("2858dcd1057d", 0, "(launch after − launch before) − (others after − others before)"),
    "did_twfe_ci": lambda v: (v[0] < (results["did_2x2"] or 0) < v[1], "the interval should contain your 2 × 2 estimate"),
    "max_abs_pre_coefficient": pm.between(0, 20, "pre-launch coefficients should hover around zero"),
    "max_smd_after_matching": pm.between(0, 0.1, "after matching every SMD should be below about 0.1"),
    "matched_effect_per_month": pm.between(40, 70, "close to the difference-in-differences estimate"),
})
