# %% [markdown]
# # Build 5: A significance test on a business question (60 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_core` (Coral Petal
# synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_core.py`
#
# PetalMart's marketing team asks: do app customers spend more than web customers?
# Write your hypotheses and alpha first, in a text cell, before you run a single test.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

rng = np.random.default_rng(2026)
tables = pm.core()
orders = pm.clean_orders(tables["orders"]).merge(tables["users"], on="user_id")
# One row per customer: a customer's orders are not independent of each other (L4.5),
# so compare customers by their average order value over FY2025-26.
customers = orders.groupby(["user_id", "device", "loyalty_member"], as_index=False).order_value.mean()
app = customers.loc[customers.device == "app", "order_value"]
web = customers.loc[customers.device == "web", "order_value"]
results = {}

# %% [markdown]
# ## Task 1 · Hypotheses and alpha (L5.2)
# In words and in symbols, and choose alpha now.

# %%
results["hypotheses"] = None  # TODO: {"H0": "...", "H1": "...", "alpha": 0.05}

# %% [markdown]
# ## Task 2 · Check the assumptions (L5.3)
# Histograms and box plots of average order value for each group; note the skew and the group sizes. Which tests fit?

# %%
results["n_app_customers"] = None  # TODO
results["n_web_customers"] = None  # TODO

# %% [markdown]
# ## Task 3 · Two tests (L5.3, L5.4)
# Welch's t-test on the means and the Mann-Whitney test on the whole distributions. Do they agree?

# %%
results["mean_diff"] = None  # TODO: app minus web, rupees
results["welch_p"] = None  # TODO
results["mannwhitney_p"] = None  # TODO

# %% [markdown]
# ## Task 4 · Intervals and effect size (L5.1, L5.5, L5.6)
# A formula 95% CI for the difference in means (`confidence_interval()` on the Welch result), a bootstrap 95% CI for the
# difference in medians (5,000 resamples, with `rng`), and Cohen's d.

# %%
results["mean_diff_ci"] = None  # TODO: [low, high]
results["median_diff_ci_bootstrap"] = None  # TODO: [low, high]
results["cohens_d"] = None  # TODO

# %% [markdown]
# ## Task 5 · Conclusion for the marketing team (at most 100 words)
# The size of the difference in rupees, how sure you are, and one reason it might not be caused by the app itself.
# Hint: compare the share of loyalty members among app and web customers.

# %%
results["conclusion"] = None  # TODO

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "n_app_customers": pm.matches("c0f2600a243d", 0, "one row per customer with at least one clean order"),
    "n_web_customers": pm.matches("36f9a4c00c33", 0, "one row per customer with at least one clean order"),
    "mean_diff": pm.matches("031b4af5197e", 0, "app mean minus web mean, rupees"),
    "welch_p": pm.between(0, 0.001, "stats.ttest_ind(app, web, equal_var=False)"),
    "mean_diff_ci": lambda v: (v[0] < 51 < v[1] and v[0] > 0, "the interval should sit above zero and contain your difference"),
    "cohens_d": pm.between(0.03, 0.10, "difference in means divided by the pooled standard deviation"),
})
