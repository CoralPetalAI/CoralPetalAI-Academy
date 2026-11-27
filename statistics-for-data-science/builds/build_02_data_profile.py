# %% [markdown]
# # Build 2: Data profile of the PetalMart sales dataset (75 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_core` (Coral Petal
# synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_core.py`
#
# PetalMart's leadership wants to know what its sales data can be trusted to say. Clean before you summarize,
# and run `describe()` before and after cleaning, side by side.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

tables = pm.core()
orders, users = tables["orders"], tables["users"]
results = {}
cleaning_log = []  # one dict per step: step, problem, rule, action, rows_affected

# %% [markdown]
# ## Task 1 · Types (L2.1)
# Report the type of every column of orders: identifier, categorical, numeric (discrete or continuous), date-time.

# %%
results["column_types"] = None  # TODO: {"order_id": "identifier", "order_value": "continuous", ...}

# %% [markdown]
# ## Task 2 · Clean, with a log (L2.6)
# 1. Exact duplicate rows. 2. Orders worth zero or less. 3. Users with a missing region: how many, are they missing
# at random (compare app and web), and what will you do? Add one row to `cleaning_log` for every step.

# %%
results["n_duplicates"] = None  # TODO: exact duplicate rows in orders
results["n_invalid_values"] = None  # TODO: rows with order_value <= 0, counted after removing duplicates
results["n_users_missing_region"] = None  # TODO
results["n_clean_orders"] = None  # TODO: rows left after removing duplicates and invalid values
clean = None  # TODO: the cleaned orders

# %% [markdown]
# ## Task 3 · Summarize order value by category (L2.2)
# Median, IQR, mean and standard deviation per category. Which would you report to leadership, and why?

# %%
summary = None  # TODO: clean.groupby("category").order_value.agg(["median", iqr, "mean", "std"])
results["highest_median_category"] = None  # TODO

# %% [markdown]
# ## Task 4 · Charts and outliers (L2.3, L2.4)
# Three charts with finding-style titles, one of them over time. Then flag outliers in `clean` with the
# 1.5 × IQR rule. Most are genuine large orders; some are errors. Look at value per item: which orders look like
# they were recorded in cents instead of dollars?

# %%
results["n_iqr_outliers"] = None  # TODO: orders outside Q1 - 1.5 IQR or Q3 + 1.5 IQR, in `clean`
results["n_suspected_errors"] = None  # TODO: outliers that look like data errors, not genuine extremes

# %% [markdown]
# ## Task 5 · What this data can and cannot tell us
# One paragraph. Then show `pd.DataFrame(cleaning_log)`.

# %%
results["can_and_cannot"] = None  # TODO: your paragraph as a string

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "n_duplicates": pm.matches("c75d3f1f5bcd", 0, "use orders.duplicated()"),
    "n_invalid_values": pm.matches("624b60c58c9d", 0, "count order_value <= 0 after dropping duplicates"),
    "n_users_missing_region": pm.matches("3538a1ef2e11", 0, "users.region.isna()"),
    "n_clean_orders": pm.matches("993f4d6c0109", 0, "drop duplicates, then values <= 0"),
    "n_iqr_outliers": pm.matches("2b3acf19e8b2", 0, "quartiles of order_value in the cleaned orders, then the two fences"),
    "n_suspected_errors": pm.matches("2c624232cdd2", 0, "sort the outliers by value per item"),
})
