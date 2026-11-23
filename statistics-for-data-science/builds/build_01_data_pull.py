# %% [markdown]
# # Build 1: PetalMart data pull (60 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_core` (Coral Petal
# synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_core.py`
#
# PetalMart's head of growth wants a weekly view of revenue by region, and to know which customers and devices drive it.

# %%
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

CORE = pm.DATA / "petalmart_core"
con = duckdb.connect()
results = {}

# %% [markdown]
# ## Task 1 · Pull with SQL (Module 01, L1.4)
# DuckDB queries CSV files directly: `con.sql(f"SELECT ... FROM '{CORE}/orders.csv' ...").df()`.
# Pull orders with a positive value, and the users and sessions tables. Check the row count against a simple total first.

# %%
orders = None  # TODO: con.sql(f"SELECT * FROM '{CORE}/orders.csv' WHERE order_value > 0").df()
users = None  # TODO
sessions = None  # TODO
results["n_positive_orders"] = None  # TODO: rows returned by your orders query

# %% [markdown]
# ## Task 2 · Join in pandas (L1.3)
# Join orders to users with a left join, `validate="many_to_one"` and `indicator=True`.
# How many orders have no matching user? Write one sentence on what that could mean for the business.

# %%
results["n_orders_without_user"] = None  # TODO: rows with _merge == "left_only"

# %% [markdown]
# ## Task 3 · Weekly revenue by region (L1.2, L1.3)
# Long format first: one row per week and region with revenue, number of orders and average order value
# (`resample("W")` on `order_ts`, or `pd.Grouper`). Then pivot to wide (one column per region) for the report.

# %%
weekly_long = None  # TODO: columns week, region, revenue, orders, aov
weekly_wide = None  # TODO: weekly_long.pivot(index="week", columns="region", values="revenue")
results["top_region_by_revenue"] = None  # TODO: region code with the highest total revenue, e.g. "N"

# %% [markdown]
# ## Task 4 · Three charts with finding-style titles (L1.5)
# 1. Weekly revenue by region. 2. Order value by device, as box plots (log scale helps). 3. Share of revenue by category.
# Every title states the finding, for example "South leads revenue in every quarter".

# %%
# TODO: three figures; save each with plt.savefig("build1_chart1.png", dpi=150, bbox_inches="tight") and so on
results["chart_titles"] = None  # TODO: list of your three titles

# %% [markdown]
# ## Check
# Then restart the kernel, run every cell, and push with a README: the question, three findings, one thing to check next.

# %%
pm.check(results, {
    "n_positive_orders": pm.matches("64753c1c6650", 0, "keep rows with order_value > 0 only, nothing else removed yet"),
    "n_orders_without_user": pm.matches("d59eced1ded0", 0, "left join orders to users and count left_only rows"),
    "top_region_by_revenue": pm.matches("657e7f22b0b4", None, "sum order_value by region and take the largest"),
})
