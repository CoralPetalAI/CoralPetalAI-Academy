# %% [markdown]
# # Build 0: Set up and meet the data (30 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells, or `jupytext --to notebook` this file).
# Data: `petalmart_core` (Coral Petal synthetic, seed 2026). Fill every `TODO`; the last cell checks your
# answers without revealing them. Generate the data first, from the course folder:
# `python datasets/generate_petalmart_core.py`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

tables = pm.core()
users, orders, sessions, costs = tables["users"], tables["orders"], tables["sessions"], tables["costs"]
results = {}

# %% [markdown]
# ## Task 1 · Load and look
# For each table print its shape, `info()` and the first five rows. If a cell fails, read the error from the bottom up.

# %%
# TODO: users.shape, users.info(), users.head() ... and the same for orders, sessions and costs
results["n_users"] = None  # TODO: rows in users
results["n_orders_raw"] = None  # TODO: rows in orders, exactly as generated
results["n_sessions"] = None  # TODO: rows in sessions
results["first_order_date"] = None  # TODO: earliest order_ts as "YYYY-MM-DD"
results["last_order_date"] = None  # TODO: latest order_ts as "YYYY-MM-DD"

# %% [markdown]
# ## Task 2 · What is one row?
# In a text cell, write one sentence per table: what does one row mean? (one order, one customer, one visit ...)

# %%
results["one_row_means"] = None  # TODO: {"users": "...", "orders": "...", "sessions": "...", "costs": "..."}

# %% [markdown]
# ## Task 3 · Three questions
# Write three specific questions you would ask of this data as PetalMart's data scientist,
# for example "Which region has the highest average order value?". Keep them: you will answer all three by the end of the course.

# %%
results["three_questions"] = None  # TODO: a list of three strings

# %% [markdown]
# ## Task 4 · Push to GitHub
# Create a public repository for this course, commit this notebook (not the data folder) with a one-line README, and push.

# %%
results["github_repo_url"] = None  # TODO

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "n_users": pm.matches("891937cb3350", 0, "count the rows of users"),
    "n_orders_raw": pm.matches("31abe7977f55", 0, "count every row of orders, duplicates included"),
    "n_sessions": pm.matches("bf79b6ec05a6", 0, "count the rows of sessions"),
    "three_questions": lambda v: (isinstance(v, list) and len(v) == 3, "three questions" if isinstance(v, list) and len(v) == 3 else "give a list of three"),
})
