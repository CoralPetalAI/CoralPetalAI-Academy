# %% [markdown]
# # Build 8: Segment PetalBank customers (75 min)
# Machine Learning for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter or VS Code, `# %%` cells). Data: `petalbank_loans` (Coral Petal synthetic,
# seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder:
# `python datasets/generate_petalbank_loans.py --out datasets/data/petalbank_loans`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalbank as pb

data = pb.loans()
results = {}

# %% [markdown]
# ## Task 1 · Customer table
# One row per customer: age, income, credit score, utilization, bounced payments, balance, employment years. Scale it.

# %%
results["n_customers"] = None

# %% [markdown]
# ## Task 2 · PCA
# How many components keep 80% of the variance? Plot the first two.

# %%
results["components_80pct"] = None

# %% [markdown]
# ## Task 3 · k-means with silhouette
# Try k from 2 to 8; choose k by silhouette and business sense.

# %%
results["chosen_k"] = None
results["silhouette"] = None

# %% [markdown]
# ## Task 4 · DBSCAN
# Compare with DBSCAN. What does it call noise?

# %%
results["dbscan_noise_share"] = None

# %% [markdown]
# ## Task 5 · Name and act
# Name 4 to 6 segments, give each a default rate and one action.

# %%
results["segment_names"] = None

# %% [markdown]
# ## Check

# %%
pb.check(results, {"chosen_k": lambda v: (4 <= v <= 6, "name 4 to 6 segments")})
