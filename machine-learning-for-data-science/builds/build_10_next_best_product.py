# %% [markdown]
# # Build 10: PetalBank next-best product (75 min)
# Machine Learning for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter or VS Code, `# %%` cells). Data: `petalbank_holdings` (Coral Petal synthetic,
# seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder:
# `python datasets/generate_petalbank_holdings.py --out datasets/data/petalbank_holdings`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalbank as pb

data = pb.holdings()
results = {}

# %% [markdown]
# ## Task 1 · Matrix and split
# Customer × product matrix from holdings started before 2026-04. Test: products adopted April to June 2026 by existing customers.

# %%
results["n_test_customers"] = None

# %% [markdown]
# ## Task 2 · Three recommenders
# Popularity, item-based CF (cosine) and ALS (`implicit` library). Recommend only eligible, not-yet-held products (see products.csv).

# %%
results["precision_at_5"] = None  # TODO: {"popularity": ..., "item_cf": ..., "als": ...}
results["ndcg_at_5"] = None

# %% [markdown]
# ## Task 3 · Recommend one
# Which would you ship, and how do eligibility rules change the lists?

# %%
results["chosen"] = None

# %% [markdown]
# ## Check

# %%
pb.check(results, {"ndcg_at_5": lambda v: (isinstance(v, dict) and v.get("item_cf", 0) > v.get("popularity", 1), "CF should beat popularity")})
