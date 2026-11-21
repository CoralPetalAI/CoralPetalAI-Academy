# %% [markdown]
# # Build 5: Leak-proof pipeline (60 min)
# Machine Learning for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter or VS Code, `# %%` cells). Data: `petalbank_loans_leaky` (Coral Petal synthetic,
# seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder:
# `python datasets/generate_petalbank_loans_leaky.py --out datasets/data/petalbank_loans_leaky`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalbank as pb

data = pb.loans_leaky()
results = {}

# %% [markdown]
# ## Task 1 · Too good to be true
# Fit gradient boosting on every column except IDs and dates with a random 80/20 split. Report PR-AUC.

# %%
results["pr_auc_naive"] = None

# %% [markdown]
# ## Task 2 · Find the three leaks
# For each column ask: would we know this at application time? Check suspiciously strong single-column scores.

# %%
results["leaks"] = None  # TODO: sorted list of the three column names

# %% [markdown]
# ## Task 3 · Rebuild honestly
# Drop the leaks, put every step in one Pipeline with a ColumnTransformer, and use TimeSeriesSplit on applications sorted by date.

# %%
results["pr_auc_honest"] = None

# %% [markdown]
# ## Task 4 · Compare
# In a markdown cell, explain each leak and how much it inflated the score.

# %%
results["explained"] = None  # TODO: True

# %% [markdown]
# ## Check

# %%
pb.check(results, {"leaks": lambda v: (len(v) == 3, "there are exactly three planted leaks"),
         "pr_auc_honest": lambda v: (v < 0.6, "an honest score this high suggests a leak is still in")})
