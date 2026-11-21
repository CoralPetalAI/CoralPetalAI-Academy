# %% [markdown]
# # Build 9: Forecast loan applications for 300 branches (75 min)
# Machine Learning for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter or VS Code, `# %%` cells). Data: `petalbank_branch_applications` (Coral Petal synthetic,
# seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder:
# `python datasets/generate_petalbank_branch_applications.py --out datasets/data/petalbank_branch_applications`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalbank as pb

data = pb.branch_applications()
results = {}

# %% [markdown]
# ## Task 1 · Weekly series
# Sum daily applications to weeks per branch. Hold out the last 13 weeks.

# %%
results["n_branches"] = None
results["test_weeks"] = None

# %% [markdown]
# ## Task 2 · Features
# Lags (1, 2, 4, 52 weeks), rolling means, week of year, festive and month-end flags from calendar.csv, branch type.

# %%
results["n_features"] = None

# %% [markdown]
# ## Task 3 · Global model
# One LightGBM or HistGradientBoosting model across all branches. Compare with seasonal naive and with SARIMA on 10 sample branches (Statistics for Data Science, Module 09).

# %%
results["mase"] = None  # TODO: {"seasonal_naive": ..., "global_model": ..., "sarima_10_branches": ...}

# %% [markdown]
# ## Task 4 · By branch type
# MASE by branch type. Where would you not trust the forecast?

# %%
results["mase_by_type"] = None

# %% [markdown]
# ## Check

# %%
pb.check(results, {"mase": lambda v: (isinstance(v, dict) and v.get("global_model", 9) < v.get("seasonal_naive", 0), "the global model should beat seasonal naive")})
