# %% [markdown]
# # Build 3: Predict approved loan amount (75 min)
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
# ## Task 1 · Split and features
# Same time split as Build 1. Predict `loan_amount` from applicant and bureau columns known at application time.

# %%
results["n_features"] = None

# %% [markdown]
# ## Task 2 · Four models
# Linear, polynomial (degree 2 on a few columns), ridge and lasso with `RidgeCV` / `LassoCV`. Report validation MAE and RMSE.

# %%
results["mae"] = None  # TODO: {"linear": ..., "poly": ..., "ridge": ..., "lasso": ...}
results["rmse"] = None
results["lasso_alpha"] = None
results["lasso_zero_coefs"] = None  # coefficients set exactly to zero

# %% [markdown]
# ## Task 3 · Error by segment
# Validation MAE by city_tier and employment_type. Where is the model weakest?

# %%
results["weakest_segment"] = None

# %% [markdown]
# ## Task 4 · Pick and justify
# One paragraph: which model you would ship and why.

# %%
results["chosen_model"] = None

# %% [markdown]
# ## Check

# %%
pb.check(results, {"mae": lambda v: (isinstance(v, dict) and len(v) == 4, "report MAE for all four models")})
