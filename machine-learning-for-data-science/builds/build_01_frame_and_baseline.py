# %% [markdown]
# # Build 1: Frame and baseline a PetalBank problem (60 min)
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
# ## Task 1 · Framing canvas
# In a markdown cell: the decision (how much to lend), target (`loan_amount`), unit (one application), when the prediction is made, the metric (MAE in rupees) and what a useful error would be.

# %%
results["framing_written"] = None  # TODO: True once your canvas is in the notebook

# %% [markdown]
# ## Task 2 · Split by time
# Train on applications before 2026-01-01, validate on January to June 2026.

# %%
results["n_train"] = None
results["n_valid"] = None

# %% [markdown]
# ## Task 3 · Dummy and linear baselines
# Fit `DummyRegressor(strategy="median")` and a `LinearRegression` on numeric applicant columns. Report validation MAE.

# %%
results["mae_dummy"] = None
results["mae_linear"] = None

# %% [markdown]
# ## Task 4 · Learning curves
# Plot training and validation MAE against training-set size with `learning_curve`. Is the linear model bias- or variance-limited?

# %%
results["diagnosis"] = None  # TODO: "bias" or "variance"

# %% [markdown]
# ## Check

# %%
pb.check(results, {"n_valid": lambda v: (v < 20_000, "validation should be the 2026 applications only"),
         "diagnosis": lambda v: (v in ("bias", "variance"), "answer bias or variance")})
