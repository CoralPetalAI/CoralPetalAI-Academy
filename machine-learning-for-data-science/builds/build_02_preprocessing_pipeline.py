# %% [markdown]
# # Build 2: A complete preprocessing pipeline (60 min)
# Machine Learning for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter or VS Code, `# %%` cells). Data: `petalbank_applicant_notes` (Coral Petal synthetic,
# seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder:
# `python datasets/generate_petalbank_applicant_notes.py --out datasets/data/petalbank_applicant_notes`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalbank as pb

data = pb.applicant_notes()
results = {}

# %% [markdown]
# ## Task 1 · Profile the columns
# Find every column with missing values and decide how to treat each. Is employment_years missing at random?

# %%
results["columns_with_missing"] = None  # TODO: sorted list of column names

# %% [markdown]
# ## Task 2 · Build the ColumnTransformer
# Numeric: median imputation with missing indicators, then scaling. Categorical: most-frequent imputation and one-hot encoding. Text: TF-IDF on `notes` (fill blanks with an empty string).

# %%
# TODO: pre = ColumnTransformer([...])
results["n_output_features"] = None  # TODO: columns after fitting on the training rows

# %% [markdown]
# ## Task 3 · Prove there is no leakage
# Write a test: fit the pipeline on training rows only, then show the imputer's median equals the training median and not the full-data median.

# %%
results["leak_test_passes"] = None  # TODO: True when your assert passes

# %% [markdown]
# ## Task 4 · Does the text help?
# Cross-validate a logistic regression with and without the TF-IDF block (time-ordered folds). Report mean PR-AUC for both.

# %%
results["pr_auc_without_notes"] = None
results["pr_auc_with_notes"] = None

# %% [markdown]
# ## Check

# %%
pb.check(results, {"columns_with_missing": lambda v: (len(v) == 4, "four columns have missing values"),
         "pr_auc_with_notes": lambda v: (0 < v < 1, "PR-AUC is between 0 and 1")})
