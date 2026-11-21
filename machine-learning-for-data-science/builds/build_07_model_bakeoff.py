# %% [markdown]
# # Build 7: The PetalBank model bake-off (60 min)
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
# ## Task 1 · Identical folds
# One TimeSeriesSplit(n_splits=5) on training applications, reused for every model. Sample 30,000 rows if your laptop is slow.

# %%
results["n_rows_used"] = None

# %% [markdown]
# ## Task 2 · Seven models
# Logistic regression, k-NN, naive Bayes, SVM (LinearSVC with calibration), MLP, random forest and LightGBM or HistGradientBoosting, each in the same preprocessing pipeline.

# %%
results["cv_pr_auc"] = None  # TODO: {model: mean PR-AUC}
results["fit_seconds"] = None  # TODO: {model: seconds}

# %% [markdown]
# ## Task 3 · Decision table
# A markdown table: PR-AUC, fit time, interpretability and ease of serving. Pick one.

# %%
results["chosen_model"] = None

# %% [markdown]
# ## Check

# %%
pb.check(results, {"cv_pr_auc": lambda v: (isinstance(v, dict) and len(v) == 7, "report all seven models")})
