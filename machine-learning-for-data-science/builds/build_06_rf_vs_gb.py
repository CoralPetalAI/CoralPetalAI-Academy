# %% [markdown]
# # Build 6: Random forest vs gradient boosting for default risk (90 min)
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
# ## Task 1 · Split
# Train before 2026-01-01, test January to June 2026. Remove `collections_flag` and `loss_given_default`.

# %%
results["n_test"] = None

# %% [markdown]
# ## Task 2 · Tune both
# RandomizedSearchCV over a small grid for a random forest and for HistGradientBoosting (or LightGBM), scored by average precision on time-ordered folds.

# %%
results["best_params"] = None  # TODO: {"random_forest": {...}, "gradient_boosting": {...}}

# %% [markdown]
# ## Task 3 · Compare on test
# Report test PR-AUC for both; add early stopping to gradient boosting and note the number of trees it kept.

# %%
results["pr_auc"] = None  # TODO: {"random_forest": ..., "gradient_boosting": ...}
results["trees_kept"] = None

# %% [markdown]
# ## Task 4 · Explain the winner
# Global mean |SHAP| and one waterfall for a declined applicant.

# %%
results["shap_top3"] = None  # TODO: list of three raw column names

# %% [markdown]
# ## Check

# %%
pb.check(results, {"pr_auc": lambda v: (isinstance(v, dict) and max(v.values()) < 0.6, "PR-AUC above 0.6 suggests a leak")})
