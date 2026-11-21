# %% [markdown]
# # Build 4: A fraud classifier with a cost-based threshold (75 min)
# Machine Learning for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter or VS Code, `# %%` cells). Data: `petalbank_card_txn` (Coral Petal synthetic,
# seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder:
# `python datasets/generate_petalbank_card_txn.py --out datasets/data/petalbank_card_txn`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalbank as pb

data = pb.card_txn()
results = {}

# %% [markdown]
# ## Task 1 · Split
# Train January to March, validate April (before the fraud ring arrives). Drop `chargeback_filed` and `days_to_report`: they are known only later.

# %%
results["n_train"] = None
results["valid_fraud_rate"] = None

# %% [markdown]
# ## Task 2 · Two models
# Logistic regression (scaled) and a decision tree or random forest. Report validation PR-AUC.

# %%
results["pr_auc"] = None  # TODO: {"logistic": ..., "tree": ...}

# %% [markdown]
# ## Task 3 · Confusion matrix at 0.5
# Show it for your better model. Why is 0.5 a poor threshold here?

# %%
results["recall_at_0_5"] = None

# %% [markdown]
# ## Task 4 · Cost-based threshold
# A missed fraud costs its amount; each alert costs ₹150 to review. Find the threshold with the lowest validation cost.

# %%
results["best_threshold"] = None
results["cost_at_best"] = None
results["cost_no_model"] = None

# %% [markdown]
# ## Check

# %%
pb.check(results, {"best_threshold": lambda v: (v < 0.5, "with rare, costly fraud the best threshold sits well below 0.5")})
