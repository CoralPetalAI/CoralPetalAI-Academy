# %% [markdown]
# # Portfolio Project 1: PetalBank Card Fraud Detection
# Machine Learning for Data Science · Coral Petal AI Academy
#
# Open this file in Jupyter or VS Code as a notebook (it uses `# %%` cells), or run
# `jupytext --to notebook project1_starter.py`.
#
# Fill in every `TODO`. The last cell writes `results.json`, which the autograder checks.
# Your report is reviewed separately.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "datasets" / "data" / "petalbank_card_txn" if "__file__" in globals() else Path("../../datasets/data/petalbank_card_txn")  # run the generator first
cards = pd.read_csv(DATA / "cards.csv")
txn = pd.read_csv(DATA / "transactions.csv")
labels = pd.read_csv(DATA / "fraud_labels.csv")
TEST_START = "2026-05-01"
REVIEW_COST = 150
CAPACITY_PER_DAY = 360
results = {}

# %% [markdown]
# ## Task 1 · Framing and split (Module 01)
# Join the tables, sort by time, and split: train before TEST_START, test from TEST_START on.

# %%
results["framing"] = {"n_transactions": None, "fraud_rate": None}  # TODO: fraud_rate rounded to 4 places
results["split"] = {"test_start": TEST_START, "n_train": None, "n_test": None, "test_fraud_rate": None}

# %% [markdown]
# ## Task 2 · Leakage audit (Modules 02, 05)
# One column is only known weeks after the transaction. Name it, show what it does to
# test PR-AUC, and keep it out of every model below.

# %%
results["leakage"] = {"leaked_feature": None, "leaked_pr_auc": None}

# %% [markdown]
# ## Task 3 · Models and metrics (Modules 04, 05)
# One leak-proof pipeline. Logistic regression and gradient boosting (or a random forest).
# Mean PR-AUC over a 4-fold TimeSeriesSplit on the training months, then PR-AUC on the test months.
# If the two disagree, say why in your report.

# %%
results["cv_pr_auc"] = {"logistic": None, "gradient_boosting": None}
results["pr_auc_test"] = {"logistic": None, "gradient_boosting": None}

# %% [markdown]
# ## Task 4 · Cost-based threshold under capacity (Module 04)
# Cost = sum of amounts of unflagged fraud + REVIEW_COST × number of alerts.
# Choose the threshold with the lowest test cost among those that keep average alerts
# per day at or below CAPACITY_PER_DAY. Search finely: fraud probabilities are small.

# %%
results["threshold"] = {"model": None, "threshold": None, "alerts_per_day": None, "expected_cost_test": None,
                        "cost_no_model": None, "recall": None, "precision": None}

# %% [markdown]
# ## Task 5 · Error analysis (Module 05)
# Group the missed frauds. One group of merchants behaves very differently in May and June.
# Report recall on that group and on all other fraud, and how many merchants are in it.

# %%
results["error_analysis"] = {"recall_ring_fraud": None, "recall_other_fraud": None, "ring_merchants": None}

# %% [markdown]
# ## Task 6 · Recommendation
# One sentence: which model, which threshold, and what you would change next.

# %%
results["recommendation"] = None
Path("results.json").write_text(json.dumps(results, indent=2, default=float))
print("results.json written")
