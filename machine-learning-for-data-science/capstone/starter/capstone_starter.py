# %% [markdown]
# # Capstone: PetalBank Credit Default Model in Production
# Machine Learning for Data Science · Coral Petal AI Academy
#
# Open this file in Jupyter or VS Code as a notebook (it uses `# %%` cells), or run
# `jupytext --to notebook capstone_starter.py`.
#
# Fill in every `TODO`. The last cell writes `results.json`, which the autograder
# checks. Your FastAPI app, MLflow export, model card and slides are reviewed separately.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "datasets" / "data" / "petalbank_loans" if "__file__" in globals() else Path("../../datasets/data/petalbank_loans")  # run the generator first
applications = pd.read_csv(DATA / "applications.csv")
applicants = pd.read_csv(DATA / "applicants.csv")
bureau = pd.read_csv(DATA / "bureau.csv")
txn = pd.read_csv(DATA / "transactions_summary.csv")
outcomes = pd.read_csv(DATA / "outcomes.csv")
q3 = pd.read_csv(DATA / "applications_2026q3.csv")
CUTOFF = "2026-01-01"
COST_OF_FUNDS = 0.08
results = {}

# %% [markdown]
# ## Task 1 · Framing and baseline (Module 01)
# Join the tables into one row per application. Split by time: train before CUTOFF,
# test from CUTOFF on. Report the overall default rate and the split sizes.

# %%
df = applications  # TODO: merge applicants, bureau, transactions_summary and outcomes
results["framing"] = {"n_applications": None, "base_rate": None}  # TODO: base_rate rounded to 4 places
results["split"] = {"train_cutoff": CUTOFF, "n_train": None, "n_test": None, "test_default_rate": None}

# %% [markdown]
# ## Task 2 · Leakage audit (Modules 02, 05)
# One column in outcomes.csv is only known after the loan goes bad. Find it, show
# how it inflates PR-AUC on the test set, then remove it from every model below.

# %%
results["leakage"] = {"leaked_feature": None, "leaked_pr_auc": None}

# %% [markdown]
# ## Task 3 · Models (Modules 04, 06, 11)
# Build one leak-proof scikit-learn pipeline (imputation with missing indicators,
# encoding, scaling where needed). Compare logistic regression, random forest and
# gradient boosting with time-based CV; tune with Optuna; log runs to MLflow.
# Report test-set PR-AUC (average precision) for each.

# %%
results["pr_auc"] = {"logistic": None, "random_forest": None, "gradient_boosting": None}

# %% [markdown]
# ## Task 4 · Profit-based threshold (Module 04)
# Approve when predicted default probability < threshold. Good approved loan:
# loan_amount × (interest_rate − COST_OF_FUNDS) × term_months / 24.
# Defaulted approved loan: −loan_amount × loss_given_default. Rejected: 0.
# Search thresholds 0.01 to 0.50 in steps of 0.01 on the test set with your best model.

# %%
results["threshold"] = {"model": None, "best_threshold": None, "expected_profit_test": None,
                        "approval_rate": None, "profit_approve_all": None}

# %% [markdown]
# ## Task 5 · Explainability (Module 06)
# Global mean |SHAP| on your best tree model; list the top three features
# (use the raw column names, for example "credit_score"). Explain three
# individual applicants with reason codes in your notebook.

# %%
results["shap_top_features"] = [None, None, None]

# %% [markdown]
# ## Task 6 · Segments (Module 08)
# k-means with k = 5 on scaled credit_score, utilization_pct, log income,
# bounced_payments_6m and age. Report the highest and lowest default rate by segment.

# %%
results["segments"] = {"k": 5, "max_segment_default_rate": None, "min_segment_default_rate": None}

# %% [markdown]
# ## Task 7 · Fairness (Module 11)
# At your chosen threshold, on the test set: approval rate by age band
# (21-25, 26-35, 36-45, 46-60) and by city tier (1, 2, 3), and the approval rate
# of good borrowers (equal opportunity) by age band. Document the trade-off.

# %%
bands = ["21-25", "26-35", "36-45", "46-60"]
results["fairness"] = {
    "approval_rate_by_age_band": {b: None for b in bands},
    "approval_rate_by_city_tier": {t: None for t in ["1", "2", "3"]},
    "tpr_good_by_age_band": {b: None for b in bands},
}

# %% [markdown]
# ## Task 8 · Serving, batch scoring and drift (Module 11)
# Population stability index (10 quantile bins from the training data) for
# income_monthly and credit_score, train versus the Q3 file. Score the Q3 file in
# batch. Serve the model with FastAPI and paste one real response below.

# %%
results["drift"] = {"psi_income_q3": None, "psi_credit_score_q3": None}
results["q3_scoring"] = {"n_scored": None, "approval_rate_q3": None}
results["api_example_response"] = {"application_id": None, "default_probability": None,
                                   "decision": None, "reason_codes": [None, None]}

# %% [markdown]
# ## Task 9 · Recommendation (Modules 11, 12)
# One sentence: which model, at which threshold, and the first thing you would monitor.

# %%
results["recommendation"] = None
Path("results.json").write_text(json.dumps(results, indent=2, default=float))
print("results.json written")
