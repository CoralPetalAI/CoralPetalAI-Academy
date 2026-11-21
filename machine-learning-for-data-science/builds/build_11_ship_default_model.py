# %% [markdown]
# # Build 11: Ship the default model (75 min)
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
# ## Task 1 · Register
# Train your Build 6 pipeline, log it to MLflow and register it as `petalbank-default`.

# %%
results["model_version"] = None

# %% [markdown]
# ## Task 2 · Serve
# A FastAPI app with POST /score that takes one application as JSON (Pydantic model) and returns probability, decision and two reason codes. Paste one real response.

# %%
results["api_response"] = None

# %% [markdown]
# ## Task 3 · Batch-score a new month
# Score applications_2026q3.csv (loans_q3()). Compute PSI for income and credit score against training.

# %%
results["psi"] = None  # TODO: {"income_monthly": ..., "credit_score": ...}

# %% [markdown]
# ## Task 4 · Fairness
# Approval rate and approval rate of good borrowers by age band at your threshold.

# %%
results["approval_by_age_band"] = None

# %% [markdown]
# ## Task 5 · Model card
# Write MODEL_CARD.md in your repository: intended use, data, metrics, fairness, limits, monitoring.

# %%
results["model_card_written"] = None

# %% [markdown]
# ## Check

# %%
pb.check(results, {"psi": lambda v: (isinstance(v, dict) and v.get("income_monthly", 0) > v.get("credit_score", 1), "income should drift more than credit score")})
