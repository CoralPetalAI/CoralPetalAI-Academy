# %% [markdown]
# # Build 0: Set up and meet PetalBank (30 min)
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
# ## Task 1 · Generate and load
# Run `python datasets/generate_petalbank_loans.py --out datasets/data/petalbank_loans` once, then load the joined table.

# %%
results["n_applications"] = None  # TODO: rows in the joined table
results["n_customers"] = None  # TODO: unique customer_id
results["keys_unique"] = None  # TODO: True if application_id is unique

# %% [markdown]
# ## Task 2 · Log a dummy run to MLflow
# Start an MLflow run, log one parameter and one metric, and open `mlflow ui` to see it.

# %%
import mlflow
# TODO: with mlflow.start_run(run_name="build-0"): mlflow.log_param(...); mlflow.log_metric(...)
results["mlflow_run_id"] = None  # TODO: the run id

# %% [markdown]
# ## Task 3 · Push to GitHub
# Create your own public repository, commit this notebook (not the data folder) and push.

# %%
results["github_repo_url"] = None  # TODO

# %% [markdown]
# ## Check

# %%
pb.check(results, {"keys_unique": lambda v: (v is True, "application_id should be unique"),
         "n_applications": lambda v: (v == 80_000, "expected 80,000 applications")})
