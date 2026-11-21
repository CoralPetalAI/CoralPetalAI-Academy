# %% [markdown]
# # Portfolio Project 2: PetalBank Cross-Sell Engine
# Machine Learning for Data Science · Coral Petal AI Academy
#
# Open this file in Jupyter or VS Code as a notebook (it uses `# %%` cells), or run
# `jupytext --to notebook project2_starter.py`.
#
# Fill in every `TODO`. The last cell writes `results.json`, which the autograder checks.
# Your report is reviewed separately.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "datasets" / "data" / "petalbank_holdings" if "__file__" in globals() else Path("../../datasets/data/petalbank_holdings")  # run the generator first
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
holdings = pd.read_csv(DATA / "holdings.csv")
offers = pd.read_csv(DATA / "offers.csv")
TEST_START = "2026-04"
FORECAST_START = "2026-04-06"
results = {}

# %% [markdown]
# ## Task 1 · Framing and split
# Train on holdings with start_month < TEST_START. Test customers: joined before TEST_START
# and adopted at least one product from TEST_START on.

# %%
results["framing"] = {"n_customers": None, "n_products": None, "n_holdings": None}
results["split"] = {"test_start": TEST_START, "n_test_customers": None, "n_test_adoptions": None}

# %% [markdown]
# ## Tasks 2–3 · Recommenders with eligibility rules (Module 10)
# Build popularity, item-based CF (cosine similarity on the customer × product matrix)
# and a hybrid of your design. Filter every list to eligible, not-yet-held products.
# Report precision@5 and NDCG@5.

# %%
results["recommenders"] = {m: {"precision_at_5": None, "ndcg_at_5": None}
                           for m in ["popularity", "item_cf", "hybrid"]}
# Share of unfiltered popularity top-5 recommendations that break an eligibility rule
results["eligibility"] = {"share_ineligible_in_unfiltered_popularity_top5": None}

# %% [markdown]
# ## Task 4 · Cold start (Module 10)
# New customers (joined from TEST_START) with 2+ products: use only the profile and the first product.
# Compare global popularity with a profile-based strategy; report precision@5 for both.

# %%
results["cold_start"] = {"n_customers": None, "popularity": None, "segment_popularity": None}

# %% [markdown]
# ## Task 5 · Weekly offer-volume forecast (Module 09)
# Count offers per week. Train on weeks before FORECAST_START; forecast the 13 weeks from it.
# MASE scale = mean absolute week-to-week change in the training weeks.
# Compare a seasonal naive (same week last year) with your model.

# %%
results["forecast"] = {"weeks_test": None, "mase_seasonal_naive": None, "mase_model": None, "total_offers_test_weeks": None}

# %% [markdown]
# ## Task 6 · Recommendation
# One sentence: which recommender, how you handle new customers, and the weekly volume to plan for.

# %%
results["recommendation"] = None
Path("results.json").write_text(json.dumps(results, indent=2, default=float))
print("results.json written")
