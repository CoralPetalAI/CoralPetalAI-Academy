# %% [markdown]
# # Capstone: The PetalMart Free-Shipping Decision
# Statistics for Data Science · Coral Petal AI Academy
#
# Open this file in Jupyter or VS Code as a notebook (it uses `# %%` cells), or run
# `jupytext --to notebook capstone_starter.py`.
#
# Fill in every `TODO`. The last cell writes `results.json`, which the autograder
# checks. Your memo and slides are reviewed separately.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA = Path(__file__).resolve().parents[2] / "datasets" / "data" / "petalmart_shipping_test" if "__file__" in globals() else Path("../../datasets/data/petalmart_shipping_test")  # run the generator first
users = pd.read_csv(DATA / "users.csv")
assign = pd.read_csv(DATA / "assignments.csv")
sessions = pd.read_csv(DATA / "sessions.csv")
orders = pd.read_csv(DATA / "orders.csv")
costs = pd.read_csv(DATA / "costs.csv")
rollout = pd.read_csv(DATA / "city_rollout.csv")
results = {}

# %% [markdown]
# ## Task 1 · Profile and clean (Modules 01, 02)
# Report duplicates, invalid order values and missing regions. Keep a cleaning log.

# %%
# TODO: drop exact duplicate orders, remove order_value <= 0, handle missing region
orders_clean = orders  # replace
results["cleaning"] = {
    "orders_raw": len(orders),
    "duplicates_removed": None,  # TODO
    "invalid_values_removed": None,  # TODO
    "orders_clean": None,  # TODO
    "users_missing_region": None,  # TODO
}

# %% [markdown]
# ## Task 2 · Validity checks (Modules 04, 05)
# Sample ratio mismatch (chi-square goodness of fit) and covariate balance.

# %%
u = users.merge(assign, on="user_id")
# TODO: SRM p-value on variant counts; chi-square of variant × customer_type
results["srm_p_value"] = None
results["balance_p_value_customer_type"] = None

# %% [markdown]
# ## Task 3 · Conversion and AOV (Modules 05, 10)
# Conversion = share of users with at least one clean order.
# Report the difference in percentage points with a 95% CI and p-value, and the
# AOV difference with a 5,000-resample bootstrap CI.

# %%
results["conversion_control"] = None
results["conversion_treatment"] = None
results["conversion_diff_pp"] = None
results["conversion_diff_ci_pp"] = [None, None]
results["conversion_p_value"] = None
results["aov_diff"] = None
results["aov_diff_ci"] = [None, None]

# %% [markdown]
# ## Task 4 · Bayesian view (Module 06)
# With Beta(1, 1) priors, estimate P(treatment conversion > control) for new and
# returning customers.

# %%
results["p_treatment_better_conversion"] = {"new": None, "returning": None}

# %% [markdown]
# ## Task 5 · Profit model (Module 03)
# Order profit = (order_value − discount) × gross_margin_pct + shipping_fee − shipping_cost_per_order.
# A returned order earns nothing and costs 2 × shipping_cost_per_order.
# Profit per user counts users with no orders as zero.

# %%
results["profit_per_user"] = {"control": None, "treatment": None}
results["profit_lift_per_user_all"] = None
results["profit_lift_per_user_by_segment"] = {"new": None, "returning": None}

# %% [markdown]
# ## Task 6 · Drivers (Modules 07, 08)
# Logistic regression: converted ~ variant × customer_type + device + region.
# Report the interaction coefficient (treatment × returning) and its p-value.

# %%
results["logit_interaction_coef"] = None
results["logit_interaction_p"] = None

# %% [markdown]
# ## Task 7 · Segments with multiple-testing correction (Module 05)
# Compare conversion lift across region × device segments with Benjamini-Hochberg.
# (Reviewed in your memo; not autograded.)

# %% [markdown]
# ## Task 8 · Observational cross-check (Module 11)
# Six cities launched same-day delivery in week 30 without randomization.
# Compute the naive post-period lift in orders per active user, then a
# difference-in-differences estimate with city and week fixed effects.

# %%
results["rollout_naive_lift"] = None
results["rollout_did_lift"] = None

# %% [markdown]
# ## Task 9 · Recommendation (Module 12)
# One of: "ship_to_all", "ship_to_new_users_only", "do_not_ship".

# %%
results["recommendation"] = None
Path("results.json").write_text(json.dumps(results, indent=2, default=float))
print("results.json written")
