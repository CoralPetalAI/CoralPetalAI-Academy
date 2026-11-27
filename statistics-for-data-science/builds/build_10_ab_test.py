# %% [markdown]
# # Build 10: An A/B test analysis and write-up (60 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_checkout_test` (Coral Petal
# synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_checkout_test.py`
#
# The product team has the results of the one-page checkout test and needs a recommendation.
# Run the SRM check before anything else; if it fails, stop and explain why.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

rng = np.random.default_rng(2026)
test = pm.checkout_test()   # one row per user who reached checkout; revenue in dollars during the two-week test
CHECKOUT_USERS_PER_MONTH = 200_000
MARGIN_RATE = 0.23          # gross margin on an extra order, for the memo
results = {}

# %% [markdown]
# ## Task 1 · Sample ratio mismatch (L5.4, L10.3)
# Chi-square goodness of fit of the two group sizes against a fifty-fifty plan. Can the analysis proceed?

# %%
results["srm_p"] = None  # TODO

# %% [markdown]
# ## Task 2 · Conversion (L10.3)
# Two-proportion z-test; absolute lift in percentage points with a 95% CI; relative lift with a bootstrap 95% CI.

# %%
results["conv_control"] = None  # TODO
results["conv_treatment"] = None  # TODO
results["abs_lift_pp"] = None  # TODO: percentage points
results["abs_lift_ci_pp"] = None  # TODO: [low, high]
results["rel_lift_ci"] = None  # TODO: [low, high] from the bootstrap, as proportions

# %% [markdown]
# ## Task 3 · Revenue per user (L10.3)
# Welch's t-test and a bootstrap CI for the difference; decompose revenue per user into conversion × revenue per
# buyer. Then CUPED with `pre_revenue` (theta from the pooled data) and compare the two intervals.

# %%
results["rpu_diff"] = None  # TODO: dollars per user, treatment minus control
results["rpu_ci"] = None  # TODO: Welch [low, high]
results["rpu_ci_cuped"] = None  # TODO: [low, high]

# %% [markdown]
# ## Task 4 · Guardrails and segments (L10.2, L10.4)
# Revenue per buyer, payment failures, page load time and returns: any harm? Then conversion ~ variant × device and
# variant × customer_type. Which findings are confirmed, and which are only hypotheses?

# %%
results["guardrails_harmed"] = None  # TODO: list of harmed guardrails, [] if none
results["device_interaction_min_p"] = None  # TODO: smallest p-value of the variant × device terms

# %% [markdown]
# ## Task 5 · The memo (L10.5, at most 200 words)
# Lead with the recommendation; translate the lift into extra margin per month at the point estimate and at both ends
# of the interval (CHECKOUT_USERS_PER_MONTH × lift × revenue per buyer × MARGIN_RATE); state the rollout plan.

# %%
results["monthly_margin"] = None  # TODO: [low, point, high] in dollars
results["memo"] = None  # TODO

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "srm_p": pm.between(0.5, 1.0, "stats.chisquare(observed counts, [n/2, n/2])"),
    "conv_control": pm.matches("4942870d7312", 3, "buyers / users in control"),
    "conv_treatment": pm.matches("59b2006806ee", 3, "buyers / users in treatment"),
    "abs_lift_pp": pm.matches("a97b042d7bd5", 1, "in percentage points: 100 × (treatment − control)"),
    "rpu_diff": pm.matches("b05e244762b1", 1, "mean revenue per user, treatment minus control, zeros included"),
    "rpu_ci_cuped": lambda v: (v[1] - v[0] < (results["rpu_ci"][1] - results["rpu_ci"][0]), "CUPED should narrow the interval"),
    "guardrails_harmed": lambda v: (v == [], "test each guardrail: none moved beyond chance"),
    "device_interaction_min_p": pm.between(0.05, 1, "segment differences here are hypotheses, not findings"),
})
