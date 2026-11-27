# %% [markdown]
# # Build 7: A regression explaining revenue drivers (75 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_weekly` and
# `petalmart_customers` (Coral Petal synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers
# without revealing them. Generate the data first, from the course folder:
# `python datasets/generate_petalmart_weekly.py` and `python datasets/generate_petalmart_customers.py`
#
# PetalMart's leadership wants to know what drives weekly revenue, and what drives churn.
# Write down your goal for each model, explanation or prediction, before you fit it.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

weekly = pm.weekly()          # spend and revenue in thousands of dollars, discount in percent; season baseline = regular
customers = pm.customers()
results = {}

# %% [markdown]
# ## Task 1 · Multiple regression (L7.1, L7.2)
# Simple regression of revenue on spend, then revenue on spend, discount, season and lead_channel. Interpret every
# coefficient with units and its CI. Why did the spend coefficient change? Then add ad_impressions and compute VIFs.

# %%
results["simple_spend_coef"] = None  # TODO
results["multiple_spend_coef"] = None  # TODO: from revenue ~ spend + discount + C(season) + C(lead_channel)
results["vif_spend_with_impressions"] = None  # TODO

# %% [markdown]
# ## Task 2 · Diagnostics (L7.3)
# Residuals against fitted values (with a lowess line), a QQ plot, residuals in time order and Cook's distance.
# Investigate the most influential week (read its note). Fix what you find: robust (HC3) standard errors or a log
# transform, and report the model with and without the special week. Say what changed.

# %%
results["funnel_in_residuals"] = None  # TODO: True or False
results["most_influential_week"] = None  # TODO: week_start as "YYYY-MM-DD"
results["spend_se_classic_vs_hc3"] = None  # TODO: [classic, HC3]

# %% [markdown]
# ## Task 3 · Logistic regression for churn (L7.4)
# churned ~ complaints + tenure_months + C(device, Treatment("web")). Odds ratios with CIs; plot predicted churn
# probability against complaints.

# %%
results["or_complaints"] = None  # TODO: odds ratio per extra complaint
results["or_app_vs_web"] = None  # TODO

# %% [markdown]
# ## Task 4 · What the data says
# Five one-sentence bullets in plain business language, and one bullet on what the data cannot tell us.

# %%
results["bullets"] = None  # TODO: list of six strings

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "simple_spend_coef": pm.matches("3135d2d71bff", 1, "smf.ols('revenue ~ spend', weekly)"),
    "multiple_spend_coef": pm.between(1.9, 2.3, "holding season and discount constant, the spend effect shrinks"),
    "vif_spend_with_impressions": pm.between(10, 1000, "spend and impressions move together: the VIF should be large"),
    "funnel_in_residuals": lambda v: (v is True, "plot residuals against fitted values; busy weeks are noisier"),
    "most_influential_week": pm.matches("7335fecd9ad0", None, "model.get_influence().cooks_distance"),
    "or_complaints": pm.matches("a97b042d7bd5", 1, "np.exp of the complaints coefficient"),
    "or_app_vs_web": pm.matches("91c20a20ec3b", 2, "web as the reference: np.exp of the app coefficient"),
    "bullets": lambda v: (isinstance(v, list) and len(v) == 6, "five findings plus one limit"),
})
