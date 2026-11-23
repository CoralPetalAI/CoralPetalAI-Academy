# %% [markdown]
# # Build 8: Modeling repeat purchases (75 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_repeat` (Coral Petal
# synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_repeat.py`
#
# PetalMart's growth team wants to know what drives how often customers come back, to decide whether to expand the
# loyalty program. Plot the distribution of order counts first: the zeros and the right tail tell you a lot.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

repeat = pm.repeat_purchases()   # one row per customer
FORMULA = "orders_6m ~ loyalty_member + C(device) + C(customer_type)"
OFFSET = np.log(repeat.active_days)
results = {}

# %% [markdown]
# ## Task 1 · Poisson regression with exposure (L8.2, L8.3)
# `smf.glm(FORMULA, repeat, family=sm.families.Poisson(), offset=OFFSET)`. Rate ratios with CIs, and the dispersion
# ratio (Pearson chi-square / residual degrees of freedom).

# %%
results["poisson_rr_loyalty"] = None  # TODO
results["dispersion_ratio"] = None  # TODO

# %% [markdown]
# ## Task 2 · Negative binomial (L8.3)
# `smf.negativebinomial(FORMULA, repeat, offset=OFFSET).fit(method="bfgs", maxiter=1000)`. Compare the loyalty rate
# ratio and its interval with the Poisson one, and explain the change.

# %%
results["nb_rr_loyalty"] = None  # TODO
results["ci_width_poisson_vs_nb"] = None  # TODO: [Poisson CI width, NB CI width] for the loyalty rate ratio

# %% [markdown]
# ## Task 3 · A random intercept for city (L8.4)
# MixedLM on orders per 30 active days with `groups=repeat.city`. What share of the variation sits between cities?

# %%
results["between_city_share"] = None  # TODO: city variance / (city variance + residual variance)

# %% [markdown]
# ## Task 4 · Compare and recommend (L8.5)
# At least three models in one table: parameters, AIC, BIC and 5-fold cross-validated mean absolute error, each customer
# in a single fold (`KFold(5, shuffle=True, random_state=2026)` on customers). Recommend one model, and say in one
# plain sentence what loyalty membership is associated with.

# %%
comparison = None  # TODO: DataFrame with model, n_params, aic, bic, cv_mae
results["recommended_model"] = None  # TODO
results["loyalty_sentence"] = None  # TODO

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "poisson_rr_loyalty": pm.between(1.30, 1.35, "np.exp of the loyalty coefficient"),
    "dispersion_ratio": pm.between(3, 5, "well above 1: the counts are overdispersed"),
    "nb_rr_loyalty": pm.between(1.29, 1.36, "about the same rate ratio, with a wider interval"),
    "ci_width_poisson_vs_nb": lambda v: (v[1] > v[0], "the negative binomial interval should be wider"),
    "between_city_share": pm.between(0.005, 0.05, "small but not zero"),
    "recommended_model": lambda v: ("negative" in str(v).lower() or "nb" in str(v).lower(), "which model wins on AIC?"),
})
