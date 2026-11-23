# %% [markdown]
# # Portfolio Project 2: PetalMart Demand Forecast
# Statistics for Data Science · Coral Petal AI Academy
#
# Open this file in Jupyter or VS Code as a notebook (it uses `# %%` cells), or run
# `jupytext --to notebook project2_starter.py`.
#
# Fill in every `TODO`. The last cell writes `results.json`, which the autograder checks.
# Your planning note is reviewed separately. Read the ground rules in the README first.

# %%
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
DATA = (Path(__file__).resolve().parents[2] / "datasets" / "data" / "petalmart_demand" if "__file__" in globals()
        else Path("../../datasets/data/petalmart_demand"))  # run the generator first
daily = pd.read_csv(DATA / "daily_orders.csv", parse_dates=["date"])
calendar = pd.read_csv(DATA / "calendar.csv", parse_dates=["date"])
H = 12
ORIGINS = list(range(104, 145, 4))
Z80 = 1.2816
results = {}

# %% [markdown]
# ## Task 1 · Weekly series and diagnostics
# Weekly orders per category (sum over regions, W-SUN) and weekly regressors from the calendar (it runs 12 weeks
# past the orders). STL seasonal strength per category; the structural break (region, week, share before and after,
# using the mean share of the 26 weeks before the break week and the 26 weeks after it).

# %%
weekly = None  # TODO: DataFrame, one column per category, 156 rows
regs = None  # TODO: weekly diwali_week and sale_day day counts, 168 rows
results["diagnostics"] = {
    "seasonal_strength": None,  # TODO: {"Appliances": ..., ...} rounded to 3 places
    "break_region": None,  # TODO: "N", "S", "E" or "W"
    "break_week": None,  # TODO: week-ending date "YYYY-MM-DD"
    "share_before_after": None,  # TODO: [before, after]
}

# %% [markdown]
# ## Task 2 · Three methods (Module 09)
# Write `seasonal_naive(train, h)`, `ets(train, h)` and `sarimax(train, x_train, x_future)`, each returning
# (point, lower_80, upper_80) as arrays. Model on logs and exponentiate. Check the SARIMAX residuals with Ljung-Box.

# %%
# TODO: the three functions

# %% [markdown]
# ## Task 3 · The honest backtest
# For each category, origin in ORIGINS and method: MAE, MASE and 80% coverage over the 12 test weeks. Average over
# origins. The best method per category has the lowest mean MASE.

# %%
results["backtest"] = {
    "n_origins": len(ORIGINS),
    "table": None,  # TODO: list of {"category", "model", "mae", "mase", "coverage_80"}, model in seasonal_naive/ets/sarimax
    "best_model": None,  # TODO: {"Appliances": "sarimax", ...}
}

# %% [markdown]
# ## Task 4 · The forecast table
# Weeks ending 6 September to 22 November 2026, each category with its best method: point, lower_80, upper_80
# (rounded to whole orders), the 12-week total and the same 12 weeks last year.

# %%
results["forecast"] = {
    "weeks": None,  # TODO: 12 week-ending dates
    "by_category": None,  # TODO: {"Appliances": {"model", "point", "lower_80", "upper_80", "total_12_weeks",
                          #        "same_weeks_last_year"}, ...}
}

# %% [markdown]
# ## Task 5 · Planning note
# Three sentences here; the full note goes in your PDF.

# %%
results["planning_note"] = None  # TODO
Path("results.json").write_text(json.dumps(results, indent=2, default=float))
print("results.json written")
