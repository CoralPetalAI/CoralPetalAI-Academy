# %% [markdown]
# # Build 9: Forecast weekly demand (75 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_demand` (Coral Petal
# synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_demand.py`
#
# PetalMart's supply team needs a 12-week forecast of weekly orders for Fashion and Electronics.
# Build the seasonal naive forecast and the backtest loop first; then every new model is one more function.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

d = pm.demand()
daily, calendar = d["daily_orders"], d["calendar"]
weekly = (daily[daily.category.isin(["Fashion", "Electronics"])]
          .groupby([pd.Grouper(key="date", freq="W-SUN"), "category"]).orders.sum().unstack())
# weekly regressors: days in the week that are Diwali-week or sale days
regs = calendar.set_index("date")[["diwali_week", "sale_day"]].resample("W-SUN").sum()
H = 12
results = {}

# %% [markdown]
# ## Task 1 · Look and decompose (L9.1, L9.2)
# Plot both series. STL on the log of each series with period 52. Describe trend, seasonality and festival peaks.

# %%
results["description"] = None  # TODO: a few sentences per category

# %% [markdown]
# ## Task 2 · Three models (L9.3, L9.4)
# Seasonal naive (last year's same week), Holt-Winters (additive trend, multiplicative yearly season) and SARIMAX
# with the Diwali-week and sale-day regressors (try order (1,1,1) and no seasonal part; the regressors carry the
# festivals). 80% prediction intervals for each. Check the SARIMAX residuals (Ljung-Box).

# %%
def seasonal_naive(train, h=H):
    """Given as a worked example: each future week repeats the same week one year earlier."""
    return train.iloc[-52:-52 + h].to_numpy()


# TODO: holt_winters(train, h) and sarimax(train, h, exog_train, exog_future), each returning point and 80% bounds

# %% [markdown]
# ## Task 3 · Rolling-origin backtest (L9.5)
# Origins every 4 weeks from week 60 to week 144, horizon 12, so the test windows cover two festive seasons.
# For each model and category: MAE, MASE (scaled by the in-sample seasonal naive error) and 80% interval coverage.

# %%
backtest = None  # TODO: DataFrame with model, category, mae, mase, coverage_80
results["backtest"] = None  # TODO: backtest.to_dict("records")

# %% [markdown]
# ## Task 4 · Recommend and plan (L9.6)
# One method per category; the forecast for the next 12 weeks (weeks ending 6 September to 22 November 2026) with a
# planning range, and one risk that could make it wrong. The calendar already covers those weeks (Diwali 2026 is on
# 8 November), so `regs` gives you the future regressors.

# %%
results["recommended"] = None  # TODO: {"Fashion": "...", "Electronics": "..."}
results["forecast_next_12"] = None  # TODO: {"Fashion": [12 numbers], "Electronics": [12 numbers]}
results["planning_note"] = None  # TODO

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "backtest": lambda v: (len(v) >= 6 and all(0 <= r["coverage_80"] <= 1 for r in v), "three models × two categories"),
    "recommended": lambda v: (set(v) == {"Fashion", "Electronics"}, "one method per category"),
    "forecast_next_12": lambda v: (all(len(x) == 12 and min(x) > 0 for x in v.values()), "twelve positive weekly numbers each"),
})
