# %% [markdown]
# # Portfolio Project 1: PetalMart Customer Survey Study
# Statistics for Data Science · Coral Petal AI Academy
#
# Open this file in Jupyter or VS Code as a notebook (it uses `# %%` cells), or run
# `jupytext --to notebook project1_starter.py`.
#
# Fill in every `TODO`. The last cell writes `results.json`, which the autograder checks.
# Your cleaning log and report are reviewed separately. Read the ground rules in the README first.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import DescrStatsW

DATA = (Path(__file__).resolve().parents[2] / "datasets" / "data" / "petalmart_survey" if "__file__" in globals()
        else Path("../../datasets/data/petalmart_survey"))  # run the generator first
raw = pd.read_csv(DATA / "responses.csv")
margins = pd.read_csv(DATA / "population_margins.csv")
RATED = [c for c in raw.columns if c[:1] == "q" and 1 <= int(c[1:3]) <= 17]     # q01 to q17
LIKERT_16 = RATED[:16]                                                           # q01 to q16
GROUPS = ["region", "age_band", "customer_type", "device", "spend_band"]
FAST_SECONDS = 150
results = {}

# %% [markdown]
# ## Task 1 · Cleaning and a cleaning log (Module 02)
# Profile first: types, missing shares per item, the distribution of duration_sec. Then remove the responses that
# answer every rated item the same way AND took under FAST_SECONDS. Keep a log row for every step.

# %%
clean = None  # TODO
results["cleaning"] = {"n_raw": int(len(raw)), "n_straightliners_removed": None, "n_clean": None,
                       "max_item_missing_share": None}  # TODO: largest share of missing answers among q01 to q17

# %% [markdown]
# ## Task 2 · Weighted versus unweighted (Modules 04, 05)
# Weight = population share of the age band / its share of the cleaned responses. Weighted mean and 95% CI with
# `DescrStatsW(x, weights=w * len(w) / w.sum()).tconfint_mean()`, dropping skipped answers first.
# Why does weighting lower app ease more than satisfaction?

# %%
results["weighting"] = {
    "age_weights": None,  # TODO: {"18-24": ..., ..., "55+": ...}
    "app_ease_unweighted": None, "app_ease_weighted": None, "app_ease_weighted_ci": None,  # TODO: q10
    "satisfaction_unweighted": None, "satisfaction_weighted": None,  # TODO: q17
}

# %% [markdown]
# ## Task 3 · Four planned tests with effect sizes (Module 05)
# 1. q01 delivery speed by region: one-way ANOVA, eta squared, the lowest region and its mean.
# 2. q06 value for money, returning minus new: Welch's t-test, mean difference, Cohen's d (pooled SD).
# 3. q19 preferred payment by region: chi-square test of independence, Cramér's V, degrees of freedom.
# 4. q10 app ease by age band: Kruskal-Wallis, epsilon squared = H / (n − 1).

# %%
results["tests"] = {
    "delivery_speed_by_region": {"anova_p": None, "eta_squared": None, "lowest_region": None, "lowest_mean": None},
    "value_returning_vs_new": {"welch_p": None, "mean_diff": None, "cohens_d": None},
    "payment_by_region": {"chi2_p": None, "cramers_v": None, "dof": None},
    "app_ease_by_age": {"kruskal_p": None, "epsilon_squared": None},
}

# %% [markdown]
# ## Task 4 · The family of 80 tests (Module 05)
# Kruskal-Wallis for every item in LIKERT_16 against every group in GROUPS. Count p < 0.05, then apply
# Benjamini-Hochberg at 5% (`multipletests(p, method="fdr_bh")`) and list the survivors as "q01_delivery_speed × region".

# %%
results["multiple_testing"] = {"n_tests": None, "n_significant_raw": None, "n_significant_bh": None,
                               "bh_survivors": None}

# %% [markdown]
# ## Task 5 · Findings
# Three to five sentences for leadership: what differs, by how much, how sure you are, and one limit of the survey.

# %%
results["findings"] = None  # TODO
Path("results.json").write_text(json.dumps(results, indent=2, default=float))
print("results.json written")
