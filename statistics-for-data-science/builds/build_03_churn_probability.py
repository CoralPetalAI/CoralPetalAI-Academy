# %% [markdown]
# # Build 3: A probability model for churn (60 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_customers` (Coral Petal
# synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_customers.py`
#
# PetalMart's retention team wants to know whether to send an offer to customers flagged as at risk.
# Tip: draw the natural-frequency tree on paper first.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

customers = pm.customers()
customers["complained"] = (customers.complaints > 0).astype(int)
# The risk flag and the offer (given by the retention team)
CATCH_RATE = 0.80        # share of customers who go on to churn that the flag catches
FALSE_ALARM = 0.15       # share of customers who stay that the flag wrongly flags
OFFER_COST = 200         # rupees per customer who gets the offer
CUSTOMER_VALUE = 2500    # rupees of margin a customer brings in the next 12 months
SUCCESS_RATE = 0.30      # share of would-be churners the offer keeps
results = {}

# %% [markdown]
# ## Task 1 · Contingency tables (L3.2, L3.3)
# Churn against complaint, and churn against customer type, with totals (`margins=True`) and row shares.

# %%
# TODO: pd.crosstab(customers.complained, customers.churned, margins=True), then normalize="index"

# %% [markdown]
# ## Task 2 · Conditional probabilities (L3.3)
# P(churn), P(churn | complaint) and P(complaint | churn). Explain in one sentence why the last two differ.

# %%
results["p_churn"] = None  # TODO
results["p_churn_given_complaint"] = None  # TODO
results["p_complaint_given_churn"] = None  # TODO
results["why_they_differ"] = None  # TODO: one sentence

# %% [markdown]
# ## Task 3 · Bayes and natural frequencies (L3.4)
# Use the data's churn rate as the prior. Out of 1,000 customers: how many churners are flagged, how many stayers
# are flagged? Then the probability that a flagged customer actually churns.

# %%
results["natural_frequencies"] = None  # TODO: {"flagged_churners": ..., "flagged_stayers": ...} per 1,000
results["p_churn_given_flag"] = None  # TODO

# %% [markdown]
# ## Task 4 · Expected value of the offer (L3.5)
# Send the offer to every flagged customer. Expected value per flagged customer = P(churn | flag) × SUCCESS_RATE ×
# CUSTOMER_VALUE − OFFER_COST. Then the success rate at which the offer breaks even.

# %%
results["ev_per_flagged_customer"] = None  # TODO: rupees
results["break_even_success_rate"] = None  # TODO
results["decision"] = None  # TODO: send or not, the number it rests on, and the assumption that could change it

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "p_churn": pm.between(0.098, 0.100, "churned.mean() over all customers"),
    "p_churn_given_complaint": pm.between(0.150, 0.155, "churn rate among customers with at least one complaint"),
    "p_complaint_given_churn": pm.between(0.195, 0.200, "complaint rate among customers who churned"),
    "p_churn_given_flag": pm.between(0.365, 0.375, "flagged churners / all flagged customers"),
    "ev_per_flagged_customer": pm.between(75, 80, "P(churn | flag) × success × value − cost"),
    "break_even_success_rate": pm.between(0.21, 0.22, "the success rate where the expected value is zero"),
})
