# %% [markdown]
# # Build 6: Bayesian conversion by city (75 min)
# Statistics for Data Science · Coral Petal AI Academy
#
# Open as a notebook (Jupyter, VS Code or Google Colab; `# %%` cells). Data: `petalmart_city_conversion` (Coral
# Petal synthetic, seed 2026). Fill every `TODO`; the last cell checks your answers without revealing them.
# Generate the data first, from the course folder: `python datasets/generate_petalmart_city_conversion.py`
#
# Marketing can advertise the new Home Fitness category in five cities and wants the five with the best conversion.
# Sort the cities by visitors before you do anything else.

# %%
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pmc
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
import petalmart as pm

cities = pm.city_conversion()
results = {}

# %% [markdown]
# ## Task 1 · Raw rates (L6.1)
# Conversion = buyers / visitors for each city. List the top five (ties broken by more visitors) and note their visitors.

# %%
results["raw_top5"] = None  # TODO: list of five city names, best first

# %% [markdown]
# ## Task 2 · Beta-binomial (L6.2, L6.3)
# One sensible prior for every city, for example Beta(4, 96) (about 4%, worth 100 visitors). Posterior mean and 95%
# credible interval per city; top five by posterior mean.

# %%
results["prior"] = None  # TODO: [a, b]
results["betabinom_top5"] = None  # TODO

# %% [markdown]
# ## Task 3 · Hierarchical model in PyMC (L6.5)
# City rates drawn from a shared population distribution, for example logit(rate) ~ Normal(mu, sigma). Sample with
# `pmc.sample(random_seed=2026)`, check r-hat and divergences, then the top five by posterior mean.
# Draw one chart: each city's raw rate and its hierarchical estimate, joined by a line, so the shrinkage shows.

# %%
results["max_rhat"] = None  # TODO: float(az.rhat(idata).max().to_array().max())
results["divergences"] = None  # TODO: int(idata.sample_stats.diverging.sum())
results["hierarchical_top5"] = None  # TODO

# %% [markdown]
# ## Task 4 · Recommendation
# Five cities, and a few sentences on why your list differs from the raw top five.

# %%
results["recommended_cities"] = None  # TODO
results["why_different"] = None  # TODO

# %% [markdown]
# ## Check

# %%
pm.check(results, {
    "raw_top5": pm.matches("fee82ed67859", None, "sort by buyers / visitors, highest first"),
    "max_rhat": pm.between(0.99, 1.01, "every r-hat should be below 1.01: sample longer or reparameterize"),
    "divergences": lambda v: (int(v) == 0, "no divergences" if int(v) == 0 else "try target_accept=0.95 or a non-centered model"),
    "hierarchical_top5": lambda v: (len(set(v) & set(results["raw_top5"] or [])) <= 3,
                                    "small lucky cities should shrink out of the top five"),
})
