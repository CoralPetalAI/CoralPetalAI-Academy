"""Generate the Coral Petal synthetic dataset `petalmart_customers`.

Used in: Build 3 (a probability model for churn) and Build 7 (logistic regression for churn),
and the lesson exercises of L3.3 and L7.4, Statistics for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- 12,000 customers active on 31 March 2026, with region, primary device, tenure in months,
  customer type (new: tenure of 6 months or less), complaints filed in the last 12 months,
  orders and spend in the last 12 months.
- churned = 1 if the customer placed no order in the six months that followed.
- About one customer in ten churns. Churn odds rise by a factor of about 1.8 with each extra
  complaint, fall with tenure, and are lower for app customers, holding the others constant.
- About one customer in seven filed at least one complaint; complaints are more common among
  heavy buyers, who simply have more orders that can go wrong.

Usage:
    python generate_petalmart_customers.py --out data/petalmart_customers --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from petalmart_common import city_table

N = 12_000
B_COMPLAINT = np.log(1.8)   # odds ratio per complaint
B_TENURE = -0.025           # per month
B_APP = np.log(0.72)        # app versus web
B0 = -1.72                  # sets overall churn near 10%


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cities = city_table()
    idx = rng.choice(len(cities), N, p=cities.weight)
    tenure = np.clip(np.round(rng.gamma(1.6, 12.0, N)), 1, 72).astype(int)
    app = rng.random(N) < 0.56
    activity = rng.gamma(2.0, 1.0, N) * (1 + 0.3 * app)
    orders = 1 + rng.poisson(2.8 * activity)
    spend = np.round(orders * rng.lognormal(np.log(900), 0.35, N), 0)
    complaints = rng.poisson(0.017 * orders + 0.012)
    lo = B0 + B_COMPLAINT * complaints + B_TENURE * tenure + B_APP * app
    churned = rng.random(N) < 1 / (1 + np.exp(-lo))
    t = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, N + 1)],
        "region": cities.region.to_numpy()[idx],
        "device": np.where(app, "app", "web"),
        "tenure_months": tenure,
        "customer_type": np.where(tenure <= 6, "new", "returning"),
        "complaints": complaints,
        "orders_12m": orders,
        "spend_12m": spend,
        "churned": churned.astype(int),
    })
    return {"customers": t}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "petalmart_customers"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
