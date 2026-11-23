"""Generate the Coral Petal synthetic dataset `petalmart_repeat`.

Used in: Build 8 (modeling repeat purchases) and the lesson exercises of L8.3 to L8.5,
Statistics for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- 15,000 customers observed from 1 October 2025 to 31 March 2026 (182 days): city, region,
  loyalty membership, primary device, customer type, the number of days each customer was
  active in the window (customers who joined late have fewer) and the orders they placed.
- Order counts are counts per customer with exposure: the expected rate per active day rises
  with loyalty membership (about 1.35 times), app use and being a returning customer.
- Customers differ far more than a Poisson model allows (overdispersion: many zeros and a long
  right tail), so a negative binomial model fits much better.
- Cities differ too: each city has its own level, so customers in the same city are more alike.

Usage:
    python generate_petalmart_repeat.py --out data/petalmart_repeat --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from petalmart_common import city_table

N = 15_000
WINDOW_DAYS = 182


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cities = city_table()
    idx = rng.choice(len(cities), N, p=cities.weight)
    city_effect = rng.normal(0, 0.18, len(cities))
    loyal = rng.random(N) < 0.30
    app = rng.random(N) < np.where(loyal, 0.68, 0.51)
    returning = rng.random(N) < 0.65
    late = rng.random(N) < 0.30
    active = np.where(late, rng.integers(14, WINDOW_DAYS, N), WINDOW_DAYS)
    log_rate = np.log(0.42) + np.log(1.35) * loyal + np.log(1.10) * app + np.log(1.25) * returning + city_effect[idx]
    mu = np.exp(log_rate) * active / 30 * rng.gamma(1.1, 1 / 1.1, N)  # gamma mixing: negative binomial counts
    t = pd.DataFrame({
        "customer_id": [f"R{i:05d}" for i in range(1, N + 1)],
        "city": cities.city.to_numpy()[idx],
        "region": cities.region.to_numpy()[idx],
        "loyalty_member": loyal.astype(int),
        "device": np.where(app, "app", "web"),
        "customer_type": np.where(returning, "returning", "new"),
        "active_days": active,
        "orders_6m": rng.poisson(mu),
    })
    return {"repeat_purchases": t}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "petalmart_repeat"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
