"""Generate the Coral Petal synthetic dataset `petalmart_same_day`.

Used in: Build 11 (did same-day delivery raise spend?) and the lesson exercises of
Module 11, Statistics for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- city_month: monthly spend per active customer for PetalMart's 40 cities, April 2024 to
  March 2026. Same-day delivery launched in April 2025 in six cities PetalMart chose
  (Bengaluru, Mumbai, Delhi, Hyderabad, Pune, Chennai), not at random: they are richer
  cities whose customers already spent more.
- All cities share the same seasonal swings (festive months, sales) and the same growth in
  rupees, so trends before the launch are parallel. Same-day delivery raised spend per customer by
  about ₹55 a month, reaching its full effect in the second month. A naive comparison of
  launch and other cities after the launch overstates it several times over.
- customers: 20,000 customers in launch and other cities, with tenure, device, age band,
  orders and spend in the six months before the launch (October 2024 to March 2025) and spend
  in the six months after (April to September 2025). Customers in launch cities spent more
  before the launch, so they need matching on their pre-launch behavior.
- retention_campaign: a separate, randomized test of a ₹100 win-back offer on 20,000 customers
  (half got the offer), with their recent behavior and whether they bought in the next 30
  days. The offer works best for customers who are starting to lapse, hardly at all for
  frequent buyers who would buy anyway, and not for long-lost customers.

Usage:
    python generate_petalmart_same_day.py --out data/petalmart_same_day --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from petalmart_common import SAME_DAY_CITIES, city_table

MONTHS = pd.period_range("2024-04", "2026-03", freq="M")
LAUNCH = pd.Period("2025-04", freq="M")
EFFECT = 55.0  # rupees per customer per month
SEASON = {1: 1.04, 2: 0.97, 3: 1.00, 4: 0.96, 5: 0.97, 6: 0.98, 7: 1.05, 8: 1.00, 9: 1.02, 10: 1.14, 11: 1.10, 12: 1.02}


def city_month(rng, cities):
    rows = []
    for c in cities.itertuples():
        treated = c.city in SAME_DAY_CITIES
        base = (1040 if treated else 820) * np.exp(rng.normal(0, 0.06))
        users = int((rng.integers(28_000, 60_000) if treated else rng.integers(3_000, 26_000)))
        for k, m in enumerate(MONTHS):
            post = treated and m >= LAUNCH
            ramp = 0.6 if (post and m == LAUNCH) else 1.0
            spend = base + 850 * (SEASON[m.month] - 1) + 5.5 * k + rng.normal(0, 10) + post * EFFECT * ramp
            active = int(users * (1.008 ** k) * np.exp(rng.normal(0, 0.02)))
            rows.append((c.city, c.region, m.strftime("%Y-%m"), int(treated), int(post), active, round(spend, 2)))
    return pd.DataFrame(rows, columns=["city", "region", "month", "launch_city", "same_day", "active_customers",
                                       "spend_per_customer"])


def customers(rng, cities, n=20_000):
    launch = cities[cities.city.isin(SAME_DAY_CITIES)]
    other = cities[~cities.city.isin(SAME_DAY_CITIES)]
    treated = rng.random(n) < 0.30
    city = np.where(treated, rng.choice(launch.city, n, p=launch.weight / launch.weight.sum()),
                    rng.choice(other.city, n, p=other.weight / other.weight.sum()))
    tenure = np.clip(np.round(rng.gamma(1.8, 11.0, n)), 1, 72).astype(int)
    app = rng.random(n) < np.where(treated, 0.64, 0.52)
    age = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n, p=[0.18, 0.36, 0.25, 0.13, 0.08])
    habit = rng.gamma(2.2, 1.0, n) * np.where(treated, 1.30, 1.0) * (1 + 0.004 * tenure) * np.where(app, 1.08, 1.0)
    pre_orders = rng.poisson(1.9 * habit)
    pre_spend = np.round(pre_orders * rng.lognormal(np.log(820), 0.25, n), 0)
    post_mu = 1.05 * pre_spend + 6 * tenure + 150 * app + treated * EFFECT * 6
    post_spend = np.round(np.clip(post_mu + rng.normal(0, 700, n), 0, None), 0)
    return pd.DataFrame({
        "customer_id": [f"D{i:05d}" for i in range(1, n + 1)], "city": city, "launch_city": treated.astype(int),
        "tenure_months": tenure, "device": np.where(app, "app", "web"), "age_band": age,
        "pre_orders": pre_orders, "pre_spend": pre_spend, "post_spend": post_spend,
    })


def retention_campaign(rng, n=20_000):
    days = np.round(rng.gamma(1.6, 55.0, n)).astype(int) + 1
    orders = rng.poisson(np.clip(9 * np.exp(-days / 70), 0.2, None))
    tenure = np.clip(np.round(rng.gamma(1.8, 11.0, n)), 1, 72).astype(int)
    complaints = rng.poisson(0.15, n)
    app = rng.random(n) < 0.56
    offer = rng.permutation(np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)])
    base = 1 / (1 + np.exp(-(0.9 - 0.018 * days + 0.08 * orders - 0.25 * complaints + 0.15 * app)))
    lapsing = np.exp(-((days - 100) / 45.0) ** 2)  # the offer works for customers starting to lapse
    uplift = 0.10 * lapsing + 0.01 * (days < 30)
    bought = rng.random(n) < np.clip(base + offer * uplift, 0, 1)
    return pd.DataFrame({
        "customer_id": [f"W{i:05d}" for i in range(1, n + 1)], "offer": offer, "days_since_last_order": days,
        "orders_6m": orders, "tenure_months": tenure, "complaints": complaints, "device": np.where(app, "app", "web"),
        "purchased_30d": bought.astype(int),
    })


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cities = city_table()
    return {"city_month": city_month(rng, cities), "customers": customers(rng, cities),
            "retention_campaign": retention_campaign(rng)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "petalmart_same_day"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
