"""Generate the Coral Petal synthetic dataset `petalmart_city_conversion`.

Used in: Build 6 (Bayesian conversion by city) and the lesson exercises of L6.2 to L6.5,
Statistics for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- PetalMart's new Home Fitness category, first four weeks: visitors to the category page
  and buyers, for each of the 40 cities PetalMart serves.
- Metros get thousands of visitors, mid-sized cities hundreds and small cities a few dozen.
- Every city has its own true conversion rate, close to 4% (most between 3% and 5%);
  buyers are binomial given visitors and that rate.
- So the raw top five is crowded with small cities that got lucky, and their estimates
  shrink most in a hierarchical model.

Usage:
    python generate_petalmart_city_conversion.py --out data/petalmart_city_conversion --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from petalmart_common import city_table

VISITORS = {1: (3000, 9000), 2: (300, 1500), 3: (25, 220)}


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    t = city_table()
    lo, hi = zip(*(VISITORS[k] for k in t.tier))
    visitors = np.round(np.exp(rng.uniform(np.log(lo), np.log(hi)))).astype(int)
    rate = 1 / (1 + np.exp(-rng.normal(np.log(0.035 / 0.965), 0.12, len(t))))
    buyers = rng.binomial(visitors, rate)
    out = pd.DataFrame({"city": t.city, "region": t.region, "tier": t.tier, "visitors": visitors, "buyers": buyers})
    return {"city_conversion": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "petalmart_city_conversion"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
