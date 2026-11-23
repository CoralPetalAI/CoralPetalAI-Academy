"""Generate the Coral Petal synthetic dataset `petalmart_checkout_test`.

Used in: Build 10 (an A/B test analysis and write-up) and the lesson exercises of L10.3
and L10.5, Statistics for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- PetalMart's one-page checkout test, the clean rerun: two full weeks, Monday 1 June to
  Sunday 14 June 2026, 20,000 users who reached checkout, randomized fifty-fifty. One row
  per user.
- The numbers L10.3 narrates: 10,012 users in control and 9,988 in treatment (the split
  passes the SRM check); 6,508 control users (65.0%) and 6,672 treatment users (66.8%)
  completed a purchase.
- Revenue per user is skewed and full of zeros; it rose by about ₹11 per user, which the
  test is too small to tell apart from zero. Revenue per buyer barely moved.
- pre_revenue is each user's spend in the four weeks before the test; it correlates about
  0.5 with revenue in the test, which is what CUPED uses.
- Guardrails: payment failures, page load time and returns are unchanged.
- The lift differs a little by device, but not by more than chance would produce.

Usage:
    python generate_petalmart_checkout_test.py --out data/petalmart_checkout_test --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N = {"control": 10_012, "treatment": 9_988}
BUYERS = {"control": 6_508, "treatment": 6_672}
RPU_LIFT = 11.0  # rupees per user, treatment minus control
RPU_SD = 585.0  # spread of revenue per user, so the 95% interval is about ±16 rupees
TEST_START = pd.Timestamp("2026-06-01")
DEVICE_P = {"app": 0.50, "mobile_web": 0.35, "desktop": 0.15}
LIFT_LOGIT = {"app": 0.11, "mobile_web": 0.09, "desktop": 0.06}


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    variant = np.array(["control"] * N["control"] + ["treatment"] * N["treatment"])
    rng.shuffle(variant)
    n = len(variant)
    device = rng.choice(list(DEVICE_P), n, p=list(DEVICE_P.values()))
    ctype = np.where(rng.random(n) < 0.35, "new", "returning")
    z = rng.normal(size=n)  # how much a user tends to buy, before and during the test

    # conversion: exactly the narrated counts, drawn in proportion to each user's odds
    logit = 0.55 + 0.6 * z + np.where(ctype == "returning", 0.15, -0.25) + np.where(device == "desktop", 0.10, 0.0)
    logit = logit + (variant == "treatment") * pd.Series(device).map(LIFT_LOGIT).to_numpy()
    converted = np.zeros(n, dtype=int)
    for v in N:
        idx = np.flatnonzero(variant == v)
        keys = logit[idx] + rng.gumbel(size=len(idx))
        converted[idx[np.argsort(-keys)[:BUYERS[v]]]] = 1

    # revenue per buyer: lognormal, larger for heavy buyers
    value = np.exp(6.62 + 0.45 * z + rng.normal(0, 0.30, n))
    revenue = np.where(converted == 1, value, 0.0)
    c, t = variant == "control", variant == "treatment"
    revenue *= RPU_SD / np.sqrt((revenue[c].var() + revenue[t].var()) / 2)  # noise level L10.3 narrates
    target_t = revenue[c].mean() + RPU_LIFT
    revenue[t] *= target_t / revenue[t].mean()

    # four weeks before the test: same habits, different weeks
    pre_buy = rng.random(n) < 1 / (1 + np.exp(-(0.2 + 0.8 * z)))
    pre_value = np.exp(6.95 + 0.8 * z + rng.normal(0, 0.55, n))
    pre_revenue = np.where(pre_buy, pre_value, 0.0)

    t_df = pd.DataFrame({
        "user_id": [f"K{i:05d}" for i in range(1, n + 1)],
        "variant": variant,
        "assigned_date": (TEST_START + pd.to_timedelta(rng.integers(0, 14, n), "D")).strftime("%Y-%m-%d"),
        "device": device,
        "customer_type": ctype,
        "pre_revenue": np.round(pre_revenue, 0),
        "converted": converted,
        "revenue": np.round(revenue, 0),
        "payment_failed": (rng.random(n) < np.where(converted == 1, 0.012, 0.035)).astype(int),
        "page_load_ms": np.round(np.random.default_rng([seed, 9]).lognormal(np.log(1400), 0.35, n), 0).astype(int),
        "returned": np.where(converted == 1, (rng.random(n) < 0.06).astype(int), 0),
    })
    return {"checkout_test": t_df}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "petalmart_checkout_test"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
