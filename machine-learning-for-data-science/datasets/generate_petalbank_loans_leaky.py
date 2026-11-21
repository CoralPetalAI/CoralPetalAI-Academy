"""Generate the Coral Petal synthetic dataset `petalbank_loans_leaky`.

Used in: Build 5 (leak-proof pipeline), Machine Learning for Data Science.

Data source: fully synthetic. It draws 30,000 applications from the `petalbank_loans`
generator (same seed, same rules) and plants three leaks, so every learner gets
identical data. Coral Petal AI owns it; no real customer data is used.

The three planted leaks (do not read this before the build if you want the challenge):
1. `collections_flag`: set only after a loan goes bad (target leakage).
2. `emis_paid_at_review`: number of EMIs paid by the 12-month review, known only after
   the outcome window (time-travel leakage).
3. `pin_default_rate`: default rate by pin-code area computed on the whole file,
   including each row's own label (leakage through preprocessing; tiny groups make it worse).
Repeat customers also appear in both early and late applications, so a random split
overstates performance compared with a time split.

Usage:
    python generate_petalbank_loans_leaky.py --out data/petalbank_loans_leaky --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generate_petalbank_loans import generate as generate_loans

N = 30_000


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    t = generate_loans(seed)
    df = (t["applications"].merge(t["applicants"], on="customer_id").merge(t["bureau"], on="customer_id")
          .merge(t["transactions_summary"], on="customer_id").merge(t["outcomes"], on="application_id"))
    rng = np.random.default_rng(seed + 5)
    df = df.sample(N, random_state=seed + 5).sort_values("application_date").reset_index(drop=True)
    paid = np.minimum(df["term_months"], 12)
    df["emis_paid_at_review"] = np.where(df["default_12m"] == 1, rng.integers(0, 9, N), paid - (rng.random(N) < 0.05))
    area = (df["pin_code"] // 100).astype(str)  # small areas: a few applications each
    df["pin_area"] = area
    df["pin_default_rate"] = df.groupby("pin_area")["default_12m"].transform("mean").round(4)
    cols = ["application_id", "customer_id", "application_date", "loan_amount", "term_months", "interest_rate", "purpose",
            "channel", "age", "income_monthly", "employment_type", "employment_years", "city_tier", "pin_area",
            "housing", "dependents", "credit_score", "open_accounts", "past_delinquencies", "utilization_pct",
            "inquiries_6m", "avg_balance_3m", "salary_credit_regular", "bounced_payments_6m",
            "pin_default_rate", "emis_paid_at_review", "collections_flag", "default_12m"]
    return {"loans": df[cols]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/petalbank_loans_leaky")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, tbl in generate(a.seed).items():
        tbl.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(tbl):,} rows")


if __name__ == "__main__":
    main()
