"""Generate the Coral Petal synthetic dataset `petalbank_applicant_notes`.

Used in: Build 2 (a complete preprocessing pipeline), Machine Learning for Data Science.

Data source: fully synthetic. It draws 20,000 applications from the `petalbank_loans`
generator (same seed, same rules) and adds an underwriter's free-text note to each, so
every learner gets identical data. Coral Petal AI owns it; no real customer data is used.

Rules:
- One flat table: numeric, categorical and text columns plus the target `default_12m`.
- Missing values on purpose: employment_years (not at random, mostly self-employed),
  income_monthly (3%), housing (2%), notes (5% blank).
- Notes mix neutral phrases with phrases that carry real signal (salary delays, EMI
  holiday requests, many recent enquiries) and phrases that sound alarming but carry none.

Usage:
    python generate_petalbank_applicant_notes.py --out data/petalbank_applicant_notes --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generate_petalbank_loans import generate as generate_loans

N = 20_000
NEUTRAL = ["documents verified", "applied through the app", "existing savings customer", "KYC complete",
           "address confirmed by video call", "requested a callback in the evening", "co-applicant not required",
           "prefers communication in Hindi", "prefers communication in Tamil", "salary account with another bank"]
RISKY = ["salary credited late twice", "asked about an EMI holiday", "several loan enquiries this month",
         "existing EMIs on two other loans", "employer could not be verified", "cash deposits only",
         "requested higher amount after rejection elsewhere", "irregular business income"]
SAFE = ["stable employer for many years", "long relationship with PetalBank", "salary credited on time every month",
        "owns the house", "low card utilization", "prepaid an earlier loan"]
RED_HERRING = ["urgent requirement", "medical expense", "wedding in the family", "called the branch twice"]


def note_for(rng, risk: float) -> str:
    parts = list(rng.choice(NEUTRAL, rng.integers(1, 3), replace=False))
    if rng.random() < 0.15 + 0.6 * risk:
        parts.append(rng.choice(RISKY))
    if rng.random() < 0.45 * (1 - risk):
        parts.append(rng.choice(SAFE))
    if rng.random() < 0.15:
        parts.append(rng.choice(RED_HERRING))
    rng.shuffle(parts)
    return "; ".join(parts).capitalize() + "."


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    t = generate_loans(seed)
    df = (t["applications"].merge(t["applicants"], on="customer_id").merge(t["bureau"], on="customer_id")
          .merge(t["transactions_summary"], on="customer_id").merge(t["outcomes"], on="application_id"))
    rng = np.random.default_rng(seed + 2)
    df = df.sample(N, random_state=seed + 2).sort_values("application_date").reset_index(drop=True)
    # Risk signal for the notes: default itself mixed with noise, so text helps but is not a leak
    risk = np.clip(0.55 * df["default_12m"] + 0.25 * (df["bounced_payments_6m"] >= 1) + rng.normal(0.1, 0.15, N), 0, 1)
    df["notes"] = [note_for(rng, r) for r in risk]
    df.loc[rng.random(N) < 0.05, "notes"] = ""
    df.loc[rng.random(N) < 0.03, "income_monthly"] = np.nan
    df.loc[rng.random(N) < 0.02, "housing"] = np.nan
    cols = ["application_id", "application_date", "loan_amount", "term_months", "interest_rate", "purpose", "channel",
            "age", "income_monthly", "employment_type", "employment_years", "city_tier", "housing", "dependents",
            "credit_score", "utilization_pct", "inquiries_6m", "bounced_payments_6m", "notes", "default_12m"]
    return {"applications_with_notes": df[cols]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/petalbank_applicant_notes")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, tbl in generate(a.seed).items():
        tbl.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(tbl):,} rows")


if __name__ == "__main__":
    main()
