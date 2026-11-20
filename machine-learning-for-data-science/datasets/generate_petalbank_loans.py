"""Generate the Coral Petal synthetic dataset `petalbank_loans`.

Capstone: PetalBank Credit Default Model in Production (Machine Learning for Data Science).

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Usage:
    python generate_petalbank_loans.py --out data/petalbank_loans --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_CUSTOMERS = 64_000
N_REPEAT = 16_000  # customers with a second application
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-30")
Q3_START, Q3_END = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-09-30")
N_Q3 = 9_000
TARGET_DEFAULT = 0.0626
SIGNAL = 0.81  # scales how predictable default is (sets the reachable PR-AUC)

PURPOSES = ["personal", "education", "medical", "home_improvement", "wedding", "travel", "debt_consolidation"]
PURPOSE_P = [0.30, 0.10, 0.12, 0.14, 0.10, 0.08, 0.16]
CHANNELS = ["app", "web", "branch", "partner"]
CHANNEL_P = [0.52, 0.22, 0.14, 0.12]
TIER_PIN = {1: (110001, 400099), 2: (226001, 302099), 3: (800001, 852199)}


def _expit(x):
    return 1 / (1 + np.exp(-x))


def make_customers(rng: np.random.Generator, n: int, income_shift: float = 0.0) -> pd.DataFrame:
    age = np.clip(rng.normal(34, 8.5, n).round(), 21, 60).astype(int)
    city_tier = rng.choice([1, 2, 3], size=n, p=[0.45, 0.35, 0.20])
    emp = rng.choice(["salaried", "self_employed", "contract"], size=n, p=[0.65, 0.25, 0.10])
    tier_mult = np.select([city_tier == 1, city_tier == 2], [1.25, 0.95], 0.75)
    income = np.exp(rng.normal(np.log(42_000) + income_shift, 0.55, n)) * tier_mult * (1 + 0.012 * (age - 34))
    income = np.round(np.clip(income, 12_000, 2_500_000), -2)
    emp_years = np.clip(rng.gamma(2.0, 2.2, n) * (age - 20) / 14, 0, age - 20).round(1)
    # 4% missing overall, mostly self-employed applicants
    p_miss = np.select([emp == "self_employed", emp == "contract"], [0.11, 0.04], 0.015)
    emp_years = np.where(rng.random(n) < p_miss, np.nan, emp_years)
    lo = np.array([TIER_PIN[t][0] for t in city_tier])
    hi = np.array([TIER_PIN[t][1] for t in city_tier])
    pin = (lo + (rng.random(n) * (hi - lo))).astype(int)
    housing = rng.choice(["own", "rent", "family"], size=n, p=[0.32, 0.43, 0.25])
    dependents = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.30, 0.22, 0.26, 0.15, 0.07])
    credit = np.clip(rng.normal(715, 68, n) + 2.2 * np.nan_to_num(emp_years, nan=3), 300, 900).round().astype(int)
    util = np.clip(rng.beta(2.0, 3.2, n) * 100 + (760 - credit) * 0.08, 0, 100).round(1)
    open_acc = rng.poisson(3.2, n)
    delinq = rng.poisson(np.clip((720 - credit) / 120, 0.05, None))
    inquiries = rng.poisson(1.1 + util / 60)
    salary_regular = np.where(emp == "salaried", rng.random(n) < 0.93, rng.random(n) < 0.35).astype(int)
    avg_balance = np.round(income * rng.lognormal(-0.2, 0.6, n), -2)
    bounced = rng.poisson(np.clip(0.15 + util / 120 + (1 - salary_regular) * 0.35, 0, None))
    return pd.DataFrame(
        {
            "age": age, "income_monthly": income, "employment_type": emp, "employment_years": emp_years,
            "city_tier": city_tier, "pin_code": pin, "housing": housing, "dependents": dependents,
            "credit_score": credit, "open_accounts": open_acc, "past_delinquencies": delinq,
            "utilization_pct": util, "inquiries_6m": inquiries,
            "avg_balance_3m": avg_balance, "salary_credit_regular": salary_regular, "bounced_payments_6m": bounced,
        }
    )


def make_applications(rng: np.random.Generator, cust: pd.DataFrame, start, end) -> pd.DataFrame:
    n = len(cust)
    days = (end - start).days
    date = start + pd.to_timedelta(rng.integers(0, days + 1, n), unit="D")
    term = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.12, 0.28, 0.32, 0.16, 0.12])
    amount = np.round(np.clip(cust["income_monthly"].to_numpy() * rng.lognormal(np.log(4.5), 0.5, n), 20_000, 4_000_000), -3)
    rate = np.round(np.clip(0.11 + (780 - cust["credit_score"].to_numpy()) * 0.00025 + rng.normal(0, 0.012, n), 0.105, 0.29), 4)
    return pd.DataFrame(
        {
            "application_date": date.strftime("%Y-%m-%d"), "loan_amount": amount, "term_months": term,
            "interest_rate": rate, "purpose": rng.choice(PURPOSES, size=n, p=PURPOSE_P),
            "channel": rng.choice(CHANNELS, size=n, p=CHANNEL_P),
        }
    )


def default_logit(df: pd.DataFrame) -> np.ndarray:
    r = df["interest_rate"].to_numpy() / 12
    t = df["term_months"].to_numpy()
    emi = df["loan_amount"].to_numpy() * r / (1 - (1 + r) ** -t)
    dti = emi / df["income_monthly"].to_numpy()
    inc = np.log(df["income_monthly"].to_numpy() / 42_000)
    ey = df["employment_years"].fillna(df["employment_years"].median()).to_numpy()
    cs = df["credit_score"].to_numpy()
    util = df["utilization_pct"].to_numpy()
    bounced = df["bounced_payments_6m"].to_numpy()
    self_emp = (df["employment_type"] == "self_employed").to_numpy()
    return (
        -0.006 * (cs - 715) + 0.022 * np.maximum(660 - cs, 0)  # risk climbs fast below 660
        + 0.008 * (util - 38) + 1.1 * (util > 85)  # maxed-out credit lines
        + 0.15 * bounced + 0.9 * (bounced >= 2)
        + 1.2 * np.maximum(dti - 0.30, 0) * 4 * (1 + self_emp)  # stretched EMIs, worse for irregular income
        - 0.06 * ey
        + 0.55 * np.minimum(inc, 0) ** 2 - 0.15 * np.maximum(inc, 0)  # low incomes risky, gains flatten above median
        + 0.30 * df["past_delinquencies"].to_numpy()
        + 0.25 * (df["salary_credit_regular"].to_numpy() == 0)
        + 0.6 * ((util > 70) & (bounced >= 1))  # interaction: high utilization plus a bounce
        + 0.25 * (pd.to_datetime(df["application_date"]) >= "2026-01-01").to_numpy()  # mild 2026 drift
    )


def generate(seed: int = 2026):
    rng = np.random.default_rng(seed)
    cust = make_customers(rng, N_CUSTOMERS)
    cust.insert(0, "customer_id", [f"C{i:06d}" for i in range(1, N_CUSTOMERS + 1)])
    repeat = rng.choice(N_CUSTOMERS, size=N_REPEAT, replace=False)
    rows = pd.concat([cust, cust.iloc[repeat]], ignore_index=True)
    apps = make_applications(rng, rows, START, END)
    df = pd.concat([rows.reset_index(drop=True), apps], axis=1).sort_values("application_date").reset_index(drop=True)
    df.insert(0, "application_id", [f"A{i:07d}" for i in range(1, len(df) + 1)])

    z = SIGNAL * default_logit(df)
    lo, hi = -10.0, 5.0
    for _ in range(60):  # calibrate the intercept to the target default rate
        mid = (lo + hi) / 2
        if _expit(z + mid).mean() > TARGET_DEFAULT:
            hi = mid
        else:
            lo = mid
    p = _expit(z + mid)
    default = (rng.random(len(df)) < p).astype(int)
    collections = np.where(default == 1, rng.random(len(df)) < 0.93, rng.random(len(df)) < 0.006).astype(int)
    lgd = np.where(default == 1, np.round(rng.beta(5.5, 4.5, len(df)), 3), 0.0)

    app_cols = ["application_id", "customer_id", "application_date", "loan_amount", "term_months", "interest_rate", "purpose", "channel"]
    tables = {
        "applications": df[app_cols],
        "applicants": cust[["customer_id", "age", "income_monthly", "employment_type", "employment_years", "city_tier", "pin_code", "housing", "dependents"]],
        "bureau": cust[["customer_id", "credit_score", "open_accounts", "past_delinquencies", "utilization_pct", "inquiries_6m"]],
        "transactions_summary": cust[["customer_id", "avg_balance_3m", "salary_credit_regular", "bounced_payments_6m"]],
        "outcomes": pd.DataFrame({"application_id": df["application_id"], "default_12m": default,
                                  "collections_flag": collections, "loss_given_default": lgd}),
    }

    # Q3 2026: new unlabeled applications, incomes shifted upward (salary revisions, new premium channel)
    q3 = make_customers(rng, N_Q3, income_shift=0.35)
    q3.insert(0, "customer_id", [f"N{i:06d}" for i in range(1, N_Q3 + 1)])
    q3 = pd.concat([q3, make_applications(rng, q3, Q3_START, Q3_END)], axis=1).sort_values("application_date")
    q3.insert(0, "application_id", [f"Q{i:06d}" for i in range(1, N_Q3 + 1)])
    tables["applications_2026q3"] = q3.drop(columns=["customer_id"]).reset_index(drop=True)
    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/petalbank_loans")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
