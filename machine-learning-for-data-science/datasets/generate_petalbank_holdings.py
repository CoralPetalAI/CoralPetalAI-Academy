"""Generate the Coral Petal synthetic dataset `petalbank_holdings`.

Used in: Build 10 (PetalBank next-best product) and Portfolio Project 2 (PetalBank
Cross-Sell Engine), Machine Learning for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- 120,000 customers, 14 products, 18 months (January 2025 to June 2026).
- Five hidden customer types (young salaried, young family, affluent, self-employed,
  senior) drive product affinity, so collaborative filtering beats popularity.
- Products follow natural paths (savings → credit card → personal loan; salary account
  → term deposit → mutual fund SIP) and have eligibility rules (age, income, credit
  score, a required product).
- 9,000 customers join in the last 3 months with at most one product (cold start).
- Weekly offer volume has a trend, a quarter-end push and a festive peak in
  October–November, for the offer-volume forecast.

Usage:
    python generate_petalbank_holdings.py --out data/petalbank_holdings --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_CUST = 120_000
N_NEW = 9_000  # joined April–June 2026
MONTHS = pd.period_range("2025-01", "2026-06", freq="M")

PRODUCTS = pd.DataFrame([
    # product_id, name, family, min_age, min_income, min_credit_score, requires
    ("P01", "savings_account", "deposits", 18, 0, 0, ""),
    ("P02", "salary_account", "deposits", 21, 15_000, 0, ""),
    ("P03", "term_deposit", "deposits", 18, 0, 0, "P01|P02"),
    ("P04", "recurring_deposit", "deposits", 18, 0, 0, "P01|P02"),
    ("P05", "credit_card", "cards", 21, 25_000, 680, ""),
    ("P06", "premium_credit_card", "cards", 25, 150_000, 760, "P05"),
    ("P07", "personal_loan", "loans", 23, 25_000, 700, ""),
    ("P08", "home_loan", "loans", 25, 60_000, 720, ""),
    ("P09", "car_loan", "loans", 23, 40_000, 700, ""),
    ("P10", "education_loan", "loans", 18, 0, 650, ""),
    ("P11", "mutual_fund_sip", "investments", 21, 30_000, 0, "P01|P02"),
    ("P12", "life_insurance", "insurance", 21, 20_000, 0, ""),
    ("P13", "health_insurance", "insurance", 18, 15_000, 0, ""),
    ("P14", "demat_account", "investments", 21, 40_000, 0, "P01|P02"),
], columns=["product_id", "name", "family", "min_age", "min_income", "min_credit_score", "requires"])

TYPES = ["young_salaried", "young_family", "affluent", "self_employed", "senior"]
TYPE_P = [0.30, 0.24, 0.12, 0.18, 0.16]
# Monthly adoption propensity (per 1,000) by type for P01..P14
AFFINITY = np.array([
    # P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14
    [3, 30, 3, 8, 22, 1, 10, 1, 4, 5, 12, 3, 6, 10],     # young_salaried
    [3, 20, 6, 10, 12, 1, 9, 12, 8, 1, 6, 14, 15, 3],    # young_family
    [3, 15, 14, 2, 8, 16, 2, 6, 6, 1, 18, 8, 8, 16],     # affluent
    [12, 2, 6, 9, 9, 2, 12, 4, 5, 1, 5, 5, 9, 6],        # self_employed
    [8, 3, 22, 6, 3, 1, 1, 1, 1, 0, 4, 5, 16, 3],        # senior
], float) / 1000
AFFINITY = AFFINITY ** 1.7 / (AFFINITY ** 1.7).mean(1, keepdims=True) * AFFINITY.mean(1, keepdims=True)  # sharpen type taste
# Holding one product lifts the next on the natural paths
PATH = {"P01": ["P05", "P04"], "P02": ["P05", "P11", "P03"], "P05": ["P07", "P06"],
        "P03": ["P11", "P14"], "P11": ["P14"], "P08": ["P13", "P12"], "P12": ["P13"]}


def eligible(prod: pd.Series, age, income, score, held: np.ndarray, pid_index: dict) -> np.ndarray:
    ok = (age >= prod["min_age"]) & (income >= prod["min_income"]) & (score >= prod["min_credit_score"])
    if prod["requires"]:
        need = np.zeros(len(age), bool)
        for r in prod["requires"].split("|"):
            need |= held[:, pid_index[r]]
        ok &= need
    return ok


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = N_CUST
    t = rng.choice(len(TYPES), n, p=TYPE_P)
    age = np.select([t == 0, t == 1, t == 2, t == 3, t == 4],
                    [rng.normal(27, 3, n), rng.normal(34, 4, n), rng.normal(44, 8, n), rng.normal(40, 9, n),
                     rng.normal(64, 5, n)])
    age = np.clip(age.round(), 18, 80).astype(int)
    inc_mu = np.log(np.array([45_000, 60_000, 220_000, 55_000, 40_000])[t])
    income = np.round(np.exp(rng.normal(inc_mu, 0.45)), -2)
    score = np.clip(rng.normal(np.array([720, 735, 785, 700, 760])[t], 45), 300, 900).round().astype(int)
    tier = rng.choice([1, 2, 3], n, p=[0.45, 0.35, 0.20])
    # Join month: existing book before 2025, then steady onboarding, last 9,000 in Apr–Jun 2026
    join = np.full(n, -1)
    j = rng.choice(n, 30_000 + N_NEW, replace=False)
    join[j[:30_000]] = rng.integers(0, 15, 30_000)
    join[j[30_000:]] = rng.integers(15, 18, N_NEW)
    channel = rng.choice(["app", "branch", "partner", "web"], n, p=[0.5, 0.25, 0.1, 0.15])
    customers = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)], "age": age, "income_monthly": income,
        "credit_score": score, "city_tier": tier, "acquisition_channel": channel,
        "join_month": [str(MONTHS[m]) if m >= 0 else "2024-12" for m in join],
        "occupation": np.where(t == 3, "self_employed", np.where(t == 4, rng.choice(["retired", "salaried"], n, p=[0.7, 0.3]), "salaried")),
        "dependents": np.clip(rng.poisson(np.array([0.3, 1.8, 1.5, 1.4, 0.4])[t]), 0, 5),
    })

    P = len(PRODUCTS)
    # Persistent personal taste by product family (what makes co-holding informative)
    fam = PRODUCTS["family"].astype("category").cat.codes.to_numpy()
    taste = rng.gamma(0.7, 1 / 0.7, (n, fam.max() + 1))[:, fam]
    pid_index = {p: i for i, p in enumerate(PRODUCTS["product_id"])}
    held = np.zeros((n, P), bool)
    start = np.full((n, P), -2)
    # Opening book: everyone has a savings or salary account; existing customers hold more
    first = np.where((t <= 1) & (rng.random(n) < 0.7), pid_index["P02"], pid_index["P01"])
    held[np.arange(n), first] = True
    start[np.arange(n), first] = np.maximum(join, -1)
    existing = join < 0
    for _ in range(6):  # simulate pre-2025 history
        lift = np.ones((n, P))
        for src, dst in PATH.items():
            for d in dst:
                lift[:, pid_index[d]] += 4.0 * held[:, pid_index[src]]
        p = AFFINITY[t] * lift * taste * 3
        for k in range(P):
            ok = existing & ~held[:, k] & eligible(PRODUCTS.iloc[k], age, income, score, held, pid_index)
            new = ok & (rng.random(n) < p[:, k])
            held[new, k] = True
            start[new, k] = -1

    hold_rows, offers = [], []
    for m in range(len(MONTHS)):
        active = join <= m
        lift = np.ones((n, P))
        for src, dst in PATH.items():
            for d in dst:
                lift[:, pid_index[d]] += 4.0 * held[:, pid_index[src]]
        festive = 1.35 if MONTHS[m].month in (10, 11) else 1.0
        p = AFFINITY[t] * lift * taste * festive * np.where(join[:, None] >= 15, 0.15, 1.0)
        for k in range(P):
            ok = active & ~held[:, k] & eligible(PRODUCTS.iloc[k], age, income, score, held, pid_index)
            new = ok & (rng.random(n) < p[:, k])
            held[new, k] = True
            start[new, k] = m
    ci, ki = np.nonzero(held)
    sm = start[ci, ki]
    holdings = pd.DataFrame({
        "customer_id": customers["customer_id"].to_numpy()[ci], "product_id": PRODUCTS["product_id"].to_numpy()[ki],
        "start_month": np.where(sm < 0, "2024-12", MONTHS.astype(str).to_numpy()[np.maximum(sm, 0)]),
    }).sort_values(["start_month", "customer_id", "product_id"]).reset_index(drop=True)
    # Some products closed (term deposits mature, loans repaid)
    closes = rng.random(len(holdings)) < np.where(holdings["product_id"].isin(["P03", "P04", "P07", "P09"]), 0.25, 0.04)
    end = pd.PeriodIndex(holdings["start_month"], freq="M") + rng.integers(3, 15, len(holdings))
    holdings["end_month"] = np.where(closes & (end <= MONTHS[-1]), end.astype(str), "")

    # Weekly offers: trend + quarter-end push + festive peak + noise; acceptance from affinity
    weeks = pd.date_range("2025-01-06", "2026-06-29", freq="W-MON")
    wk = np.arange(len(weeks))
    base = 6_000 + 25 * wk
    qend = np.isin(weeks.month, [3, 6, 9, 12]) & (weeks.day >= 15)
    fest = np.isin(weeks.month, [10, 11])
    vol = np.round(base * (1 + 0.18 * qend + 0.35 * fest) * rng.normal(1, 0.06, len(wk))).astype(int)
    for w, v in zip(weeks, vol):
        c = rng.integers(0, n, v)
        k = rng.integers(0, P, v)
        acc_p = np.clip(AFFINITY[t[c], k] * 6 + 0.01, 0, 0.6)
        offers.append(pd.DataFrame({"sent_date": w.strftime("%Y-%m-%d"), "customer_id": customers["customer_id"].to_numpy()[c],
                                    "product_id": PRODUCTS["product_id"].to_numpy()[k],
                                    "channel": rng.choice(["app_push", "email", "sms", "rm_call"], v, p=[0.45, 0.3, 0.15, 0.1]),
                                    "accepted": (rng.random(v) < acc_p).astype(int)}))
    offers = pd.concat(offers, ignore_index=True)
    offers.insert(0, "offer_id", [f"O{i:07d}" for i in range(1, len(offers) + 1)])
    return {"customers": customers, "products": PRODUCTS, "holdings": holdings, "offers": offers}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/petalbank_holdings")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
