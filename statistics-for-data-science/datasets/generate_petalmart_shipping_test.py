"""Generate the Coral Petal synthetic dataset `petalmart_shipping_test`.

Capstone: The PetalMart Free-Shipping Decision (Statistics for Data Science).

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Usage:
    python generate_petalmart_shipping_test.py --out data/petalmart_shipping_test --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2026-08-03")  # Monday
TEST_DAYS = 28
CONTROL_SHARE = 0.502  # planted 50.2 / 49.8 split, passes an SRM check

CITIES = {
    "N": ["Delhi", "Jaipur", "Lucknow", "Chandigarh", "Dehradun", "Amritsar", "Kanpur", "Agra", "Ludhiana", "Meerut"],
    "S": ["Bengaluru", "Chennai", "Hyderabad", "Kochi", "Coimbatore", "Mysuru", "Madurai", "Vizag", "Mangaluru", "Trichy"],
    "E": ["Kolkata", "Bhubaneswar", "Patna", "Guwahati", "Ranchi", "Siliguri", "Cuttack", "Durgapur", "Gaya", "Jamshedpur"],
    "W": ["Mumbai", "Pune", "Ahmedabad", "Surat", "Nagpur", "Indore", "Vadodara", "Nashik", "Rajkot", "Bhopal"],
}
CATEGORIES = pd.DataFrame(
    {
        "category": ["Fashion", "Home", "Beauty", "Electronics", "Grocery", "Kids", "Sports", "Books"],
        "gross_margin_pct": [0.42, 0.36, 0.48, 0.14, 0.16, 0.38, 0.32, 0.26],
        "shipping_cost_per_order": [72, 95, 58, 88, 80, 70, 85, 55],
        "weight": [0.24, 0.14, 0.14, 0.12, 0.16, 0.08, 0.06, 0.06],
    }
)
SHIPPING_FEE = 79  # charged when the basket is below the free-shipping threshold
THRESHOLD = {"control": 999, "treatment": 699}

# Conversion rates over the 4-week test (share of users who place at least one order)
BASE_CONV = {"new": 0.060, "returning": 0.085}
LIFT_PP = {"new": 0.014, "returning": 0.001}
DEVICE_ODDS = {"app": 1.12, "web": 0.88}
REGION_ODDS = {"N": 1.00, "S": 1.06, "E": 0.90, "W": 1.04}


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


def make_users(rng: np.random.Generator, n: int = 60_000) -> pd.DataFrame:
    region = rng.choice(list(CITIES), size=n, p=[0.27, 0.29, 0.18, 0.26])
    city = np.array([rng.choice(CITIES[r]) for r in region])
    signup = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 1004, size=n), unit="D")
    users = pd.DataFrame(
        {
            "user_id": [f"U{i:06d}" for i in range(1, n + 1)],
            "signup_date": signup.date,
            "city": city,
            "region": region,
            "device": rng.choice(["app", "web"], size=n, p=[0.56, 0.44]),
            "customer_type": rng.choice(["new", "returning"], size=n, p=[0.40, 0.60]),
            "age_band": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], size=n, p=[0.18, 0.36, 0.25, 0.13, 0.08]),
        }
    )
    return users


def make_assignments(rng, users):
    n = len(users)
    n_control = int(round(n * CONTROL_SHARE))
    variant = np.array(["control"] * n_control + ["treatment"] * (n - n_control))
    rng.shuffle(variant)
    assigned = TEST_START + pd.to_timedelta(rng.integers(0, 7 * 24 * 60, size=n), unit="min")
    return pd.DataFrame({"user_id": users.user_id, "variant": variant, "assigned_at": assigned})


def make_orders_and_sessions(rng, users, assign):
    df = users.merge(assign, on="user_id")
    ctype = df.customer_type.to_numpy()
    treat = (df.variant == "treatment").to_numpy()
    p = np.array([BASE_CONV[c] for c in ctype]) + treat * np.array([LIFT_PP[c] for c in ctype])
    # device and region shift the odds but are balanced across variants
    lo = _logit(p) + np.log(df.device.map(DEVICE_ODDS).to_numpy()) + np.log(df.region.map(REGION_ODDS).to_numpy())
    p_adj = _expit(lo - 0.035)  # centre so overall rates stay on target
    converted = rng.random(len(df)) < p_adj

    # sessions: everyone browses; converters browse more
    n_sess = rng.poisson(3.7, len(df)) + 1 + converted * rng.poisson(1.5, len(df))
    sess_user = np.repeat(df.user_id.to_numpy(), n_sess)
    sess_day = rng.integers(0, TEST_DAYS, size=len(sess_user))
    sessions = pd.DataFrame(
        {
            "session_id": [f"S{i:07d}" for i in range(1, len(sess_user) + 1)],
            "user_id": sess_user,
            "date": (TEST_START + pd.to_timedelta(sess_day, unit="D")).date,
            "pages_viewed": rng.geometric(0.18, size=len(sess_user)),
            "added_to_cart": (rng.random(len(sess_user)) < 0.11).astype(int),
        }
    )

    # orders: converters place 1 + Poisson(0.9) orders
    conv = df[converted].reset_index(drop=True)
    k = 1 + rng.poisson(0.9, len(conv))
    o = conv.loc[np.repeat(conv.index, k)].reset_index(drop=True)
    o["date"] = (TEST_START + pd.to_timedelta(rng.integers(0, TEST_DAYS, len(o)), unit="D")).date
    o["category"] = rng.choice(CATEGORIES.category, size=len(o), p=CATEGORIES.weight)
    base_value = rng.lognormal(mean=np.log(780), sigma=0.62, size=len(o))
    is_t = (o.variant == "treatment").to_numpy()
    # treatment: people stop padding baskets to reach ₹999, so baskets shrink by ₹38 on average
    value = base_value - is_t * rng.normal(38, 12, len(o))
    # control shoppers just under ₹999 often pad up to qualify
    pad = (~is_t) & (value > 820) & (value < 999) & (rng.random(len(o)) < 0.45)
    value = np.where(pad, value + rng.uniform(0, 260, len(o)), value)
    value = np.round(np.clip(value, 99, None), 2)
    threshold = np.where(is_t, THRESHOLD["treatment"], THRESHOLD["control"])
    o["order_value"] = value
    o["shipping_fee"] = np.where(value < threshold, SHIPPING_FEE, 0)
    o["discount"] = np.round(value * rng.choice([0, 0, 0.05, 0.10], len(o)), 2)
    o["items"] = 1 + rng.poisson(np.clip(value / 600, 0.2, None))
    o["returned"] = (rng.random(len(o)) < 0.06).astype(int)
    o["order_id"] = [f"O{i:06d}" for i in range(1, len(o) + 1)]
    orders = o[["order_id", "user_id", "date", "category", "order_value", "shipping_fee", "discount", "items", "returned"]].copy()
    return orders, sessions


def plant_errors(rng, users, orders):
    users = users.copy()
    miss = rng.choice(len(users), size=int(0.01 * len(users)), replace=False)
    users.loc[miss, "region"] = np.nan
    orders = orders.copy()
    neg = rng.choice(len(orders), size=30, replace=False)
    orders.loc[neg, "order_value"] = -orders.loc[neg, "order_value"]
    dup = orders.sample(n=int(0.005 * len(orders)), random_state=int(rng.integers(1e9)))
    orders = pd.concat([orders, dup]).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    return users, orders


def make_city_rollout(rng):
    """52 weeks × 40 cities. Six richer cities got same-day delivery in week 30 (not randomized)."""
    rows = []
    cities = [(c, r) for r, cs in CITIES.items() for c in cs]
    rolled = {"Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai"}
    for city, _ in cities:
        rich = city in rolled
        users = int(rng.integers(9_000, 16_000) * (1.35 if rich else 1.0))
        rate = 0.20 * (1.16 if rich else 1.0)  # richer cities order more per user, before any rollout
        for w in range(1, 53):
            season = 1 + 0.12 * np.sin(2 * np.pi * (w - 40) / 52) + (0.18 if w in (43, 44) else 0)
            growth = 1 + 0.002 * w
            treated = rich and w >= 30
            effect = 1.08 if treated else 1.0  # true causal effect: +8% orders per active user
            active = int(users * growth * rng.normal(1, 0.02))
            orders = rng.poisson(active * rate * season * effect)
            revenue = round(orders * rng.normal(1050 * (1.05 if rich else 1), 25), 0)
            rows.append((city, w, int(treated), orders, revenue, active))
    return pd.DataFrame(rows, columns=["city", "week", "rolled_out", "orders", "revenue", "active_users"])


def main(out: Path, seed: int = 2026):
    rng = np.random.default_rng(seed)
    users = make_users(rng)
    assign = make_assignments(rng, users)
    orders, sessions = make_orders_and_sessions(rng, users, assign)
    users_out, orders_out = plant_errors(rng, users, orders)
    rollout = make_city_rollout(rng)
    out.mkdir(parents=True, exist_ok=True)
    users_out.to_csv(out / "users.csv", index=False)
    assign.to_csv(out / "assignments.csv", index=False)
    sessions.to_csv(out / "sessions.csv", index=False)
    orders_out.to_csv(out / "orders.csv", index=False)
    CATEGORIES.drop(columns="weight").to_csv(out / "costs.csv", index=False)
    rollout.to_csv(out / "city_rollout.csv", index=False)
    print(f"users {len(users_out):,} · sessions {len(sessions):,} · orders {len(orders_out):,} · rollout {len(rollout):,} rows → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "petalmart_shipping_test")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    main(a.out, a.seed)
