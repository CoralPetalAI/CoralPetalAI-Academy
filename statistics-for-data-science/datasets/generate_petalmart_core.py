"""Generate the Coral Petal synthetic dataset `petalmart_core`.

Used in: Builds 0, 1, 2, 4 and 5 and the lesson exercises of Modules 01 to 05 (and L7.3),
Statistics for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- 30,000 users (a random sample of PetalMart's customers) in 40 cities across four regions
  (N, S, E, W), signed up between April 2023 and March 2026; primary device app (about 56%)
  or web; 28% are loyalty members, who prefer the app and order more often.
- One financial year of orders, 1 April 2025 to 31 March 2026 (FY2025-26), about 50,000.
  Orders arrive at random: steady between 10:00 and 22:00, quiet at night, busier at
  weekends, in the festive season before Diwali and in PetalMart's sales, growing through the year.
- All money is in US dollars. Order values are right-skewed (lognormal), median about $70,
  mean about $97, recorded to the nearest 10 cents. Electronics
  and Home baskets are larger, Grocery and Books smaller; South and West spend a little more per
  order and East less; loyalty members and app orders spend a little more. A few genuine
  bulk orders sit far out in the right tail.
- Shipping: orders below the free-shipping threshold of $100 pay an $8 fee (shipping_fee); from $100
  shipping is free.
- Deliveries take about 48 hours on average (SD about 8); a few late deliveries make a longer right tail.
- Sessions: visits on app or web with pages viewed, add to cart and whether the visit ended in an order.
- Planted problems to find and clean (Builds 1 and 2): 0.5% exact duplicate order rows, 30 orders
  with a value of zero or less, 8 orders recorded in cents instead of dollars (100 times too large),
  40 orders from deleted accounts (no matching user), 1.5% of users with a missing region (more
  often app users), and Bengaluru also written as "Bangalore" and "BLR".

Usage:
    python generate_petalmart_core.py --out data/petalmart_core --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from petalmart_common import CATEGORIES, city_table, festival_calendar

FY_START, FY_END = "2025-04-01", "2026-03-31"
N_USERS = 30_000
TARGET_ORDERS = 50_000
REGION_VALUE = {"N": 1.00, "S": 1.06, "E": 0.92, "W": 1.03}
HOURLY = np.array([0.10, 0.06, 0.04, 0.04, 0.05, 0.10, 0.25, 0.50, 0.80, 0.90] + [1.0] * 12 + [0.60, 0.30])
COURIERS = ["PetalExpress", "RapidShip", "SwiftKart"]
FREE_SHIPPING_FROM, SHIPPING_FEE = 100, 8  # dollars


def make_users(rng: np.random.Generator) -> pd.DataFrame:
    cities = city_table()
    idx = rng.choice(len(cities), N_USERS, p=cities.weight)
    days = (pd.Timestamp(FY_END) - pd.Timestamp("2023-04-01")).days
    # more recent sign-ups: PetalMart is growing
    signup = pd.Timestamp("2023-04-01") + pd.to_timedelta(np.floor(days * rng.beta(1.3, 1.0, N_USERS)), "D")
    loyal = rng.random(N_USERS) < 0.28
    app = rng.random(N_USERS) < np.where(loyal, 0.70, 0.505)
    return pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(1, N_USERS + 1)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "city": cities.city.to_numpy()[idx],
        "region": cities.region.to_numpy()[idx],
        "device": np.where(app, "app", "web"),
        "loyalty_member": loyal.astype(int),
        "age_band": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], N_USERS, p=[0.18, 0.36, 0.25, 0.13, 0.08]),
    })


def daily_weights() -> pd.DataFrame:
    cal = festival_calendar(FY_START, FY_END)
    d = cal.date
    w = np.ones(len(cal))
    w *= np.where(d.dt.day_of_week >= 5, 1.15, 1.0)
    w *= np.where(cal.festive_season == 1, 1.55, 1.0)
    w *= np.where(cal.sale_day == 1, 1.30, 1.0)
    w *= np.where((cal.diwali_week == 1) & ((d - pd.Timestamp("2025-10-21")).dt.days.between(0, 1)), 0.45, 1.0)
    w *= 1 + 0.15 * np.arange(len(cal)) / len(cal)  # growth through the year
    cal["weight"] = w / w.sum()
    return cal


def make_orders(rng, users):
    cal = daily_weights()
    signup = pd.to_datetime(users.signup_date).to_numpy()
    start = np.maximum(signup, np.datetime64(FY_START))
    exposure = (np.datetime64(FY_END) - start).astype("timedelta64[D]").astype(int) + 1
    rate = rng.gamma(0.9, 1.0, len(users)) * np.where(users.loyalty_member == 1, 1.6, 1.0) * exposure / 365
    rate *= TARGET_ORDERS / rate.sum()
    n = rng.poisson(rate)
    u = users.loc[np.repeat(users.index, n)].reset_index(drop=True)
    u_start = np.repeat(start, n)

    # order day: drawn from the seasonal day weights, restricted to days after sign-up
    cum = np.cumsum(cal.weight.to_numpy())
    first_day = np.searchsorted(cal.date.to_numpy(), u_start)
    lo = np.where(first_day > 0, cum[np.maximum(first_day - 1, 0)], 0.0)
    pick = lo + rng.random(len(u)) * (1 - lo)
    day = np.minimum(np.searchsorted(cum, pick), len(cal) - 1)
    hour = rng.choice(24, len(u), p=HOURLY / HOURLY.sum())
    secs = rng.integers(0, 3600, len(u))
    ts = cal.date.to_numpy()[day] + (hour * 3600 + secs).astype("timedelta64[s]")

    cat = rng.choice(len(CATEGORIES), len(u), p=CATEGORIES.weight)
    mult = (CATEGORIES.value_multiplier.to_numpy()[cat] * u.region.map(REGION_VALUE).to_numpy()
            * np.where(u.device == "app", 1.03, 1.0) * np.where(u.loyalty_member == 1, 1.08, 1.0))
    value = rng.lognormal(np.log(70), 0.60, len(u)) * mult  # dollars
    bulk = rng.random(len(u)) < 0.002  # genuine bulk orders: offices, events, resellers
    value = np.where(bulk, value * rng.uniform(8, 15, len(u)), value)
    value = np.round(np.clip(value, 4.9, None), 1)  # to the nearest 10 cents
    items = np.where(bulk, rng.integers(12, 41, len(u)), 1 + rng.poisson(np.clip(value / 90, 0.1, 4)))
    in_sale = cal.sale_day.to_numpy()[day] == 1
    disc_rate = np.where(in_sale, rng.choice([0, 0.05, 0.10, 0.20], len(u), p=[0.25, 0.25, 0.30, 0.20]),
                         rng.choice([0, 0.05, 0.10, 0.20], len(u), p=[0.60, 0.20, 0.14, 0.06]))
    ret_p = np.array([0.11, 0.05, 0.05, 0.04, 0.01, 0.07, 0.06, 0.02])[cat]
    orders = pd.DataFrame({
        "order_id": "",
        "user_id": u.user_id.to_numpy(),
        "order_ts": pd.to_datetime(ts),
        "category": CATEGORIES.category.to_numpy()[cat],
        "order_value": value,
        "shipping_fee": np.where(value < FREE_SHIPPING_FROM, SHIPPING_FEE, 0),
        "discount": np.round(value * 10 * disc_rate) / 10,  # to the nearest 10 cents
        "items": items,
        "payment_method": rng.choice(["UPI", "card", "COD", "wallet"], len(u), p=[0.52, 0.22, 0.18, 0.08]),
        "returned": (rng.random(len(u)) < ret_p).astype(int),
    }).sort_values("order_ts").reset_index(drop=True)
    orders["order_id"] = [f"O{i:06d}" for i in range(1, len(orders) + 1)]
    return orders


def make_deliveries(rng, orders, users):
    region = orders.user_id.map(users.set_index("user_id").region).fillna("N").to_numpy()
    courier = rng.choice(COURIERS, len(orders), p=[0.5, 0.3, 0.2])
    hours = rng.normal(47.5, 7.2, len(orders)) + np.where(region == "E", 2.5, 0) + np.where(courier == "SwiftKart", 1.5, 0)
    late = rng.random(len(orders)) < 0.04  # weather, address problems, missed handovers
    hours = np.where(late, hours + rng.gamma(2.0, 8.0, len(orders)), hours)
    return pd.DataFrame({"order_id": orders.order_id, "courier": courier, "promised_hours": 48,
                         "delivery_hours": np.round(np.clip(hours, 12, None), 1)})


def make_sessions(rng, users, orders):
    # one session ends in each order, plus browsing sessions that do not
    buy = orders[["user_id", "order_ts"]].copy()
    buy["ordered"] = 1
    n_browse = rng.poisson(2.2, len(orders))
    br = orders.loc[np.repeat(orders.index, n_browse), ["user_id", "order_ts"]].reset_index(drop=True)
    br["order_ts"] = br.order_ts - pd.to_timedelta(rng.exponential(5 * 24 * 3600, len(br)).astype(int), "s")
    nb = users.loc[np.repeat(users.index, rng.poisson(1.6, len(users))), ["user_id"]].reset_index(drop=True)
    nb["order_ts"] = pd.Timestamp(FY_START) + pd.to_timedelta(rng.integers(0, 365 * 24 * 3600, len(nb)), "s")
    s = pd.concat([buy, br.assign(ordered=0), nb.assign(ordered=0)], ignore_index=True)
    s = s[s.order_ts >= pd.Timestamp(FY_START)].sort_values("order_ts").reset_index(drop=True)
    primary = s.user_id.map(users.set_index("user_id").device)
    other = np.where(primary == "app", "web", "app")
    dev = np.where(rng.random(len(s)) < 0.85, primary, other)
    cart = np.where(s.ordered == 1, 1, (rng.random(len(s)) < 0.12).astype(int))
    return pd.DataFrame({
        "session_id": [f"S{i:07d}" for i in range(1, len(s) + 1)],
        "user_id": s.user_id, "session_start": s.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S"), "device": dev,
        "pages_viewed": rng.geometric(0.16, len(s)) + 2 * s.ordered.to_numpy(),
        "added_to_cart": cart, "ordered": s.ordered.to_numpy(),
    })


def plant_problems(rng, users, orders):
    users, orders = users.copy(), orders.copy()
    # missing region, more often for app users (missing at random given device, not completely at random)
    p_miss = np.where(users.device == "app", 0.022, 0.006)
    users.loc[rng.random(len(users)) < p_miss, "region"] = np.nan
    blr = users.index[users.city == "Bengaluru"]
    lab = rng.random(len(blr))
    users.loc[blr[lab < 0.06], "city"] = "Bangalore"
    users.loc[blr[(lab >= 0.06) & (lab < 0.09)], "city"] = "BLR"
    # invalid values: 20 negative and 10 zero
    bad = rng.choice(len(orders), 30, replace=False)
    orders.loc[bad[:20], "order_value"] = -orders.loc[bad[:20], "order_value"]
    orders.loc[bad[20:], "order_value"] = 0
    # recorded in cents instead of dollars
    cents = rng.choice(np.setdiff1d(np.arange(len(orders)), bad), 8, replace=False)
    orders.loc[cents, "order_value"] = orders.loc[cents, "order_value"] * 100
    # orders from 40 deleted accounts
    orphan = orders.sample(40, random_state=int(rng.integers(1e9))).copy()
    orphan["user_id"] = [f"U{i:05d}" for i in range(N_USERS + 1, N_USERS + 41)]
    orphan["order_id"] = [f"O{i:06d}" for i in range(len(orders) + 1, len(orders) + 41)]
    orders = pd.concat([orders, orphan], ignore_index=True)
    # exact duplicate rows (a pipeline retry wrote some orders twice)
    dup = orders.sample(frac=0.005, random_state=int(rng.integers(1e9)))
    orders = pd.concat([orders, dup]).sort_values(["order_ts", "order_id"]).reset_index(drop=True)
    orders["order_ts"] = orders.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    return users, orders


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    users = make_users(rng)
    orders = make_orders(rng, users)
    deliveries = make_deliveries(rng, orders, users)
    sessions = make_sessions(rng, users, orders)
    users_out, orders_out = plant_problems(rng, users, orders)
    costs = CATEGORIES[["category", "gross_margin_pct", "shipping_cost_per_order"]]
    return {"users": users_out, "orders": orders_out, "sessions": sessions, "costs": costs, "deliveries": deliveries}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "petalmart_core"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
