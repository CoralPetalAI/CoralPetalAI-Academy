"""Generate the Coral Petal synthetic dataset `petalmart_demand`.

Used in: Build 9 (forecast weekly demand), Portfolio Project 2 (PetalMart Demand Forecast)
and the lesson exercises of Module 09, Statistics for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- Three years of daily orders, Monday 4 September 2023 to Sunday 30 August 2026 (156 full
  weeks), for 10 categories in 4 regions (40 series), with the average selling price and a
  promotion flag, plus a daily festival calendar (known 12 weeks beyond the orders, to
  22 November 2026) and the list of PetalMart's warehouses.
- Each series has its own level and growth. Every category has a weekly pattern (busier
  at weekends), a yearly pattern of its own (Sports peaks in summer, Books before exams in
  February and March, Fashion in the end-of-season sales) and festival peaks: the 30 days
  before Diwali lift Electronics, Appliances, Fashion and Jewellery most; Diwali day and the
  day after are quiet.
- PetalMart's sales (PetalMart Days, end-of-season, Republic Day, Independence Day) lift
  orders, and so do the category promotions, through lower prices.
- Structural break: a new warehouse in the East opened on 2 June 2025, and East orders
  stepped up by about a fifth over the following month, as delivery got faster.
- Daily counts are noisy (negative binomial); weekly totals are much smoother.

Usage:
    python generate_petalmart_demand.py --out data/petalmart_demand --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from petalmart_common import REGION_SHARE, REGIONS, festival_calendar

START, END = "2023-09-04", "2026-08-30"
CALENDAR_END = "2026-11-22"  # 12 weeks past the last day of orders
EAST_WAREHOUSE = pd.Timestamp("2025-06-02")
# category: daily orders (all regions, Sept 2023), yearly growth, festive lift, sale lift, base price (₹)
CATS = pd.DataFrame(
    [
        ("Fashion", 4200, 0.16, 0.85, 0.55, 780),
        ("Home", 2300, 0.12, 0.55, 0.35, 1150),
        ("Beauty", 2400, 0.18, 0.45, 0.30, 520),
        ("Electronics", 1900, 0.14, 1.30, 0.75, 2900),
        ("Grocery", 3100, 0.22, 0.25, 0.15, 410),
        ("Kids", 1200, 0.10, 0.50, 0.30, 690),
        ("Sports", 800, 0.12, 0.20, 0.30, 950),
        ("Books", 700, 0.06, 0.10, 0.20, 340),
        ("Footwear", 1300, 0.15, 0.60, 0.50, 1050),
        ("Appliances", 600, 0.13, 1.10, 0.60, 6400),
    ],
    columns=["category", "level", "growth", "festive_lift", "sale_lift", "price"],
)
WEEKDAY = np.array([0.96, 0.94, 0.95, 0.97, 1.00, 1.08, 1.10])  # Mon..Sun
REGION_GROWTH = {"N": 0.00, "S": 0.02, "E": -0.02, "W": 0.01}
WAREHOUSES = pd.DataFrame({
    "warehouse": ["Gurugram", "Bhiwandi", "Hoskote", "Kolkata Dankuni"],
    "region": ["N", "W", "S", "E"],
    "opened": ["2019-03-01", "2019-08-01", "2020-01-15", "2025-06-02"],
})


def yearly_shape(cat: str, doy: np.ndarray) -> np.ndarray:
    """Smooth within-year pattern by category (day of year 1..366)."""
    a = 2 * np.pi * doy / 365.25
    if cat == "Sports":
        return 1 + 0.25 * np.cos(a - 2 * np.pi * 140 / 365.25)   # summer
    if cat == "Books":
        return 1 + 0.30 * np.exp(-((doy - 55) / 25.0) ** 2)      # before exams
    if cat == "Fashion":
        return 1 + 0.08 * np.cos(2 * a)                           # two seasons a year
    if cat == "Grocery":
        return 1 + 0.05 * np.cos(a - 2 * np.pi * 200 / 365.25)   # monsoon stocking
    return 1 + 0.04 * np.cos(a)


def promotions(rng, cats, dates):
    """Seven to ten promotions per category per year, three to seven days long, 10-30% off."""
    flag = np.zeros((len(cats), len(dates)), dtype=int)
    depth = np.zeros((len(cats), len(dates)))
    for i in range(len(cats)):
        for _ in range(rng.integers(21, 31)):
            s = rng.integers(0, len(dates) - 7)
            L = rng.integers(3, 8)
            flag[i, s:s + L] = 1
            depth[i, s:s + L] = rng.choice([0.10, 0.15, 0.20, 0.30])
    return flag, depth


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cal = festival_calendar(START, END)
    d = pd.DatetimeIndex(cal.date)
    t_years = (d - d[0]).days.to_numpy() / 365.25
    doy = d.day_of_year.to_numpy()
    wk = WEEKDAY[d.day_of_week.to_numpy()]
    festive, diwali_wk, sale = cal.festive_season.to_numpy(), cal.diwali_week.to_numpy(), cal.sale_day.to_numpy()
    # build-up inside the festive season: strongest in the last ten days before Diwali
    days_to_diwali = np.full(len(d), 999)
    for s in ["2023-11-12", "2024-11-01", "2025-10-21"]:
        k = (pd.Timestamp(s) - d).days.to_numpy()
        days_to_diwali = np.where((k >= 0) & (k < days_to_diwali), k, days_to_diwali)
    ramp = np.where(festive == 1, 0.6 + 0.4 * np.clip(1 - days_to_diwali / 30, 0, 1), 0)
    quiet = np.isin(days_to_diwali, [0]) | np.isin(np.roll(days_to_diwali, 1), [0])
    east_step = np.where(d >= EAST_WAREHOUSE, 1 + 0.20 * np.clip((d - EAST_WAREHOUSE).days.to_numpy() / 30, 0, 1), 1.0)
    promo_flag, promo_depth = promotions(rng, CATS, d)

    rows = []
    for i, c in CATS.iterrows():
        shape = yearly_shape(c.category, doy)
        price_drift = c.price * (1 + 0.03 * t_years) * np.exp(np.cumsum(rng.normal(0, 0.002, len(d))))
        for r, share in zip(REGIONS, REGION_SHARE):
            level = c.level * share * np.exp(rng.normal(0, 0.06))
            trend = np.exp((c.growth + REGION_GROWTH[r]) * t_years)
            price = price_drift * (1 - promo_depth[i]) * np.exp(rng.normal(0, 0.01, len(d)))
            elasticity = 1.6 if c.category in ("Electronics", "Appliances", "Fashion", "Footwear") else 1.1
            promo_lift = (1 - promo_depth[i]) ** (-elasticity)
            mu = (level * trend * wk * shape * (1 + c.festive_lift * ramp) * (1 + c.sale_lift * sale)
                  * promo_lift * np.where(quiet, 0.45, 1.0) * np.where(cal.national_holiday == 1, 0.85, 1.0))
            if r == "E":
                mu = mu * east_step
            k = 25.0  # negative binomial size: variance = mu + mu^2 / k
            y = rng.poisson(mu * rng.gamma(k, 1 / k, len(d)))
            rows.append(pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "category": c.category, "region": r,
                                      "orders": y, "avg_price": np.round(price, 0), "promo": promo_flag[i]}))
    daily = pd.concat(rows, ignore_index=True)
    future = festival_calendar(START, CALENDAR_END)  # the calendar is known 12 weeks ahead, for forecasting
    cal_out = future.assign(date=future.date.dt.strftime("%Y-%m-%d"))
    return {"daily_orders": daily, "calendar": cal_out, "warehouses": WAREHOUSES}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "petalmart_demand"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
