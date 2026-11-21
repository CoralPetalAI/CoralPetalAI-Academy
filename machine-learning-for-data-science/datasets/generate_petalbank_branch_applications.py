"""Generate the Coral Petal synthetic dataset `petalbank_branch_applications`.

Used in: Build 9 (forecast loan applications for 300 branches), Machine Learning for
Data Science; Portfolio Project 2 reuses the forecasting approach on weekly offers.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- 300 branches (metro, urban, semi-urban, rural), daily loan applications from
  January 2023 to June 2026.
- Each branch has its own level and trend; branch types share a weekly pattern
  (rural branches are busy on market day and closed on Sundays).
- Month-end salary days, the Diwali festive season and the March financial-year end lift
  demand; national holidays close branches.
- 24 branches open during the period (short histories), and a few run local promotions.
- Counts are Poisson around the expected level, so small branches are noisy.

Usage:
    python generate_petalbank_branch_applications.py --out data/petalbank_branch_applications --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BRANCHES = 300
DATES = pd.date_range("2023-01-01", "2026-06-30", freq="D")
DIWALI = {2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-21"}
HOLIDAYS = ["01-26", "08-15", "10-02"]
TYPES = ["metro", "urban", "semi_urban", "rural"]
TYPE_P = [0.20, 0.30, 0.28, 0.22]
LEVEL = {"metro": 38, "urban": 20, "semi_urban": 10, "rural": 5}
WEEKLY = {  # Mon..Sun multipliers
    "metro": [1.10, 1.05, 1.0, 1.0, 1.05, 0.85, 0.45],
    "urban": [1.10, 1.05, 1.0, 1.0, 1.05, 0.90, 0.35],
    "semi_urban": [1.05, 1.0, 1.0, 1.05, 1.0, 1.05, 0.25],
    "rural": [0.9, 0.9, 1.0, 1.6, 0.9, 1.1, 0.0],  # Thursday market day, closed Sunday
}


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    btype = rng.choice(TYPES, N_BRANCHES, p=TYPE_P)
    states = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Uttar Pradesh", "Gujarat", "West Bengal",
              "Telangana", "Rajasthan", "Bihar", "Kerala", "Madhya Pradesh"]
    opened = np.full(N_BRANCHES, DATES[0])
    newb = rng.choice(N_BRANCHES, 24, replace=False)
    opened[newb] = DATES[0] + pd.to_timedelta(rng.integers(200, 1100, 24), "D")
    branches = pd.DataFrame({
        "branch_id": [f"B{i:03d}" for i in range(1, N_BRANCHES + 1)], "branch_type": btype,
        "state": rng.choice(states, N_BRANCHES), "open_date": pd.to_datetime(opened).strftime("%Y-%m-%d"),
        "staff": np.array([14, 9, 6, 4])[[TYPES.index(b) for b in btype]] + rng.integers(-1, 3, N_BRANCHES),
    })

    # Calendar effects
    d = DATES
    dom, dim = d.day.to_numpy(), d.days_in_month.to_numpy()
    month_end = np.where(dom >= dim - 2, 1.25, np.where(dom <= 3, 1.15, 1.0))
    festive = np.ones(len(d))
    for y, s in DIWALI.items():
        t = (d - pd.Timestamp(s)).days.to_numpy()
        festive *= np.where((t >= -21) & (t < 0), 1.45, np.where((t >= 0) & (t <= 2), 0.3, 1.0))
    fy_end = np.where((d.month == 3) & (dom >= 15), 1.2, 1.0)
    holiday = np.isin(d.strftime("%m-%d"), HOLIDAYS)
    calendar = pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "national_holiday": holiday.astype(int),
                             "festive_season": (festive > 1).astype(int), "month_end_window": (month_end > 1).astype(int)})

    rows = []
    t_years = (d - d[0]).days.to_numpy() / 365
    for i in range(N_BRANCHES):
        bt = btype[i]
        level = LEVEL[bt] * np.exp(rng.normal(0, 0.35))
        trend = rng.normal(0.08, 0.07)
        wk = np.array(WEEKLY[bt])[d.day_of_week]
        mu = level * np.exp(trend * t_years) * wk * month_end * festive * fy_end * np.where(holiday, 0, 1)
        if rng.random() < 0.12:  # a local promotion of 4–8 weeks
            s = rng.integers(100, len(d) - 60)
            mu[s:s + rng.integers(28, 57)] *= 1.3
        live = d >= opened[i]
        ramp = np.clip((d - pd.Timestamp(opened[i])).days.to_numpy() / 120, 0.2, 1.0)
        y = rng.poisson(mu * ramp * rng.lognormal(0, 0.08, len(d)))
        rows.append(pd.DataFrame({"date": calendar["date"][live], "branch_id": branches["branch_id"][i],
                                  "applications": y[live]}))
    daily = pd.concat(rows, ignore_index=True)
    return {"branches": branches, "calendar": calendar, "daily_applications": daily}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/petalbank_branch_applications")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
