"""Generate the Coral Petal synthetic dataset `petalmart_weekly`.

Used in: Build 7 (a regression explaining revenue drivers) and the lesson exercises of
L7.1, L7.2 and L7.5, Statistics for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- 52 weeks of FY2025-26 (weeks starting Monday 7 April 2025 to Monday 30 March 2026):
  marketing spend and revenue in thousands of US dollars, average discount in percent, season
  (regular, festive, sale), the marketing channel that took the largest share of the
  week's spend, and ad impressions in millions.
- The 52 spend values are the ones behind the L7.1 lesson (between $20,000 and $80,000), and the
  simple regression of revenue on spend reproduces it exactly: revenue = 400 + 3.2 x spend,
  R squared 0.64.
- The team spends most in the festive weeks before Diwali, when revenue is high anyway, so
  part of that 3.2 belongs to the season: holding season and discount constant, each $1,000 of
  spend goes with about $2,100 of revenue.
- Discount and the lead channel add little once spend and season are known.
- Ad impressions are bought with spend, so the two move almost together (multicollinearity).
- Busy weeks are noisier than quiet ones (a funnel in the residuals), and one December week
  lost two days of sales to a site outage (see the note column).

Usage:
    python generate_petalmart_weekly.py --out data/petalmart_weekly --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WEEKS = pd.date_range("2025-04-07", periods=52, freq="7D")
FESTIVE = pd.to_datetime(["2025-09-15", "2025-09-22", "2025-09-29", "2025-10-06", "2025-10-13", "2025-10-20"])
SALE = pd.to_datetime(["2025-06-30", "2025-07-07", "2025-08-11", "2025-12-29", "2026-01-05", "2026-01-19", "2026-03-16"])
OUTAGE_WEEK = pd.Timestamp("2025-12-08")
DESIGN = 508  # fixed design stream, chosen so the multiple regression shows the L7.2 story
LESSON_SEED = 7  # the L7.1 lesson's 52 spend values


def lesson_spend() -> np.ndarray:
    return np.round(np.random.default_rng(LESSON_SEED).uniform(2, 8, 52), 1) * 10  # $000s, whole thousands


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng([seed, DESIGN])
    season = np.where(WEEKS.isin(FESTIVE), "festive", np.where(WEEKS.isin(SALE), "sale", "regular"))
    f, s_, r = (np.flatnonzero(season == k) for k in ("festive", "sale", "regular"))

    # festive weeks get six of the ten biggest budgets, sale weeks budgets from the upper middle
    s = lesson_spend()
    by_size = list(np.argsort(-s))
    top = rng.permutation(by_size[:10])
    spend = np.empty(52)
    spend[f] = s[top[:6]]
    rest = [i for i in by_size if i not in top[:6]]
    mid = rng.permutation(rest[:20])
    spend[s_] = s[mid[:7]]
    spend[r] = s[rng.permutation([i for i in rest if i not in mid[:7]])]

    discount = np.where(season == "sale", rng.normal(22, 3, 52),
                        np.where(season == "festive", rng.normal(14, 2, 52), rng.normal(8, 1.5, 52)))
    level = 300 + 2.1 * spend + 120 * (season == "festive") + 40 * (season == "sale")
    revenue = level + rng.normal(0, 1, 52) * 34 * (level / level.mean()) ** 1.5
    revenue[WEEKS == OUTAGE_WEEK] -= 160.0

    # exact lesson view: the simple regression is 400 + 3.2 x spend with R squared 0.64
    x = np.column_stack([np.ones(52), spend])
    e = revenue - (400 + 3.2 * spend)
    e -= x @ np.linalg.lstsq(x, e, rcond=None)[0]
    sxx = ((spend - spend.mean()) ** 2).sum()
    e *= np.sqrt(3.2 ** 2 * sxx * 0.36 / 0.64 / (e ** 2).sum())
    revenue = 400 + 3.2 * spend + e

    extra = np.random.default_rng([seed, DESIGN, 1])
    weekly = pd.DataFrame({
        "week_start": WEEKS.strftime("%Y-%m-%d"),
        "week": np.arange(1, 53),
        "season": season,
        "spend": spend,
        "discount": np.round(discount, 1),
        "lead_channel": extra.choice(["search", "social", "video"], 52, p=[0.5, 0.3, 0.2]),
        "ad_impressions": np.round(0.12 * spend + extra.normal(0, 0.25, 52), 2),
        "revenue": np.round(revenue, 1),
        "note": np.where(WEEKS == OUTAGE_WEEK, "site outage (2 days)", ""),
    })
    return {"weekly_revenue": weekly}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "petalmart_weekly"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
