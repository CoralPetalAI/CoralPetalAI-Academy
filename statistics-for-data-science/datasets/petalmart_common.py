"""Shared constants for the PetalMart dataset generators (Statistics for Data Science).

Every generator in this folder imports this file, so cities, regions, categories and the
festival calendar are the same in every dataset of the course. All money is in US dollars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIONS = ["N", "S", "E", "W"]
REGION_SHARE = [0.27, 0.29, 0.18, 0.26]
CITIES = {
    "N": ["Delhi", "Jaipur", "Lucknow", "Chandigarh", "Dehradun", "Amritsar", "Kanpur", "Agra", "Ludhiana", "Meerut"],
    "S": ["Bengaluru", "Chennai", "Hyderabad", "Kochi", "Coimbatore", "Mysuru", "Madurai", "Vizag", "Mangaluru", "Trichy"],
    "E": ["Kolkata", "Bhubaneswar", "Patna", "Guwahati", "Ranchi", "Siliguri", "Cuttack", "Durgapur", "Gaya", "Jamshedpur"],
    "W": ["Mumbai", "Pune", "Ahmedabad", "Surat", "Nagpur", "Indore", "Vadodara", "Nashik", "Rajkot", "Bhopal"],
}
# within each region the first cities are the big ones
CITY_WEIGHT = np.array([0.22, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06])
METROS = {"Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Mumbai", "Pune"}
SAME_DAY_CITIES = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai"]

CATEGORIES = pd.DataFrame(
    {
        "category": ["Fashion", "Home", "Beauty", "Electronics", "Grocery", "Kids", "Sports", "Books"],
        "gross_margin_pct": [0.42, 0.36, 0.48, 0.14, 0.16, 0.38, 0.32, 0.26],
        "shipping_cost_per_order": [7.2, 9.5, 5.8, 8.8, 8.0, 7.0, 8.5, 5.5],  # US dollars
        "weight": [0.24, 0.14, 0.14, 0.12, 0.16, 0.08, 0.06, 0.06],
        "value_multiplier": [1.0, 1.35, 0.7, 2.4, 0.55, 0.9, 1.1, 0.5],
    }
)

# Diwali (Lakshmi Puja) dates; the festive season is the 30 days before Diwali
DIWALI = {2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-21", 2026: "2026-11-08"}


def city_table() -> pd.DataFrame:
    rows = []
    for r, cs in CITIES.items():
        for i, c in enumerate(cs):
            tier = 1 if c in METROS else (2 if i < 6 else 3)
            rows.append((c, r, tier, CITY_WEIGHT[i] * REGION_SHARE[REGIONS.index(r)]))
    t = pd.DataFrame(rows, columns=["city", "region", "tier", "weight"])
    t["weight"] /= t["weight"].sum()
    return t


def festival_calendar(start: str, end: str) -> pd.DataFrame:
    """Daily calendar: festive season (30 days to Diwali), Diwali week, PetalMart's big sales, national holidays."""
    d = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"date": d})
    cal["festive_season"] = 0
    cal["diwali_week"] = 0
    for s in DIWALI.values():
        t = (d - pd.Timestamp(s)).days
        cal.loc[(t >= -30) & (t <= 0), "festive_season"] = 1
        cal.loc[(t >= -3) & (t <= 3), "diwali_week"] = 1
    md = d.strftime("%m-%d")
    # PetalMart Days (first ten days of the festive season), end-of-season sales, Republic Day and Independence Day sales
    sale = np.zeros(len(d), dtype=int)
    for s in DIWALI.values():
        t = (d - pd.Timestamp(s)).days
        sale[(t >= -30) & (t < -20)] = 1
    sale[(md >= "01-01") & (md <= "01-10")] = 1
    sale[(md >= "07-01") & (md <= "07-10")] = 1
    sale[(md >= "01-20") & (md <= "01-26")] = 1
    sale[(md >= "08-10") & (md <= "08-15")] = 1
    cal["sale_day"] = sale
    cal["national_holiday"] = np.isin(md, ["01-26", "08-15", "10-02"]).astype(int)
    return cal
