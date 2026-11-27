"""Shared helpers for the Statistics for Data Science builds.

Every build notebook imports this file. It finds the generated PetalMart data, loads
each dataset with the right types, applies the standard cleaning from Build 2 when a
later build needs clean orders, and checks your answers without revealing them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "datasets" / "data"


def _need(name: str) -> Path:
    d = DATA / name
    if not d.exists():
        raise FileNotFoundError(f"{d} not found. From the course folder run: python datasets/generate_{name}.py")
    return d


def core() -> dict[str, pd.DataFrame]:
    """Raw PetalMart tables, problems included: users, orders, sessions, costs, deliveries."""
    d = _need("petalmart_core")
    return {
        "users": pd.read_csv(d / "users.csv", parse_dates=["signup_date"]),
        "orders": pd.read_csv(d / "orders.csv", parse_dates=["order_ts"]),
        "sessions": pd.read_csv(d / "sessions.csv", parse_dates=["session_start"]),
        "costs": pd.read_csv(d / "costs.csv"),
        "deliveries": pd.read_csv(d / "deliveries.csv"),
    }


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """The standard cleaning you worked out in Build 2, for later builds.

    Drops exact duplicate rows and orders worth zero or less, and divides the orders recorded
    in cents instead of dollars (value per item above $2,000) by 100. Orders from deleted accounts are kept;
    they drop out when you join to users.
    """
    o = orders.drop_duplicates()
    o = o[o.order_value > 0].copy()
    cents = o.order_value / o["items"] > 2_000
    o.loc[cents, "order_value"] = o.loc[cents, "order_value"] / 100
    return o.reset_index(drop=True)


def customers() -> pd.DataFrame:
    return pd.read_csv(_need("petalmart_customers") / "customers.csv")


def city_conversion() -> pd.DataFrame:
    return pd.read_csv(_need("petalmart_city_conversion") / "city_conversion.csv")


def weekly() -> pd.DataFrame:
    w = pd.read_csv(_need("petalmart_weekly") / "weekly_revenue.csv", parse_dates=["week_start"], keep_default_na=False)
    w["season"] = pd.Categorical(w.season, categories=["regular", "festive", "sale"])  # regular is the baseline
    return w


def repeat_purchases() -> pd.DataFrame:
    return pd.read_csv(_need("petalmart_repeat") / "repeat_purchases.csv")


def demand() -> dict[str, pd.DataFrame]:
    d = _need("petalmart_demand")
    return {"daily_orders": pd.read_csv(d / "daily_orders.csv", parse_dates=["date"]),
            "calendar": pd.read_csv(d / "calendar.csv", parse_dates=["date"]),
            "warehouses": pd.read_csv(d / "warehouses.csv", parse_dates=["opened"])}


def checkout_test() -> pd.DataFrame:
    return pd.read_csv(_need("petalmart_checkout_test") / "checkout_test.csv", parse_dates=["assigned_date"])


def same_day() -> dict[str, pd.DataFrame]:
    d = _need("petalmart_same_day")
    return {n: pd.read_csv(d / f"{n}.csv") for n in ["city_month", "customers", "retention_campaign"]}


def _digest(value, digits):
    if digits is not None:
        value = round(float(value), digits)
        value = int(value) if digits == 0 else value
    return hashlib.sha256(repr(value).encode()).hexdigest()[:12]


def matches(digest: str, digits: int | None = None, hint: str = "check your working"):
    """A check rule that compares your answer, rounded to `digits`, with the right one, without showing it."""
    return lambda v: (_digest(v, digits) == digest, "matches" if _digest(v, digits) == digest else hint)


def between(lo: float, hi: float, hint: str):
    """A check rule for answers that depend on your choices: is it in a sensible range?"""
    return lambda v: (lo <= float(v) <= hi, "in a sensible range" if lo <= float(v) <= hi else hint)


def check(results: dict, rules: dict | None = None) -> None:
    """Print which answers are still TODO, then run the build's checks on the filled ones."""
    todo = [k for k, v in results.items() if v is None]
    for k, rule in (rules or {}).items():
        if results.get(k) is not None:
            try:
                ok, msg = rule(results[k])
            except (TypeError, ValueError):
                ok, msg = False, "unexpected type: see the TODO comment for the format"
            print(("✅ " if ok else "❌ ") + f"{k}: {msg}")
    print("Still TODO: " + ", ".join(todo) if todo else "All answers filled. Restart, run all, then push your notebook to GitHub.")
