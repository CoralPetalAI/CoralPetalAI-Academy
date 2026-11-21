"""Shared helpers for the Machine Learning for Data Science builds.

Every build notebook imports this file. It finds the generated data, joins the
PetalBank tables and checks your answers without revealing them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "datasets" / "data"


def _need(name: str) -> Path:
    d = DATA / name
    if not d.exists():
        raise FileNotFoundError(f"{d} not found. Run: python datasets/generate_{name}.py --out datasets/data/{name}")
    return d


def loans() -> pd.DataFrame:
    """One row per loan application, with applicant, bureau, account and outcome columns."""
    d = _need("petalbank_loans")
    df = pd.read_csv(d / "applications.csv", parse_dates=["application_date"])
    for t in ["applicants", "bureau", "transactions_summary"]:
        df = df.merge(pd.read_csv(d / f"{t}.csv"), on="customer_id", how="left")
    return df.merge(pd.read_csv(d / "outcomes.csv"), on="application_id", how="left")


def loans_q3() -> pd.DataFrame:
    return pd.read_csv(_need("petalbank_loans") / "applications_2026q3.csv", parse_dates=["application_date"])


def applicant_notes() -> pd.DataFrame:
    return pd.read_csv(_need("petalbank_applicant_notes") / "applications_with_notes.csv", parse_dates=["application_date"])


def loans_leaky() -> pd.DataFrame:
    return pd.read_csv(_need("petalbank_loans_leaky") / "loans.csv", parse_dates=["application_date"])


def card_txn() -> pd.DataFrame:
    d = _need("petalbank_card_txn")
    t = pd.read_csv(d / "transactions.csv", parse_dates=["timestamp"])
    t = t.merge(pd.read_csv(d / "fraud_labels.csv"), on="txn_id").merge(pd.read_csv(d / "cards.csv"), on="card_id")
    return t.sort_values("timestamp").reset_index(drop=True)


def branch_applications() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = _need("petalbank_branch_applications")
    return (pd.read_csv(d / "daily_applications.csv", parse_dates=["date"]), pd.read_csv(d / "branches.csv"),
            pd.read_csv(d / "calendar.csv", parse_dates=["date"]))


def holdings() -> dict[str, pd.DataFrame]:
    d = _need("petalbank_holdings")
    return {n: pd.read_csv(d / f"{n}.csv") for n in ["customers", "products", "holdings", "offers"]}


def check(results: dict, rules: dict | None = None) -> None:
    """Print which answers are still TODO, then run the build's sanity rules on the filled ones."""
    todo = [k for k, v in results.items() if v is None]
    for k, rule in (rules or {}).items():
        if results.get(k) is not None:
            ok, msg = rule(results[k])
            print(("✅ " if ok else "❌ ") + f"{k}: {msg}")
    print("Still TODO: " + ", ".join(todo) if todo else "All answers filled. Push your notebook to GitHub.")
