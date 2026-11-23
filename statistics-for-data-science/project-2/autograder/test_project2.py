"""Autograder for Portfolio Project 2, PetalMart Demand Forecast.

Self-check (learners):
    pytest autograder/test_project2.py --results starter/results.json
    Runs the completeness and sanity checks only; the expected values are held by the Academy grader.

Full grading (Academy):
    pytest autograder/test_project2.py --results <learner results.json> --expected <expected_results.json>

The planning note (criterion 4) is reviewed by an instructor.
"""
import json
import math
import os
from datetime import date

import pytest

CATEGORIES = ["Appliances", "Beauty", "Books", "Electronics", "Fashion", "Footwear", "Grocery", "Home", "Kids", "Sports"]
MODELS = {"seasonal_naive", "ets", "sarimax"}


@pytest.fixture(scope="session")
def res(request):
    return json.load(open(request.config.getoption("--results")))


@pytest.fixture(scope="session")
def exp(request):
    path = request.config.getoption("--expected")
    if not path or not os.path.exists(path):
        pytest.skip("expected results are held by the Academy grader")
    return json.load(open(path))


REQUIRED = ["diagnostics", "backtest", "forecast", "planning_note"]


def _filled(v):
    if isinstance(v, dict):
        return all(_filled(x) for x in v.values())
    if isinstance(v, list):
        return all(_filled(x) for x in v)
    return v is not None


def close(a, b, rel=0.01, abs_=1e-3):
    return a is not None and math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def test_results_complete(res):
    missing = [k for k in REQUIRED if k not in res or not _filled(res[k])]
    assert not missing, f"fill these in results.json: {missing}"


def test_sanity(res):
    if not all(_filled(res.get(k)) for k in ["backtest", "forecast"]):
        pytest.skip("fill in results.json first")
    assert len(res["forecast"]["weeks"]) == 12
    for c, f in res["forecast"]["by_category"].items():
        assert len(f["point"]) == 12, c
        assert all(lo <= p <= hi for lo, p, hi in zip(f["lower_80"], f["point"], f["upper_80"])), f"{c}: interval order"
    for row in res["backtest"]["table"]:
        assert 0 <= row["coverage_80"] <= 1 and row["model"] in MODELS


# Criterion 1 · Diagnostics (20)
def test_diagnostics(res, exp):
    d, e = res["diagnostics"], exp["diagnostics"]
    assert d["break_region"] == e["break_region"], "compare each region's share of weekly orders before and after"
    gap = abs((date.fromisoformat(d["break_week"]) - date.fromisoformat(e["break_week"])).days)
    assert gap <= 21, "the break week is off by more than three weeks"
    assert d["share_before_after"][1] > d["share_before_after"][0]
    for c in CATEGORIES:
        assert close(d["seasonal_strength"][c], e["seasonal_strength"][c], abs_=0.02), f"STL strength for {c}"


# Criterion 2 · Models and forecast table (25)
def test_forecast_table(res, exp):
    f = res["forecast"]
    assert f["weeks"] == exp["forecast"]["weeks"], "weeks ending 6 September to 22 November 2026"
    assert sorted(f["by_category"]) == CATEGORIES
    for c, row in f["by_category"].items():
        assert row["model"] in MODELS
        assert row["same_weeks_last_year"] == exp["forecast"]["by_category"][c]["same_weeks_last_year"], c
        ratio = row["total_12_weeks"] / row["same_weeks_last_year"]
        assert 0.85 <= ratio <= 1.40, f"{c}: the 12-week total is far from last year's"
        assert all(x > 0 for x in row["lower_80"]), f"{c}: negative lower bound; model on logs"


# Criterion 3 · Honest backtest (30)
def test_backtest(res, exp):
    b, e = res["backtest"], exp["backtest"]
    assert b["n_origins"] == e["n_origins"] == 11
    mine = {(r["category"], r["model"]): r for r in b["table"]}
    for r in e["table"]:
        if r["model"] == "seasonal_naive":
            got = mine[(r["category"], "seasonal_naive")]
            assert close(got["mase"], r["mase"], rel=0.01), f"seasonal naive MASE for {r['category']}: check the scale"
            assert close(got["coverage_80"], r["coverage_80"], abs_=0.02), f"seasonal naive coverage for {r['category']}"
    for c in CATEGORIES:
        rows = {m: mine[(c, m)]["mase"] for m in MODELS}
        assert b["best_model"][c] == min(rows, key=rows.get), f"{c}: best model should have the lowest mean MASE"
    sn = sum(mine[(c, "seasonal_naive")]["mase"] for c in CATEGORIES)
    best = sum(min(mine[(c, m)]["mase"] for m in MODELS) for c in CATEGORIES)
    assert best < 0.85 * sn, "a model with the festival regressors should clearly beat seasonal naive"
