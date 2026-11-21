"""Autograder for Portfolio Project 2, PetalBank Cross-Sell Engine.

Self-check (learners):
    pytest autograder/test_project2.py --results starter/results.json
    Runs the completeness check only; the expected values are held by the Academy grader.

Full grading (Academy):
    pytest autograder/test_project2.py --results <learner results.json> --expected <expected_results.json>

The report (criterion 4) is reviewed by an instructor.
"""
import json
import math
import os

import pytest


@pytest.fixture(scope="session")
def res(request):
    return json.load(open(request.config.getoption("--results")))


@pytest.fixture(scope="session")
def exp(request):
    path = request.config.getoption("--expected")
    if not path or not os.path.exists(path):
        pytest.skip("expected results are held by the Academy grader")
    return json.load(open(path))


REQUIRED = ["framing", "split", "recommenders", "eligibility", "cold_start", "forecast", "recommendation"]


def _filled(v):
    if isinstance(v, dict):
        return all(_filled(x) for x in v.values())
    if isinstance(v, list):
        return all(_filled(x) for x in v)
    return v is not None


def close(a, b, rel=0.05, abs_=1e-3):
    return a is not None and math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def test_results_complete(res):
    missing = [k for k in REQUIRED if k not in res or not _filled(res[k])]
    assert not missing, f"fill these in results.json: {missing}"


# Criterion 2 · Evaluation (25)
def test_split_and_baseline(res, exp):
    assert res["framing"] == exp["framing"]
    assert res["split"]["n_test_customers"] == exp["split"]["n_test_customers"], "check the test population"
    assert res["split"]["n_test_adoptions"] == exp["split"]["n_test_adoptions"]
    p = res["recommenders"]["popularity"]
    assert close(p["precision_at_5"], exp["recommenders"]["popularity"]["precision_at_5"], rel=0.03)
    assert close(p["ndcg_at_5"], exp["recommenders"]["popularity"]["ndcg_at_5"], rel=0.05)


# Criterion 1 · Recommender quality (30)
def test_recommenders(res, exp):
    r, e = res["recommenders"], exp["recommenders"]
    assert r["item_cf"]["ndcg_at_5"] > r["popularity"]["ndcg_at_5"]
    best = max(r[m]["ndcg_at_5"] for m in ["item_cf", "hybrid"])
    assert best >= e["hybrid"]["ndcg_at_5"] - 0.03
    assert best <= 1


# Criterion 3 · Cold start and rules (20)
def test_cold_start_and_rules(res, exp):
    assert res["cold_start"]["n_customers"] == exp["cold_start"]["n_customers"]
    assert res["cold_start"]["segment_popularity"] >= res["cold_start"]["popularity"]
    assert close(res["eligibility"]["share_ineligible_in_unfiltered_popularity_top5"],
                 exp["eligibility"]["share_ineligible_in_unfiltered_popularity_top5"], rel=0.1)


# Forecast numbers (feed criteria 2 and 4)
def test_forecast(res, exp):
    f = res["forecast"]
    assert f["weeks_test"] == 13 and f["total_offers_test_weeks"] == exp["forecast"]["total_offers_test_weeks"]
    assert close(f["mase_seasonal_naive"], exp["forecast"]["mase_seasonal_naive"], rel=0.03)
    assert f["mase_model"] < f["mase_seasonal_naive"], "the model should beat the seasonal naive"
