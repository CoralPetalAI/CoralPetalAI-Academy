"""Autograder for Portfolio Project 1, PetalBank Card Fraud Detection.

Self-check (learners):
    pytest autograder/test_project1.py --results starter/results.json
    Runs the completeness and sanity checks only; the expected values are held by the Academy grader.

Full grading (Academy):
    pytest autograder/test_project1.py --results <learner results.json> --expected <expected_results.json>

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


REQUIRED = ["framing", "split", "leakage", "cv_pr_auc", "pr_auc_test", "threshold", "error_analysis", "recommendation"]


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


def test_sanity(res):
    if not all(_filled(res.get(k)) for k in ["threshold", "pr_auc_test"]):
        pytest.skip("fill in results.json first")
    assert res["threshold"]["alerts_per_day"] <= 360, "the threshold breaks the review capacity"
    assert 0 < res["threshold"]["threshold"] < 1


# Criterion 1 · Validation and leakage (25)
def test_split_and_leakage(res, exp):
    assert res["framing"]["n_transactions"] == exp["framing"]["n_transactions"]
    assert close(res["framing"]["fraud_rate"], exp["framing"]["fraud_rate"], abs_=2e-4)
    assert res["split"]["n_train"] == exp["split"]["n_train"], "split by timestamp at 2026-05-01"
    assert res["split"]["n_test"] == exp["split"]["n_test"]
    assert res["leakage"]["leaked_feature"] == exp["leakage"]["leaked_feature"]
    assert res["leakage"]["leaked_pr_auc"] > 0.8
    assert max(res["pr_auc_test"].values()) < 0.7, "a test PR-AUC this high suggests the leak is still in"


# Criterion 2 · Models and metrics (25)
def test_models(res, exp):
    for k in ["logistic", "gradient_boosting"]:
        assert 0.6 <= res["cv_pr_auc"][k] <= 0.85, f"{k} CV PR-AUC out of range"
        assert 0.35 <= res["pr_auc_test"][k] <= 0.6, f"{k} test PR-AUC out of range"
    assert min(res["cv_pr_auc"].values()) > max(res["pr_auc_test"].values()), "the test months should score lower (drift)"


# Criterion 3 · Cost-based threshold under capacity (25)
def test_threshold(res, exp):
    t, e = res["threshold"], exp["threshold"]
    assert close(t["cost_no_model"], e["cost_no_model"], rel=0.001), "check the cost rule"
    assert t["alerts_per_day"] <= 360
    assert t["expected_cost_test"] <= 1.15 * e["expected_cost_test"]
    assert t["expected_cost_test"] < 0.5 * t["cost_no_model"]


# Error analysis numbers (feed criterion 4)
def test_error_analysis(res, exp):
    ea = res["error_analysis"]
    assert ea["recall_ring_fraud"] < ea["recall_other_fraud"], "the ring should be the weak spot"
    assert ea["ring_merchants"] == exp["error_analysis"]["ring_merchants"]
