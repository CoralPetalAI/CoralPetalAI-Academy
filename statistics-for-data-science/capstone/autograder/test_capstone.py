"""Autograder for the PetalMart Free-Shipping capstone.

Self-check (learners):
    pytest autograder/test_capstone.py --results starter/results.json
    Runs the completeness check only; the expected values are held by the Academy grader.

Full grading (Academy):
    pytest autograder/test_capstone.py --results <learner results.json> --expected <expected_results.json>

Each test maps to a rubric criterion. The memo and readout (criterion 8) are reviewed by an instructor.
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


REQUIRED = [
    "cleaning", "srm_p_value", "balance_p_value_customer_type", "conversion_diff_pp", "conversion_diff_ci_pp",
    "conversion_p_value", "aov_diff", "aov_diff_ci", "p_treatment_better_conversion", "profit_per_user",
    "profit_lift_per_user_by_segment", "logit_interaction_coef", "logit_interaction_p",
    "rollout_naive_lift", "rollout_did_lift", "recommendation",
]


def _filled(v):
    if isinstance(v, dict):
        return all(_filled(x) for x in v.values())
    if isinstance(v, list):
        return all(_filled(x) for x in v)
    return v is not None


def test_results_complete(res):
    missing = [k for k in REQUIRED if k not in res or not _filled(res[k])]
    assert not missing, f"fill these in results.json: {missing}"


def close(a, b, rel=0.05, abs_=1e-3):
    return a is not None and math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


# Criterion 1 · Data pull and cleaning (10)
def test_cleaning(res, exp):
    for k in ["duplicates_removed", "invalid_values_removed", "orders_clean", "users_missing_region"]:
        assert res["cleaning"][k] == exp["cleaning"][k], k


# Criterion 2 · Validity checks (10)
def test_validity(res, exp):
    assert close(res["srm_p_value"], exp["srm_p_value"], rel=0.02)
    assert res["srm_p_value"] > 0.05, "the split should pass the SRM check"
    assert close(res["balance_p_value_customer_type"], exp["balance_p_value_customer_type"], rel=0.05)


# Criterion 3 · Tests, CIs and effect sizes (15)
def test_conversion_and_aov(res, exp):
    assert close(res["conversion_diff_pp"], exp["conversion_diff_pp"], rel=0.02)
    lo, hi = res["conversion_diff_ci_pp"]
    assert close(lo, exp["conversion_diff_ci_pp"][0], rel=0.05) and close(hi, exp["conversion_diff_ci_pp"][1], rel=0.05)
    assert close(res["conversion_p_value"], exp["conversion_p_value"], rel=0.25, abs_=1e-4)
    assert close(res["aov_diff"], exp["aov_diff"], rel=0.02)
    lo, hi = res["aov_diff_ci"]
    assert lo < exp["aov_diff"] < hi and hi < 0, "bootstrap CI should exclude zero"


# Criterion 4 · Bayesian analysis (10)
def test_bayesian(res, exp):
    for seg in ["new", "returning"]:
        assert abs(res["p_treatment_better_conversion"][seg] - exp["p_treatment_better_conversion"][seg]) < 0.03, seg


# Criterion 5 · Profit model (10)
def test_profit(res, exp):
    for v in ["control", "treatment"]:
        assert close(res["profit_per_user"][v], exp["profit_per_user"][v], rel=0.02), v
    for seg in ["new", "returning"]:
        assert close(res["profit_lift_per_user_by_segment"][seg], exp["profit_lift_per_user_by_segment"][seg], rel=0.05, abs_=0.05), seg


# Criterion 6 · Regression (10)
def test_regression(res, exp):
    assert close(res["logit_interaction_coef"], exp["logit_interaction_coef"], rel=0.05)
    assert res["logit_interaction_p"] < 0.01


# Criterion 7 · Causal cross-check (15)
def test_causal(res, exp):
    assert close(res["rollout_naive_lift"], exp["rollout_naive_lift"], rel=0.05)
    assert close(res["rollout_did_lift"], exp["rollout_did_lift"], rel=0.10)
    assert res["rollout_naive_lift"] > 2 * res["rollout_did_lift"], "naive comparison should overstate the effect"


# Criterion 8 · Recommendation (memo and slides reviewed by an instructor)
def test_recommendation(res, exp):
    assert res["recommendation"] == exp["recommendation"]
