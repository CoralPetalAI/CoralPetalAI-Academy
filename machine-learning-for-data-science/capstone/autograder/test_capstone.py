"""Autograder for the PetalBank Credit Default capstone.

Self-check (learners):
    pytest autograder/test_capstone.py --results starter/results.json
    Runs the completeness and sanity checks only; the expected values are held by the Academy grader.

Full grading (Academy):
    pytest autograder/test_capstone.py --results <learner results.json> --expected <expected_results.json>

Each test maps to a rubric criterion. Pipeline quality (3) and the model card and readout (9)
are reviewed by an instructor.
"""
import json
import math
import os

import pytest

DRIVERS = {"credit_score", "dti", "bounced_payments_6m", "utilization_pct", "employment_years",
           "log_income", "income_monthly", "emi", "debt_to_income"}
BANDS = ["21-25", "26-35", "36-45", "46-60"]


@pytest.fixture(scope="session")
def res(request):
    return json.load(open(request.config.getoption("--results")))


@pytest.fixture(scope="session")
def exp(request):
    path = request.config.getoption("--expected")
    if not path or not os.path.exists(path):
        pytest.skip("expected results are held by the Academy grader")
    return json.load(open(path))


REQUIRED = ["framing", "split", "leakage", "pr_auc", "threshold", "shap_top_features", "segments",
            "fairness", "drift", "q3_scoring", "api_example_response", "recommendation"]


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
    if not all(_filled(res.get(k)) for k in ["framing", "threshold", "api_example_response"]):
        pytest.skip("fill in results.json first")
    assert 0 < res["framing"]["base_rate"] < 0.2
    assert 0 < res["threshold"]["best_threshold"] <= 0.5
    assert res["api_example_response"]["decision"] in {"approve", "reject"}
    assert 0 <= res["api_example_response"]["default_probability"] <= 1


# Criterion 1 · Framing and honest validation (10)
def test_framing_and_split(res, exp):
    assert res["framing"]["n_applications"] == exp["framing"]["n_applications"]
    assert close(res["framing"]["base_rate"], exp["framing"]["base_rate"], abs_=5e-4)
    assert res["split"]["train_cutoff"] == exp["split"]["train_cutoff"]
    assert res["split"]["n_train"] == exp["split"]["n_train"], "split by application_date, not at random"
    assert res["split"]["n_test"] == exp["split"]["n_test"]


# Criterion 2 · Leakage found and removed (10)
def test_leakage(res, exp):
    assert res["leakage"]["leaked_feature"] == exp["leakage"]["leaked_feature"]
    assert res["leakage"]["leaked_pr_auc"] > 0.85, "show how much the leak inflates PR-AUC"
    assert max(res["pr_auc"].values()) < 0.6, "a PR-AUC this high suggests the leak is still in"


# Criterion 4 · Model comparison with correct metrics (15)
def test_models(res, exp):
    gb = res["pr_auc"]["gradient_boosting"]
    assert 0.33 <= gb <= 0.48, f"gradient boosting PR-AUC {gb} is outside the plausible range"
    assert res["pr_auc"]["logistic"] > exp["framing"]["base_rate"] * 3, "logistic should beat random by far"
    assert max(res["pr_auc"].values()) >= exp["pr_auc"]["gradient_boosting"] - 0.03


# Criterion 5 · Profit-based threshold (10)
def test_threshold(res, exp):
    t = res["threshold"]
    assert close(t["profit_approve_all"], exp["threshold"]["profit_approve_all"], rel=0.01), "check the profit rule"
    assert t["expected_profit_test"] >= 0.85 * exp["threshold"]["expected_profit_test"]
    assert t["expected_profit_test"] > t["profit_approve_all"], "the threshold should beat approving everyone"
    assert 0.05 <= t["best_threshold"] <= 0.2


# Criterion 6 · Explainability (10)
def test_shap(res, exp):
    top = res["shap_top_features"]
    assert len(top) == 3 and top[0] in DRIVERS, f"top SHAP feature {top[0]} is not a true driver"
    assert len(set(top) & DRIVERS) >= 2


# Criterion 7 · Fairness check (10)
def test_fairness(res, exp):
    f = res["fairness"]
    for k in BANDS:
        assert 0 <= f["approval_rate_by_age_band"][k] <= 1 and 0 <= f["tpr_good_by_age_band"][k] <= 1
    assert set(f["approval_rate_by_city_tier"]) == {"1", "2", "3"}
    assert res["segments"]["max_segment_default_rate"] > 2 * res["segments"]["min_segment_default_rate"]


# Criterion 8 · Serving, batch scoring and drift (10)
def test_drift_and_serving(res, exp):
    assert close(res["drift"]["psi_income_q3"], exp["drift"]["psi_income_q3"], abs_=0.05)
    assert res["drift"]["psi_income_q3"] > 0.2, "income has shifted in Q3"
    assert res["drift"]["psi_credit_score_q3"] < 0.1
    assert res["q3_scoring"]["n_scored"] == exp["q3_scoring"]["n_scored"]
    api = res["api_example_response"]
    assert set(api) >= {"application_id", "default_probability", "decision", "reason_codes"}
    assert api["application_id"].startswith("Q")
