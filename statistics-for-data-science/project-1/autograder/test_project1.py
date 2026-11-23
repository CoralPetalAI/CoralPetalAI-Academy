"""Autograder for Portfolio Project 1, PetalMart Customer Survey Study.

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


REQUIRED = ["cleaning", "weighting", "tests", "multiple_testing", "findings"]


def _filled(v):
    if isinstance(v, dict):
        return all(_filled(x) for x in v.values())
    if isinstance(v, list):
        return all(_filled(x) for x in v)
    return v is not None


def close(a, b, rel=0.02, abs_=1e-3):
    return a is not None and math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def test_results_complete(res):
    missing = [k for k in REQUIRED if k not in res or not _filled(res[k])]
    assert not missing, f"fill these in results.json: {missing}"


def test_sanity(res):
    if not all(_filled(res.get(k)) for k in ["cleaning", "weighting", "multiple_testing"]):
        pytest.skip("fill in results.json first")
    c, w, m = res["cleaning"], res["weighting"], res["multiple_testing"]
    assert c["n_clean"] + c["n_straightliners_removed"] == c["n_raw"]
    assert 0 < c["n_straightliners_removed"] < 0.1 * c["n_raw"], "remove only straight-lined AND fast responses"
    lo, hi = w["app_ease_weighted_ci"]
    assert lo < w["app_ease_weighted"] < hi
    assert m["n_significant_bh"] <= m["n_significant_raw"] <= m["n_tests"]


# Criterion 1 · Cleaning and weighting (25)
def test_cleaning_and_weighting(res, exp):
    c, e = res["cleaning"], exp["cleaning"]
    assert c["n_straightliners_removed"] == e["n_straightliners_removed"], "apply the README rule exactly"
    assert c["n_clean"] == e["n_clean"]
    w, ew = res["weighting"], exp["weighting"]
    for band, v in ew["age_weights"].items():
        assert close(w["age_weights"][band], v), f"weight for {band}"
    assert close(w["app_ease_weighted"], ew["app_ease_weighted"], rel=0.005)
    assert close(w["app_ease_unweighted"], ew["app_ease_unweighted"], rel=0.005)
    assert w["app_ease_weighted"] < w["app_ease_unweighted"], "older customers rate the app lower and were under-sampled"


# Criterion 2 · Correct tests and confidence intervals (30)
def test_planned_tests(res, exp):
    t, e = res["tests"], exp["tests"]
    assert t["delivery_speed_by_region"]["lowest_region"] == e["delivery_speed_by_region"]["lowest_region"]
    assert t["delivery_speed_by_region"]["anova_p"] < 1e-10
    assert close(t["value_returning_vs_new"]["mean_diff"], e["value_returning_vs_new"]["mean_diff"], rel=0.02)
    assert t["value_returning_vs_new"]["welch_p"] < 1e-8
    assert t["payment_by_region"]["dof"] == e["payment_by_region"]["dof"]
    assert t["payment_by_region"]["chi2_p"] < 1e-20
    assert t["app_ease_by_age"]["kruskal_p"] < 1e-20
    lo, hi = res["weighting"]["app_ease_weighted_ci"]
    elo, ehi = exp["weighting"]["app_ease_weighted_ci"]
    assert close(lo, elo, rel=0.005) and close(hi, ehi, rel=0.005), "weighted CI with DescrStatsW as in the README"


# Criterion 3 · Effect sizes and multiple testing (20)
def test_effect_sizes_and_multiple_testing(res, exp):
    t, e = res["tests"], exp["tests"]
    assert close(t["delivery_speed_by_region"]["eta_squared"], e["delivery_speed_by_region"]["eta_squared"], rel=0.05)
    assert close(t["value_returning_vs_new"]["cohens_d"], e["value_returning_vs_new"]["cohens_d"], rel=0.05)
    assert close(t["payment_by_region"]["cramers_v"], e["payment_by_region"]["cramers_v"], rel=0.05)
    assert close(t["app_ease_by_age"]["epsilon_squared"], e["app_ease_by_age"]["epsilon_squared"], rel=0.05)
    m, em = res["multiple_testing"], exp["multiple_testing"]
    assert m["n_tests"] == em["n_tests"]
    assert m["n_significant_raw"] == em["n_significant_raw"]
    assert m["n_significant_bh"] == em["n_significant_bh"]
    assert sorted(m["bh_survivors"]) == sorted(em["bh_survivors"])
