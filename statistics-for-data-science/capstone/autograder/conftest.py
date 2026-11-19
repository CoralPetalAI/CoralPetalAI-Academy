def pytest_addoption(parser):
    parser.addoption("--results", default="results.json", help="learner results.json")
    parser.addoption("--expected", default="", help="expected results (Academy grader only)")
