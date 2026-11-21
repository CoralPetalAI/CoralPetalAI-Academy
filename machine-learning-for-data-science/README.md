# Machine Learning for Data Science

Intermediate · about 52 hours · 12 modules, 2 portfolio projects and a capstone · Coral Petal AI Academy

You are the data scientist on the Coral Petal analytics team's engagement with **PetalBank**, a digital lender. Every module solves one of its problems: loan sizing, credit risk, card fraud, customer segments, branch demand and product recommendations. All code is Python.

## Modules

| # | Module |
|---|---|
| 1 | How machines learn |
| 2 | Preparing data for machine learning |
| 3 | Regression |
| 4 | Classification |
| 5 | Features, validation and leakage |
| 6 | Ensembles |
| 7 | More algorithms and choosing between them |
| 8 | Unsupervised learning |
| 9 | Forecasting with machine learning |
| 10 | Recommender systems and ranking |
| 11 | Machine learning in production |
| 12 | Career lab |

Portfolio Project 1 (after Module 05): PetalBank Card Fraud Detection. Portfolio Project 2 (after Module 10): PetalBank Cross-Sell Engine. Capstone: PetalBank Credit Default Model in Production.

## Datasets

All data is Coral Petal synthetic, generated with seed 2026 so every learner gets identical files. Run the generators from this folder, for example `python datasets/generate_petalbank_loans.py --out datasets/data/petalbank_loans`.

| # | Generator | Dataset | Used in |
|---|---|---|---|
| 1 | `generate_petalbank_loans.py` | 80,000 loan applications in 6 tables, 6.2% default, Q3 2026 drift file | Builds 0, 1, 3, 6, 7, 8, 11 and the capstone |
| 2 | `generate_petalbank_applicant_notes.py` | 20,000 applications with underwriter notes and missing values | Build 2 |
| 3 | `generate_petalbank_card_txn.py` | 1.2 million card transactions, 0.8% fraud, fraud ring in May–June | Build 4, Portfolio Project 1 |
| 4 | `generate_petalbank_loans_leaky.py` | 30,000 loans with three planted leaks | Build 5 |
| 5 | `generate_petalbank_branch_applications.py` | daily applications for 300 branches, 2023–2026 | Build 9 |
| 6 | `generate_petalbank_holdings.py` | 120,000 customers, 14 products, 18 months of holdings and weekly offers | Build 10, Portfolio Project 2 |

## Folder

```
datasets/                               data generators (output goes to datasets/data/, not committed)
builds/                                 starter notebooks for Builds 0–12 and the shared loader petalbank.py
project-1/                              Portfolio Project 1 brief, starter and autograder
project-2/                              Portfolio Project 2 brief, starter and autograder
capstone/
  README.md                             brief, tasks, rubric and how to submit
  starter/capstone_starter.py           notebook with TODOs; writes results.json
  autograder/                           pytest checks for the autograded rubric criteria
```

