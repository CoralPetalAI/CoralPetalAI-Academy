# Statistics for Data Science

Intermediate · about 54 hours · 12 modules, 2 portfolio projects and a capstone · Coral Petal AI Academy

You are the data scientist on the Coral Petal analytics team's engagement with **PetalMart**, an online retailer selling across India. Every module answers one question PetalMart's leadership asks of its data.

## Modules

| # | Module |
|---|---|
| 1 | The data scientist's toolkit (NumPy, pandas, SQL, plotting) |
| 2 | Describing data |
| 3 | Probability |
| 4 | Distributions |
| 5 | Inference |
| 6 | Bayesian statistics |
| 7 | Regression |
| 8 | Beyond linear models |
| 9 | Time series analysis |
| 10 | Experiments |
| 11 | Causal inference |
| 12 | Career lab |

Portfolio Project 1 (after Module 05): PetalMart Customer Survey Study. Portfolio Project 2 (after Module 09): PetalMart Demand Forecast. Capstone: The PetalMart Free-Shipping Decision.

## Datasets

All data is Coral Petal synthetic, generated with seed 2026 so every learner gets identical files. No real customer data is used. Run the generators from this folder, for example `python datasets/generate_petalmart_core.py`; output goes to `datasets/data/<dataset>/` (not committed).

| # | Generator | Dataset | Used in |
|---|---|---|---|
| 1 | `generate_petalmart_core.py` | 30,000 users, 50,000 orders in FY2025-26, sessions, deliveries and category costs, with planted problems to clean | Builds 0, 1, 2, 4, 5 |
| 2 | `generate_petalmart_customers.py` | 12,000 customers with complaints, tenure, device and churn | Builds 3, 7 |
| 3 | `generate_petalmart_survey.py` | 4,800 survey responses, 22 items, non-response bias and straight-lining | Portfolio Project 1 |
| 4 | `generate_petalmart_city_conversion.py` | visitors and buyers of a new category in 40 cities | Build 6 |
| 5 | `generate_petalmart_weekly.py` | 52 weeks of marketing spend, discount, season and revenue | Build 7 |
| 6 | `generate_petalmart_repeat.py` | 15,000 customers' orders over six months with exposure | Build 8 |
| 7 | `generate_petalmart_demand.py` | three years of daily orders for 10 categories in 4 regions, prices, promotions, festival calendar | Build 9, Portfolio Project 2 |
| 8 | `generate_petalmart_checkout_test.py` | the one-page checkout A/B test, 20,000 users | Build 10 |
| 9 | `generate_petalmart_same_day.py` | same-day delivery launch: 40 cities by month, matched customers, a randomized retention offer | Build 11 |
| 10 | `generate_petalmart_shipping_test.py` | the free-shipping A/B test and the city rollout | Capstone |

`petalmart_common.py` holds what every generator shares: cities, regions, categories and the festival calendar.

## Folder

```
datasets/                               data generators (output goes to datasets/data/, not committed)
capstone/
  README.md                             brief, tasks, rubric and how to submit
  starter/capstone_starter.py           notebook with TODOs; writes results.json
  autograder/                           pytest checks for rubric criteria 1 to 7
```
