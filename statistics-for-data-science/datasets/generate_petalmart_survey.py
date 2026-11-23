"""Generate the Coral Petal synthetic dataset `petalmart_survey`.

Used in: Portfolio Project 1 (PetalMart Customer Survey Study), Statistics for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- 4,800 responses to PetalMart's January 2026 customer survey: region, age band, tenure,
  spend band, customer type and device, then 22 items: 16 five-point Likert items on
  delivery, pricing, the app and support, overall satisfaction (1 to 5), likelihood to
  recommend (0 to 10) and four categorical items.
- Non-response bias: older customers answered less often, so customers aged 45 and over are
  under-represented compared with the customer base (population_margins.csv gives the base).
  Older customers rate the app lower, so unweighted estimates flatter the app.
- About 3% of responses are straight-lined: the same answer to every Likert item, submitted
  in under two and a half minutes. They should be found and removed.
- A few differences are real: delivery speed is rated lower in the East, the app lower by
  older customers, value for money lower by new customers, and the preferred payment method
  differs by region (more cash on delivery in the East). Most other items do not differ by
  group, so testing every item against every group will throw up chance "findings".
- About 2% of answers to each item are missing (skipped questions).

Usage:
    python generate_petalmart_survey.py --out data/petalmart_survey --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from petalmart_common import REGION_SHARE, REGIONS

N = 4_800
AGES = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_POP = [0.18, 0.36, 0.25, 0.13, 0.08]       # customer base
AGE_RESP = [0.24, 0.40, 0.23, 0.09, 0.04]      # who answered
LIKERT = [
    "q01_delivery_speed", "q02_delivery_tracking", "q03_packaging", "q04_courier_courtesy", "q05_delivery_slots",
    "q06_value_for_money", "q07_discount_clarity", "q08_price_vs_competitors", "q09_shipping_fee_fair",
    "q10_app_ease", "q11_search_results", "q12_checkout_ease", "q13_payment_options", "q14_app_stability",
    "q15_support_response", "q16_returns_process",
]
CUTS = np.array([-1.6, -0.75, 0.15, 1.05])  # latent score to 1..5


def likert(latent: np.ndarray) -> np.ndarray:
    return 1 + np.searchsorted(CUTS, latent)


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    age = rng.choice(AGES, N, p=AGE_RESP)
    region = rng.choice(REGIONS, N, p=REGION_SHARE)
    tenure = rng.choice(["<6m", "6-24m", "2y+"], N, p=[0.22, 0.38, 0.40])
    ctype = np.where(tenure == "<6m", "new", "returning")
    spend = rng.choice(["low", "medium", "high"], N, p=[0.40, 0.40, 0.20])
    older = np.isin(age, ["45-54", "55+"])
    app = rng.random(N) < np.where(older, 0.38, 0.62)
    mood = rng.normal(0, 0.55, N)  # some people rate everything higher

    shift = {k: np.zeros(N) for k in LIKERT}
    shift["q01_delivery_speed"] += np.where(region == "E", -0.40, 0) - 0.10 * older
    shift["q04_courier_courtesy"] += np.where(region == "E", -0.10, 0)
    shift["q06_value_for_money"] += np.where(ctype == "new", -0.30, 0)
    shift["q10_app_ease"] += np.select([age == "55+", age == "45-54", age == "35-44"], [-0.90, -0.55, -0.15], 0.10)
    shift["q12_checkout_ease"] += -0.30 * older
    shift["q14_app_stability"] += np.where(app, 0.10, -0.05)
    shift["q09_shipping_fee_fair"] += np.where(spend == "low", -0.20, 0)
    out = {}
    for k in LIKERT:
        out[k] = likert(0.35 + mood + shift[k] + rng.normal(0, 0.85, N))
    sat_latent = (0.25 * (out["q01_delivery_speed"] - 3) + 0.2 * (out["q06_value_for_money"] - 3)
                  + 0.2 * (out["q10_app_ease"] - 3) + 0.15 * (out["q15_support_response"] - 3) + mood + rng.normal(0, 0.5, N))
    out["q17_overall_satisfaction"] = likert(0.30 + sat_latent)
    out["q18_recommend_0_10"] = np.clip(np.round(6.8 + 1.6 * sat_latent + rng.normal(0, 1.2, N)), 0, 10).astype(int)
    pay_p = {"N": [0.55, 0.20, 0.17, 0.08], "S": [0.60, 0.21, 0.12, 0.07], "E": [0.40, 0.17, 0.35, 0.08],
             "W": [0.58, 0.22, 0.12, 0.08]}
    out["q19_preferred_payment"] = np.array([rng.choice(["UPI", "card", "COD", "wallet"], p=pay_p[r]) for r in region])
    out["q20_preferred_slot"] = rng.choice(["morning", "afternoon", "evening", "weekend"], N, p=[0.22, 0.18, 0.38, 0.22])
    out["q21_shop_frequency"] = np.where(spend == "high", rng.choice(["weekly", "monthly", "occasionally"], N, p=[0.45, 0.45, 0.10]),
                                         rng.choice(["weekly", "monthly", "occasionally"], N, p=[0.15, 0.50, 0.35]))
    out["q22_main_category"] = rng.choice(["Fashion", "Home", "Beauty", "Electronics", "Grocery", "Kids", "Sports", "Books"], N,
                                          p=[0.24, 0.14, 0.14, 0.12, 0.16, 0.08, 0.06, 0.06])
    duration = np.round(rng.lognormal(np.log(420), 0.35, N)).astype(int)

    # straight-liners: same answer everywhere, very fast
    sl = rng.random(N) < 0.03
    val = rng.choice([3, 4, 5], N, p=[0.25, 0.30, 0.45])
    for k in LIKERT + ["q17_overall_satisfaction"]:
        out[k] = np.where(sl, val, out[k])
    duration = np.where(sl, rng.integers(55, 150, N), duration)

    t = pd.DataFrame({
        "response_id": [f"SV{i:05d}" for i in range(1, N + 1)],
        "submitted_at": (pd.Timestamp("2026-01-12") + pd.to_timedelta(rng.integers(0, 14 * 24 * 3600, N), "s")).strftime("%Y-%m-%d %H:%M"),
        "duration_sec": duration, "region": region, "age_band": age, "tenure_band": tenure, "customer_type": ctype,
        "spend_band": spend, "device": np.where(app, "app", "web"), **out,
    })
    items = [c for c in t.columns if c.startswith("q")]
    for k in items:  # skipped questions
        t[k] = t[k].astype(object)
        t.loc[rng.random(N) < 0.02, k] = np.nan
    t = t.sort_values("submitted_at").reset_index(drop=True)

    margins = pd.DataFrame(
        [("age_band", a, p) for a, p in zip(AGES, AGE_POP)] + [("region", r, p) for r, p in zip(REGIONS, REGION_SHARE)],
        columns=["variable", "level", "population_share"])
    codebook = pd.DataFrame({"item": items, "scale": ["1-5 (1 = very poor, 5 = excellent)"] * 17 + ["0-10"] + ["categorical"] * 4})
    return {"responses": t, "population_margins": margins, "codebook": codebook}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "petalmart_survey"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
