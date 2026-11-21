"""Generate the Coral Petal synthetic dataset `petalbank_card_txn`.

Used in: Build 4 (fraud classifier with a cost-based threshold) and Portfolio Project 1
(PetalBank Card Fraud Detection), Machine Learning for Data Science.

Data source: fully synthetic, produced by this script from the rules below with a
fixed seed, so every learner gets identical data. Coral Petal AI owns it; no real
customer data is used.

Rules:
- 40,000 cards, 1.2 million transactions from January to June 2026, about 0.8% fraud.
- Classic fraud: a compromised card sees a short burst of high-value online or
  electronics/jewellery spends, often at night, on a new device, far from home.
- Fraud ring in May and June: small digital-goods and gift-card spends on new devices
  close to home, through a handful of ring merchants. A model trained on January to
  April misses much of it (drift).
- Legitimate travel and big-ticket purchases look risky too, so the problem is not trivial.
- `fraud_labels.csv` carries a planted leak: `chargeback_filed` is only known weeks after
  the transaction.

Usage:
    python generate_petalbank_card_txn.py --out data/petalbank_card_txn --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_CARDS = 40_000
N_TXN = 1_200_000
START, END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-01")
RING_START = pd.Timestamp("2026-05-01")
N_CLASSIC_CARDS = 1_350      # compromised cards with a classic burst
N_RING_TXN = 1_900           # fraud-ring transactions in May and June

CITIES = {  # name: (lat, lon, tier)
    "Mumbai": (19.08, 72.88, 1), "Delhi": (28.61, 77.21, 1), "Bengaluru": (12.97, 77.59, 1),
    "Chennai": (13.08, 80.27, 1), "Hyderabad": (17.39, 78.49, 1), "Kolkata": (22.57, 88.36, 1),
    "Pune": (18.52, 73.86, 1), "Ahmedabad": (23.02, 72.57, 1), "Jaipur": (26.91, 75.79, 2),
    "Lucknow": (26.85, 80.95, 2), "Indore": (22.72, 75.86, 2), "Kochi": (9.93, 76.27, 2),
    "Coimbatore": (11.02, 76.96, 2), "Bhopal": (23.26, 77.41, 2), "Nagpur": (21.15, 79.09, 2),
    "Patna": (25.59, 85.14, 3), "Ranchi": (23.34, 85.31, 3), "Guwahati": (26.14, 91.74, 3),
    "Dehradun": (30.32, 78.03, 3), "Madurai": (9.93, 78.12, 3),
}
CITY_P = np.array([10, 10, 9, 7, 7, 6, 5, 5, 4, 4, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3], float)
CATS = ["grocery", "fuel", "restaurants", "apparel", "electronics", "travel", "utilities",
        "pharmacy", "entertainment", "jewellery", "digital_goods", "gift_cards"]
CAT_P = np.array([0.22, 0.12, 0.15, 0.09, 0.05, 0.04, 0.10, 0.07, 0.06, 0.02, 0.06, 0.02])
CAT_MU = dict(zip(CATS, np.log([900, 1800, 1100, 2400, 9000, 7500, 1600, 600, 900, 18000, 450, 1500])))


def _km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _times(rng, n, start, end, night_bias=0.0):
    """Random timestamps with a daytime-heavy hour profile."""
    days = rng.integers(0, (end - start).days, n)
    hour_p = np.array([1, .5, .4, .3, .3, .5, 1.5, 3, 4, 5, 5.5, 6, 6.5, 6, 5.5, 5.5, 6, 6.5, 7, 7, 6, 5, 3.5, 2])
    hour_p = hour_p + night_bias * np.r_[np.full(6, 8), np.zeros(18)]
    hours = rng.choice(24, n, p=hour_p / hour_p.sum())
    secs = rng.integers(0, 3600, n)
    return start + pd.to_timedelta(days, "D") + pd.to_timedelta(hours, "h") + pd.to_timedelta(secs, "s")


def generate(seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = list(CITIES)

    # Cards
    home = rng.choice(len(names), N_CARDS, p=CITY_P / CITY_P.sum())
    card_type = rng.choice(["classic", "gold", "platinum"], N_CARDS, p=[0.6, 0.3, 0.1])
    limit = np.round(np.select([card_type == "classic", card_type == "gold"], [75_000, 200_000], 500_000)
                     * rng.uniform(0.6, 1.4, N_CARDS), -3)
    opened = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 3650, N_CARDS), "D")
    cards = pd.DataFrame({
        "card_id": [f"K{i:06d}" for i in range(1, N_CARDS + 1)],
        "customer_id": [f"C{i:06d}" for i in rng.permutation(N_CARDS) + 1],
        "card_type": card_type, "credit_limit": limit, "open_date": opened.strftime("%Y-%m-%d"),
        "home_city": np.array(names)[home], "city_tier": [CITIES[names[h]][2] for h in home],
    })
    lat_home = np.array([CITIES[names[h]][0] for h in home])
    lon_home = np.array([CITIES[names[h]][1] for h in home])
    spend_mult = np.select([card_type == "classic", card_type == "gold"], [0.85, 1.15], 1.6)

    # Fraud volumes
    burst = rng.integers(3, 9, N_CLASSIC_CARDS)
    n_classic = int(burst.sum())
    n_legit = N_TXN - n_classic - N_RING_TXN

    # Legitimate transactions
    activity = rng.gamma(1.6, 1.0, N_CARDS)
    c = rng.choice(N_CARDS, n_legit, p=activity / activity.sum())
    cat = rng.choice(len(CATS), n_legit, p=CAT_P)
    mu = np.array([CAT_MU[CATS[k]] for k in cat]) + np.log(spend_mult[c])
    amount = np.exp(rng.normal(mu, 0.75))
    online = rng.random(n_legit) < np.where(np.isin(cat, [10, 11]), 0.95, np.where(cat == 4, 0.45, 0.18))
    travelling = rng.random(n_legit) < 0.06
    dest = rng.choice(len(names), n_legit, p=CITY_P / CITY_P.sum())
    lat = np.where(travelling, [CITIES[names[d]][0] for d in dest], lat_home[c]) + rng.normal(0, 0.04, n_legit)
    lon = np.where(travelling, [CITIES[names[d]][1] for d in dest], lon_home[c]) + rng.normal(0, 0.04, n_legit)
    legit = pd.DataFrame({
        "ci": c, "ts": _times(rng, n_legit, START, END), "category": np.array(CATS)[cat], "amount": amount,
        "online": online, "lat": lat, "lon": lon,
        "new_device": online & (rng.random(n_legit) < 0.09), "is_fraud": 0, "kind": "legit",
    })

    # Classic fraud bursts: high value, online/electronics/jewellery, often night, far away, new device
    fc = rng.choice(N_CARDS, N_CLASSIC_CARDS, replace=False, p=activity / activity.sum())
    t0 = _times(rng, N_CLASSIC_CARDS, START, END, night_bias=0.6)
    ci = np.repeat(fc, burst)
    ts = np.repeat(t0.values, burst) + pd.to_timedelta(rng.exponential(2400, n_classic).cumsum() % 14400, "s").values
    fcat = rng.choice(["electronics", "jewellery", "digital_goods", "travel", "apparel", "gift_cards"],
                      n_classic, p=[0.34, 0.16, 0.16, 0.12, 0.12, 0.10])
    famt = np.exp(rng.normal([CAT_MU[k] for k in fcat], 0.6)) * rng.uniform(0.8, 2.0, n_classic)
    far = rng.random(n_classic) < 0.40
    fdest = rng.choice(len(names), n_classic)
    classic = pd.DataFrame({
        "ci": ci, "ts": ts, "category": fcat, "amount": famt, "online": rng.random(n_classic) < 0.6,
        "lat": np.where(far, [CITIES[names[d]][0] for d in fdest], lat_home[ci]) + rng.normal(0, 0.05, n_classic),
        "lon": np.where(far, [CITIES[names[d]][1] for d in fdest], lon_home[ci]) + rng.normal(0, 0.05, n_classic),
        "new_device": rng.random(n_classic) < 0.35, "is_fraud": 1, "kind": "classic",
    })

    # Fraud ring (May and June): small digital/gift-card spends near home on new devices at ring merchants
    rc = rng.choice(N_CARDS, N_RING_TXN, p=activity / activity.sum())
    rcat = rng.choice(["digital_goods", "gift_cards"], N_RING_TXN, p=[0.6, 0.4])
    ring = pd.DataFrame({
        "ci": rc, "ts": _times(rng, N_RING_TXN, RING_START, END), "category": rcat,
        "amount": rng.uniform(700, 2600, N_RING_TXN), "online": True,
        "lat": lat_home[rc] + rng.normal(0, 0.04, N_RING_TXN), "lon": lon_home[rc] + rng.normal(0, 0.04, N_RING_TXN),
        "new_device": rng.random(N_RING_TXN) < 0.45, "is_fraud": 1, "kind": "ring",
    })

    df = pd.concat([legit, classic, ring], ignore_index=True)
    df = df[df["ts"] < END].sort_values(["ci", "ts"], kind="stable").reset_index(drop=True)

    # Merchants: ring transactions use 12 ring merchants, everything else a category pool
    m_id = np.empty(len(df), dtype=object)
    for k in CATS:
        idx = np.flatnonzero((df["category"] == k).to_numpy() & (df["kind"] != "ring").to_numpy())
        pool = 400 if k in ("grocery", "restaurants", "fuel") else 150
        m_id[idx] = [f"M_{k[:3].upper()}{j:04d}" for j in rng.integers(1, pool + 1, len(idx))]
    ri = np.flatnonzero((df["kind"] == "ring").to_numpy())
    m_id[ri] = [f"M_{'DIG' if k == 'digital_goods' else 'GIF'}{j:04d}"
                for k, j in zip(df.loc[ri, "category"], rng.integers(9001, 9013, len(ri)))]

    # Sequence features within each card
    same = df["ci"].eq(df["ci"].shift())
    secs = (df["ts"] - df["ts"].shift()).dt.total_seconds().where(same)
    km_last = pd.Series(_km(df["lat"].shift(), df["lon"].shift(), df["lat"], df["lon"])).where(same)
    km_home = _km(lat_home[df["ci"]], lon_home[df["ci"]], df["lat"], df["lon"])
    cnt_24h = (df.assign(one=1).set_index("ts").groupby("ci")["one"].rolling("24h").sum()
               .reset_index(level=0, drop=True).to_numpy())

    n = len(df)
    txn = pd.DataFrame({
        "txn_id": [f"T{i:07d}" for i in range(1, n + 1)],
        "card_id": cards["card_id"].to_numpy()[df["ci"]],
        "timestamp": df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": np.round(df["amount"], 2),
        "merchant_id": m_id, "merchant_category": df["category"],
        "channel": np.where(df["online"], "online", "pos"),
        "card_present": (~df["online"]).astype(int),
        "new_device": df["new_device"].astype(int),
        "km_from_home": np.round(km_home, 1),
        "km_from_last_txn": np.round(km_last, 1),
        "secs_since_last_txn": secs.round(),
        "txns_last_24h": cnt_24h.astype(int),
        "amount_to_limit": np.round(df["amount"].to_numpy() / limit[df["ci"]], 4),
    })
    txn = txn.sort_values("timestamp", kind="stable").reset_index(drop=True)
    order = txn["txn_id"].str[1:].astype(int).to_numpy() - 1
    fraud = df["is_fraud"].to_numpy()[order]
    txn["txn_id"] = [f"T{i:07d}" for i in range(1, n + 1)]  # renumber in time order

    # Labels with a planted leak: chargebacks arrive weeks later and almost only on fraud
    chargeback = np.where(fraud == 1, rng.random(n) < 0.9, rng.random(n) < 0.0015).astype(int)
    labels = pd.DataFrame({
        "txn_id": txn["txn_id"], "is_fraud": fraud, "chargeback_filed": chargeback,
        "days_to_report": np.where(fraud == 1, rng.integers(1, 45, n), np.nan),
    })
    return {"cards": cards, "transactions": txn, "fraud_labels": labels}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/petalbank_card_txn")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in generate(a.seed).items():
        t.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv  {len(t):,} rows")


if __name__ == "__main__":
    main()
