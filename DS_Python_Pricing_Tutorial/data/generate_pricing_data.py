"""
generate_pricing_data.py
========================
Creates a synthetic — but realistic — freight-pricing dataset used throughout
this tutorial. The data is modeled on the kind of quoting data a brokerage
pricing team (e.g., Uber Freight) works with: each row is a *quote* we sent to
a shipper for moving one truckload on one lane.

WHY SYNTHETIC DATA?
  Real pricing data is confidential and messy. A synthetic generator lets us
  (a) practice on data that looks real, and (b) KNOW the true relationships, so
  later you can check whether your regression / causal / ML models recover them.

The "ground truth" relationships are documented inline. When you fit a model in
lessons 03-05, compare your estimated coefficients to the TRUE_* values here.

Run this file once before starting the lessons:
    python data/generate_pricing_data.py

It writes:  data/freight_quotes.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility: same seed => same data every run. Critical for a team so that
# everyone debugging the same notebook sees identical numbers.
RNG = np.random.default_rng(42)

N = 20_000  # number of quotes

# ----------------------------------------------------------------------------
# GROUND-TRUTH PARAMETERS (the "answer key")
# ----------------------------------------------------------------------------
# Carrier cost (what WE pay the trucker) per load, built up from drivers below.
TRUE_BASE_COST = 150.0           # fixed cost component per load ($)
TRUE_COST_PER_MILE = 1.85        # variable linehaul cost ($/mile)
TRUE_FUEL_SENSITIVITY = 28.0     # extra cost per $1 increase in diesel, per 100mi-equiv
TRUE_EQUIP_PREMIUM = {           # cost premium by trailer type ($/mile add-on)
    "Dry Van": 0.00,
    "Reefer": 0.45,             # refrigerated -> more expensive
    "Flatbed": 0.30,
}
TRUE_WEIGHT_COST = 0.0008        # $ per lb (heavier = slightly more)
TRUE_PEAK_COST = 180.0           # peak-season capacity crunch surcharge ($)
TRUE_TIGHTNESS_COST = 220.0      # cost added per unit of market-tightness index

# Price elasticity of winning the load. Shippers accept cheaper quotes more often.
# Modeled as a logistic function of our margin %. These are the "true" elasticity
# parameters you'll try to recover in the regression & ML lessons.
TRUE_WIN_INTERCEPT = 2.2
TRUE_WIN_MARGIN_SLOPE = -14.0    # higher margin (pricier quote) -> lower win prob

# Causal experiment baked in: a "Dynamic Pricing" algorithm was piloted on a
# subset of lanes starting 2024-07-01. On treated lanes after that date it raised
# our target MARKUP by 3 pts. Two things follow (and you'll recover both with
# difference-in-differences in lesson 04):
#   - realized (revenue) margin rises ~2.2 pts, because realized margin =
#     markup/(1+markup), so a +3pt markup is a slightly smaller realized lift;
#   - win rate falls a few points, because charging more loses some loads (the
#     price elasticity in the win model below). The interesting question DiD
#     answers is whether the pilot was NET profitable despite the lost volume.
TRUE_DID_MARGIN_LIFT = 0.03

# ----------------------------------------------------------------------------
# Build the columns
# ----------------------------------------------------------------------------
dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
    RNG.integers(0, 730, size=N), unit="D"
)

cities = [
    "Chicago", "Atlanta", "Dallas", "Los Angeles", "Newark",
    "Seattle", "Denver", "Miami", "Memphis", "Columbus",
]
origin = RNG.choice(cities, size=N)
# Pick a destination different from origin
dest = RNG.choice(cities, size=N)
same = origin == dest
while same.any():
    dest[same] = RNG.choice(cities, size=same.sum())
    same = origin == dest

distance_miles = np.clip(RNG.normal(900, 450, size=N), 80, 2800).round(0)

equipment = RNG.choice(["Dry Van", "Reefer", "Flatbed"], size=N, p=[0.6, 0.25, 0.15])
weight_lbs = np.clip(RNG.normal(28_000, 9_000, size=N), 2_000, 45_000).round(0)

# Diesel price drifts over the 2-year window plus weekly noise (real fuel series-ish)
day_index = (dates - dates.min()).days.values
fuel_price = 3.6 + 0.6 * np.sin(day_index / 115.0) + RNG.normal(0, 0.12, size=N)
fuel_price = fuel_price.round(2)

# Market tightness: 0 = loose (lots of trucks), ~1.5 = very tight. Seasonal + noise.
market_tightness = np.clip(
    0.7 + 0.4 * np.sin((day_index - 60) / 95.0) + RNG.normal(0, 0.18, size=N),
    0.0, 1.8,
).round(3)

# Peak season = produce season (May-July) + holiday push (Nov-Dec)
month = dates.month
is_peak = np.isin(month, [5, 6, 7, 11, 12]).astype(int)

customer_segment = RNG.choice(
    ["Enterprise", "Mid-Market", "SMB"], size=N, p=[0.35, 0.4, 0.25]
)
shipper_id = RNG.integers(1000, 1300, size=N)  # ~300 distinct shippers

# ----------------------------------------------------------------------------
# Carrier cost (structural equation + noise)
# ----------------------------------------------------------------------------
equip_premium_arr = np.array([TRUE_EQUIP_PREMIUM[e] for e in equipment])

carrier_cost = (
    TRUE_BASE_COST
    + TRUE_COST_PER_MILE * distance_miles
    + equip_premium_arr * distance_miles
    + TRUE_FUEL_SENSITIVITY * (fuel_price - 3.6) * (distance_miles / 100.0)
    + TRUE_WEIGHT_COST * weight_lbs
    + TRUE_PEAK_COST * is_peak
    + TRUE_TIGHTNESS_COST * market_tightness
    + RNG.normal(0, 120, size=N)  # idiosyncratic noise
)
carrier_cost = np.clip(carrier_cost, 200, None).round(2)

# ----------------------------------------------------------------------------
# Pricing decision: target margin varies by segment + a little randomness.
# This is the markup WE chose. Enterprise customers negotiate harder (thinner).
# ----------------------------------------------------------------------------
base_margin = np.select(
    [customer_segment == "Enterprise",
     customer_segment == "Mid-Market",
     customer_segment == "SMB"],
    [0.11, 0.15, 0.19],
)
target_margin = base_margin + RNG.normal(0, 0.03, size=N)

# Difference-in-differences treatment: roughly half the LANES are in the pilot.
lane = pd.Series(origin) + " -> " + pd.Series(dest)
treated_lane = (pd.util.hash_pandas_object(lane, index=False) % 2 == 0).values
post_period = np.asarray(dates >= "2024-07-01")
in_treatment = treated_lane & post_period
target_margin = target_margin + TRUE_DID_MARGIN_LIFT * in_treatment

quoted_price = (carrier_cost * (1 + target_margin)).round(2)
realized_margin = (quoted_price - carrier_cost) / quoted_price

# ----------------------------------------------------------------------------
# Win/loss: logistic in realized margin (price elasticity of demand)
# ----------------------------------------------------------------------------
logit = TRUE_WIN_INTERCEPT + TRUE_WIN_MARGIN_SLOPE * realized_margin
# Tighter market => shipper has fewer options => slightly higher win prob
logit = logit + 0.8 * market_tightness
win_prob = 1 / (1 + np.exp(-logit))
won = (RNG.random(N) < win_prob).astype(int)

# ----------------------------------------------------------------------------
# Assemble, then sprinkle in realistic messiness for the cleaning lesson
# ----------------------------------------------------------------------------
df = pd.DataFrame({
    "quote_id": np.arange(1, N + 1),
    "quote_date": dates,
    "origin": origin,
    "destination": dest,
    "lane": lane.values,
    "equipment_type": equipment,
    "distance_miles": distance_miles,
    "weight_lbs": weight_lbs,
    "fuel_price_usd_gal": fuel_price,
    "market_tightness": market_tightness,
    "is_peak_season": is_peak,
    "customer_segment": customer_segment,
    "shipper_id": shipper_id,
    "carrier_cost_usd": carrier_cost,
    "quoted_price_usd": quoted_price,
    "linehaul_rate_per_mile": (quoted_price / distance_miles).round(3),
    "realized_margin": realized_margin.round(4),
    "is_dynamic_pricing_lane": treated_lane.astype(int),
    "won": won,
})

# Messiness for the data-cleaning exercises (lesson 01):
#  - Some missing weights (sensor/EDI gaps)
miss_idx = RNG.choice(N, size=int(0.03 * N), replace=False)
df.loc[miss_idx, "weight_lbs"] = np.nan
#  - Inconsistent casing/whitespace in equipment type
dirty_idx = RNG.choice(N, size=int(0.02 * N), replace=False)
df.loc[dirty_idx, "equipment_type"] = df.loc[dirty_idx, "equipment_type"].str.upper()
#  - A few duplicate rows (double-logged quotes)
dupes = df.sample(50, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

out_path = Path(__file__).parent / "freight_quotes.csv"
df.to_csv(out_path, index=False)

print(f"Wrote {len(df):,} rows -> {out_path}")
print(f"Overall win rate: {df['won'].mean():.1%}")
print(f"Avg realized margin: {df['realized_margin'].mean():.1%}")
print("\nGround-truth coefficients you'll try to recover later:")
print(f"  cost per mile      = ${TRUE_COST_PER_MILE}")
print(f"  reefer premium     = ${TRUE_EQUIP_PREMIUM['Reefer']}/mile")
print(f"  win~margin slope   = {TRUE_WIN_MARGIN_SLOPE}")
print(f"  DiD margin lift    = {TRUE_DID_MARGIN_LIFT:+.0%}")
