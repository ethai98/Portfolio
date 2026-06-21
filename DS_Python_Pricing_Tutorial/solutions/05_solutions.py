# %% [markdown]
# # Solutions — Lesson 05 (machine learning)
#   python solutions/05_solutions.py
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              GradientBoostingClassifier)
from sklearn.metrics import mean_absolute_error
from sklearn.calibration import calibration_curve

df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
df = df.drop_duplicates().reset_index(drop=True)
df["weight_lbs"] = df.groupby("equipment_type")["weight_lbs"].transform(
    lambda s: s.fillna(s.median()))

num = ["distance_miles", "weight_lbs", "fuel_price_usd_gal",
       "market_tightness", "is_peak_season"]
cat = ["equipment_type", "customer_segment"]
X, y = df[num + cat], df["carrier_cost_usd"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
prep = ColumnTransformer([("num", StandardScaler(), num),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

# %% E1 — add GradientBoostingRegressor to the comparison
results = {}
for name, est in [("RandomForest", RandomForestRegressor(
                        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)),
                  ("GradBoost", GradientBoostingRegressor(random_state=0))]:
    m = Pipeline([("prep", prep), ("model", est)]).fit(Xtr, ytr)
    results[name] = mean_absolute_error(yte, m.predict(Xte))
print("E1: Test MAE by model:")
for k, v in results.items():
    print(f"  {k:14s} ${v:,.0f}")
print()

# %% E2 — tune max_depth with CV
cv = KFold(5, shuffle=True, random_state=0)
print("E2: RF CV MAE by max_depth:")
for d in [4, 8, 12, None]:
    rf = Pipeline([("prep", prep),
                   ("model", RandomForestRegressor(n_estimators=150, max_depth=d,
                                                   min_samples_leaf=5, n_jobs=-1,
                                                   random_state=0))])
    s = -cross_val_score(rf, Xtr, ytr, cv=cv,
                         scoring="neg_mean_absolute_error").mean()
    print(f"  max_depth={str(d):4s} -> CV MAE ${s:,.0f}")
print("  CV MAE improves with depth then plateaus; very deep trees overfit"
      " (train keeps improving, CV doesn't).\n")

# %% E3 — domain baseline
baseline_pred = 1.85 * Xte["distance_miles"]
print(f"E3: baseline (1.85*distance) MAE = ${mean_absolute_error(yte, baseline_pred):,.0f}"
      f" vs RF ${results['RandomForest']:,.0f}. ML adds value by capturing fuel,"
      " tightness, equipment & peak effects the flat rate misses.\n")

# %% E4 — calibration of win classifier
cnum = ["realized_margin", "market_tightness", "distance_miles", "fuel_price_usd_gal"]
ccat = ["equipment_type", "customer_segment"]
Xc, yc = df[cnum + ccat], df["won"]
Xctr, Xcte, yctr, ycte = train_test_split(Xc, yc, test_size=0.2,
                                          random_state=0, stratify=yc)
cprep = ColumnTransformer([("num", StandardScaler(), cnum),
                           ("cat", OneHotEncoder(handle_unknown="ignore"), ccat)])
gb = Pipeline([("prep", cprep),
               ("model", GradientBoostingClassifier(random_state=0))]).fit(Xctr, yctr)
proba = gb.predict_proba(Xcte)[:, 1]
frac_pos, mean_pred = calibration_curve(ycte, proba, n_bins=10)
fig, ax = plt.subplots()
ax.plot(mean_pred, frac_pos, marker="o", label="model")
ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
ax.set_xlabel("Predicted P(win)"); ax.set_ylabel("Actual win rate")
ax.set_title("Calibration curve"); ax.legend(); plt.show()
print("E4: Points near the diagonal => probabilities are trustworthy. Calibration"
      " matters because expected profit = P(win)*margin; a miscalibrated P(win)"
      " biases every price recommendation.\n")

# %% E5 — price recommender vs actual quoted prices
def recommend_price(row):
    cost = row["carrier_cost_usd"]
    margins = np.linspace(0.03, 0.30, 40)
    grid = pd.DataFrame({"realized_margin": margins,
                         "market_tightness": row["market_tightness"],
                         "distance_miles": row["distance_miles"],
                         "fuel_price_usd_gal": row["fuel_price_usd_gal"],
                         "equipment_type": row["equipment_type"],
                         "customer_segment": row["customer_segment"]})
    p = gb.predict_proba(grid[cnum + ccat])[:, 1]
    price = cost / (1 - margins)
    exp_profit = p * (price - cost)
    best = np.argmax(exp_profit)
    return pd.Series({"rec_margin": margins[best], "rec_exp_profit": exp_profit[best]})

sample = df.sample(200, random_state=11).copy()
recs = sample.apply(recommend_price, axis=1)
# Expected profit at the price we ACTUALLY quoted (use model's P(win)):
actual_p = gb.predict_proba(sample[cnum + ccat])[:, 1]
actual_exp = actual_p * (sample["quoted_price_usd"] - sample["carrier_cost_usd"])
print("E5: total expected profit over 200 loads")
print(f"  actual quotes : ${actual_exp.sum():,.0f}")
print(f"  recommender   : ${recs['rec_exp_profit'].sum():,.0f}")
print(f"  uplift        : ${recs['rec_exp_profit'].sum() - actual_exp.sum():,.0f}")
print("  (Illustrative — assumes the win model is correct; validate with a live"
      " A/B test before trusting any engine's uplift, per lesson 04.)")
