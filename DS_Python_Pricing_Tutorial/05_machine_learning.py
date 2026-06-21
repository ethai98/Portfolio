# %% [markdown]
# # Lesson 05 — Machine Learning for Pricing
#
# Now the prediction toolkit. On a pricing team ML powers things like:
#  - a **rate predictor** (what will a lane cost us to cover?),
#  - a **win-probability model** (will the shipper accept this quote?),
#  - feeding both into a **price-recommendation / optimization** engine.
#
# We'll build both a regression and a classification model the *professional* way:
# train/validation/test discipline, pipelines, proper preprocessing, cross-
# validation, model comparison, and interpretation (feature importance). The
# emphasis is on the WORKFLOW, which transfers to any business problem.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             classification_report, confusion_matrix)

df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
df = df.drop_duplicates().reset_index(drop=True)
df["weight_lbs"] = df.groupby("equipment_type")["weight_lbs"].transform(
    lambda s: s.fillna(s.median())
)

# %% [markdown]
# ## 1. Frame the problem & avoid leakage
# **Task A (regression):** predict `carrier_cost_usd` — our cost to move a load —
# from features known *at quote time*.
#
# **Data leakage** is the #1 ML mistake in industry: using a feature that won't
# exist when you actually predict, or that encodes the answer. Here we must NOT
# use `quoted_price_usd`, `realized_margin`, or `linehaul_rate_per_mile` to
# predict cost — they're derived from cost itself. Only use genuine inputs.

# %%
num_features = ["distance_miles", "weight_lbs", "fuel_price_usd_gal",
                "market_tightness", "is_peak_season"]
cat_features = ["equipment_type", "customer_segment"]
target = "carrier_cost_usd"

X = df[num_features + cat_features]
y = df[target]

# Hold out a test set we DON'T touch until the very end (honest performance).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# %% [markdown]
# ## 2. Preprocessing with a ColumnTransformer + Pipeline
# A **Pipeline** chains preprocessing + model into one object. Why pros always do
# this: it (a) prevents leakage (scaling is fit on train only, applied to test),
# and (b) makes the whole thing reproducible and deployable as a single artifact.

# %%
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])

linreg = Pipeline([
    ("prep", preprocess),
    ("model", LinearRegression()),
])
linreg.fit(X_train, y_train)
pred_lin = linreg.predict(X_test)
print(f"Linear  -> Test MAE ${mean_absolute_error(y_test, pred_lin):,.0f} | "
      f"R² {r2_score(y_test, pred_lin):.3f}")

# %% [markdown]
# ## 3. A nonlinear model — Random Forest
# Trees capture interactions (fuel × distance) and nonlinearities automatically,
# no manual feature engineering. Same pipeline, swap the estimator.

# %%
rf = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=None,
                                    min_samples_leaf=5, n_jobs=-1,
                                    random_state=0)),
])
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print(f"Forest  -> Test MAE ${mean_absolute_error(y_test, pred_rf):,.0f} | "
      f"R² {r2_score(y_test, pred_rf):.3f}")

# %% [markdown]
# ## 4. Cross-validation — don't trust a single split
# One train/test split is noisy. K-fold CV averages performance over K splits for
# a stabler estimate and a sense of variance. Report mean ± std.

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(rf, X_train, y_train, cv=cv,
                         scoring="neg_mean_absolute_error")
print(f"RF 5-fold CV MAE: ${-scores.mean():,.0f} ± ${scores.std():,.0f}")

# %% [markdown]
# ## 5. Which features drive cost? (interpretation)
# A model you can't explain is a hard sell to a pricing director. Random-forest
# feature importances rank what matters. (Expect distance to dominate, then
# market tightness / fuel — matching how the generator built cost.)

# %%
ohe_names = (rf.named_steps["prep"]
             .named_transformers_["cat"]
             .get_feature_names_out(cat_features))
feat_names = num_features + list(ohe_names)
importances = rf.named_steps["model"].feature_importances_
imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
print(imp.head(10))

# %% [markdown]
# ## 6. Task B — win-probability classifier
# Predict `won` from the quote terms. This model + the cost model = the
# ingredients of a price-recommendation engine: for any candidate price, predict
# P(win), then choose the price that maximizes expected profit.
#
# Here `realized_margin` IS a legitimate feature — at quote time we know the
# margin we're about to charge (that's the lever we're choosing).

# %%
clf_features_num = ["realized_margin", "market_tightness", "distance_miles",
                    "fuel_price_usd_gal"]
clf_features_cat = ["equipment_type", "customer_segment"]
Xc = df[clf_features_num + clf_features_cat]
yc = df["won"]
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2,
                                              random_state=0, stratify=yc)

clf_prep = ColumnTransformer([
    ("num", StandardScaler(), clf_features_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), clf_features_cat),
])

# %%
logit = Pipeline([("prep", clf_prep),
                  ("model", LogisticRegression(max_iter=1000))]).fit(Xc_tr, yc_tr)
gb = Pipeline([("prep", clf_prep),
               ("model", GradientBoostingClassifier(random_state=0))]).fit(Xc_tr, yc_tr)

for name, m in [("Logistic", logit), ("GradBoost", gb)]:
    proba = m.predict_proba(Xc_te)[:, 1]
    print(f"{name:10s} ROC-AUC = {roc_auc_score(yc_te, proba):.3f}")

# %% [markdown]
# **ROC-AUC** measures ranking quality: the probability the model scores a random
# won quote higher than a random lost one. 0.5 = coin flip, 1.0 = perfect. For
# pricing, calibrated *probabilities* matter more than hard 0/1 labels, because
# we multiply P(win) by profit — so prefer models with good probability output.

# %%
print(classification_report(yc_te, gb.predict(Xc_te)))
print("Confusion matrix:\n", confusion_matrix(yc_te, gb.predict(Xc_te)))

# %% [markdown]
# ## 7. Bringing it together — a price recommendation
# For one example load, sweep candidate margins, predict P(win) with the
# classifier, and pick the margin that maximizes expected profit
# = P(win) × (price − cost). This is, in miniature, what a pricing engine does.

# %%
example = df.iloc[[0]].copy()
cost = example["carrier_cost_usd"].values[0]
candidate_margins = np.linspace(0.03, 0.30, 40)

rows = []
for mgn in candidate_margins:
    e = example.copy()
    e["realized_margin"] = mgn
    p = gb.predict_proba(e[clf_features_num + clf_features_cat])[:, 1][0]
    price = cost / (1 - mgn)
    rows.append((mgn, p, p * (price - cost)))

rec = pd.DataFrame(rows, columns=["margin", "p_win", "expected_profit"])
best = rec.loc[rec["expected_profit"].idxmax()]
print(f"Recommended margin: {best['margin']:.1%}  "
      f"(P(win)={best['p_win']:.2f}, E[profit]=${best['expected_profit']:,.0f})")

# %% [markdown]
# ## 8. Professional hygiene checklist
# - **Persist the model:** `joblib.dump(rf, "cost_model.pkl")` so it's reusable
#   and versioned (alongside the training code/data version).
# - **Monitor drift:** fuel and market regimes change; a model trained on 2023
#   data degrades. Re-evaluate on recent data periodically.
# - **Baseline first:** always compare ML against a dumb baseline (e.g., cost =
#   $/mile × distance). If the fancy model barely beats it, ship the simple one.
# - **Don't optimize a metric blind to the business:** a 1% AUC gain that needs a
#   black-box model may be worse than an interpretable model the team trusts.
#
# ---
# # EXERCISES
# Solutions in `solutions/05_solutions.py`.
#
# **E1.** Add `GradientBoostingRegressor` to the cost-model comparison (section
# 3). Does it beat the random forest on test MAE? Compare all three models in one
# printed table.
#
# **E2.** Tune the random forest's `max_depth` over [4, 8, 12, None] using
# cross-validation. Plot/print CV MAE vs depth. Where does it start overfitting
# (train improves but CV stops improving)?
#
# **E3.** Build a **baseline** cost model: `predicted_cost = 1.85 * distance`.
# Compute its test MAE and compare to the ML models. How much does ML actually
# add over domain knowledge?
#
# **E4.** For the win classifier, compute and plot a **calibration curve**
# (`sklearn.calibration.calibration_curve`). Are predicted probabilities
# trustworthy? Why does calibration matter when you multiply P(win) by profit?
#
# **E5.** Turn section 7 into a function `recommend_price(load_row)` that returns
# the profit-maximizing price for any quote. Run it across 200 sampled loads and
# compare total expected profit under your recommendations vs the prices we
# actually quoted. How much uplift could the engine capture?

# %%
# Your answers here:
