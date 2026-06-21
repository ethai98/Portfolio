# %% [markdown]
# # Lesson 03 — Regression for Pricing
#
# Regression is the bread and butter of a pricing analyst: *how much does cost
# change per mile? what's the fuel pass-through? how does price affect win rate?*
#
# We use two libraries on purpose:
#  - **statsmodels** — gives you coefficients, standard errors, p-values, R² and
#    a regression *summary table*. Use it when you need to INTERPRET and explain.
#  - **scikit-learn** — clean train/test API, used when you care about PREDICTION
#    (lesson 05 leans on this).
#
# Throughout, compare your estimated coefficients to the TRUE_* values printed by
# `data/generate_pricing_data.py`. Recovering them ≈ you did it right.

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
df = df.drop_duplicates().reset_index(drop=True)
df["weight_lbs"] = df.groupby("equipment_type")["weight_lbs"].transform(
    lambda s: s.fillna(s.median())
)

# %% [markdown]
# ## 1. Simple linear regression — cost vs distance
# Model: `carrier_cost = b0 + b1 * distance + error`. The slope b1 estimates the
# marginal cost per mile. The generator's TRUE cost-per-mile is **$1.85** (plus a
# blended equipment premium, so expect a touch higher).

# %%
model1 = smf.ols("carrier_cost_usd ~ distance_miles", data=df).fit()
print(model1.summary())

# %% [markdown]
# **How to read the summary:**
# - `coef` on `distance_miles` ≈ marginal $/mile. `Intercept` ≈ fixed cost/load.
# - `std err` / `t` / `P>|t|`: is the effect statistically distinguishable from 0?
#   (p < 0.05 is the usual threshold.) With 20k rows everything looks significant
#   — at scale, **effect size matters more than significance**.
# - `R-squared`: share of cost variance explained. Distance alone explains a lot,
#   but not all — fuel, equipment, and market drive the rest. Let's add them.

# %% [markdown]
# ## 2. Multiple regression — the real cost model
# Add fuel, weight, peak season, market tightness, and equipment type. `C(...)`
# tells the formula to treat a column as categorical (auto one-hot encoding).

# %%
cost_model = smf.ols(
    "carrier_cost_usd ~ distance_miles + fuel_price_usd_gal + weight_lbs "
    "+ is_peak_season + market_tightness + C(equipment_type)",
    data=df,
).fit()
print(cost_model.summary())

# %% [markdown]
# **Interpreting categoricals:** `C(equipment_type)` creates dummy variables with
# "Dry Van" as the omitted baseline. So the `Reefer` coefficient is the *extra*
# cost of a reefer load **vs a dry van**, holding distance etc. constant. (In the
# generator, reefer adds $0.45/mile — on an ~900-mile average haul that's ≈ $400,
# which is roughly what the dummy coefficient should reflect.)
#
# **Holding constant** is the magic phrase: each coefficient is the effect of its
# variable *with all the others fixed* — exactly what you want when isolating,
# say, the pure fuel pass-through.

# %% [markdown]
# ## 3. Log transforms & elasticities
# Pricing people love *elasticities* (% change in y per % change in x). A
# log-log regression gives coefficients you read directly as elasticities.

# %%
df["ln_cost"] = np.log(df["carrier_cost_usd"])
df["ln_dist"] = np.log(df["distance_miles"])
elastic = smf.ols("ln_cost ~ ln_dist", data=df).fit()
print(f"Distance elasticity of cost: {elastic.params['ln_dist']:.3f}")
print("(≈ a 1% longer haul costs ~{:.2f}% more)".format(elastic.params["ln_dist"]))

# %% [markdown]
# ## 4. Interaction terms — does fuel hit long hauls harder?
# Intuitively, a fuel-price jump costs more on a 2,000-mile load than a 200-mile
# one. An interaction `distance:fuel` tests that. (The generator builds fuel cost
# as fuel × distance, so this interaction SHOULD be strongly positive.)

# %%
interact = smf.ols(
    "carrier_cost_usd ~ distance_miles + fuel_price_usd_gal "
    "+ distance_miles:fuel_price_usd_gal",
    data=df,
).fit()
print(interact.params)
print("\nThe positive interaction = fuel's per-mile bite grows with distance.")

# %% [markdown]
# ## 5. Robust standard errors
# Real pricing data has heteroskedasticity (cost variance grows with haul size).
# OLS coefficients stay fine, but standard errors can be wrong. Use HC3 robust
# SEs — cheap insurance, standard practice in applied econ/pricing work.

# %%
robust = smf.ols("carrier_cost_usd ~ distance_miles + fuel_price_usd_gal",
                 data=df).fit(cov_type="HC3")
print(robust.summary().tables[1])     # just the coefficient table

# %% [markdown]
# ## 6. Logistic regression — the price-elasticity-of-demand model
# `won` is binary, so we model the *probability* of winning with a logit. This is
# arguably the most valuable model on a pricing team: how does win probability
# respond to the margin we charge? TRUE win~margin slope in the generator = -14.

# %%
win_model = smf.logit("won ~ realized_margin + market_tightness", data=df).fit()
print(win_model.summary())
print(f"\nEstimated margin slope: {win_model.params['realized_margin']:.2f} "
      f"(true value = -14.0)")

# %% [markdown]
# **Reading a logit:** coefficients are in log-odds, not probabilities. Two
# friendlier views:
#  - **Odds ratio** = exp(coef). For margin: a 1.0 (i.e. 100-pt) margin increase
#    multiplies the odds of winning by exp(-14) — huge. Per +1 percentage point
#    (0.01), odds multiply by exp(-0.14) ≈ 0.87, i.e. ~13% lower odds.
#  - **Marginal effects** — average change in win *probability* per unit of x:

# %%
mfx = win_model.get_margeff()
print(mfx.summary())
print("\n^ 'dy/dx' for realized_margin = avg change in P(win) per +1.0 margin.")

# %% [markdown]
# ## 7. Putting it to work: optimal-price intuition
# Expected profit of a quote = P(win | price) × (price − cost). Raising price
# lifts margin-per-win but lowers P(win). The logit model lets us trace that
# trade-off and find a profit-maximizing markup. Here's the curve for a typical
# quote (you'll formalize this in the exercises).

# %%
typical_cost = df["carrier_cost_usd"].median()
typical_tight = df["market_tightness"].median()
margins = np.linspace(0.02, 0.30, 50)

# Predict P(win) across candidate margins, holding tightness at its median.
grid = pd.DataFrame({"realized_margin": margins, "market_tightness": typical_tight})
p_win = win_model.predict(grid)
exp_profit = p_win * (typical_cost * margins / (1 - margins))  # profit per quote
best = margins[np.argmax(exp_profit)]
print(f"Profit-maximizing margin for a median quote ≈ {best:.1%}")

# %% [markdown]
# ## 8. scikit-learn version (prediction-focused)
# Same idea, sklearn API: train/test split + a metric. This is the muscle memory
# you'll reuse for the ML lesson.

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

features = ["distance_miles", "fuel_price_usd_gal", "weight_lbs",
            "is_peak_season", "market_tightness"]
X = df[features]
y = df["carrier_cost_usd"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)
lin = LinearRegression().fit(X_train, y_train)
pred = lin.predict(X_test)
print(f"Test R²:  {r2_score(y_test, pred):.3f}")
print(f"Test MAE: ${mean_absolute_error(y_test, pred):,.0f} per load")
print("\nCoefficients:")
for f, c in zip(features, lin.coef_):
    print(f"  {f:22s} {c:>10.3f}")

# %% [markdown]
# ---
# # EXERCISES
# Solutions in `solutions/03_solutions.py`.
#
# **E1.** Fit an OLS of `quoted_price_usd` on `distance_miles` and
# `C(customer_segment)`. Holding distance constant, which segment is quoted the
# highest prices? Does that match the margin story from lesson 01?
#
# **E2.** Estimate the **fuel pass-through**: regress `carrier_cost_usd` on
# `fuel_price_usd_gal` and `distance_miles`. By how many dollars does cost rise
# per +$1 of diesel for an average-length haul?
#
# **E3.** Re-run the win logit but add `C(customer_segment)`. Are some segments
# more likely to accept at the same margin? Convert the margin coefficient to an
# odds ratio and interpret it in one sentence.
#
# **E4.** Add an interaction `realized_margin:market_tightness` to the win logit.
# Business question: are shippers *less* price-sensitive when capacity is tight?
# What sign do you expect, and what do you get?
#
# **E5.** Using section 7's framework, write a small function
# `optimal_margin(cost, tightness)` that returns the profit-maximizing margin for
# any quote. Try it for a cheap loose-market load vs an expensive tight-market
# load. Do the recommendations differ sensibly?

# %%
# Your answers here:
