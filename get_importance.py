import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
import shap

from tester import load_all, extract_promo_signals, extract_signup_velocity, compute_monthly_revenue_prior, vectorised_features, build_weights, _make_lgb, N_BAGS, RNG_SEED, HOLDOUT_DAYS

sales, sample, promos, customers, orders, web = load_all()
promo_signals = extract_promo_signals(promos)
signup_vel = extract_signup_velocity(customers)
monthly_prior = compute_monthly_revenue_prior(sales)

feat = vectorised_features(sales, "Date", "Revenue", promo_signals, signup_vel, monthly_prior)
target = np.log1p(sales["Revenue"].astype(float))

merged = pd.concat([feat, target.rename("y")], axis=1).dropna().reset_index(drop=True)
feature_cols = [c for c in merged.columns if c != "y"]
X_all = merged[feature_cols]
y_all = merged["y"].values

dates_all = sales["Date"].iloc[-len(X_all):].reset_index(drop=True)
w_all = build_weights(dates_all)

# Train 1 model to get importance quickly
m = _make_lgb(RNG_SEED)
m.fit(X_all, y_all, sample_weight=w_all)

importances = m.feature_importances_
df_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
df_imp = df_imp.sort_values('importance', ascending=False)
print("--- LIGHTGBM FEATURE IMPORTANCE ---")
print(df_imp.head(15))

try:
    print("\n--- SHAP VALUES ---")
    explainer = shap.TreeExplainer(m)
    # Use a sample of 1000 rows to speed up SHAP calculation
    X_sample = X_all.sample(1000, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([feature_cols, shap_sum.tolist()]).T
    importance_df.columns = ['feature', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    print(importance_df.head(15))
except Exception as e:
    print(f"SHAP error: {e}")
