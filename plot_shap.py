import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import matplotlib.pyplot as plt
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

m = _make_lgb(RNG_SEED)
m.fit(X_all, y_all, sample_weight=w_all)

explainer = shap.TreeExplainer(m)
X_sample = X_all.sample(1000, random_state=42)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Revenue Drivers)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
print("Saved shap_summary.png")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Impact on Model Output", fontsize=14)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=300)
print("Saved shap_beeswarm.png")
