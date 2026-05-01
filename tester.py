from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

RNG_SEED = 42
N_BAGS = 20
FORECAST_HORIZON = 548
HOLDOUT_DAYS = 300

def load_all():
    sales = (pd.read_csv(DATA / "sales.csv", parse_dates=["Date"])
               .sort_values("Date").reset_index(drop=True))
    sample = pd.read_csv(DATA / "sample_submission.csv", parse_dates=["Date"])

    promo_path = DATA / "promotions.csv"
    promos = pd.read_csv(promo_path, engine="python",
                         on_bad_lines="skip") if promo_path.exists() else None

    cust_path = DATA / "customers.csv"
    customers = pd.read_csv(cust_path) if cust_path.exists() else None

    orders_path = DATA / "orders.csv"
    orders = pd.read_csv(orders_path) if orders_path.exists() else None

    web_path = DATA / "web_traffic.csv"
    web = pd.read_csv(web_path, parse_dates=["date"]) if web_path.exists() else None

    return sales, sample, promos, customers, orders, web

def extract_promo_signals(promos: pd.DataFrame | None) -> dict[int, dict]:
    if promos is None:
        return {}
    needed = {"start_date", "end_date", "discount_value", "promo_type", "stackable_flag"}
    if not needed.issubset(promos.columns):
        return {}

    df = promos.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.dropna(subset=["start_date", "end_date"])

    records = []
    for _, r in df.iterrows():
        s, e = pd.Timestamp(r["start_date"]), pd.Timestamp(r["end_date"])
        if e < s:
            continue
        ptype = str(r.get("promo_type", "")).strip().lower()
        disc = float(r.get("discount_value", 0) or 0)
        pct = disc if ptype == "percentage" else 0.0
        stack = float(r.get("stackable_flag", 0) or 0)
        for day in pd.date_range(s, e, freq="D"):
            records.append({"dt": day, "n": 1, "pct": pct, "stack": stack})

    if not records:
        return {}

    tmp = pd.DataFrame(records)
    daily = tmp.groupby("dt").agg(
        count=("n", "sum"), pct=("pct", "mean"), stack=("stack", "mean"),
    )
    daily["doy"] = daily.index.dayofyear
    by_doy = daily.groupby("doy").agg(
        intensity=("count", "mean"),
        depth=("pct", "mean"),
        stackability=("stack", "mean"),
    )
    return by_doy.to_dict("index")

def extract_signup_velocity(customers: pd.DataFrame | None) -> dict:
    if customers is None or "signup_date" not in customers.columns:
        return {}
    c = customers.copy()
    c["signup_date"] = pd.to_datetime(c["signup_date"], errors="coerce").dt.normalize()
    c = c.dropna(subset=["signup_date"])
    daily = c.groupby("signup_date").size().sort_index()
    smooth = daily.rolling(7, min_periods=1).mean()
    return {pd.Timestamp(k): float(v) for k, v in smooth.items()}

def compute_monthly_revenue_prior(sales: pd.DataFrame) -> dict[int, float]:
    s = sales.copy()
    s["mo"] = s["Date"].dt.month
    return s.groupby("mo")["Revenue"].mean().to_dict()

def _derive_target_mean(sales: pd.DataFrame) -> float:
    annual = sales.groupby(sales["Date"].dt.year)["Revenue"].mean()
    tail5 = annual.tail(5)
    cagr = float((tail5.iloc[-1] / tail5.iloc[0]) ** (1.0 / max(len(tail5) - 1, 1)))
    return float(tail5.iloc[-1] * (cagr ** 1.5))

def build_seasonal_baseline(sales: pd.DataFrame, target_dates: pd.Series) -> np.ndarray:
    s = sales.copy()
    s["yr"] = s["Date"].dt.year
    s["mo"] = s["Date"].dt.month
    s["dy"] = s["Date"].dt.day

    annual = s.groupby("yr")["Revenue"].sum()
    full = annual.loc[2013:2022]
    geo_growth = float((1 + full.pct_change().dropna()).prod() ** (1 / 9))
    ann_mean = s.groupby("yr")["Revenue"].transform("mean")
    s["idx"] = s["Revenue"] / ann_mean
    seasonal = s.groupby(["mo", "dy"])["idx"].mean().reset_index()

    base = float(annual.loc[2022]) / 365

    td = pd.DataFrame({"Date": target_dates})
    td["mo"] = td["Date"].dt.month
    td["dy"] = td["Date"].dt.day
    td["ahead"] = td["Date"].dt.year - 2022
    td = td.merge(seasonal, on=["mo", "dy"], how="left")
    td["idx"] = td["idx"].fillna(1.0)

    return base * geo_growth ** td["ahead"].values * td["idx"].values

def vectorised_features(df: pd.DataFrame, date_col: str,
                        revenue_col: str | None,
                        promo_signals: dict, signup_vel: dict,
                        monthly_prior: dict) -> pd.DataFrame:

    dates = pd.to_datetime(df[date_col])
    F = pd.DataFrame(index=df.index)

    if revenue_col is not None:
        rev = df[revenue_col].astype(float)
        prev = rev.shift(1)

        for k in [1, 2, 7, 14, 21, 28]:
            F[f"L{k}"] = rev.shift(k)

        for w in [7, 14, 21]:
            F[f"RM{w}"] = prev.rolling(w).mean()
            F[f"RS{w}"] = prev.rolling(w).std()

        for sp in [7, 14]:
            F[f"EM{sp}"] = prev.ewm(span=sp, adjust=False).mean()

        F["MOM_1d7d"] = F["L1"] / (F["L7"] + 1e-8)
        F["MOM_1d14d"] = F["L1"] / (F["RM14"] + 1e-8)
        F["DIFF_1d7d"] = F["L1"] - F["L7"]

    doy = dates.dt.dayofyear.astype(float)
    dow = dates.dt.dayofweek.astype(float)
    dom = dates.dt.day.astype(float)
    mo = dates.dt.month.astype(float)

    F["SIN_Y"] = np.sin(2 * np.pi * doy / 365.25)
    F["COS_Y"] = np.cos(2 * np.pi * doy / 365.25)
    F["SIN_W"] = np.sin(2 * np.pi * dow / 7)
    F["COS_W"] = np.cos(2 * np.pi * dow / 7)
    F["SIN_M"] = np.sin(2 * np.pi * mo / 12)
    F["COS_M"] = np.cos(2 * np.pi * mo / 12)

    F["WE"] = (dates.dt.dayofweek >= 5).astype(float)
    F["PAY"] = ((dates.dt.day >= 25) | (dates.dt.day <= 3)).astype(float)
    F["M1111"] = ((dates.dt.month == 11) & (dates.dt.day == 11)).astype(float)
    F["M1212"] = ((dates.dt.month == 12) & (dates.dt.day == 12)).astype(float)
    F["MEDGE"] = (dates.dt.is_month_start | dates.dt.is_month_end).astype(float)
    F["WK"] = dates.dt.isocalendar().week.astype(float).values
    F["QTR"] = dates.dt.quarter.astype(float)

    F["PAY_WE"] = F["PAY"] * F["WE"]
    F["PAY_M11"] = F["PAY"] * F["M1111"]

    doy_int = dates.dt.dayofyear
    F["PR_N"] = [promo_signals.get(d, {}).get("intensity", 0.0) for d in doy_int]
    F["PR_PCT"] = [promo_signals.get(d, {}).get("depth", 0.0) for d in doy_int]
    F["PR_STK"] = [promo_signals.get(d, {}).get("stackability", 0.0) for d in doy_int]

    F["USR_V"] = [signup_vel.get(pd.Timestamp(d) - pd.Timedelta(days=7), 0.0)
                  for d in dates]

    F["M_PRIOR"] = [np.log1p(max(monthly_prior.get(int(m), 0.0), 0.0))
                    for m in mo]

    return F

def _row_features(dt: pd.Timestamp, hist: list[float],
                  cols: list[str],
                  promo_signals: dict, signup_vel: dict,
                  monthly_prior: dict) -> pd.DataFrame:
    r: dict[str, float] = {}

    def _g(lag):
        return hist[-lag] if len(hist) >= lag else hist[0]

    for k in [1, 2, 7, 14, 21, 28]:
        r[f"L{k}"] = _g(k)
    for w in [7, 14, 21]:
        chunk = hist[-w:] if len(hist) >= w else hist
        r[f"RM{w}"] = float(np.mean(chunk))
        r[f"RS{w}"] = float(np.std(chunk)) if len(chunk) > 1 else 0.0
    for sp in [7, 14]:
        alpha = 2.0 / (sp + 1)
        ema = hist[0]
        for v in hist[1:]:
            ema = alpha * v + (1 - alpha) * ema
        r[f"EM{sp}"] = float(ema)

    r["MOM_1d7d"] = r["L1"] / (r["L7"] + 1e-8)
    r["MOM_1d14d"] = r["L1"] / (r["RM14"] + 1e-8)
    r["DIFF_1d7d"] = r["L1"] - r["L7"]

    doy = dt.dayofyear
    dow = dt.dayofweek
    dom = dt.day
    mo = dt.month

    r["SIN_Y"] = float(np.sin(2 * np.pi * doy / 365.25))
    r["COS_Y"] = float(np.cos(2 * np.pi * doy / 365.25))
    r["SIN_W"] = float(np.sin(2 * np.pi * dow / 7))
    r["COS_W"] = float(np.cos(2 * np.pi * dow / 7))
    r["SIN_M"] = float(np.sin(2 * np.pi * mo / 12))
    r["COS_M"] = float(np.cos(2 * np.pi * mo / 12))

    r["WE"] = 1.0 if dow >= 5 else 0.0
    r["PAY"] = 1.0 if dom >= 25 or dom <= 3 else 0.0
    r["M1111"] = 1.0 if (mo == 11 and dom == 11) else 0.0
    r["M1212"] = 1.0 if (mo == 12 and dom == 12) else 0.0
    r["MEDGE"] = 1.0 if dt.is_month_start or dt.is_month_end else 0.0
    r["WK"] = float(dt.isocalendar().week)
    r["QTR"] = float(dt.quarter)
    r["PAY_WE"] = r["PAY"] * r["WE"]
    r["PAY_M11"] = r["PAY"] * r["M1111"]

    pp = promo_signals.get(doy, {})
    r["PR_N"] = pp.get("intensity", 0.0)
    r["PR_PCT"] = pp.get("depth", 0.0)
    r["PR_STK"] = pp.get("stackability", 0.0)

    r["USR_V"] = signup_vel.get(dt - pd.Timedelta(days=7), 0.0)
    r["M_PRIOR"] = float(np.log1p(max(monthly_prior.get(mo, 0.0), 0.0)))

    return pd.DataFrame([[r.get(c, 0.0) for c in cols]], columns=cols)

def build_weights(dates: pd.Series) -> np.ndarray:
    w = np.ones(len(dates))
    for i, d in enumerate(pd.to_datetime(dates)):
        if (d.month == 11 and d.day == 11) or (d.month == 12 and d.day == 12):
            w[i] = 3.0
        elif d.day >= 25 or d.day <= 3:
            w[i] = 1.5
    return w

def _make_lgb(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=15,
        subsample=0.88,
        colsample_bytree=0.86,
        reg_alpha=0.05,
        reg_lambda=1.2,
        objective="tweedie",
        tweedie_variance_power=1.35,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

def train_evaluate_and_forecast(
    sales: pd.DataFrame, future_dates: pd.Series,
    promo_signals: dict, signup_vel: dict, monthly_prior: dict,
) -> tuple[np.ndarray, float]:

    feat = vectorised_features(sales, "Date", "Revenue",
                               promo_signals, signup_vel, monthly_prior)
    target = np.log1p(sales["Revenue"].astype(float))

    merged = pd.concat([feat, target.rename("y")], axis=1).dropna().reset_index(drop=True)
    feature_cols = [c for c in merged.columns if c != "y"]
    X_all = merged[feature_cols]
    y_all = merged["y"].values

    dates_all = sales["Date"].iloc[-len(X_all):].reset_index(drop=True)
    w_all = build_weights(dates_all)

    cut = len(X_all) - HOLDOUT_DAYS
    X_tr, X_ho = X_all.iloc[:cut], X_all.iloc[cut:]
    y_tr, y_ho = y_all[:cut], y_all[cut:]
    w_tr = w_all[:cut]

    bags_ho = []
    for i in range(N_BAGS):
        m = _make_lgb(RNG_SEED + i * 7)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        bags_ho.append(m)

    ho_pred_log = np.mean([m.predict(X_ho) for m in bags_ho], axis=0)
    ho_pred = np.expm1(np.clip(ho_pred_log, -8, 18))
    ho_true = np.expm1(y_ho)
    ho_mae = mean_absolute_error(ho_true, ho_pred)
    print(f"  Holdout MAE (last {HOLDOUT_DAYS} d): {ho_mae:,.0f}")

    bags_full = []
    for i in range(N_BAGS):
        m = _make_lgb(RNG_SEED + i * 7)
        m.fit(X_all, y_all, sample_weight=w_all)
        bags_full.append(m)

    rev_hist = sales["Revenue"].astype(float).tolist()
    preds = []
    for dt in future_dates:
        row = _row_features(pd.Timestamp(dt), rev_hist, feature_cols,
                            promo_signals, signup_vel, monthly_prior)
        yhat_log = np.mean([m.predict(row)[0] for m in bags_full])
        yhat = max(float(np.expm1(np.clip(yhat_log, -8, 18))), 0.0)
        preds.append(yhat)
        rev_hist.append(yhat)

    return np.asarray(preds), ho_mae

def main() -> None:
    print("=" * 60)
    print("  Revenue Forecasting – Seasonal + Bagged LGB Residual")
    print("=" * 60)

    sales, sample, promos, customers, orders, web = load_all()
    future_dates = sample["Date"]
    target_mean = _derive_target_mean(sales)

    print("\n[1/4] Extracting external signals ...")
    promo_signals = extract_promo_signals(promos)
    signup_vel = extract_signup_velocity(customers)
    monthly_prior = compute_monthly_revenue_prior(sales)
    print(f"      Promo DOY entries: {len(promo_signals)}")
    print(f"      Signup velocity entries: {len(signup_vel)}")

    print("\n[2/4] Building seasonal baseline ...")
    baseline = build_seasonal_baseline(sales, future_dates)
    print(f"      Baseline mean: {baseline.mean():,.0f}")

    print("\n[3/4] Training bagged LightGBM ({} bags) ...".format(N_BAGS))
    ml_forecast, ho_mae = train_evaluate_and_forecast(
        sales, future_dates, promo_signals, signup_vel, monthly_prior,
    )
    print(f"      ML forecast mean (raw): {ml_forecast.mean():,.0f}")

    print("\n[4/4] Blending ML + baseline (adaptive) ...")
    n = len(ml_forecast)
    ml_weight = np.linspace(0.82, 0.58, n)
    blended = ml_weight * ml_forecast + (1 - ml_weight) * baseline

    scale = target_mean / blended.mean()
    final_revenue = blended * scale
    print(f"      Scale factor: {scale:.4f}")
    print(f"      Final mean Revenue: {final_revenue.mean():,.0f}")

    out = sample.copy()
    out["Revenue"] = np.round(final_revenue, 2)
    out["COGS"] = 0.0
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")

    out_path = ROOT / "submission.csv"
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"  DONE → {out_path}")
    print(f"  Revenue mean={final_revenue.mean():,.0f}  COGS mean=0")
    print("=" * 60)

if __name__ == "__main__":
    main()
