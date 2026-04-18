import numpy as np
import pandas as pd
from scipy.stats import kendalltau, normaltest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


def _safe_float(value):
    try:
        v = float(value)
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


def run_preanalysis_tests(
    date,
    values,
    station_name,
    missing_ratio_threshold=0.05,
    outlier_sigma=4.0,
    ljungbox_lag=30,
):
    """
    Q1-grade pre-analysis tests for temperature series.

    Returns a one-row DataFrame with:
    - data quality: completeness / outlier ratio
    - trend: Mann-Kendall (tau, p)
    - stationarity: ADF p-value
    - autocorrelation: Ljung-Box p-value
    - normality: D'Agostino K^2 p-value
    - global quality flag
    """
    d = pd.to_datetime(date)
    s = pd.Series(values, index=d, dtype=float)

    n_total = int(len(s))
    n_valid = int(s.notna().sum())
    missing_ratio = 1.0 - (n_valid / max(n_total, 1))

    x = s.dropna().to_numpy(dtype=float)
    if x.size > 1:
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=1)
    else:
        mu = np.nan
        sd = np.nan

    if np.isfinite(sd) and sd > 0:
        outlier_ratio = float(np.mean(np.abs((x - mu) / sd) > outlier_sigma))
    else:
        outlier_ratio = np.nan

    if x.size >= 10:
        t = np.arange(x.size, dtype=float)
        mk = kendalltau(t, x)
        mk_tau = _safe_float(mk.statistic)
        mk_p = _safe_float(mk.pvalue)
    else:
        mk_tau = np.nan
        mk_p = np.nan

    if x.size >= 30:
        try:
            adf_p = _safe_float(adfuller(x, autolag="AIC")[1])
        except Exception:
            adf_p = np.nan
        try:
            lag_eff = int(min(max(ljungbox_lag, 1), max(x.size // 4, 1)))
            lb = acorr_ljungbox(x, lags=[lag_eff], return_df=True)
            lb_p = _safe_float(lb["lb_pvalue"].iloc[0])
        except Exception:
            lb_p = np.nan
    else:
        adf_p = np.nan
        lb_p = np.nan

    if x.size >= 20:
        try:
            normal_p = _safe_float(normaltest(x).pvalue)
        except Exception:
            normal_p = np.nan
    else:
        normal_p = np.nan

    pass_missing = bool(missing_ratio <= missing_ratio_threshold)
    pass_outlier = bool((n_valid > 0) and (np.isnan(outlier_ratio) or outlier_ratio <= 0.02))

    reasons = []
    if n_valid == 0:
        reasons.append("ALL_NAN_AFTER_PREPROCESS")
    if not pass_missing:
        reasons.append("HIGH_MISSING_RATIO")
    if not pass_outlier:
        reasons.append("HIGH_OUTLIER_RATIO")
    if np.isfinite(mk_p) and mk_p < 0.05:
        reasons.append("SIGNIFICANT_MONOTONIC_TREND")
    if np.isfinite(adf_p) and adf_p > 0.05:
        reasons.append("NON_STATIONARY")
    if np.isfinite(lb_p) and lb_p < 0.05:
        reasons.append("SERIAL_CORRELATION")
    if np.isfinite(normal_p) and normal_p < 0.05:
        reasons.append("NON_NORMAL_DISTRIBUTION")
    reason_code = "OK" if not reasons else "|".join(reasons)

    row = {
        "station_name": station_name,
        "n_total": n_total,
        "n_valid": n_valid,
        "missing_ratio": float(missing_ratio),
        "outlier_ratio": float(outlier_ratio) if np.isfinite(outlier_ratio) else np.nan,
        "mk_tau": mk_tau,
        "mk_pvalue": mk_p,
        "adf_pvalue": adf_p,
        "ljungbox_pvalue": lb_p,
        "normaltest_pvalue": normal_p,
        "pass_missing": int(pass_missing),
        "pass_outlier": int(pass_outlier),
        "quality_pass": int(pass_missing and pass_outlier),
        "reason_code": reason_code,
    }
    return pd.DataFrame([row])


def summarize_preanalysis(df_tests):
    if df_tests.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "mean",
                "median",
                "p05",
                "p95",
            ]
        )

    metrics = [
        "missing_ratio",
        "outlier_ratio",
        "mk_tau",
        "mk_pvalue",
        "adf_pvalue",
        "ljungbox_pvalue",
        "normaltest_pvalue",
    ]

    rows = []
    for m in metrics:
        vals = pd.to_numeric(df_tests[m], errors="coerce").dropna()
        if vals.empty:
            rows.append({"metric": m, "mean": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan})
            continue
        rows.append(
            {
                "metric": m,
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "p05": float(vals.quantile(0.05)),
                "p95": float(vals.quantile(0.95)),
            }
        )

    quality_rate = float(df_tests["quality_pass"].mean()) if "quality_pass" in df_tests else np.nan
    rows.append({"metric": "quality_pass_rate", "mean": quality_rate, "median": quality_rate, "p05": quality_rate, "p95": quality_rate})

    if "reason_code" in df_tests.columns:
        reason_freq = df_tests["reason_code"].fillna("UNKNOWN").value_counts(normalize=True)
        for reason, frac in reason_freq.items():
            rows.append({
                "metric": f"reason_freq::{reason}",
                "mean": float(frac),
                "median": float(frac),
                "p05": float(frac),
                "p95": float(frac),
            })

    return pd.DataFrame(rows)


def build_preanalysis_publication_table(df_tests):
    """
    Build a manuscript-ready diagnostic table with human-readable labels.
    """
    if df_tests.empty:
        return pd.DataFrame(
            columns=[
                "station_name",
                "quality_status",
                "main_issue",
                "missing_percent",
                "outlier_percent",
                "mk_pvalue",
                "adf_pvalue",
                "ljungbox_pvalue",
                "normaltest_pvalue",
                "recommendation",
            ]
        )

    out = df_tests.copy()
    out["missing_percent"] = pd.to_numeric(out.get("missing_ratio"), errors="coerce") * 100.0
    out["outlier_percent"] = pd.to_numeric(out.get("outlier_ratio"), errors="coerce") * 100.0

    out["quality_status"] = np.where(out.get("quality_pass", 0).astype(int) == 1, "PASS", "REVIEW")

    def _main_issue(reason):
        if pd.isna(reason):
            return "Unknown"
        if reason == "OK":
            return "None"
        parts = str(reason).split("|")
        return parts[0]

    out["main_issue"] = out.get("reason_code").apply(_main_issue)

    def _recommend(row):
        r = str(row.get("reason_code", ""))
        if "ALL_NAN_AFTER_PREPROCESS" in r:
            return "Exclude station; recover raw source data"
        if "HIGH_MISSING_RATIO" in r:
            return "Apply stricter gap handling / metadata review"
        if "HIGH_OUTLIER_RATIO" in r:
            return "Recheck sensor metadata and outlier thresholds"
        if "NON_STATIONARY" in r:
            return "Use detrending / differencing before inference"
        if "SERIAL_CORRELATION" in r:
            return "Use block bootstrap / autocorrelation-aware CI"
        if "NON_NORMAL_DISTRIBUTION" in r:
            return "Prefer robust/non-parametric estimators"
        if "SIGNIFICANT_MONOTONIC_TREND" in r:
            return "Report trend explicitly in baseline table"
        return "Ready for downstream modeling"

    out["recommendation"] = out.apply(_recommend, axis=1)

    keep = [
        "station_name",
        "quality_status",
        "main_issue",
        "missing_percent",
        "outlier_percent",
        "mk_pvalue",
        "adf_pvalue",
        "ljungbox_pvalue",
        "normaltest_pvalue",
        "recommendation",
    ]
    out = out[keep].sort_values(["quality_status", "main_issue", "station_name"]).reset_index(drop=True)
    return out
