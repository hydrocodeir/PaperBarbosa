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
    pass_outlier = bool(np.isnan(outlier_ratio) or outlier_ratio <= 0.02)

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
    return pd.DataFrame(rows)
