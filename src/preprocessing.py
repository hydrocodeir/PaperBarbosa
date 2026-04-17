import pandas as pd
import numpy as np


def load_data(path, date_cols):
    df = pd.read_csv(path)
    missing = [c for c in date_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing date columns: {missing}")
    df["date"] = pd.to_datetime(df[date_cols])
    return df

def fill_tmean(df):
    df = df.copy()
    if "tmean" not in df.columns:
        df["tmean"] = np.nan
    if {"tmin", "tmax"}.issubset(df.columns):
        est = (df["tmin"] + df["tmax"]) / 2.0
        df["tmean"] = df["tmean"].fillna(est)
    return df


def filter_station_coverage(
    df,
    station_col="station_name",
    date_col="date",
    target_col="tmean",
    min_start_date=None,
    max_end_date=None,
    max_missing_ratio=None,
):
    """
    Optional pre-filter that mirrors the paper's station selection idea:
    keep stations with sufficient temporal coverage and limited missingness.
    """
    if station_col not in df.columns:
        raise ValueError(f"Missing station column '{station_col}'")
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'")

    keep_ids = []
    for sid, sub in df.groupby(station_col):
        s = sub.sort_values(date_col)
        start_ok = True if min_start_date is None else (s[date_col].min() <= pd.Timestamp(min_start_date))
        end_ok = True if max_end_date is None else (s[date_col].max() >= pd.Timestamp(max_end_date))
        miss_ratio = s[target_col].isna().mean()
        missing_ok = True if max_missing_ratio is None else (miss_ratio <= float(max_missing_ratio))
        if start_ok and end_ok and missing_ok:
            keep_ids.append(sid)
    return df[df[station_col].isin(keep_ids)].copy()


def reindex_station_daily(sub):
    sub = sub.sort_values("date").copy()
    idx = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
    out = sub.set_index("date").reindex(idx)
    out.index.name = "date"
    out = out.reset_index()
    out["station_name"] = sub["station_name"].iloc[0]
    if "station_id" in sub.columns:
        out["station_id"] = sub["station_id"].iloc[0]
    return out

def fill_short_gaps(series, max_gap=3):
    s = pd.Series(series, dtype=float)
    return s.interpolate(limit=max_gap, limit_direction="both").to_numpy()

def fill_with_doy_climatology(dates, values):
    s = pd.Series(values, index=pd.to_datetime(dates), dtype=float)
    doy = s.index.dayofyear
    clim = s.groupby(doy).mean()
    filled = s.copy()
    miss = filled.isna()
    if miss.any():
        filled.loc[miss] = [clim.get(d, np.nan) for d in doy[miss]]
    return filled.to_numpy()

def screen_outliers_sigma(values, sigma=7.0):
    s = pd.Series(values, dtype=float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s.to_numpy(), np.zeros(len(s), dtype=bool)
    mask = ((s - mu).abs() > sigma * sd).fillna(False).to_numpy()
    s.loc[mask] = np.nan
    return s.to_numpy(), mask

def prepare_station_series(sub, target_variable="tmean", max_gap=3, outlier_sigma=7.0):
    sub = reindex_station_daily(sub)
    vals = sub[target_variable].to_numpy(dtype=float) if target_variable in sub.columns else np.full(len(sub), np.nan)
    vals = fill_short_gaps(vals, max_gap=max_gap)
    vals = fill_with_doy_climatology(sub["date"], vals)
    vals, outlier_mask = screen_outliers_sigma(vals, sigma=outlier_sigma)
    vals = fill_short_gaps(vals, max_gap=max_gap)
    vals = fill_with_doy_climatology(sub["date"], vals)
    sub[target_variable] = vals
    sub["outlier_flag"] = outlier_mask.astype(int)
    return sub
