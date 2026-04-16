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
