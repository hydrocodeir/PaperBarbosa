import pandas as pd
import numpy as np
from pathlib import Path
import re


COLUMN_ALIASES = {
    "station": ["station", "station_name", "stationid", "station_id", "stname", "name"],
    "tmean": ["tmean", "tas", "temp", "temperature", "tavg", "t_mean"],
    "tmin": ["tmin", "tn", "tasmin", "minimum_temperature", "t_min"],
    "tmax": ["tmax", "tx", "tasmax", "maximum_temperature", "t_max"],
}

def _read_tabular(path):
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    # sep=None lets pandas sniff delimiter (comma/semicolon/tab/...)
    return pd.read_csv(path, sep=None, engine="python")



def _normalize_columns(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _find_first_present(df, candidates):
    lc_to_orig = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lc_to_orig:
            return lc_to_orig[c.lower()]
    return None


def _normalize_numeric_text(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan

    # Persian/Arabic digits -> ASCII
    trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    s = s.translate(trans)

    # normalize decimal/thousand/minus variants
    s = (
        s.replace("٫", ".")
        .replace("،", ".")
        .replace(",", ".")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    # keep only numeric-relevant chars
    s = re.sub(r"[^0-9eE+\-\.]", "", s)
    if s in {"", ".", "-", "+"}:
        return np.nan
    return s


def _coerce_numeric(series):
    normalized = series.map(_normalize_numeric_text)
    return pd.to_numeric(normalized, errors="coerce")


def load_data(path, date_cols):
    df = _read_tabular(path)
    df = _normalize_columns(df)

    # Resolve date columns case-insensitively
    lc_to_orig = {str(c).strip().lower(): c for c in df.columns}
    resolved_date_cols = []
    for c in date_cols:
        key = str(c).strip().lower()
        if key not in lc_to_orig:
            raise ValueError(f"Missing date columns: {date_cols}")
        resolved_date_cols.append(lc_to_orig[key])

    df["date"] = pd.to_datetime(df[resolved_date_cols], errors="coerce")

    # Harmonize common climate column names to pipeline defaults
    st_col = _find_first_present(df, COLUMN_ALIASES["station"])
    if st_col is not None and st_col != "station_name":
        df = df.rename(columns={st_col: "station_name"})

    for key in ["tmin", "tmax", "tmean"]:
        src = _find_first_present(df, COLUMN_ALIASES[key])
        if src is not None and src != key:
            df = df.rename(columns={src: key})

    for col in ["tmin", "tmax", "tmean"]:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

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
    if target_variable not in sub.columns:
        raise ValueError(
            f"Target variable '{target_variable}' not found after load/preprocess. "
            f"Available columns: {list(sub.columns)}"
        )
    vals = sub[target_variable].to_numpy(dtype=float)
    vals = fill_short_gaps(vals, max_gap=max_gap)
    vals = fill_with_doy_climatology(sub["date"], vals)
    vals, outlier_mask = screen_outliers_sigma(vals, sigma=outlier_sigma)
    vals = fill_short_gaps(vals, max_gap=max_gap)
    vals = fill_with_doy_climatology(sub["date"], vals)
    sub[target_variable] = vals
    sub["outlier_flag"] = outlier_mask.astype(int)
    return sub


def run_input_precheck(df, station_col="station_name", target_col="tmean", date_col="date"):
    """Return one-row dataframe with input health checks before station processing."""
    row_count = int(len(df))
    station_missing_ratio = float(df[station_col].isna().mean()) if station_col in df.columns and row_count else 1.0
    target_missing_ratio = float(df[target_col].isna().mean()) if target_col in df.columns and row_count else 1.0
    invalid_date_ratio = float(df[date_col].isna().mean()) if date_col in df.columns and row_count else 1.0
    n_stations = int(df[station_col].nunique(dropna=True)) if station_col in df.columns else 0

    issues = []
    if row_count == 0:
        issues.append("EMPTY_INPUT")
    if len(df.columns) <= 1:
        issues.append("BAD_DELIMITER_OR_SINGLE_COLUMN")
    if invalid_date_ratio > 0:
        issues.append("INVALID_DATES_PRESENT")
    if station_missing_ratio > 0:
        issues.append("MISSING_STATION_IDS")
    if target_missing_ratio >= 0.95:
        issues.append("TARGET_ALMOST_ALL_MISSING")
    if n_stations == 0:
        issues.append("NO_STATION_FOUND")

    status = "OK" if not issues else ("FAIL" if "TARGET_ALMOST_ALL_MISSING" in issues or "NO_STATION_FOUND" in issues or "EMPTY_INPUT" in issues or "BAD_DELIMITER_OR_SINGLE_COLUMN" in issues else "WARN")

    return pd.DataFrame([
        {
            "status": status,
            "issues": "|".join(issues) if issues else "NONE",
            "row_count": row_count,
            "station_count": n_stations,
            "invalid_date_ratio": invalid_date_ratio,
            "station_missing_ratio": station_missing_ratio,
            "target_missing_ratio": target_missing_ratio,
        }
    ])


def summarize_preprocess_health(df, station_col="station_name", target_col="tmean"):
    rows = []
    if station_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(columns=["station_name", "n_rows", "n_valid_target", "missing_ratio_target"])

    for sid, sub in df.groupby(station_col):
        n = int(len(sub))
        valid = int(pd.to_numeric(sub[target_col], errors="coerce").notna().sum())
        miss = 1.0 - (valid / max(n, 1))
        rows.append(
            {
                "station_name": sid,
                "n_rows": n,
                "n_valid_target": valid,
                "missing_ratio_target": miss,
            }
        )
    return pd.DataFrame(rows)
