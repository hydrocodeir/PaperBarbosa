import numpy as np
import pandas as pd

def build_harmonic_design(n, periods_per_year=2):
    t = np.arange(n)
    cols = [np.ones(n)]
    for k in range(1, periods_per_year + 1):
        cols.append(np.sin(2 * np.pi * k * t / 365.25))
        cols.append(np.cos(2 * np.pi * k * t / 365.25))
    return np.column_stack(cols)

def deseasonalize(values, periods_per_year=2):
    y = np.asarray(values, dtype=float)
    X = build_harmonic_design(len(y), periods_per_year=periods_per_year)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    seasonal = X @ beta
    resid = y - seasonal
    return resid, seasonal, beta

def decimal_year(dates):
    dt = pd.to_datetime(dates)
    years = dt.dt.year
    start = pd.to_datetime(years.astype(str) + "-01-01")
    end = pd.to_datetime((years + 1).astype(str) + "-01-01")
    return years + ((dt - start) / (end - start)).astype(float)

def decade_index(dates):
    yr = decimal_year(dates)
    return (yr - yr.min()) / 10.0