import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def l2_wasserstein_empirical(a, b):
    """
    Empirical L2-Wasserstein distance between two 1D distributions.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))

    if len(a) == 0 or len(b) == 0:
        return np.nan

    n = max(len(a), len(b))
    p = np.linspace(0.0, 1.0, n)

    qa = np.quantile(a, p)
    qb = np.quantile(b, p)

    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def distance_matrix_from_bootstrap(boot_df, quantile):
    sub = boot_df[boot_df["quantile"] == quantile].copy()
    stations = sorted(sub["station_name"].unique())

    D = np.zeros((len(stations), len(stations)), dtype=float)

    for i, s1 in enumerate(stations):
        a = sub.loc[sub["station_name"] == s1, "slope_per_decade"].to_numpy()
        for j in range(i + 1, len(stations)):
            s2 = stations[j]
            b = sub.loc[sub["station_name"] == s2, "slope_per_decade"].to_numpy()
            d = l2_wasserstein_empirical(a, b)
            D[i, j] = d
            D[j, i] = d

    return stations, pd.DataFrame(D, index=stations, columns=stations)


def linkage_from_distance_matrix(D_df):
    condensed = squareform(D_df.to_numpy(), checks=True)
    return linkage(condensed, method="average")