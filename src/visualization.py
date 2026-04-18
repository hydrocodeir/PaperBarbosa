from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.cluster.hierarchy import dendrogram

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_publication_style():
    """
    Style tuned for journal-ready figures (readable labels, thin lines, high contrast).
    """
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _compute_hist_bins(values, min_bins=12, max_bins=50):
    """
    Compute histogram bin edges using the Freedman-Diaconis rule with robust fallbacks.
    This avoids degenerate histograms when bootstrap ranges are very narrow.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.array([0.0, 1.0])

    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    data_range = vals.max() - vals.min()

    if data_range <= 0:
        eps = max(abs(vals[0]) * 1e-4, 1e-6)
        return np.linspace(vals[0] - eps, vals[0] + eps, min_bins + 1)

    if iqr > 0:
        width = 2 * iqr * (vals.size ** (-1 / 3))
    else:
        width = data_range / min_bins

    if width <= 0:
        width = data_range / min_bins

    n_bins = int(np.ceil(data_range / width))
    n_bins = int(np.clip(n_bins, min_bins, max_bins))
    return np.linspace(vals.min(), vals.max(), n_bins + 1)


def _iter_geojson_paths(geometry):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            yield np.asarray(ring, dtype=float)
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                yield np.asarray(ring, dtype=float)
    elif gtype == "LineString":
        yield np.asarray(coords, dtype=float)
    elif gtype == "MultiLineString":
        for line in coords:
            yield np.asarray(line, dtype=float)
    elif gtype == "GeometryCollection":
        for subgeom in geometry.get("geometries", []):
            yield from _iter_geojson_paths(subgeom)


def plot_figure1_station_map(stations_df, geojson_path, outpath):
    """
    Paper-style station map: country boundary + station points.
    """
    ensure_dir(Path(outpath).parent)
    set_publication_style()

    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    fig, ax = plt.subplots(figsize=(8.5, 8))

    features = gj.get("features", [])
    for feat in features:
        geom = feat.get("geometry", {})
        for ring in _iter_geojson_paths(geom):
            if ring.ndim != 2 or ring.shape[1] < 2:
                continue
            ax.plot(ring[:, 0], ring[:, 1], color="0.35", linewidth=0.8)

    st = stations_df.dropna(subset=["longitude", "latitude"]).copy()
    ax.scatter(st["longitude"], st["latitude"], s=30, color="black", zorder=3)

    # number labels similar to paper map notation
    for idx, row in st.reset_index(drop=True).iterrows():
        ax.text(
            row["longitude"] + 0.08,
            row["latitude"] + 0.08,
            str(idx + 1),
            fontsize=7,
            color="black",
            zorder=4,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Figure 1. Station map over Iran")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_figure3_quantile_slopes(
    x_decades,
    y,
    quantiles,
    station_name,
    outpath
):
    """
    بازتولید Fig 3:
    - quantile slopes
    - standard error shading
    - OLS slope (horizontal dashed line)
    """

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    x = np.asarray(x_decades, dtype=float)
    y = np.asarray(y, dtype=float)

    X = sm.add_constant(x)

    slopes = []
    ses = []

    # --- Quantile regression ---
    for q in quantiles:
        model = sm.QuantReg(y, X)
        res = model.fit(q=q)
        slopes.append(res.params[1])
        ses.append(res.bse[1])

    slopes = np.array(slopes)
    ses = np.array(ses)

    # --- OLS slope ---
    ols = sm.OLS(y, X).fit()
    ols_slope = ols.params[1]

    # --- Plot ---
    plt.figure(figsize=(7, 5))

    # main curve
    plt.plot(quantiles, slopes, marker='o', color='black')

    # error band
    plt.fill_between(
        quantiles,
        slopes - ses,
        slopes + ses,
        color='gray',
        alpha=0.3
    )

    # OLS horizontal dashed line
    plt.axhline(
        y=ols_slope,
        linestyle='--',
        color='black',
        linewidth=1.5
    )

    plt.xlabel("Quantile")
    plt.ylabel("Slope (°C/decade)")
    plt.title(f"{station_name}", pad=24)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_figure4_bootstrap(
    boot_df,
    fit_df,
    station_name,
    quantiles,
    outpath,
    x_mode="absolute",
    center_scale=1.0,
):
    """
    Fig 4:
    - Adaptive histogram bins for each quantile
    - Shared x-limits across quantiles (for direct comparison)
    - dashed vertical line = point estimate
    """

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    fig, axes = plt.subplots(len(quantiles), 1, figsize=(6, 8), sharex=True)
    if len(quantiles) == 1:
        axes = [axes]

    x_mode = str(x_mode).lower()
    if x_mode not in {"absolute", "centered"}:
        raise ValueError("x_mode must be one of {'absolute', 'centered'}")

    transformed_values_all = []
    per_quantile = []

    # --- رسم subplot ها ---
    for i, q in enumerate(quantiles):
        sub_boot = boot_df[
            (boot_df["station_name"] == station_name) &
            (boot_df["quantile"] == q)
        ]

        sub_fit = fit_df[
            (fit_df["station_name"] == station_name) &
            (fit_df["quantile"] == q)
        ]

        if sub_boot.empty or sub_fit.empty:
            per_quantile.append((q, None, None))
            continue

        slope = float(sub_fit["slope_per_decade"].iloc[0])
        values = sub_boot["slope_per_decade"].to_numpy(dtype=float)
        if x_mode == "centered":
            transformed = (values - slope) * center_scale
            ref_line_x = 0.0
        else:
            transformed = values
            ref_line_x = slope

        transformed_values_all.extend(transformed.tolist())
        per_quantile.append((q, transformed, ref_line_x))

    all_values = np.asarray(transformed_values_all, dtype=float)
    all_values = all_values[np.isfinite(all_values)]
    global_bins = _compute_hist_bins(all_values) if all_values.size else np.array([0.0, 1.0])

    for i, (q, transformed, ref_line_x) in enumerate(per_quantile):
        ax = axes[i]
        if transformed is None:
            ax.set_title(f"τ = {q} (no data)")
            continue
        # --- histogram ---
        ax.hist(
            transformed,
            bins=global_bins,
            color="gray",
            edgecolor="black"
        )

        # --- zoom on robust range for readability ---
        # --- خط vertical dashed ---
        ax.axvline(
            x=ref_line_x,
            linestyle="--",
            color="black",
            linewidth=1.5
        )

        ax.set_title(f"τ = {q}")
        ax.set_ylabel("Count")

    if all_values.size:
        x_low, x_high = np.percentile(all_values, [0.5, 99.5])
        x_span = x_high - x_low
        x_pad = max(0.12 * x_span, 1e-6)

        if x_high <= x_low:
            center = float(np.median(all_values))
            for ax in axes:
                ax.set_xlim(center - max(x_pad, 1e-6), center + max(x_pad, 1e-6))
        else:
            for ax in axes:
                ax.set_xlim(x_low - x_pad, x_high + x_pad)

    if x_mode == "centered":
        axes[-1].set_xlabel(f"Δ Slope relative to τ estimate (×{center_scale:g} °C/decade)")
    else:
        axes[-1].set_xlabel("Slope (°C/decade)")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_station_timeseries(date, raw, seasonal, anomaly, station_name, outpath):
    ensure_dir(Path(outpath).parent)
    set_publication_style()
    plt.figure(figsize=(12, 5))
    plt.plot(date, raw, linewidth=0.8, label="raw tmean")
    plt.plot(date, seasonal, linewidth=1.2, label="seasonal fit")
    plt.plot(date, anomaly, linewidth=0.8, label="deseasoned")
    plt.title(f"{station_name}: raw, seasonal fit, and deseasoned series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_quantile_grid(grid_df, station_name, outpath):
    ensure_dir(Path(outpath).parent)
    set_publication_style()
    sub = grid_df[grid_df["station_name"] == station_name].sort_values("quantile")
    plt.figure(figsize=(7, 5))
    plt.plot(sub["quantile"], sub["slope_per_decade"], marker="o")
    if "std_error" in sub.columns:
        y = sub["slope_per_decade"].to_numpy()
        se = sub["std_error"].to_numpy()
        plt.fill_between(sub["quantile"], y - se, y + se, alpha=0.2)
    plt.xlabel("Quantile")
    plt.ylabel("Slope (°C/decade)")
    plt.title(f"{station_name}: quantile slope curve")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bootstrap_hist(boot_df, station_name, quantile, outpath):
    ensure_dir(Path(outpath).parent)
    set_publication_style()
    sub = boot_df[
        (boot_df["station_name"] == station_name)
        & (boot_df["quantile"] == quantile)
    ]
    plt.figure(figsize=(6, 4))
    values = sub["slope_per_decade"].to_numpy(dtype=float)
    bins = _compute_hist_bins(values)
    plt.hist(values, bins=bins)
    if values.size:
        q_low, q_high = np.percentile(values, [0.5, 99.5])
        span = q_high - q_low
        pad = max(0.15 * span, 1e-6)
        if span <= 0:
            center = float(np.median(values))
            plt.xlim(center - pad, center + pad)
        else:
            plt.xlim(q_low - pad, q_high + pad)
    plt.xlabel("Slope (°C/decade)")
    plt.ylabel("Count")
    plt.title(f"{station_name}: bootstrap slopes q={quantile}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dendrogram(Z, labels, quantile, outpath):
    ensure_dir(Path(outpath).parent)
    set_publication_style()
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=labels)
    plt.title(f"Dendrogram for quantile {quantile}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_figure2_deseasoned(date, deseasoned, time_decades, station_name, outpath):
    """
    بازتولید شکل 2 مقاله برای داده های deseasoned:
    - سری زمانی anomaly
    - خط روند quantile regression برای 0.05، 0.5، 0.95
    - 0.05 dashed
    - 0.5 solid
    - 0.95 dotted
    """
    ensure_dir(Path(outpath).parent)
    set_publication_style()

    date = pd.to_datetime(date)
    y = np.asarray(deseasoned, dtype=float)
    x = np.asarray(time_decades, dtype=float)

    X = sm.add_constant(x)

    q05 = sm.QuantReg(y, X).fit(q=0.05)
    q50 = sm.QuantReg(y, X).fit(q=0.50)
    q95 = sm.QuantReg(y, X).fit(q=0.95)

    yhat05 = q05.predict(X)
    yhat50 = q50.predict(X)
    yhat95 = q95.predict(X)

    plt.figure(figsize=(12, 5))
    plt.plot(date, y, color="black", linewidth=0.5)

    # مطابق کپشن مقاله
    plt.plot(date, yhat05, linestyle="--", linewidth=1.4, color="black")
    plt.plot(date, yhat50, linestyle="-",  linewidth=1.4, color="black")
    plt.plot(date, yhat95, linestyle=":",  linewidth=1.8, color="black")

    slope_text = (
        f"Slope/decade  τ=0.05: {q05.params[1]:+.3f} °C"
        f" | τ=0.50: {q50.params[1]:+.3f} °C"
        f" | τ=0.95: {q95.params[1]:+.3f} °C"
    )
    plt.gca().text(
        0.5,
        1.005,
        slope_text,
        transform=plt.gca().transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.title(f"{station_name}", pad=20)
    plt.ylabel("Deseasoned daily mean temperature anomaly (°C)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_homogenization_breaks(date, raw_values, adjusted_values, break_dates, station_name, outpath):
    ensure_dir(Path(outpath).parent)
    set_publication_style()

    d = pd.to_datetime(date)
    raw = np.asarray(raw_values, dtype=float)
    adj = np.asarray(adjusted_values, dtype=float)

    plt.figure(figsize=(12, 5))
    plt.plot(d, raw, color="0.45", linewidth=0.8, label="Raw")
    plt.plot(d, adj, color="black", linewidth=1.0, label="Homogenized")

    for i, bd in enumerate(break_dates):
        bd = pd.to_datetime(bd)
        plt.axvline(bd, color="tab:red", linestyle="--", linewidth=1.0)
        if i == 0:
            plt.text(bd, np.nanmax(adj), " Break", color="tab:red", fontsize=8, va="top")

    plt.title(f"{station_name}: RHtests-like break detection (SNHT)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_preanalysis_heatmap(test_df, outpath):
    ensure_dir(Path(outpath).parent)
    set_publication_style()

    if test_df.empty:
        return

    cols = ["missing_ratio", "outlier_ratio", "mk_pvalue", "adf_pvalue", "ljungbox_pvalue", "normaltest_pvalue"]
    available = [c for c in cols if c in test_df.columns]
    mat = test_df.set_index("station_name")[available].copy()
    mat = mat.apply(pd.to_numeric, errors="coerce")

    fig_h = max(3.5, 0.45 * len(mat) + 1.6)
    plt.figure(figsize=(10, fig_h))
    im = plt.imshow(mat.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Value")
    plt.xticks(range(len(available)), available, rotation=35, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.title("Pre-analysis diagnostics heatmap")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_station_preanalysis_panel(date, values, station_name, outpath):
    ensure_dir(Path(outpath).parent)
    set_publication_style()

    d = pd.to_datetime(date)
    v = np.asarray(values, dtype=float)
    if len(d) != len(v):
        raise ValueError(f"date/value length mismatch: {len(d)} vs {len(v)}")
    y = pd.Series(v, index=d, dtype=float)
    y_valid = y.dropna()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    if y_valid.empty:
        axes[0].text(
            0.5,
            0.5,
            "No valid observations after preprocessing",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[0].set_title(f"{station_name}: pre-analysis quality panel")
        axes[0].set_ylabel("Temperature (°C)")

        axes[1].text(
            0.5,
            0.5,
            "Histogram unavailable (all values are NaN)",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        axes[1].set_xlabel("Temperature (°C)")
        axes[1].set_ylabel("Count")
    else:
        roll = y_valid.rolling(30, min_periods=10).mean()
        axes[0].plot(y_valid.index, y_valid.to_numpy(dtype=float), color="0.55", linewidth=0.7, label="Daily")
        axes[0].plot(roll.index, roll.to_numpy(dtype=float), color="black", linewidth=1.2, label="30-day mean")
        axes[0].set_title(f"{station_name}: pre-analysis quality panel")
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].legend()

        yy = y_valid.to_numpy(dtype=float)
        bins = min(35, max(10, int(np.sqrt(yy.size))))
        axes[1].hist(yy, bins=bins, color="gray", edgecolor="black")
        axes[1].set_xlabel("Temperature (°C)")
        axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
