from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import gaussian_kde

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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
    plt.title(f"{station_name}")

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_figure4_bootstrap(
    boot_df,
    fit_df,
    station_name,
    quantiles,
    outpath
):
    """
    Fig 4 دقیق:
    - x-axis مشترک برای همه subplot ها
    - bin width = 0.005
    - dashed vertical line = punctual estimate
    """

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(quantiles), 1, figsize=(6, 8), sharex=True)

    # --- جمع کردن کل داده برای تعیین محدوده مشترک ---
    all_values = []

    for q in quantiles:
        sub = boot_df[
            (boot_df["station_name"] == station_name) &
            (boot_df["quantile"] == q)
        ]
        all_values.extend(sub["slope_per_decade"].values)

    all_values = np.array(all_values)

    # --- تعیین بازه مشترک ---
    xmin = np.nanmin(all_values)
    xmax = np.nanmax(all_values)

    # --- ایجاد bin با گام 0.005 ---
    bin_width = 0.001
    bins = np.arange(xmin, xmax + bin_width, bin_width)

    # --- رسم subplot ها ---
    for i, q in enumerate(quantiles):
        ax = axes[i]

        sub_boot = boot_df[
            (boot_df["station_name"] == station_name) &
            (boot_df["quantile"] == q)
        ]

        sub_fit = fit_df[
            (fit_df["station_name"] == station_name) &
            (fit_df["quantile"] == q)
        ]

        if sub_boot.empty or sub_fit.empty:
            ax.set_title(f"τ = {q} (no data)")
            continue

        slope = sub_fit["slope_per_decade"].iloc[0]

        # --- histogram ---
        ax.hist(
            sub_boot["slope_per_decade"],
            bins=bins,
            color="gray",
            edgecolor="black"
        )

        # --- خط vertical dashed ---
        ax.axvline(
            x=slope,
            linestyle="--",
            color="black",
            linewidth=1.5
        )

        ax.set_title(f"τ = {q}")
        ax.set_ylabel("Count")

    axes[-1].set_xlabel("Slope (°C/decade)")

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_station_timeseries(date, raw, seasonal, anomaly, station_name, outpath):
    ensure_dir(Path(outpath).parent)
    plt.figure(figsize=(12, 5))
    plt.plot(date, raw, linewidth=0.8, label="raw tmean")
    plt.plot(date, seasonal, linewidth=1.2, label="seasonal fit")
    plt.plot(date, anomaly, linewidth=0.8, label="deseasoned")
    plt.title(f"{station_name}: raw, seasonal fit, and deseasoned series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_quantile_grid(grid_df, station_name, outpath):
    ensure_dir(Path(outpath).parent)
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
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_bootstrap_hist(boot_df, station_name, quantile, outpath):
    ensure_dir(Path(outpath).parent)
    sub = boot_df[
        (boot_df["station_name"] == station_name)
        & (boot_df["quantile"] == quantile)
    ]
    plt.figure(figsize=(6, 4))
    plt.hist(sub["slope_per_decade"], bins=20)
    plt.xlabel("Slope (°C/decade)")
    plt.ylabel("Count")
    plt.title(f"{station_name}: bootstrap slopes q={quantile}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_dendrogram(Z, labels, quantile, outpath):
    ensure_dir(Path(outpath).parent)
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=labels)
    plt.title(f"Dendrogram for quantile {quantile}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
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

    plt.title(f"{station_name}")
    plt.ylabel("Deseasoned daily mean temperature anomaly (°C)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()