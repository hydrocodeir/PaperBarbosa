from pathlib import Path
import warnings
import yaml
import numpy as np
import pandas as pd

from src.preprocessing import (
    load_data,
    fill_tmean,
    prepare_station_series,
    filter_station_coverage,
    run_input_precheck,
    summarize_preprocess_health,
)
from src.feature_engineering import deseasonalize, decade_index
from src.homogenization import detect_breakpoints_snht, mean_shift_adjustment
from src.diagnostics import run_preanalysis_tests, summarize_preanalysis, build_preanalysis_publication_table
from src.modeling import fit_quantiles, maximum_entropy_bootstrap_slopes
from src.evaluation import summarize_bootstrap, merge_fit_and_bootstrap
from src.clustering import distance_matrix_from_bootstrap, linkage_from_distance_matrix
from src.visualization import (
    plot_station_timeseries,
    plot_quantile_grid,
    plot_bootstrap_hist,
    plot_dendrogram,
    plot_figure2_deseasoned,
    plot_figure3_quantile_slopes,
    plot_figure4_bootstrap,
    plot_figure1_station_map,
    plot_homogenization_breaks,
    plot_preanalysis_heatmap,
    plot_station_preanalysis_panel,
)


def process_single_station(station_name, sub, cfg, fig_dir):
    target = cfg["target_variable"]
    sub = prepare_station_series(
        sub,
        target_variable=target,
        max_gap=cfg["max_interp_gap"],
        outlier_sigma=cfg["outlier_sigma"],
    )

    hom_cfg = cfg.get("homogenization", {})
    use_hom = bool(hom_cfg.get("enabled", False))

    break_rows = []
    adjust_rows = []
    if use_hom:
        breaks = detect_breakpoints_snht(
            sub[target].to_numpy(),
            min_segment=hom_cfg.get("min_segment", 365),
            threshold=hom_cfg.get("snht_threshold", 120.0),
            max_breaks=hom_cfg.get("max_breaks", 5),
        )
        break_indices = [b[0] for b in breaks]
        sub[f"{target}_raw"] = sub[target].to_numpy(dtype=float)
        homogenized, adjust_df = mean_shift_adjustment(sub[target].to_numpy(), break_indices)
        sub[target] = homogenized

        for b_idx, score in breaks:
            break_rows.append(
                {
                    "station_name": station_name,
                    "break_index": int(b_idx),
                    "break_date": pd.Timestamp(sub["date"].iloc[int(b_idx)]),
                    "snht_score": float(score),
                }
            )

        if not adjust_df.empty:
            adjust_df["station_name"] = station_name
            adjust_rows = adjust_df.to_dict("records")

        plot_homogenization_breaks(
            sub["date"],
            sub[f"{target}_raw"],
            sub[target],
            [sub["date"].iloc[i] for i in break_indices],
            station_name,
            fig_dir / f"{station_name}_homogenization_breaks.png",
        )

    diag_cfg = cfg.get("diagnostics", {})
    diag_df = run_preanalysis_tests(
        sub["date"],
        sub[target],
        station_name,
        missing_ratio_threshold=diag_cfg.get("max_missing_ratio", 0.05),
        outlier_sigma=diag_cfg.get("outlier_sigma", 4.0),
        ljungbox_lag=diag_cfg.get("ljungbox_lag", 30),
    )
    plot_station_preanalysis_panel(
        sub["date"],
        sub[target],
        station_name,
        fig_dir / f"{station_name}_preanalysis_panel.png",
    )

    anomaly, seasonal, _ = deseasonalize(
        sub[target].to_numpy(),
        periods_per_year=cfg["harmonics_per_year"],
    )
    sub["seasonal_fit"] = seasonal
    sub["deseasoned"] = anomaly
    sub["time_decades"] = decade_index(sub["date"])

    plot_figure2_deseasoned(
        sub["date"],
        sub["deseasoned"],
        sub["time_decades"],
        station_name,
        fig_dir / f"{station_name}_figure2_deseasoned.png",
    )

    fit_df = fit_quantiles(sub["time_decades"], sub["deseasoned"], cfg["quantiles"])
    fit_df["station_name"] = station_name

    grid_df = fit_quantiles(sub["time_decades"], sub["deseasoned"], cfg["quantile_grid"])
    grid_df["station_name"] = station_name

    plot_figure3_quantile_slopes(
        sub["time_decades"],
        sub["deseasoned"],
        cfg["quantile_grid"],
        station_name,
        fig_dir / f"{station_name}_figure3.png",
    )

    boot_df = maximum_entropy_bootstrap_slopes(
        sub["time_decades"],
        sub["deseasoned"],
        cfg["quantiles"],
        n_boot=cfg["bootstrap_samples"],
        random_seed=cfg["random_seed"],
        method=cfg.get("bootstrap_method", "meboot"),
        block_length=cfg.get("bootstrap_block_length"),
    )
    boot_df["station_name"] = station_name

    plot_figure4_bootstrap(
        boot_df,
        fit_df,
        station_name,
        cfg["quantiles"],
        fig_dir / f"{station_name}_figure4.png",
        x_mode="absolute",
    )
    plot_figure4_bootstrap(
        boot_df,
        fit_df,
        station_name,
        cfg["quantiles"],
        fig_dir / f"{station_name}_figure4_centered.png",
        x_mode="centered",
        center_scale=cfg.get("bootstrap_centered_scale", 10000.0),
    )

    plot_station_timeseries(
        sub["date"],
        sub[target],
        sub["seasonal_fit"],
        sub["deseasoned"],
        station_name,
        fig_dir / f"{station_name}_timeseries.png",
    )
    plot_quantile_grid(grid_df, station_name, fig_dir / f"{station_name}_quantile_curve.png")
    for q in cfg["quantiles"]:
        plot_bootstrap_hist(boot_df, station_name, q, fig_dir / f"{station_name}_bootstrap_q{q}.png")

    return sub, fit_df, grid_df, boot_df, break_rows, adjust_rows, diag_df


def main():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

    cfg["quantile_grid"] = np.arange(0.1, 0.91, 0.02)
    out = Path("outputs")
    fig_dir = out / "figures"
    table_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    station_info_path = Path(cfg.get("station_info_path", "data/stationsInfo.csv"))
    iran_geojson_path = Path(cfg.get("iran_geojson_path", "data/Iran.geojson"))
    if station_info_path.exists() and iran_geojson_path.exists():
        stations_df = pd.read_csv(station_info_path)
        plot_figure1_station_map(
            stations_df,
            iran_geojson_path,
            fig_dir / "figure1_station_map.png",
        )
    else:
        warnings.warn(
            f"Figure 1 skipped: required files not found ({station_info_path}, {iran_geojson_path})."
        )

    df = load_data(cfg["data_path"], cfg["date_cols"])
    df = fill_tmean(df)

    precheck_cfg = cfg.get("precheck", {})
    precheck_df = run_input_precheck(
        df,
        station_col=cfg.get("station_col", "station_name"),
        target_col=cfg.get("target_variable", "tmean"),
        date_col="date",
    )
    precheck_df.to_csv(table_dir / "precheck_input_report.csv", index=False)
    if bool(precheck_cfg.get("stop_on_fail", True)) and str(precheck_df["status"].iloc[0]).upper() == "FAIL":
        raise ValueError(
            "Input precheck failed. See outputs/tables/precheck_input_report.csv for details."
        )

    summarize_preprocess_health(
        df,
        station_col=cfg.get("station_col", "station_name"),
        target_col=cfg.get("target_variable", "tmean"),
    ).to_csv(table_dir / "preprocess_health_before_station_processing.csv", index=False)

    df = filter_station_coverage(
        df,
        station_col=cfg["station_col"],
        date_col="date",
        target_col=cfg["target_variable"],
        min_start_date=cfg.get("min_start_date"),
        max_end_date=cfg.get("max_end_date"),
        max_missing_ratio=cfg.get("max_missing_ratio"),
    )

    station_col = cfg["station_col"]

    fit_rows = []
    grid_rows = []
    boot_rows = []
    cleaned_rows = []
    break_rows_all = []
    adjust_rows_all = []
    diagnostics_rows = []

    for station_name, sub in df.groupby(station_col):
        sub, fit_df, grid_df, boot_df, break_rows, adjust_rows, diag_df = process_single_station(station_name, sub, cfg, fig_dir)
        cleaned_rows.append(sub)
        fit_rows.append(fit_df)
        grid_rows.append(grid_df)
        boot_rows.append(boot_df)
        break_rows_all.extend(break_rows)
        adjust_rows_all.extend(adjust_rows)
        diagnostics_rows.append(diag_df)

    cleaned_df = pd.concat(cleaned_rows, ignore_index=True)
    fit_df = pd.concat(fit_rows, ignore_index=True)
    grid_df = pd.concat(grid_rows, ignore_index=True)
    boot_df = pd.concat(boot_rows, ignore_index=True)
    diagnostics_df = pd.concat(diagnostics_rows, ignore_index=True)

    boot_summary = summarize_bootstrap(boot_df)
    final_summary = merge_fit_and_bootstrap(fit_df, boot_summary)

    cleaned_df.to_csv(table_dir / "cleaned_station_data.csv", index=False)
    summarize_preprocess_health(
        cleaned_df,
        station_col=cfg.get("station_col", "station_name"),
        target_col=cfg.get("target_variable", "tmean"),
    ).to_csv(table_dir / "preprocess_health_after_station_processing.csv", index=False)
    fit_df.to_csv(table_dir / "slope_summary.csv", index=False)
    grid_df.to_csv(table_dir / "quantile_grid_summary.csv", index=False)
    boot_df.to_csv(table_dir / "bootstrap_slopes.csv", index=False)
    boot_summary.to_csv(table_dir / "bootstrap_summary.csv", index=False)
    final_summary.to_csv(table_dir / "final_summary_with_ci.csv", index=False)

    diagnostics_df.to_csv(table_dir / "preanalysis_station_tests.csv", index=False)
    summarize_preanalysis(diagnostics_df).to_csv(table_dir / "preanalysis_summary.csv", index=False)
    build_preanalysis_publication_table(diagnostics_df).to_csv(table_dir / "preanalysis_publication_table.csv", index=False)
    plot_preanalysis_heatmap(diagnostics_df, fig_dir / "preanalysis_heatmap.png")

    if break_rows_all:
        pd.DataFrame(break_rows_all).to_csv(table_dir / "homogenization_breakpoints.csv", index=False)
    if adjust_rows_all:
        pd.DataFrame(adjust_rows_all).to_csv(table_dir / "homogenization_adjustments.csv", index=False)

    # Dendrogram per quantile
    for q in cfg["quantiles"]:
        labels, D = distance_matrix_from_bootstrap(boot_df, q)
        D.to_csv(table_dir / f"cluster_distance_matrix_tau_{q}.csv")
        Z = linkage_from_distance_matrix(D)
        plot_dendrogram(
            Z,
            labels,
            q,
            fig_dir / f"dendrogram_tau_{q}.png",
        )

    print("Pipeline complete. Outputs written to outputs/")


if __name__ == "__main__":
    main()
