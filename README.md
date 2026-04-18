# Climate paper replication

This project adapts the workflow from:
"Summarising changes in air temperature over Central Europe by quantile regression and clustering"
to a station dataset with daily observations.

## Expected input
Place your CSV at:

`data/data.csv`

For Figure 1 (station map), also provide:
- `data/stationsInfo.csv` (station metadata with `latitude` and `longitude`)
- `data/Iran.geojson` (Iran boundary geometry)

Expected columns:
- station_id
- station_name
- year
- month
- day
- tmin
- tmean
- tmax

### Input robustness notes
- Column names are now matched case-insensitively for `year/month/day`, station id/name, and temperature fields.
- Common aliases are auto-mapped (e.g., `Temperature`, `TAVG`, `tas` -> `tmean`; `TX` -> `tmax`; `TN` -> `tmin`).
- Numeric temperature strings are aggressively normalized (comma/Arabic decimal, Unicode minus, Persian/Arabic digits).
- Delimiter is auto-detected for CSV files (comma/semicolon/tab), and Excel files (`.xlsx/.xls`) are supported directly.
- If `target_variable` is not found after normalization, pipeline now raises a clear error instead of silently producing all-NaN station series.

## Run
```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Outputs
The pipeline writes outputs into `outputs/`:
- cleaned_station_data.csv
- slope_summary.csv
- quantile_grid_summary.csv
- bootstrap_summary.csv
- final_summary_with_ci.csv
- precheck_input_report.csv *(input schema/health gate before station processing)*
- preprocess_health_before_station_processing.csv *(station-wise valid target count right after load)*
- preprocess_health_after_station_processing.csv *(station-wise valid target count after station preprocessing)*
- preanalysis_station_tests.csv *(Q1 pre-analysis checks)*
- preanalysis_summary.csv *(Q1 pre-analysis aggregated stats)*
- preanalysis_publication_table.csv *(manuscript-ready diagnostic table)*
- q1_station_rankings.csv *(Q1 discussion-priority table based on trend+CI+QC+breaks)*
- q1_taylor_metrics.csv *(signal-vs-uncertainty metrics for Q1 panel)*
- homogenization_breakpoints.csv *(when homogenization is enabled)*
- homogenization_adjustments.csv *(when homogenization is enabled)*
- cluster_distance_matrix_tau_*.csv
- figures/*.png

## RHtests-like homogenization (SNHT)
A practical RHtests-style step is included and can be toggled in `config.yaml`.

```yaml
homogenization:
  enabled: true
  method: snht
  min_segment: 365
  snht_threshold: 120.0
  max_breaks: 5
```

What it does per station:
1. Detects candidate changepoints with recursive SNHT scans.
2. Applies mean-shift adjustment so all earlier segments align to the most recent segment mean.
3. Saves publication-ready outputs:
   - `outputs/tables/homogenization_breakpoints.csv`
   - `outputs/tables/homogenization_adjustments.csv`
   - `outputs/figures/<station>_homogenization_breaks.png`

These can be inserted directly into your paper as a table/figure set similar to RHtests reporting.

## Paper-aligned defaults
- Quantile grid follows the article setup: 0.10 to 0.90 with step 0.02.
- Figure styles are configured for publication-readability (higher DPI, consistent typography).
- Optional station coverage filters can be set in `config.yaml`:
  - `min_start_date`
  - `max_end_date`
  - `max_missing_ratio`


## Q1 pre-analysis test suite (before trend/quantile modeling)
The pipeline now runs a mandatory pre-analysis diagnostic stage for each station before deseasonalization/modeling:

- Completeness (`missing_ratio`)
- Outlier rate (`outlier_ratio`)
- Mann-Kendall trend significance (`mk_tau`, `mk_pvalue`)
- ADF stationarity test (`adf_pvalue`)
- Ljung-Box autocorrelation test (`ljungbox_pvalue`)
- D’Agostino normality test (`normaltest_pvalue`)
- Diagnostic reason tags (`reason_code`) مثل `ALL_NAN_AFTER_PREPROCESS`, `HIGH_MISSING_RATIO`, ...

Outputs for paper reporting:
- `outputs/tables/preanalysis_station_tests.csv`
- `outputs/tables/preanalysis_summary.csv` *(includes `reason_freq::...` rows for station diagnostic codes)*
- `outputs/tables/preanalysis_publication_table.csv` *(final paper-style QC table with recommendations)*
- `outputs/figures/preanalysis_heatmap.png`
- `outputs/figures/<station>_preanalysis_panel.png`

You can tune thresholds in `config.yaml`:

```yaml
diagnostics:
  max_missing_ratio: 0.05
  outlier_sigma: 4.0
  ljungbox_lag: 30
```


## Input precheck gate
Before per-station preprocessing, the pipeline writes:
- `outputs/tables/precheck_input_report.csv`

This report includes `status` (`OK`/`WARN`/`FAIL`) and issue codes such as:
- `INVALID_DATES_PRESENT`
- `MISSING_STATION_IDS`
- `TARGET_ALMOST_ALL_MISSING`
- `BAD_DELIMITER_OR_SINGLE_COLUMN`

Behavior is configurable:
```yaml
precheck:
  stop_on_fail: true
```
If `stop_on_fail=true` and status is `FAIL`, the pipeline stops early with a clear error message.


## Q1 manuscript figure/table add-on
A publication-oriented ranking artifact is now generated:
- Table: `outputs/tables/q1_station_rankings.csv`
- Figure: `outputs/figures/q1_forest_top_stations.png`
- Figure: `outputs/figures/q1_trend_vs_breaks.png`
- Figure: `outputs/figures/q1_taylor_like_panel.png`

Design intent:
- Focuses on median trend (`quantile=0.5`)
- Shows slope and 95% bootstrap CI
- Includes QC state (`quality_pass`, `reason_code`)
- Includes homogenization complexity (`n_breaks`)
- Adds `q1_priority_score` to prioritize stations for Results/Discussion sections
