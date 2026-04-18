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
