# Climate paper replication

This project adapts the workflow from:
"Summarising changes in air temperature over Central Europe by quantile regression and clustering"
to a station dataset with daily observations.

## Expected input
Place your CSV at:

`data/raw_data.csv`

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
- cluster_distance_matrix_tau_*.csv
- figures/*.png
