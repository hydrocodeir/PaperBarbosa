import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.preprocessing import load_data, fill_tmean, prepare_station_series, run_input_precheck
from src.diagnostics import run_preanalysis_tests


def test_load_data_semicolon_and_aliases(tmp_path):
    df = pd.DataFrame(
        {
            "station_id": [40769, 40769, 40769],
            "station_name": ["Arak", "Arak", "Arak"],
            "year": [2022, 2022, 2022],
            "month": [1, 1, 1],
            "day": [1, 2, 3],
            "tmin": [-1.2, 1.6, 0.0],
            "tmean": [4.4, 3.7, 3.2],
            "tmax": [10.0, 5.8, 6.4],
        }
    )
    p = tmp_path / "sample_semicolon.csv"
    df.to_csv(p, sep=";", index=False)

    loaded = load_data(str(p), ["year", "month", "day"])
    loaded = fill_tmean(loaded)
    sub = prepare_station_series(loaded[loaded["station_name"] == "Arak"])

    assert "date" in loaded.columns
    assert sub["tmean"].isna().sum() == 0


def test_precheck_detects_single_column_failure():
    raw = pd.DataFrame({"everything": ["a,b,c", "1,2,3"]})
    rep = run_input_precheck(raw)
    assert rep["status"].iloc[0] == "FAIL"
    assert "BAD_DELIMITER_OR_SINGLE_COLUMN" in rep["issues"].iloc[0]


def test_coerce_persian_digits_and_unicode_minus(tmp_path):
    df = pd.DataFrame(
        {
            "station_name": ["Arak", "Arak", "Arak"],
            "year": [2022, 2022, 2022],
            "month": [1, 1, 1],
            "day": [1, 2, 3],
            "tmean": ["۴٫۴", "−۱٫۵", "3,2"],
        }
    )
    p = tmp_path / "unicode_digits.csv"
    df.to_csv(p, index=False)

    loaded = load_data(str(p), ["year", "month", "day"])
    vals = loaded["tmean"].to_numpy()
    assert vals[0] == 4.4
    assert vals[1] == -1.5
    assert vals[2] == 3.2


def test_run_preanalysis_tests_series_alignment_bugfix():
    # values passed as Series (index 0..n-1) should not be reindexed to NaN against datetime index
    date = pd.date_range("2022-01-01", periods=5, freq="D")
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = run_preanalysis_tests(date, values, "S1")
    assert int(out["n_valid"].iloc[0]) == 5
    assert float(out["missing_ratio"].iloc[0]) == 0.0
