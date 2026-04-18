import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.reporting import build_q1_station_table, build_q1_taylor_metrics


def test_build_q1_station_table_basic_merge_and_sort():
    final_summary = pd.DataFrame(
        {
            "station_name": ["A", "B", "A", "B"],
            "quantile": [0.5, 0.5, 0.95, 0.95],
            "slope_per_decade": [0.3, -0.1, 0.4, -0.2],
            "ci_2_5": [0.1, -0.3, 0.2, -0.4],
            "ci_97_5": [0.5, 0.1, 0.6, 0.0],
        }
    )
    diag = pd.DataFrame(
        {
            "station_name": ["A", "B"],
            "quality_pass": [1, 0],
            "reason_code": ["OK", "HIGH_MISSING_RATIO"],
        }
    )
    breaks = pd.DataFrame({"station_name": ["A", "A", "B"]})

    out = build_q1_station_table(final_summary, diag, breakpoints_df=breaks, focus_quantile=0.5)

    assert list(out.columns) == [
        "station_name",
        "quantile",
        "slope_per_decade",
        "ci_2_5",
        "ci_97_5",
        "trend_significant",
        "quality_pass",
        "reason_code",
        "n_breaks",
        "q1_priority_score",
    ]
    assert set(out["station_name"].tolist()) == {"A", "B"}
    assert int(out.loc[out["station_name"] == "A", "n_breaks"].iloc[0]) == 2
    assert int(out.loc[out["station_name"] == "B", "n_breaks"].iloc[0]) == 1


def test_build_q1_taylor_metrics_extracts_focus_quantile():
    final_summary = pd.DataFrame(
        {
            "station_name": ["A", "A", "B"],
            "quantile": [0.5, 0.95, 0.5],
            "slope_per_decade": [0.3, 0.4, -0.1],
            "boot_std": [0.07, 0.08, 0.05],
            "ci_2_5": [0.1, 0.2, -0.3],
            "ci_97_5": [0.5, 0.6, 0.1],
        }
    )
    diag = pd.DataFrame({"station_name": ["A", "B"], "quality_pass": [1, 0], "reason_code": ["OK", "X"]})

    out = build_q1_taylor_metrics(final_summary, diag, focus_quantile=0.5)
    assert set(out["station_name"]) == {"A", "B"}
    assert "ci_width" in out.columns
    assert float(out.loc[out["station_name"] == "A", "ci_width"].iloc[0]) == 0.4
