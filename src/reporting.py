import numpy as np
import pandas as pd


def build_q1_station_table(
    final_summary_df,
    diagnostics_df,
    breakpoints_df=None,
    focus_quantile=0.5,
):
    """
    Build a manuscript-grade station table focused on the median (tau=0.5) trend.
    """
    fs = final_summary_df.copy()
    fs = fs[np.isclose(fs["quantile"].astype(float), float(focus_quantile))].copy()

    if fs.empty:
        return pd.DataFrame(
            columns=[
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
        )

    fs["trend_significant"] = ((fs["ci_2_5"] > 0) | (fs["ci_97_5"] < 0)).astype(int)

    diag_keep = [c for c in ["station_name", "quality_pass", "reason_code"] if c in diagnostics_df.columns]
    out = fs.merge(diagnostics_df[diag_keep], on="station_name", how="left")

    if breakpoints_df is not None and not breakpoints_df.empty and "station_name" in breakpoints_df.columns:
        bcount = breakpoints_df.groupby("station_name").size().rename("n_breaks").reset_index()
        out = out.merge(bcount, on="station_name", how="left")
    else:
        out["n_breaks"] = np.nan

    out["n_breaks"] = out["n_breaks"].fillna(0).astype(int)

    # Higher is more concerning/interesting for discussion in manuscript
    abs_slope = out["slope_per_decade"].abs().fillna(0.0)
    sig_bonus = out["trend_significant"].fillna(0).astype(int)
    qual_penalty = 1 - out["quality_pass"].fillna(0).astype(int)
    out["q1_priority_score"] = abs_slope + 0.2 * sig_bonus + 0.15 * out["n_breaks"] + 0.1 * qual_penalty

    cols = [
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
    out = out[cols].sort_values(["q1_priority_score", "station_name"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_q1_taylor_metrics(final_summary_df, diagnostics_df, focus_quantile=0.5):
    """
    Build compact metrics for a Taylor-like summary plot at the focus quantile.
    """
    fs = final_summary_df.copy()
    fs = fs[np.isclose(fs["quantile"].astype(float), float(focus_quantile))].copy()
    if fs.empty:
        return pd.DataFrame(columns=["station_name", "slope_per_decade", "boot_std", "ci_width", "quality_pass", "reason_code"])

    fs["ci_width"] = pd.to_numeric(fs["ci_97_5"], errors="coerce") - pd.to_numeric(fs["ci_2_5"], errors="coerce")
    keep = [c for c in ["station_name", "slope_per_decade", "boot_std", "ci_width"] if c in fs.columns]
    out = fs[keep].copy()

    diag_keep = [c for c in ["station_name", "quality_pass", "reason_code"] if c in diagnostics_df.columns]
    out = out.merge(diagnostics_df[diag_keep], on="station_name", how="left")
    return out
