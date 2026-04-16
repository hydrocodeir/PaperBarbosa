import pandas as pd

def summarize_bootstrap(boot_df):
    g = boot_df.groupby(["station_name", "quantile"])["slope_per_decade"]
    out = g.agg(["mean", "std", "median",
                 lambda x: x.quantile(0.025),
                 lambda x: x.quantile(0.975)]).reset_index()
    out.columns = ["station_name", "quantile", "boot_mean", "boot_std", "boot_median", "ci_2_5", "ci_97_5"]
    return out

def merge_fit_and_bootstrap(fit_df, boot_summary_df):
    return fit_df.merge(boot_summary_df, on=["station_name", "quantile"], how="left")
