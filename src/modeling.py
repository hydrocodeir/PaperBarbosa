import numpy as np
import pandas as pd
import statsmodels.api as sm


def quantile_fit(x_decades, y, q):
    X = sm.add_constant(np.asarray(x_decades, dtype=float))
    model = sm.QuantReg(np.asarray(y, dtype=float), X)
    res = model.fit(q=q)
    intercept, slope = res.params
    bse = getattr(res, "bse", [np.nan, np.nan])
    pvalues = getattr(res, "pvalues", [np.nan, np.nan])

    return {
        "quantile": float(q),
        "intercept": float(intercept),
        "slope_per_decade": float(slope),
        "std_error": float(bse[1]) if len(bse) > 1 else np.nan,
        "pvalue": float(pvalues[1]) if len(pvalues) > 1 else np.nan,
        "result": res,
    }


def fit_quantiles(x_decades, y, quantiles):
    rows = []
    for q in quantiles:
        fit = quantile_fit(x_decades, y, q)
        fit.pop("result", None)
        rows.append(fit)
    return pd.DataFrame(rows)


def _meboot_single(series, rng):
    """
    Practical maximum-entropy-style bootstrap for a univariate time series.
    This is a rank-preserving, smooth bootstrap inspired by Vinod & de Lacalle.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)

    if n < 3:
        return x.copy()

    # sort values and keep original rank positions
    order = np.argsort(x)
    xs = x[order]

    # midpoints between adjacent sorted observations
    mids = 0.5 * (xs[:-1] + xs[1:])

    # tail extensions
    left_width = mids[0] - xs[0]
    right_width = xs[-1] - mids[-1]
    z = np.empty(n + 1, dtype=float)
    z[0] = xs[0] - left_width
    z[1:-1] = mids
    z[-1] = xs[-1] + right_width

    # sample uniformly from each entropy interval
    u = rng.uniform(size=n)
    ys = z[:-1] + u * (z[1:] - z[:-1])

    # restore original temporal rank structure
    y = np.empty(n, dtype=float)
    y[order] = np.sort(ys)

    # preserve mean approximately
    y = y - y.mean() + x.mean()
    return y


def maximum_entropy_bootstrap_slopes(
    x_decades,
    y,
    quantiles,
    n_boot=200,
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)
    x = np.asarray(x_decades, dtype=float)
    y = np.asarray(y, dtype=float)

    rows = []
    for b in range(n_boot):
        y_boot = _meboot_single(y, rng)
        for q in quantiles:
            fit = quantile_fit(x, y_boot, q)
            rows.append(
                {
                    "bootstrap_id": b,
                    "quantile": float(q),
                    "slope_per_decade": fit["slope_per_decade"],
                }
            )
    return pd.DataFrame(rows)