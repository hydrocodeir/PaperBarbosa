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


def _residual_bootstrap_single(x_decades, y, rng):
    """
    Residual bootstrap around an OLS trend:
    y_boot = y_hat + e*, where e* are centered residuals sampled with replacement.
    """
    x = np.asarray(x_decades, dtype=float)
    y = np.asarray(y, dtype=float)
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    y_hat = ols.predict(X)
    resid = ols.resid - np.mean(ols.resid)
    sampled_resid = rng.choice(resid, size=len(resid), replace=True)
    return y_hat + sampled_resid


def _moving_block_bootstrap_single(series, rng, block_length=None):
    """
    Moving block bootstrap for time series that preserves short-range dependence.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 3:
        return x.copy()

    if block_length is None:
        block_length = max(5, int(np.sqrt(n)))
    block_length = int(np.clip(block_length, 2, n))

    starts = np.arange(0, n - block_length + 1)
    n_blocks = int(np.ceil(n / block_length))
    pieces = []
    for _ in range(n_blocks):
        s = rng.choice(starts)
        pieces.append(x[s : s + block_length])
    out = np.concatenate(pieces)[:n]
    return out


def maximum_entropy_bootstrap_slopes(
    x_decades,
    y,
    quantiles,
    n_boot=200,
    random_seed=42,
    method="meboot",
    block_length=None,
):
    rng = np.random.default_rng(random_seed)
    x = np.asarray(x_decades, dtype=float)
    y = np.asarray(y, dtype=float)
    method = str(method).lower()
    valid_methods = {"meboot", "residual", "moving_block"}
    if method not in valid_methods:
        raise ValueError(f"Unknown bootstrap method '{method}'. Valid options: {sorted(valid_methods)}")

    rows = []
    for b in range(n_boot):
        if method == "meboot":
            y_boot = _meboot_single(y, rng)
        elif method == "residual":
            y_boot = _residual_bootstrap_single(x, y, rng)
        else:  # moving_block
            y_boot = _moving_block_bootstrap_single(y, rng, block_length=block_length)

        for q in quantiles:
            fit = quantile_fit(x, y_boot, q)
            rows.append(
                {
                    "bootstrap_id": b,
                    "quantile": float(q),
                    "slope_per_decade": fit["slope_per_decade"],
                    "bootstrap_method": method,
                }
            )
    return pd.DataFrame(rows)
