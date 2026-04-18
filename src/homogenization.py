import numpy as np
import pandas as pd


def _snht_scores(values, min_segment=365):
    """
    Standard Normal Homogeneity Test (single-break scan).
    Returns SNHT score for each candidate split index k.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    scores = np.full(n, np.nan, dtype=float)
    if n < 2 * min_segment + 1:
        return scores

    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return scores

    z = (x - mu) / sd
    csum = np.cumsum(z)

    for k in range(min_segment, n - min_segment):
        left_n = k
        right_n = n - k
        mean_left = csum[k - 1] / left_n
        mean_right = (csum[-1] - csum[k - 1]) / right_n
        scores[k] = left_n * (mean_left ** 2) + right_n * (mean_right ** 2)

    return scores


def _detect_breaks_recursive(values, start_idx, min_segment, threshold, max_breaks, out_breaks):
    if len(out_breaks) >= max_breaks:
        return

    scores = _snht_scores(values, min_segment=min_segment)
    if np.all(~np.isfinite(scores)):
        return

    local_k = int(np.nanargmax(scores))
    max_score = float(scores[local_k])
    if not np.isfinite(max_score) or max_score < threshold:
        return

    absolute_k = start_idx + local_k
    out_breaks.append((absolute_k, max_score))

    left = values[:local_k]
    right = values[local_k:]

    if left.size >= 2 * min_segment + 1:
        _detect_breaks_recursive(
            left,
            start_idx=start_idx,
            min_segment=min_segment,
            threshold=threshold,
            max_breaks=max_breaks,
            out_breaks=out_breaks,
        )

    if right.size >= 2 * min_segment + 1 and len(out_breaks) < max_breaks:
        _detect_breaks_recursive(
            right,
            start_idx=absolute_k,
            min_segment=min_segment,
            threshold=threshold,
            max_breaks=max_breaks,
            out_breaks=out_breaks,
        )


def detect_breakpoints_snht(values, min_segment=365, threshold=120.0, max_breaks=5):
    """
    RHtests-like practical approximation:
    - SNHT scan over candidate breakpoints
    - Recursive segmentation when score exceeds threshold

    Parameters
    ----------
    values : array-like
        Temperature (or anomaly) series.
    min_segment : int
        Minimum number of daily observations in each segment.
    threshold : float
        Breakpoint acceptance threshold for SNHT statistic.
    max_breaks : int
        Maximum number of breakpoints to report.
    """
    x = np.asarray(values, dtype=float)
    breaks = []
    _detect_breaks_recursive(
        x,
        start_idx=0,
        min_segment=int(min_segment),
        threshold=float(threshold),
        max_breaks=int(max_breaks),
        out_breaks=breaks,
    )
    breaks = sorted(breaks, key=lambda t: t[0])
    return breaks


def mean_shift_adjustment(values, break_indices):
    """
    Adjust earlier segments to match the mean of the final segment.
    This is a common climate-homogenization convention for relative consistency.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    bk = sorted(int(b) for b in break_indices if 0 < int(b) < n)

    edges = [0] + bk + [n]
    seg_means = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        seg_means.append(float(np.nanmean(x[a:b])))

    ref_mean = seg_means[-1]
    adjustments = [ref_mean - m for m in seg_means]

    y = x.copy()
    detail_rows = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        y[a:b] = x[a:b] + adjustments[i]
        detail_rows.append(
            {
                "segment_id": i + 1,
                "segment_start_index": a,
                "segment_end_index": b - 1,
                "segment_mean_raw": seg_means[i],
                "applied_adjustment": adjustments[i],
                "segment_mean_adjusted": seg_means[i] + adjustments[i],
            }
        )

    return y, pd.DataFrame(detail_rows)
