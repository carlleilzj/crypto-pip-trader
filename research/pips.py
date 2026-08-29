"""Perceptually Important Points (PIP) extraction.

PIP is a dimensionality-reduction technique that identifies the most
"visually important" points in a time series by iteratively selecting
the point with the greatest perpendicular distance from the line
connecting its two neighbouring PIPs.

Used in this project to capture the shape of a price window, which is
then compared against the recorded shape at entry time to detect when
the price trajectory has deviated enough to warrant an early exit.
"""

import numpy as np


def find_pips(
    data: np.ndarray,
    n_pips: int,
    dist_measure: int = 3,
) -> tuple[list[int], list[float]]:
    """Extract Perceptually Important Points from a 1-D time series.

    The algorithm starts with the first and last points as the initial
    PIPs, then iteratively picks the point with the largest distance
    from the line segment connecting its two neighbouring PIPs, until
    ``n_pips`` points are selected.

    Parameters
    ----------
    data : np.ndarray
        1-D array of price values.
    n_pips : int
        Desired number of PIPs (including endpoints).  Must be >= 2.
    dist_measure : int, default 3
        Distance metric:
        - 1: Euclidean distance to both neighbour endpoints (summed).
        - 2: Perpendicular distance (divided by sqrt(slope² + 1)).
        - 3: Vertical distance only (|y - y_hat|).  Fastest, and the
             default because price is the dimension of interest.

    Returns
    -------
    pips_x : list[int]
        Indices of the selected PIPs, sorted.
    pips_y : list[float]
        Values of the selected PIPs at those indices.
    """
    pips_x = [0, len(data) - 1]
    pips_y = [float(data[0]), float(data[-1])]

    for _ in range(2, n_pips):
        md = 0.0
        md_i = -1
        insert_index = -1

        for k in range(0, len(pips_x) - 1):
            left_adj = k
            right_adj = k + 1
            time_diff = pips_x[right_adj] - pips_x[left_adj]
            if time_diff == 0:
                continue
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                # Distance metric selection
                if dist_measure == 1:
                    # Euclidean: sum of distances to both endpoints
                    d = (
                        (pips_x[left_adj] - i) ** 2
                        + (pips_y[left_adj] - data[i]) ** 2
                    ) ** 0.5
                    d += (
                        (pips_x[right_adj] - i) ** 2
                        + (pips_y[right_adj] - data[i]) ** 2
                    ) ** 0.5
                elif dist_measure == 2:
                    # Perpendicular distance (signed) — normalised by slope
                    d = abs((slope * i + intercept) - data[i]) / (
                        slope ** 2 + 1
                    ) ** 0.5
                else:
                    # Vertical distance — fastest, equivalent to absolute
                    # residual from the line segment
                    d = abs((slope * i + intercept) - data[i])

                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj

        if md_i < 0:
            break
        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, float(data[md_i]))

    return pips_x, pips_y
