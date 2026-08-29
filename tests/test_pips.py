import numpy as np

from research.pips import find_pips


def test_find_pips_endpoints():
    x = np.array([1.0, 2.0, 1.5, 3.0, 2.2, 2.8, 1.1], dtype=float)
    px, py = find_pips(x, 5, 3)
    assert px[0] == 0
    assert px[-1] == len(x) - 1
    assert len(px) == 5
    assert len(py) == 5
