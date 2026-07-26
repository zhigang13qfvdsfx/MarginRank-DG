import numpy as np

from transsurv.baseline import breslow_cumulative_hazard


def test_breslow_hazard_is_monotone():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 1, 1], dtype=bool)
    risk = np.zeros(4)
    values = breslow_cumulative_hazard(time, event, risk, [1, 2, 3, 4])
    assert np.all(np.diff(values) >= 0)
    assert values[-1] > values[0] > 0
