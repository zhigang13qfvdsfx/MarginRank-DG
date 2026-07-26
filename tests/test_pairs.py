import numpy as np
import pandas as pd

from transsurv.pairs import make_pair_features


def test_pair_features_are_sample_local_and_monotone_invariant():
    x = pd.DataFrame(
        [[1.0, 3.0, 2.0], [5.0, 4.0, 8.0]], columns=["a", "b", "c"]
    )
    binary = make_pair_features(x, x.columns, mode="binary")
    transformed = make_pair_features(np.exp(x), x.columns, mode="binary")
    assert binary.equals(transformed)
    assert binary.shape == (2, 3)
