from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


def make_pair_features(
    expression: pd.DataFrame,
    genes: Iterable[str],
    mode: str = "difference",
) -> pd.DataFrame:
    """Create within-sample relative-expression features.

    ``difference`` returns rank/expression differences. ``binary`` returns
    relative expression ordering indicators. Both are independent of other
    samples once the input has been transformed sample-wise.
    """
    genes = [str(g) for g in genes if str(g) in expression.columns]
    pairs = list(combinations(genes, 2))
    if not pairs:
        raise ValueError("At least two available genes are required")
    left = expression[[a for a, _ in pairs]].to_numpy(dtype=float)
    right = expression[[b for _, b in pairs]].to_numpy(dtype=float)
    if mode == "difference":
        values = left - right
    elif mode == "binary":
        values = (left > right).astype(np.float32)
    else:
        raise ValueError("mode must be 'difference' or 'binary'")
    columns = [f"{a}|{b}" for a, b in pairs]
    return pd.DataFrame(values, index=expression.index, columns=columns)

