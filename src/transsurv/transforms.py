from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SingleSampleRankTransformer:
    """Map expression to within-sample percentiles over a fixed gene universe.

    The transformation of a sample does not use outcomes or other samples. If
    some universe genes are unavailable, ranks are computed over the available
    subset and missing requested features are returned as NaN.
    """

    universe_: tuple[str, ...] | None = None

    def fit(self, gene_universe: Iterable[str]) -> "SingleSampleRankTransformer":
        universe = tuple(dict.fromkeys(map(str, gene_universe)))
        if len(universe) < 2:
            raise ValueError("At least two genes are required for ranking")
        self.universe_ = universe
        return self

    def transform(
        self, expression: pd.DataFrame, features: Iterable[str] | None = None
    ) -> pd.DataFrame:
        if self.universe_ is None:
            raise RuntimeError("Transformer has not been fitted")
        available = [g for g in self.universe_ if g in expression.columns]
        if len(available) < 2:
            raise ValueError("Fewer than two reference genes are available")
        numeric = expression.loc[:, available].apply(pd.to_numeric, errors="coerce")
        ranked = numeric.rank(axis=1, method="average", pct=True, na_option="keep")
        requested = list(self.universe_ if features is None else map(str, features))
        return ranked.reindex(columns=requested).astype(np.float32)

    def fit_transform(
        self,
        expression: pd.DataFrame,
        gene_universe: Iterable[str],
        features: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit(gene_universe).transform(expression, features=features)


@dataclass
class CohortZTransformer:
    """Outcome-free, cohort-wise gene standardisation baseline."""

    def transform(
        self, expression: pd.DataFrame, features: Iterable[str]
    ) -> pd.DataFrame:
        features = list(map(str, features))
        data = expression.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
        means = data.mean(axis=0)
        scales = data.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        return ((data - means) / scales).replace([np.inf, -np.inf], np.nan)

