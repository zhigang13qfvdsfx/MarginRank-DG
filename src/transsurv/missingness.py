from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def available_pair_features(
    columns: Iterable[str], missing_genes: Iterable[str]
) -> list[str]:
    missing = set(map(str, missing_genes))
    output = []
    for column in map(str, columns):
        genes = set(column.split("|"))
        if genes.isdisjoint(missing):
            output.append(column)
    return output


@dataclass
class RiskDistiller:
    alpha: float = 1.0

    def fit(self, x: pd.DataFrame, teacher_risk: np.ndarray) -> "RiskDistiller":
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.features_ = list(x.columns.astype(str))
        frame = x[self.features_].astype(float)
        self.center_ = frame.mean(axis=0).fillna(0.0)
        self.scale_ = frame.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        z = ((frame - self.center_) / self.scale_).fillna(0.0).to_numpy(dtype=float)
        target = np.asarray(teacher_risk, dtype=float)
        self.intercept_ = float(target.mean())
        centered_target = target - self.intercept_
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.coef_ = np.linalg.solve(gram, z.T @ centered_target)
        fitted = self.intercept_ + z @ self.coef_
        self.training_rmse_ = float(np.sqrt(np.mean((fitted - target) ** 2)))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        frame = x.reindex(columns=self.features_).astype(float)
        z = (
            ((frame - self.center_) / self.scale_)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return self.intercept_ + z @ self.coef_

