from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM

from .metrics import make_y
from .model import DomainData


@dataclass
class PooledSurvivalLearner:
    kind: str
    parameters: dict[str, object] = field(default_factory=dict)

    def fit(self, domains: dict[str, DomainData]) -> "PooledSurvivalLearner":
        first = next(iter(domains.values()))
        self.features_ = list(first.x.columns.astype(str))
        x = pd.concat([d.x[self.features_] for d in domains.values()], axis=0)
        time = np.concatenate([d.time for d in domains.values()])
        event = np.concatenate([d.event for d in domains.values()])
        self.center_ = x.mean(axis=0).fillna(0.0)
        self.scale_ = x.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        z = ((x - self.center_) / self.scale_).fillna(0.0).to_numpy(dtype=float)
        y = make_y(time, event)
        p = dict(self.parameters)
        if self.kind == "coxph":
            estimator = CoxPHSurvivalAnalysis(**p)
        elif self.kind == "coxnet":
            estimator = CoxnetSurvivalAnalysis(**p)
        elif self.kind == "svm":
            estimator = FastSurvivalSVM(**p)
        elif self.kind == "gradient_boosting":
            estimator = GradientBoostingSurvivalAnalysis(**p)
        elif self.kind == "random_forest":
            estimator = RandomSurvivalForest(**p)
        else:
            raise ValueError(f"Unknown learner kind: {self.kind}")
        estimator.fit(z, y)
        self.estimator_ = estimator
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        frame = x.reindex(columns=self.features_)
        return (
            ((frame - self.center_) / self.scale_)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator_.predict(self.transform(x)), dtype=float)

