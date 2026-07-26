from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .model import DomainData


def _censoring_survival_left(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Kaplan-Meier censoring survival immediately before each observed time."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    unique = np.sort(np.unique(time))
    survival = 1.0
    before: dict[float, float] = {}
    for value in unique:
        before[float(value)] = survival
        at_risk = int(np.sum(time >= value))
        censored = int(np.sum((time == value) & (~event)))
        if at_risk > 0:
            survival *= 1.0 - censored / at_risk
    return np.asarray([before[float(value)] for value in time], dtype=float)


def _comparable_pairs(
    x: np.ndarray, time: np.ndarray, event: np.ndarray, min_censor_survival: float
) -> tuple[np.ndarray, np.ndarray]:
    g_left = np.maximum(_censoring_survival_left(time, event), min_censor_survival)
    differences = []
    weights = []
    for i in np.flatnonzero(event):
        later = np.flatnonzero(time > time[i])
        if len(later) == 0:
            continue
        differences.append(x[i][None, :] - x[later])
        weights.append(np.repeat(1.0 / (g_left[i] ** 2), len(later)))
    if not differences:
        raise ValueError("No comparable survival pairs")
    return np.vstack(differences), np.concatenate(weights)


def _pair_loss_gradient(
    beta: np.ndarray, differences: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    margin = differences @ beta
    normalizer = float(weights.sum())
    loss = float(np.sum(weights * np.logaddexp(0.0, -margin)) / normalizer)
    gradient = -(
        differences.T @ (weights * expit(-margin))
    ) / normalizer
    return loss, gradient


@dataclass
class IPCWPairwiseRanker:
    robustness: float = 0.0
    l1: float = 0.001
    l2: float = 0.05
    min_censor_survival: float = 0.05
    max_iter: int = 400

    def fit(self, domains: dict[str, DomainData]) -> "IPCWPairwiseRanker":
        first = next(iter(domains.values()))
        self.features_ = list(first.x.columns.astype(str))
        pooled = pd.concat([domain.x[self.features_] for domain in domains.values()])
        self.center_ = pooled.mean(axis=0).fillna(0.0)
        self.scale_ = pooled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        pairs = {}
        for name, domain in domains.items():
            z = (
                ((domain.x[self.features_] - self.center_) / self.scale_)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            pairs[name] = _comparable_pairs(
                z,
                np.asarray(domain.time, dtype=float),
                np.asarray(domain.event, dtype=bool),
                self.min_censor_survival,
            )

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            values = []
            gradients = []
            for differences, weights in pairs.values():
                value, gradient = _pair_loss_gradient(beta, differences, weights)
                values.append(value)
                gradients.append(gradient)
            losses = np.asarray(values)
            grads = np.vstack(gradients)
            if self.robustness == 0:
                domain_weights = np.repeat(1.0 / len(losses), len(losses))
                data_loss = float(losses.mean())
            else:
                logits = self.robustness * losses
                domain_weights = np.exp(logits - logsumexp(logits))
                data_loss = float(
                    (logsumexp(logits) - np.log(len(losses))) / self.robustness
                )
            data_gradient = domain_weights @ grads
            smooth = np.sqrt(beta * beta + 1e-10)
            penalty = self.l1 * smooth.sum() + 0.5 * self.l2 * np.dot(beta, beta)
            penalty_gradient = self.l1 * beta / smooth + self.l2 * beta
            return data_loss + float(penalty), data_gradient + penalty_gradient

        result = minimize(
            lambda beta: objective(beta)[0],
            np.zeros(len(self.features_)),
            jac=lambda beta: objective(beta)[1],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-9, "gtol": 1e-8},
        )
        self.coef_ = np.asarray(result.x, dtype=float)
        self.optimization_ = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "pairs_by_domain": {name: len(value[1]) for name, value in pairs.items()},
        }
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        frame = x.reindex(columns=self.features_)
        z = (
            ((frame - self.center_) / self.scale_)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return z @ self.coef_

