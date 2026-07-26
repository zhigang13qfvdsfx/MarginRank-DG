from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class DomainData:
    x: pd.DataFrame
    time: np.ndarray
    event: np.ndarray


def _cox_loss_gradient(
    beta: np.ndarray, x: np.ndarray, time: np.ndarray, event: np.ndarray
) -> tuple[float, np.ndarray]:
    """Event-normalised Breslow negative partial log-likelihood and gradient."""
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    n_events = int(event.sum())
    if n_events < 1:
        raise ValueError("A survival domain must contain at least one event")

    order = np.argsort(-time, kind="mergesort")
    x_sorted = x[order]
    time_sorted = time[order]
    event_sorted = event[order]
    eta = x_sorted @ beta
    offset = float(np.max(eta))
    weights = np.exp(np.clip(eta - offset, -60.0, 60.0))
    risk_sum = np.cumsum(weights)
    risk_x_sum = np.cumsum(weights[:, None] * x_sorted, axis=0)

    log_likelihood = 0.0
    score = np.zeros_like(beta, dtype=float)
    _, starts, counts = np.unique(time_sorted, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        stop = int(start + count)
        group_events = event_sorted[start:stop]
        d = int(group_events.sum())
        if d == 0:
            continue
        risk_position = stop - 1
        denominator = float(risk_sum[risk_position])
        risk_mean = risk_x_sum[risk_position] / denominator
        event_x = x_sorted[start:stop][group_events]
        event_eta = eta[start:stop][group_events]
        log_likelihood += float(event_eta.sum()) - d * (
            np.log(denominator) + offset
        )
        score += event_x.sum(axis=0) - d * risk_mean

    return -log_likelihood / n_events, -score / n_events


class MultiDomainCox:
    """Shared Cox predictor with a smooth worst-domain objective.

    ``robustness=0`` gives the equally weighted mean of event-normalised
    domain losses. Positive values use log-sum-exp and increasingly emphasize
    the worst-fitting study.
    """

    def __init__(
        self,
        robustness: float = 5.0,
        l1: float = 0.001,
        l2: float = 0.05,
        domain_l2: float | None = None,
        l1_smoothing: float = 1e-5,
        max_iter: int = 400,
        tolerance: float = 1e-8,
    ) -> None:
        if robustness < 0 or l1 < 0 or l2 < 0:
            raise ValueError("Penalties and robustness must be non-negative")
        if domain_l2 is not None and domain_l2 <= 0:
            raise ValueError("domain_l2 must be positive or None")
        self.robustness = float(robustness)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.domain_l2 = None if domain_l2 is None else float(domain_l2)
        self.l1_smoothing = float(l1_smoothing)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    def fit(self, domains: dict[str, DomainData]) -> "MultiDomainCox":
        if len(domains) < 1:
            raise ValueError("At least one domain is required")
        first = next(iter(domains.values()))
        features = list(map(str, first.x.columns))
        if not features:
            raise ValueError("At least one feature is required")
        for name, domain in domains.items():
            if list(map(str, domain.x.columns)) != features:
                raise ValueError(f"Feature order mismatch in {name}")
            if len(domain.x) != len(domain.time) or len(domain.x) != len(domain.event):
                raise ValueError(f"Length mismatch in {name}")

        pooled = pd.concat([d.x for d in domains.values()], axis=0)
        center = pooled.mean(axis=0).fillna(0.0)
        scale = pooled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, domain in domains.items():
            z = ((domain.x - center) / scale).replace([np.inf, -np.inf], np.nan)
            z = z.fillna(0.0)
            prepared[name] = (
                z.to_numpy(dtype=float),
                np.asarray(domain.time, dtype=float),
                np.asarray(domain.event, dtype=bool),
            )

        domain_names = list(prepared)
        p = len(features)

        def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
            beta = parameters[:p]
            if self.domain_l2 is None:
                return beta, None
            offsets = parameters[p:].reshape(len(domain_names), p)
            return beta, offsets

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            beta, offsets = unpack(parameters)
            values = []
            gradients = []
            for k, values_k in enumerate(prepared.values()):
                x, time, event = values_k
                beta_k = beta if offsets is None else beta + offsets[k]
                value, gradient = _cox_loss_gradient(beta_k, x, time, event)
                values.append(value)
                gradients.append(gradient)
            losses = np.asarray(values, dtype=float)
            grads = np.vstack(gradients)
            if self.robustness == 0.0:
                data_loss = float(losses.mean())
                weights = np.repeat(1.0 / len(losses), len(losses))
            else:
                logits = self.robustness * losses
                weights = np.exp(logits - logsumexp(logits))
                data_loss = float(
                    (logsumexp(logits) - np.log(len(losses))) / self.robustness
                )
            data_grad = weights @ grads
            smooth_abs = np.sqrt(beta * beta + self.l1_smoothing**2)
            penalty = self.l1 * smooth_abs.sum() + 0.5 * self.l2 * np.dot(beta, beta)
            penalty_grad = self.l1 * beta / smooth_abs + self.l2 * beta
            beta_grad = data_grad + penalty_grad
            if offsets is None:
                return data_loss + float(penalty), beta_grad
            offset_penalty = 0.5 * self.domain_l2 * float(np.sum(offsets * offsets))
            offset_grad = weights[:, None] * grads + self.domain_l2 * offsets
            full_grad = np.concatenate([beta_grad, offset_grad.ravel()])
            return data_loss + float(penalty) + offset_penalty, full_grad

        parameter_count = p if self.domain_l2 is None else p * (1 + len(domain_names))
        initial = np.zeros(parameter_count, dtype=float)
        result = minimize(
            fun=lambda b: objective(b)[0],
            jac=lambda b: objective(b)[1],
            x0=initial,
            method="L-BFGS-B",
            options={
                "maxiter": self.max_iter,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
                "maxls": 50,
            },
        )
        self.features_ = features
        self.center_ = center
        self.scale_ = scale
        self.coef_, offsets = unpack(np.asarray(result.x, dtype=float))
        self.domain_offsets_ = (
            {}
            if offsets is None
            else {name: offsets[k].copy() for k, name in enumerate(domain_names)}
        )
        self.optimization_ = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(result.fun),
        }
        self.domain_losses_ = {}
        self.consensus_domain_losses_ = {}
        for name, values in prepared.items():
            local_beta = self.coef_ + self.domain_offsets_.get(name, 0.0)
            self.domain_losses_[name] = float(
                _cox_loss_gradient(local_beta, *values)[0]
            )
            self.consensus_domain_losses_[name] = float(
                _cox_loss_gradient(self.coef_, *values)[0]
            )
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        frame = x.reindex(columns=self.features_)
        z = ((frame - self.center_) / self.scale_).replace([np.inf, -np.inf], np.nan)
        return z.fillna(0.0).to_numpy(dtype=float)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.transform(x) @ self.coef_

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Model has not been fitted")
