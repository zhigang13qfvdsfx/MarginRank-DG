from __future__ import annotations

import numpy as np
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv


def make_y(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(np.asarray(event, dtype=bool), np.asarray(time, dtype=float))


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            np.asarray(event, dtype=bool),
            np.asarray(time, dtype=float),
            np.asarray(risk, dtype=float),
        )[0]
    )


def uno_c(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    tau: float | None = None,
) -> float:
    y_train = make_y(train_time, train_event)
    y_test = make_y(test_time, test_event).copy()
    upper = float(np.max(y_train["time"]) - 1e-6)
    beyond = y_test["time"] >= upper
    y_test["time"][beyond] = upper
    y_test["event"][beyond] = False
    if tau is None:
        tau = min(36.0, float(np.quantile(y_test["time"], 0.9)), upper)
    return float(concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0])


def time_dependent_metrics(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    survival: np.ndarray,
    times: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    y_train = make_y(train_time, train_event)
    y_test = make_y(test_time, test_event).copy()
    upper = float(np.max(y_train["time"]) - 1e-6)
    beyond = y_test["time"] >= upper
    y_test["time"][beyond] = upper
    y_test["event"][beyond] = False
    times = np.asarray(times, dtype=float)
    valid = (times > float(np.min(y_test["time"]))) & (
        times < min(float(np.max(y_test["time"])), float(np.max(y_train["time"])))
    )
    use_times = times[valid]
    if len(use_times) == 0:
        return {"mean_auc": np.nan, "mean_brier": np.nan}, []
    use_survival = survival[:, valid]
    aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, use_times)
    _, briers = brier_score(y_train, y_test, use_survival, use_times)
    rows = [
        {"time": float(t), "auc": float(a), "brier": float(b)}
        for t, a, b in zip(use_times, aucs, briers)
    ]
    return {
        "mean_auc": float(mean_auc),
        "mean_brier": float(np.mean(briers)),
        "n_administratively_truncated": int(beyond.sum()),
    }, rows

