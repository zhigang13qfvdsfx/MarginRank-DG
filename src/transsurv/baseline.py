from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .model import DomainData, MultiDomainCox


def breslow_cumulative_hazard(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    evaluation_times: Iterable[float],
) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    evaluation_times = np.asarray(list(evaluation_times), dtype=float)
    values = np.zeros_like(evaluation_times, dtype=float)
    cumulative = 0.0
    unique_event_times = np.unique(time[event])
    for event_time in np.sort(unique_event_times):
        deaths = int(np.sum(event & np.isclose(time, event_time)))
        denominator = float(np.exp(np.clip(risk[time >= event_time], -40, 40)).sum())
        if denominator > 0:
            cumulative += deaths / denominator
        values[evaluation_times >= event_time] = cumulative
    return values


def estimate_meta_baseline(
    model: MultiDomainCox,
    domains: dict[str, DomainData],
    evaluation_times: Iterable[float],
) -> pd.DataFrame:
    evaluation_times = np.asarray(list(evaluation_times), dtype=float)
    rows = []
    for name, domain in domains.items():
        risk = model.predict(domain.x)
        hazard = breslow_cumulative_hazard(
            domain.time, domain.event, risk, evaluation_times
        )
        for time, value in zip(evaluation_times, hazard):
            rows.append(
                {
                    "domain": name,
                    "time": float(time),
                    "cumulative_hazard": float(value),
                    "events": int(np.sum(domain.event)),
                    "supported": bool(np.max(domain.time) > time),
                }
            )
    table = pd.DataFrame(rows)
    combined_rows = []
    for time, group in table.groupby("time", sort=True):
        supported = group[group["supported"]]
        if supported.empty:
            supported = group
        combined_rows.append(
            {
                "time": float(time),
                "cumulative_hazard": float(
                    np.average(
                        supported["cumulative_hazard"], weights=supported["events"]
                    )
                ),
                "n_source_domains": int(len(supported)),
            }
        )
    combined = pd.DataFrame(combined_rows)
    combined.attrs["domain_table"] = table
    return combined


def predict_survival(
    model: MultiDomainCox, x: pd.DataFrame, baseline: pd.DataFrame
) -> np.ndarray:
    risk = model.predict(x)
    cumulative_hazard = baseline["cumulative_hazard"].to_numpy(dtype=float)
    return np.exp(-np.exp(np.clip(risk, -40, 40))[:, None] * cumulative_hazard[None, :])

