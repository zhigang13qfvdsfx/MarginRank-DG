from __future__ import annotations

import numpy as np
import pandas as pd

from .model import DomainData


def _cox_score_z(x: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Vectorised diagonal Cox score statistic at beta=0."""
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    order = np.argsort(-time, kind="mergesort")
    xs = x[order]
    ts = time[order]
    es = event[order]
    csum = np.cumsum(xs, axis=0)
    csum2 = np.cumsum(xs * xs, axis=0)
    score = np.zeros(xs.shape[1], dtype=float)
    information = np.zeros(xs.shape[1], dtype=float)
    _, starts, counts = np.unique(ts, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        stop = int(start + count)
        group_events = es[start:stop]
        d = int(group_events.sum())
        if d == 0:
            continue
        n_risk = float(stop)
        mean = csum[stop - 1] / n_risk
        variance = np.maximum(csum2[stop - 1] / n_risk - mean * mean, 1e-12)
        score += xs[start:stop][group_events].sum(axis=0) - d * mean
        information += d * variance
    return score / np.sqrt(np.maximum(information, 1e-12))


def screen_features(
    domains: dict[str, DomainData],
    n_features: int,
    force_include: list[str] | None = None,
    method: str = "soft",
) -> pd.DataFrame:
    if n_features < 1:
        raise ValueError("n_features must be positive")
    columns = list(map(str, next(iter(domains.values())).x.columns))
    z_by_domain = []
    names = []
    for name, domain in domains.items():
        frame = domain.x.reindex(columns=columns).astype(float)
        center = frame.mean(axis=0)
        scale = frame.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        zframe = ((frame - center) / scale).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        z_by_domain.append(
            _cox_score_z(zframe.to_numpy(), domain.time, domain.event)
        )
        names.append(name)
    matrix = np.vstack(z_by_domain)
    mean_z = matrix.mean(axis=0)
    dispersion = matrix.std(axis=0, ddof=0)
    sign_consistency = np.abs(np.sign(matrix).mean(axis=0))
    min_abs_z = np.min(np.abs(matrix), axis=0)
    meta_z = matrix.sum(axis=0) / np.sqrt(len(names))
    soft_stability = (
        np.abs(meta_z) * (0.5 + 0.5 * sign_consistency) / (1.0 + dispersion)
    )
    strict_stability = np.where(
        np.isclose(sign_consistency, 1.0),
        min_abs_z * np.abs(meta_z) / (1.0 + dispersion),
        0.0,
    )
    if method == "soft":
        stability = soft_stability
    elif method == "strict":
        stability = strict_stability
    else:
        raise ValueError("method must be 'soft' or 'strict'")
    table = pd.DataFrame(
        {
            "gene": columns,
            "meta_z": meta_z,
            "mean_z": mean_z,
            "z_dispersion": dispersion,
            "sign_consistency": sign_consistency,
            "min_abs_z": min_abs_z,
            "soft_stability_score": soft_stability,
            "strict_stability_score": strict_stability,
            "stability_score": stability,
        }
    )
    for i, name in enumerate(names):
        table[f"z_{name}"] = matrix[i]
    table = table.sort_values(
        ["stability_score", "gene"], ascending=[False, True]
    ).reset_index(drop=True)
    selected = table.head(n_features).copy()
    if force_include:
        forced = table[table["gene"].isin(force_include)]
        selected = pd.concat([selected, forced], ignore_index=True).drop_duplicates("gene")
    selected["selected"] = True
    return selected
