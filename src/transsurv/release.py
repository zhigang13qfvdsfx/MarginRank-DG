"""Public release helpers for the frozen TransSurv-PDAC v4 model."""

from __future__ import annotations

import json
from importlib.resources import as_file, files
from typing import Any

import pandas as pd

from .marginrank import LockedMarginRankModel


def load_default_model() -> LockedMarginRankModel:
    """Load the packaged, immutable v4 MarginRank-DG research model."""
    resource = files("transsurv").joinpath(
        "models/transsurv_pdac_v4_marginrank_locked.joblib"
    )
    with as_file(resource) as model_path:
        return LockedMarginRankModel.load(model_path)


def load_model_card() -> dict[str, Any]:
    """Return the machine-readable model card bundled with the package."""
    resource = files("transsurv").joinpath("resources/model_card.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def score_expression(
    expression: pd.DataFrame,
    model: LockedMarginRankModel | None = None,
) -> pd.DataFrame:
    """Score samples-by-genes expression without fitting or outcome access.

    The returned ``raw_risk`` is composition invariant. Batch-relative z scores,
    percentiles and tertiles are emitted only for batches of at least 40 samples.
    Absolute survival probabilities are intentionally unavailable.
    """
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a pandas DataFrame")
    if expression.empty:
        raise ValueError("expression contains no samples or genes")
    scorer = load_default_model() if model is None else model
    return scorer.predict_from_expression(expression)
