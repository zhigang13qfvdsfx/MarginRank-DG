from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .transforms import SingleSampleRankTransformer


def anchor_rank_features(
    rank_expression: pd.DataFrame,
    selected_genes: list[str],
    anchor_genes: list[str],
) -> pd.DataFrame:
    """Construct continuous gene-rank margins relative to locked anchors.

    Missing selected genes remain NaN and are mapped to the training-centred
    value by the fitted learner. The anchor centre is the per-sample median of
    available anchors; no outcomes or other samples are used.
    """
    rank_expression = rank_expression.copy()
    rank_expression.index = rank_expression.index.astype(str)
    selected = rank_expression.reindex(columns=selected_genes)
    anchors = rank_expression.reindex(columns=anchor_genes)
    anchor_center = anchors.median(axis=1, skipna=True)
    output = pd.DataFrame(index=rank_expression.index)
    for gene in selected_genes:
        output[f"anchor::{gene}"] = selected[gene] - anchor_center
    return output


class LockedMarginRankModel:
    """Inference wrapper for the frozen relative-risk-only MarginRank model."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        if bool(bundle.get("absolute_survival_enabled", False)):
            raise ValueError("MarginRank bundles must disable absolute survival")

    @classmethod
    def load(cls, path: str | Path) -> "LockedMarginRankModel":
        return cls(joblib.load(Path(path)))

    @property
    def selected_genes(self) -> list[str]:
        return list(self.bundle["feature_definition"]["genes"])

    @property
    def anchor_genes(self) -> list[str]:
        return list(self.bundle["feature_definition"]["anchors"])

    @property
    def required_input_genes(self) -> set[str]:
        return set(self.selected_genes) | set(self.anchor_genes)

    def predict_from_rank(self, rank_expression: pd.DataFrame) -> pd.DataFrame:
        rank_expression = rank_expression.copy()
        rank_expression.index = rank_expression.index.astype(str)
        selected = rank_expression.reindex(columns=self.selected_genes)
        anchors = rank_expression.reindex(columns=self.anchor_genes)
        features = anchor_rank_features(
            rank_expression, self.selected_genes, self.anchor_genes
        )
        raw_risk = np.asarray(self.bundle["model"].predict(features), dtype=float)
        selected_coverage = selected.notna().mean(axis=1).to_numpy(float)
        anchor_coverage = anchors.notna().mean(axis=1).to_numpy(float)
        minimum_batch_n = int(self.bundle["relative_risk"]["minimum_batch_n"])
        dispersion = float(np.std(raw_risk, ddof=0))
        relative_available = len(raw_risk) >= minimum_batch_n and dispersion > 1e-12
        if relative_available:
            within_batch_z = (raw_risk - float(np.mean(raw_risk))) / dispersion
            percentile = (
                pd.Series(raw_risk, index=rank_expression.index)
                .rank(method="average", pct=True)
                .to_numpy(float)
            )
            tier = np.select(
                [percentile < (1.0 / 3.0), percentile < (2.0 / 3.0)],
                ["low", "middle"],
                default="high",
            ).astype(object)
        else:
            within_batch_z = np.full(len(raw_risk), np.nan)
            percentile = np.full(len(raw_risk), np.nan)
            tier = np.full(len(raw_risk), None, dtype=object)
        minimum_selected_coverage = float(
            self.bundle.get("minimum_selected_gene_coverage", 0.80)
        )
        minimum_anchor_coverage = float(
            self.bundle.get("minimum_anchor_gene_coverage", 0.50)
        )
        return pd.DataFrame(
            {
                "raw_risk": raw_risk,
                "within_batch_z": within_batch_z,
                "risk_percentile": percentile,
                "risk_tier": tier,
                "relative_outputs_available": relative_available,
                "selected_gene_coverage": selected_coverage,
                "anchor_gene_coverage": anchor_coverage,
                "input_degraded": (selected_coverage < minimum_selected_coverage)
                | (anchor_coverage < minimum_anchor_coverage),
                "absolute_survival_disabled": True,
                "research_use_only": True,
            },
            index=rank_expression.index,
        )

    def predict_from_expression(self, expression: pd.DataFrame) -> pd.DataFrame:
        expression = expression.copy()
        expression.index = expression.index.astype(str)
        expression.columns = expression.columns.astype(str)
        universe = list(self.bundle["rank_universe"])
        available_reference = [gene for gene in universe if gene in expression.columns]
        numeric_reference = expression.reindex(columns=available_reference).apply(
            pd.to_numeric, errors="coerce"
        )
        reference_coverage = (
            numeric_reference.notna().sum(axis=1).to_numpy(float) / len(universe)
        )
        transformer = SingleSampleRankTransformer().fit(universe)
        ranks = transformer.transform(
            expression,
            features=sorted(self.required_input_genes),
        )
        output = self.predict_from_rank(ranks)
        output["reference_gene_coverage"] = reference_coverage
        output["reference_coverage_warning"] = reference_coverage < float(
            self.bundle.get("minimum_reference_gene_coverage", 0.80)
        )
        return output
