from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def selected_pair_features(
    rank_expression: pd.DataFrame,
    pairs: list[str],
    available_genes: set[str] | None = None,
) -> pd.DataFrame:
    """Build locked binary pairs while preserving unavailable pairs as missing.

    Missing endpoints must not be interpreted as a biological ordering. The
    fitted learners subsequently map NaN to their training-centred value.
    """
    if available_genes is None:
        available_genes = {
            str(column)
            for column in rank_expression.columns
            if not rank_expression[column].isna().all()
        }
    output = pd.DataFrame(index=rank_expression.index)
    for pair in pairs:
        left, right = pair.split("|", 1)
        if left not in available_genes or right not in available_genes:
            output[pair] = np.nan
        else:
            comparison = rank_expression[left] > rank_expression[right]
            missing = rank_expression[left].isna() | rank_expression[right].isna()
            output[pair] = comparison.astype(float).mask(missing)
    return output


class LockedPDACModel:
    """Inference wrapper for the frozen TransSurv-PDAC research model."""

    def __init__(self, bundle: dict):
        self.bundle = bundle

    @classmethod
    def load(cls, path: str | Path) -> "LockedPDACModel":
        return cls(joblib.load(Path(path)))

    @property
    def required_genes(self) -> set[str]:
        return {
            gene
            for component in self.bundle["components"]
            for pair in component["pairs"]
            for gene in pair.split("|")
        }

    def predict_from_rank(
        self,
        rank_expression: pd.DataFrame,
        available_genes: set[str] | None = None,
    ) -> pd.DataFrame:
        rank_expression = rank_expression.copy()
        rank_expression.index = rank_expression.index.astype(str)
        if available_genes is None:
            available_genes = {
                str(column)
                for column in rank_expression.columns
                if not rank_expression[column].isna().all()
            }
        branch_values: dict[str, list[np.ndarray]] = {
            "multi_learner": [],
            "ipcw": [],
        }
        component_values = []
        coverage_values: list[np.ndarray] = []
        for component in self.bundle["components"]:
            x = selected_pair_features(
                rank_expression,
                component["pairs"],
                available_genes=available_genes,
            )
            raw = component["model"].predict(x)
            standardized = (raw - component["location"]) / component["scale"]
            branch_values[component["branch"]].append(standardized)
            component_values.append(standardized)
            coverage_values.append(x.notna().mean(axis=1).to_numpy(float))
        ml_risk = np.mean(np.vstack(branch_values["multi_learner"]), axis=0)
        ipcw_risk = np.mean(np.vstack(branch_values["ipcw"]), axis=0)
        weight = float(self.bundle["ipcw_weight"])
        raw_risk = (1.0 - weight) * ml_risk + weight * ipcw_risk
        relative_config = self.bundle.get("relative_risk", {})
        relative_minimum_n = int(
            relative_config.get(
                "minimum_batch_n",
                self.bundle.get("batchscale", {}).get("minimum_batch_n", 40),
            )
        )
        raw_dispersion = float(np.std(raw_risk, ddof=0))
        relative_outputs_available = (
            len(raw_risk) >= relative_minimum_n and raw_dispersion > 1e-12
        )
        if relative_outputs_available:
            within_batch_z = (raw_risk - float(np.mean(raw_risk))) / raw_dispersion
            risk_percentile = (
                pd.Series(raw_risk, index=rank_expression.index)
                .rank(method="average", pct=True)
                .to_numpy(float)
            )
            risk_tier = np.select(
                [risk_percentile < (1.0 / 3.0), risk_percentile < (2.0 / 3.0)],
                ["low", "middle"],
                default="high",
            ).astype(object)
        else:
            within_batch_z = np.full(len(raw_risk), np.nan)
            risk_percentile = np.full(len(raw_risk), np.nan)
            risk_tier = np.full(len(raw_risk), None, dtype=object)
        batchscale = self.bundle.get("batchscale")
        batchscale_applied = False
        calibration_warning = False
        if batchscale is not None:
            minimum_n = int(batchscale.get("minimum_batch_n", 40))
            dispersion = raw_dispersion
            if len(raw_risk) >= minimum_n and dispersion > 1e-12:
                coefficient = float(batchscale["coefficient"])
                risk = coefficient * (raw_risk - float(np.mean(raw_risk))) / dispersion
                batchscale_applied = True
            else:
                risk = raw_risk.copy()
                calibration_warning = True
        else:
            risk = raw_risk
        output = pd.DataFrame(
            {
                "ml_risk": ml_risk,
                "ipcw_risk": ipcw_risk,
                "raw_risk": raw_risk,
                "risk": risk,
                "within_batch_z": within_batch_z,
                "risk_percentile": risk_percentile,
                "risk_tier": risk_tier,
                "relative_outputs_available": relative_outputs_available,
                "branch_disagreement": np.abs(ml_risk - ipcw_risk),
                "minimum_component_pair_coverage": np.min(
                    np.vstack(coverage_values), axis=0
                ),
                "batchscale_applied": batchscale_applied,
                "calibration_warning": calibration_warning,
                "absolute_survival_disabled": not bool(
                    self.bundle.get("absolute_survival_enabled", True)
                ),
            },
            index=rank_expression.index,
        )
        for index, values in enumerate(component_values, start=1):
            output[f"component_{index}_risk"] = values
        baseline = np.asarray(self.bundle["baseline_cumulative_hazard"], float)
        survival = np.exp(-np.exp(np.clip(risk, -40, 40))[:, None] * baseline[None, :])
        absolute_survival_enabled = bool(
            self.bundle.get("absolute_survival_enabled", True)
        )
        if (calibration_warning and batchscale is not None) or not absolute_survival_enabled:
            survival[:] = np.nan
        for position, time in enumerate(self.bundle["baseline_times_months"]):
            output[f"survival_{int(time)}m"] = survival[:, position]
        return output
