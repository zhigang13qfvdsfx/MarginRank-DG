from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from transsurv.marginrank import LockedMarginRankModel


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
MODEL_PATH = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"
PLAN_PATH = CONFIG / "V4_GSE77435_TECHNICAL_REPRODUCIBILITY_PLAN.lock"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_gene_z(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    median = frame.median(axis=0)
    mad = frame.sub(median, axis=1).abs().median(axis=0)
    scale = (1.4826 * mad).replace(0.0, np.nan)
    return frame.sub(median, axis=1).div(scale, axis=1), mad


def row_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - np.nanmean(left, axis=1, keepdims=True)
    right = right - np.nanmean(right, axis=1, keepdims=True)
    left = left / np.sqrt(np.nansum(left**2, axis=1, keepdims=True))
    right = right / np.sqrt(np.nansum(right**2, axis=1, keepdims=True))
    return np.nan_to_num(left, nan=0.0) @ np.nan_to_num(right, nan=0.0).T


def tertiles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average", pct=True).to_numpy(float)
    return np.where(ranks <= 1 / 3, 0, np.where(ranks <= 2 / 3, 1, 2))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(spearmanr(left, right).statistic)


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["status"] != "LOCKED_BEFORE_GSE77435_IDENTITY_MAPPING_AND_RISK_SCORING":
        raise RuntimeError("Unexpected GSE77435 technical plan")
    model_spec = json.loads(
        (CONFIG / "frozen_model_v4_marginrank_spec.json").read_text(encoding="utf-8")
    )
    if sha256(MODEL_PATH) != model_spec["model_sha256"]:
        raise RuntimeError("Frozen model hash mismatch")

    universe = pd.read_csv(
        ROOT / "data" / "manifests" / "v4_strict_common_gene_universe.csv"
    )["gene"].astype(str).tolist()
    rna = pd.read_parquet(DATA / "gse77435_v4_fresh_expression.parquet").reindex(
        columns=universe
    )
    array = pd.read_parquet(DATA / "gse62452_expression.parquet").reindex(
        columns=universe
    )
    rna.index = rna.index.astype(str)
    array.index = array.index.astype(str)
    rna = rna.apply(pd.to_numeric, errors="coerce")
    array = array.apply(pd.to_numeric, errors="coerce")

    rna_z, rna_mad = robust_gene_z(rna)
    array_z, array_mad = robust_gene_z(array)
    valid = (
        rna_mad.notna()
        & array_mad.notna()
        & (rna_mad > 0)
        & (array_mad > 0)
        & rna.notna().all(axis=0)
        & array.notna().all(axis=0)
    )
    rna_pct = rna_mad[valid].rank(pct=True)
    array_pct = array_mad[valid].rank(pct=True)
    information = pd.concat([rna_pct, array_pct], axis=1).min(axis=1)
    fingerprint_genes = information.nlargest(min(5000, len(information))).index.tolist()
    if len(fingerprint_genes) < 3000:
        raise RuntimeError("Too few outcome-free fingerprint genes")

    primary_similarity = row_pearson(
        rna_z[fingerprint_genes].to_numpy(float),
        array_z[fingerprint_genes].to_numpy(float),
    )
    rna_ranks = rna[fingerprint_genes].rank(axis=1, method="average", pct=True)
    array_ranks = array[fingerprint_genes].rank(axis=1, method="average", pct=True)
    secondary_similarity = row_pearson(
        rna_ranks.to_numpy(float), array_ranks.to_numpy(float)
    )

    row_ind, col_ind = linear_sum_assignment(-primary_similarity)
    assigned = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    rna_top = np.argmax(primary_similarity, axis=1)
    array_top = np.argmax(primary_similarity, axis=0)
    secondary_top = np.argmax(secondary_similarity, axis=1)
    rules = plan["identity_mapping"]
    mapping_rows = []
    for i, rna_id in enumerate(rna.index):
        order = np.argsort(primary_similarity[i])[::-1]
        j = assigned[i]
        top1 = int(order[0])
        top2 = int(order[1])
        primary_rnn = bool(rna_top[i] == j and array_top[j] == i)
        is_primary_top = bool(j == top1)
        secondary_agreement = bool(secondary_top[i] == j)
        margin = float(primary_similarity[i, top1] - primary_similarity[i, top2])
        confident = bool(
            primary_rnn
            and is_primary_top
            and primary_similarity[i, j] >= rules["minimum_primary_similarity"]
            and margin >= rules["minimum_primary_top1_minus_top2_margin"]
        )
        mapping_rows.append(
            {
                "gse77435_id": rna_id,
                "gse62452_id": array.index[j],
                "hungarian_primary_similarity": float(primary_similarity[i, j]),
                "primary_top1_similarity": float(primary_similarity[i, top1]),
                "primary_top2_similarity": float(primary_similarity[i, top2]),
                "primary_top1_margin": margin,
                "secondary_similarity": float(secondary_similarity[i, j]),
                "primary_reciprocal_nearest": primary_rnn,
                "hungarian_is_primary_top1": is_primary_top,
                "secondary_top1_agrees": secondary_agreement,
                "confident_identity": confident,
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(RESULTS / "43_gse77435_gse62452_identity_map.csv", index=False)
    n_confident = int(mapping["confident_identity"].sum())
    agreement_fraction = float(mapping["secondary_top1_agrees"].mean())
    mapping_conditions = {
        "minimum_confident_pairs": n_confident >= rules["minimum_confident_pairs"],
        "primary_secondary_agreement": agreement_fraction
        >= rules["minimum_primary_secondary_top1_agreement_fraction"],
        "one_to_one_assignment": mapping["gse62452_id"].nunique() == len(mapping),
    }

    if not all(mapping_conditions.values()):
        summary = {
            "status": "GSE77435_IDENTITY_MAPPING_FAILED_STOP_BEFORE_RISK_SCORING",
            "overall_gate": "FAIL",
            "mapping_gate": "FAIL",
            "mapping_conditions": mapping_conditions,
            "n_rnaseq": int(len(rna)),
            "n_microarray": int(len(array)),
            "confident_pairs": n_confident,
            "primary_secondary_top1_agreement_fraction": agreement_fraction,
            "risk_scores_generated": False,
            "model_changed": False,
            "clinical_validation_claimed": False,
        }
        (RESULTS / "43_v4_gse77435_crossplatform_gate.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return

    confident = mapping.loc[mapping["confident_identity"]].copy()
    model = LockedMarginRankModel.load(MODEL_PATH)
    rna_score = model.predict_from_expression(rna)
    array_score = model.predict_from_expression(array)
    paired = confident.copy()
    paired["rnaseq_raw_risk"] = rna_score.loc[paired["gse77435_id"], "raw_risk"].to_numpy(float)
    paired["microarray_raw_risk"] = array_score.loc[paired["gse62452_id"], "raw_risk"].to_numpy(float)
    paired["rnaseq_selected_coverage"] = rna_score.loc[
        paired["gse77435_id"], "selected_gene_coverage"
    ].to_numpy(float)
    paired["microarray_selected_coverage"] = array_score.loc[
        paired["gse62452_id"], "selected_gene_coverage"
    ].to_numpy(float)
    paired["rnaseq_anchor_coverage"] = rna_score.loc[
        paired["gse77435_id"], "anchor_gene_coverage"
    ].to_numpy(float)
    paired["microarray_anchor_coverage"] = array_score.loc[
        paired["gse62452_id"], "anchor_gene_coverage"
    ].to_numpy(float)
    for prefix in ("rnaseq", "microarray"):
        values = paired[f"{prefix}_raw_risk"].to_numpy(float)
        paired[f"{prefix}_z_risk"] = (values - values.mean()) / values.std(ddof=0)
        paired[f"{prefix}_tertile"] = tertiles(values)
    paired["z_risk_difference"] = paired["rnaseq_z_risk"] - paired["microarray_z_risk"]
    paired.to_csv(RESULTS / "43_v4_gse77435_crossplatform_pairs.csv", index=False)

    rna_values = paired["rnaseq_raw_risk"].to_numpy(float)
    array_values = paired["microarray_raw_risk"].to_numpy(float)
    spearman = safe_spearman(rna_values, array_values)
    pearson = float(pearsonr(rna_values, array_values).statistic)
    standardized_rmse = float(np.sqrt(np.mean(paired["z_risk_difference"] ** 2)))
    weighted_kappa = float(
        cohen_kappa_score(
            paired["rnaseq_tertile"], paired["microarray_tertile"], weights="quadratic"
        )
    )

    gate = plan["technical_gate"]
    rng = np.random.default_rng(int(gate["seed"]))
    bootstrap_rows = []
    for replicate in range(int(gate["bootstrap_replicates"])):
        index = rng.integers(0, len(paired), size=len(paired))
        left = rna_values[index]
        right = array_values[index]
        if len(np.unique(left)) < 3 or len(np.unique(right)) < 3:
            continue
        bootstrap_rows.append(
            {"replicate": replicate, "spearman": safe_spearman(left, right)}
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(RESULTS / "43_v4_gse77435_crossplatform_bootstrap.csv", index=False)
    spearman_ci = np.quantile(bootstrap["spearman"], [0.025, 0.975]).tolist()
    minimum_selected_coverage = float(
        paired[["rnaseq_selected_coverage", "microarray_selected_coverage"]].min().min()
    )
    minimum_anchor_coverage = float(
        paired[["rnaseq_anchor_coverage", "microarray_anchor_coverage"]].min().min()
    )
    technical_conditions = {
        "minimum_confident_pairs": len(paired) >= gate["minimum_confident_pairs"],
        "raw_risk_spearman": spearman >= gate["minimum_raw_risk_spearman"],
        "raw_risk_pearson": pearson >= gate["minimum_raw_risk_pearson"],
        "bootstrap_spearman_lower": spearman_ci[0]
        >= gate["minimum_bootstrap_spearman_lower_95"],
        "standardized_risk_rmse": standardized_rmse
        <= gate["maximum_standardized_risk_rmse"],
        "tertile_weighted_kappa": weighted_kappa
        >= gate["minimum_tertile_weighted_kappa"],
        "selected_gene_coverage": minimum_selected_coverage
        >= gate["minimum_selected_gene_coverage"],
        "anchor_gene_coverage": minimum_anchor_coverage
        >= gate["minimum_anchor_gene_coverage"],
    }
    summary = {
        "status": "GSE77435_SOURCE_OVERLAP_TECHNICAL_REPRODUCIBILITY_COMPLETE",
        "overall_gate": "PASS" if all(technical_conditions.values()) else "FAIL",
        "mapping_gate": "PASS",
        "mapping_conditions": mapping_conditions,
        "technical_gate": "PASS" if all(technical_conditions.values()) else "FAIL",
        "technical_conditions": technical_conditions,
        "n_rnaseq": int(len(rna)),
        "n_microarray": int(len(array)),
        "fingerprint_genes": len(fingerprint_genes),
        "confident_pairs": int(len(paired)),
        "primary_secondary_top1_agreement_fraction": agreement_fraction,
        "risk_spearman": spearman,
        "risk_spearman_ci_95": spearman_ci,
        "risk_pearson": pearson,
        "standardized_risk_rmse": standardized_rmse,
        "tertile_quadratic_weighted_kappa": weighted_kappa,
        "minimum_selected_gene_coverage": minimum_selected_coverage,
        "minimum_anchor_gene_coverage": minimum_anchor_coverage,
        "risk_scores_generated_after_identity_mapping": True,
        "outcomes_used": False,
        "survival_groups_used": False,
        "independent_clinical_validation": False,
        "model_changed": False,
        "paca_ca_used_for_tuning": False,
        "hashes": {
            "plan": sha256(PLAN_PATH),
            "model": sha256(MODEL_PATH),
            "gse77435_expression": sha256(DATA / "gse77435_v4_fresh_expression.parquet"),
            "gse62452_expression": sha256(DATA / "gse62452_expression.parquet"),
            "script": sha256(Path(__file__)),
        },
    }
    (RESULTS / "43_v4_gse77435_crossplatform_gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    completion = {
        "status": "GSE77435_TECHNICAL_DATA_CONSUMED_ONCE",
        "result_gate": summary["overall_gate"],
        "source_overlap_with_gse62452": True,
        "may_be_used_for_model_tuning": False,
        "may_be_claimed_as_independent_validation": False,
        "result_sha256": sha256(RESULTS / "43_v4_gse77435_crossplatform_gate.json"),
    }
    (CONFIG / "V4_GSE77435_TECHNICAL_COMPLETED.lock").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
