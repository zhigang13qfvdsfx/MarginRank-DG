from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from transsurv.inference import LockedPDACModel
from transsurv.marginrank import LockedMarginRankModel
from transsurv.transforms import SingleSampleRankTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
MODEL = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"
V2_MODEL = ROOT / "models" / "predecessors" / "transsurv_pdac_v2_1_locked.joblib"
EXPRESSION = DATA / "gse77435_v4_fresh_expression.parquet"
CLINICAL = DATA / "gse77435_v4_fresh_clinical.csv"
CORRECTION = CONFIG / "V4_GSE77435_FACTUAL_CORRECTION_AMENDMENT.lock"
PLAN = CONFIG / "V4_GSE77435_EXTREME_SURVIVAL_ONE_TIME_PLAN.lock"
COMPLETION = CONFIG / "V4_GSE77435_EXTREME_SURVIVAL_COMPLETED.lock"

LEGACY_COEFFICIENTS = {
    "LY6D": 0.05732909198204257,
    "PLA2G2A": -0.047712655849770504,
    "GHRL": -0.039489341632882174,
    "EREG": 0.012040159816165675,
    "FAM83A": 0.009252212452749452,
    "UCA1": 0.008674303585955168,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_score(expression: pd.DataFrame) -> tuple[np.ndarray, float]:
    genes = list(LEGACY_COEFFICIENTS)
    frame = expression.reindex(columns=genes).apply(pd.to_numeric, errors="coerce")
    available = frame.columns[frame.notna().any(axis=0)]
    location = frame.mean(axis=0)
    scale = frame.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    z = ((frame - location) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    coefficients = np.asarray([LEGACY_COEFFICIENTS[gene] for gene in genes])
    return z.to_numpy(float) @ coefficients, float(len(available) / len(genes))


def logistic_or(label: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    standardized = (risk - risk.mean()) / risk.std(ddof=0)
    design = sm.add_constant(standardized)
    fit = sm.GLM(label, design, family=sm.families.Binomial()).fit()
    coefficient = float(fit.params[1])
    se = float(fit.bse[1])
    return {
        "or_per_sd": float(np.exp(coefficient)),
        "or_ci_low": float(np.exp(coefficient - 1.96 * se)),
        "or_ci_high": float(np.exp(coefficient + 1.96 * se)),
        "or_p": float(fit.pvalues[1]),
    }


def metric(label: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(label, risk))
    short = risk[label == 1]
    long = risk[label == 0]
    mann = mannwhitneyu(short, long, alternative="greater", method="exact")
    threshold = float(np.median(risk))
    predicted = (risk > threshold).astype(int)
    return {
        "auc": auc,
        "rank_biserial": float(2.0 * auc - 1.0),
        "mann_whitney_one_sided_p": float(mann.pvalue),
        "median_short": float(np.median(short)),
        "median_long": float(np.median(long)),
        "median_difference_short_minus_long": float(np.median(short) - np.median(long)),
        "median_split_threshold": threshold,
        "median_split_balanced_accuracy": float(
            balanced_accuracy_score(label, predicted)
        ),
        **logistic_or(label, risk),
    }


def main() -> None:
    if COMPLETION.exists():
        raise RuntimeError("GSE77435 extreme-survival endpoint has already been consumed")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    correction = json.loads(CORRECTION.read_text(encoding="utf-8"))
    if plan["status"] != "LOCKED_BEFORE_FIRST_GSE77435_RISK_SCORE":
        raise RuntimeError("Unexpected GSE77435 one-time plan")
    if correction["status"] != "FACTUAL_CORRECTION_LOCKED_BEFORE_ANY_GSE77435_RISK_SCORE":
        raise RuntimeError("Factual correction not locked")
    observed_hashes = {
        "model_sha256": sha256(MODEL),
        "v2_model_sha256": sha256(V2_MODEL),
        "expression_sha256": sha256(EXPRESSION),
        "clinical_sha256": sha256(CLINICAL),
        "correction_sha256": sha256(CORRECTION),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    for key, value in observed_hashes.items():
        if plan[key].lower() != value.lower():
            raise RuntimeError(f"Locked hash mismatch: {key}")

    clinical = pd.read_csv(CLINICAL)
    expression = pd.read_parquet(EXPRESSION)
    expression.index = expression.index.astype(str)
    clinical["SAMPLE_ID"] = clinical["SAMPLE_ID"].astype(str)
    if clinical["SAMPLE_ID"].duplicated().any() or expression.index.duplicated().any():
        raise RuntimeError("Duplicate GSE77435 sample identifier")
    clinical = clinical.set_index("SAMPLE_ID", drop=False)
    if set(clinical.index) != set(expression.index):
        raise RuntimeError("GSE77435 clinical/expression identifiers do not align")
    clinical = clinical.loc[expression.index].copy()
    label = pd.to_numeric(clinical["SHORT_SURVIVAL_LABEL"], errors="raise").to_numpy(int)
    if list(np.bincount(label)) != [11, 13]:
        raise RuntimeError("Unexpected locked extreme-survival group counts")

    v4 = LockedMarginRankModel.load(MODEL)
    v4_prediction = v4.predict_from_expression(expression)
    if v4_prediction["input_degraded"].any() or v4_prediction[
        "reference_coverage_warning"
    ].any():
        raise RuntimeError("GSE77435 v4 inference coverage failed")
    if any(column.startswith("survival_") for column in v4_prediction):
        raise RuntimeError("Absolute survival was unexpectedly emitted")

    v2 = LockedPDACModel.load(V2_MODEL)
    v2_universe = list(v2.bundle["rank_universe"])
    v2_rank = SingleSampleRankTransformer().fit(v2_universe).transform(
        expression, features=v2_universe
    )
    v2_prediction = v2.predict_from_rank(v2_rank)
    legacy, legacy_coverage = legacy_score(expression)
    scores = {
        "v4_marginrank": v4_prediction["raw_risk"].to_numpy(float),
        "v2_predecessor": v2_prediction["raw_risk"].to_numpy(float),
        "legacy_six_gene": legacy,
    }
    metrics = {name: metric(label, values) for name, values in scores.items()}

    fpr, tpr, thresholds = roc_curve(label, scores["v4_marginrank"])
    pd.DataFrame(
        {"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds}
    ).to_csv(RESULTS / "44_v4_gse77435_roc_curve.csv", index=False)

    rng = np.random.default_rng(int(plan["bootstrap_seed"]))
    bootstrap_rows = []
    for replicate in range(int(plan["bootstrap_replicates"])):
        short_index = rng.choice(np.where(label == 1)[0], size=int((label == 1).sum()), replace=True)
        long_index = rng.choice(np.where(label == 0)[0], size=int((label == 0).sum()), replace=True)
        index = np.concatenate([short_index, long_index])
        sampled_label = label[index]
        row: dict[str, float | int] = {"replicate": replicate}
        for name, values in scores.items():
            row[f"{name}_auc"] = float(roc_auc_score(sampled_label, values[index]))
        row["delta_auc_v4_vs_v2"] = row["v4_marginrank_auc"] - row["v2_predecessor_auc"]
        row["delta_auc_v4_vs_legacy"] = row["v4_marginrank_auc"] - row["legacy_six_gene_auc"]
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(RESULTS / "44_v4_gse77435_bootstrap.csv", index=False)

    permutation_rng = np.random.default_rng(int(plan["permutation_seed"]))
    observed_auc = metrics["v4_marginrank"]["auc"]
    exceed = 0
    permutation_rows = []
    for replicate in range(int(plan["permutation_replicates"])):
        permuted = permutation_rng.permutation(label)
        auc = float(roc_auc_score(permuted, scores["v4_marginrank"]))
        exceed += int(auc >= observed_auc)
        permutation_rows.append({"replicate": replicate, "auc": auc})
    permutation_p = float((exceed + 1) / (int(plan["permutation_replicates"]) + 1))
    pd.DataFrame(permutation_rows).to_csv(
        RESULTS / "44_v4_gse77435_permutation.csv", index=False
    )

    prediction_frame = clinical[
        ["SAMPLE_ID", "PATIENT_ID", "TITLE", "SURVIVAL_GROUP", "SHORT_SURVIVAL_LABEL"]
    ].copy()
    for name, values in scores.items():
        prediction_frame[f"{name}_risk"] = values
    prediction_frame["selected_gene_coverage"] = v4_prediction[
        "selected_gene_coverage"
    ].to_numpy(float)
    prediction_frame["anchor_gene_coverage"] = v4_prediction[
        "anchor_gene_coverage"
    ].to_numpy(float)
    prediction_frame["reference_gene_coverage"] = v4_prediction[
        "reference_gene_coverage"
    ].to_numpy(float)
    prediction_frame.to_csv(RESULTS / "44_v4_gse77435_predictions.csv", index=False)
    pd.DataFrame(
        [{"model": name, **values} for name, values in metrics.items()]
    ).to_csv(RESULTS / "44_v4_gse77435_metrics.csv", index=False)

    v4_ci = np.quantile(bootstrap["v4_marginrank_auc"], [0.025, 0.975]).tolist()
    delta_v2_ci = np.quantile(bootstrap["delta_auc_v4_vs_v2"], [0.025, 0.975]).tolist()
    delta_legacy_ci = np.quantile(
        bootstrap["delta_auc_v4_vs_legacy"], [0.025, 0.975]
    ).tolist()
    gate = plan["validation_gate"]
    v4_metric = metrics["v4_marginrank"]
    validation_conditions = {
        "auc": v4_metric["auc"] >= gate["minimum_auc"],
        "bootstrap_auc_lower": v4_ci[0] >= gate["minimum_bootstrap_auc_lower_95"],
        "permutation_p": permutation_p <= gate["maximum_permutation_p"],
        "odds_ratio_ci_lower": v4_metric["or_ci_low"] > gate["minimum_or_ci_lower"],
        "correct_median_direction": v4_metric["median_difference_short_minus_long"] > 0,
        "minimum_balanced_accuracy": v4_metric["median_split_balanced_accuracy"]
        >= gate["minimum_median_split_balanced_accuracy"],
        "continuous_scores": len(np.unique(np.round(scores["v4_marginrank"], 12)))
        >= gate["minimum_distinct_risks"],
        "coverage": bool(
            v4_prediction["selected_gene_coverage"].min() >= 1.0
            and v4_prediction["anchor_gene_coverage"].min() >= 1.0
        ),
    }
    readiness = plan["additional_evidence_gate"]
    delta_v2 = v4_metric["auc"] - metrics["v2_predecessor"]["auc"]
    delta_legacy = v4_metric["auc"] - metrics["legacy_six_gene"]["auc"]
    additional_conditions = {
        "validation_gate_green": all(validation_conditions.values()),
        "auc_gain_over_v2": delta_v2 >= readiness["minimum_auc_gain_over_v2"],
        "bootstrap_probability_v4_gt_v2": float(
            (bootstrap["delta_auc_v4_vs_v2"] > 0).mean()
        )
        >= readiness["minimum_probability_v4_gt_v2"],
        "auc_gain_over_legacy": delta_legacy
        >= readiness["minimum_auc_gain_over_legacy"],
        "bootstrap_probability_v4_gt_legacy": float(
            (bootstrap["delta_auc_v4_vs_legacy"] > 0).mean()
        )
        >= readiness["minimum_probability_v4_gt_legacy"],
    }
    summary = {
        "status": "GSE77435_EXTREME_SURVIVAL_VALIDATED_EXACTLY_ONCE",
        "external_validation_gate": "GREEN_EXTREME_PHENOTYPE_VALIDATION"
        if all(validation_conditions.values())
        else "AMBER_OR_RED_EXTREME_PHENOTYPE_RESULT",
        "validation_conditions": validation_conditions,
        "additional_evidence_gate": "PASS"
        if all(additional_conditions.values())
        else "FAIL",
        "additional_evidence_conditions": additional_conditions,
        "cohort": "GSE77435",
        "n": int(len(label)),
        "short_survivors_lt10m": int(label.sum()),
        "long_survivors_gt20m": int((label == 0).sum()),
        "primary": v4_metric,
        "comparators": metrics,
        "v4_auc_ci_95": v4_ci,
        "permutation_p": permutation_p,
        "v4_auc_gain_over_v2": delta_v2,
        "v4_vs_v2_delta_auc_ci_95": delta_v2_ci,
        "bootstrap_probability_v4_gt_v2": float(
            (bootstrap["delta_auc_v4_vs_v2"] > 0).mean()
        ),
        "v4_auc_gain_over_legacy": delta_legacy,
        "v4_vs_legacy_delta_auc_ci_95": delta_legacy_ci,
        "bootstrap_probability_v4_gt_legacy": float(
            (bootstrap["delta_auc_v4_vs_legacy"] > 0).mean()
        ),
        "legacy_gene_coverage": legacy_coverage,
        "absolute_survival_values_emitted": False,
        "time_to_event_claimed": False,
        "calibration_claimed": False,
        "clinical_utility_claimed": False,
        "model_changed": False,
        "paca_ca_used_for_tuning": False,
        "interpretation": "Independent small-cohort validation of a prespecified extreme-survival binary phenotype; not representative time-to-event validation and not suitable for calibration or clinical-decision claims.",
        "hashes": observed_hashes,
    }
    result_path = RESULTS / "44_v4_gse77435_extreme_survival_gate.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    completion = {
        "status": "V4_GSE77435_EXTREME_SURVIVAL_CONSUMED_ONCE",
        "result_gate": summary["external_validation_gate"],
        "model_sha256": observed_hashes["model_sha256"],
        "result_sha256": sha256(result_path),
        "may_be_reused_for_model_tuning": False,
        "may_be_claimed_as_time_to_event_validation": False,
    }
    COMPLETION.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
