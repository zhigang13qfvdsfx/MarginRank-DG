from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc

from transsurv.inference import LockedPDACModel
from transsurv.marginrank import LockedMarginRankModel
from transsurv.metrics import harrell_c, make_y, uno_c
from transsurv.transforms import SingleSampleRankTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
MODEL = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"
V2_MODEL = ROOT / "models" / "predecessors" / "transsurv_pdac_v2_1_locked.joblib"
SPEC = CONFIG / "frozen_model_v4_marginrank_spec.json"
V2_SPEC = CONFIG / "frozen_model_v2_1_spec.json"
ELIGIBILITY = RESULTS / "40_v4_fresh_eligibility.json"
EXPRESSION = DATA / "paca_ca_v4_fresh_expression.parquet"
CLINICAL = DATA / "paca_ca_v4_fresh_clinical.csv"
ANALYSIS_LOCK = CONFIG / "V4_PACA_CA_ONE_TIME_ANALYSIS_PLAN.lock"
COMPLETION_LOCK = CONFIG / "V4_PACA_CA_VALIDATION_COMPLETED.lock"
SCRIPT = Path(__file__).resolve()

DEVELOPMENT_CLINICAL = [
    "tcga_paad_clinical.csv",
    "cptac_pdacc_clinical.csv",
    "gse62452_clinical.csv",
    "gse85916_clinical.csv",
    "emtab6134_clinical.csv",
    "gse183795_v3_development_clinical.csv",
    "gse57495_external_clinical.csv",
    "gse79668_external_clinical.csv",
    "paca_au_v2_confirmation_clinical.csv",
    "gse71729_v2_confirmation_clinical.csv",
]
LEGACY_COEFFICIENTS = {
    "LY6D": 0.05732909198204257,
    "PLA2G2A": -0.047712655849770504,
    "GHRL": -0.039489341632882174,
    "EREG": 0.012040159816165675,
    "FAM83A": 0.009252212452749452,
    "UCA1": 0.008674303585955168,
}
TIMES = np.asarray([12.0, 24.0, 36.0])
N_BOOT = 2000
BOOT_SEED = 410040
CV_REPEATS = 10
CV_FOLDS = 5
CV_SEED = 410168
CV_ALPHA = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_locks() -> tuple[dict, dict]:
    if COMPLETION_LOCK.exists():
        raise RuntimeError("PACA-CA validation has already been consumed")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    v2_spec = json.loads(V2_SPEC.read_text(encoding="utf-8"))
    eligibility = json.loads(ELIGIBILITY.read_text(encoding="utf-8"))
    plan = json.loads(ANALYSIS_LOCK.read_text(encoding="utf-8"))
    observed = {
        "model_sha256": sha256(MODEL),
        "v2_model_sha256": sha256(V2_MODEL),
        "eligibility_sha256": sha256(ELIGIBILITY),
        "expression_sha256": sha256(EXPRESSION),
        "clinical_sha256": sha256(CLINICAL),
        "analysis_script_sha256": sha256(SCRIPT),
    }
    for key, value in observed.items():
        if plan[key] != value:
            raise RuntimeError(f"Locked hash mismatch: {key}")
    if observed["model_sha256"] != spec["model_sha256"]:
        raise RuntimeError("Frozen v4 model hash mismatch")
    if observed["v2_model_sha256"] != v2_spec["model_sha256"]:
        raise RuntimeError("Frozen v2 predecessor hash mismatch")
    if eligibility["status"] != "V4_FRESH_ELIGIBILITY_COMPLETE_NO_RISK_SCORES":
        raise RuntimeError("Eligibility status is not pre-prediction")
    if eligibility["any_fresh_risk_score_generated"] is not False:
        raise RuntimeError("A fresh risk score was generated before locking")
    if eligibility["paca_ca"]["eligibility"] != "ELIGIBLE_PRIMARY_FRESH":
        raise RuntimeError("PACA-CA is not eligible")
    fixed = {
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "clinical_crossfit_repeats": CV_REPEATS,
        "clinical_crossfit_folds": CV_FOLDS,
        "clinical_crossfit_seed": CV_SEED,
        "clinical_cox_alpha": CV_ALPHA,
    }
    for key, value in fixed.items():
        if plan[key] != value:
            raise RuntimeError(f"Locked analysis constant differs: {key}")
    return spec, plan


def pooled_development_survival() -> tuple[np.ndarray, np.ndarray]:
    frames = [pd.read_csv(DATA / filename) for filename in DEVELOPMENT_CLINICAL]
    time = np.concatenate(
        [pd.to_numeric(frame["OS_TIME"], errors="coerce").to_numpy(float) for frame in frames]
    )
    event = np.concatenate(
        [pd.to_numeric(frame["OS_EVENT"], errors="coerce").to_numpy(float) for frame in frames]
    )
    valid = np.isfinite(time) & np.isfinite(event) & (time > 0) & np.isin(event, [0, 1])
    return time[valid], event[valid].astype(bool)


def legacy_score(expression: pd.DataFrame) -> tuple[np.ndarray, float]:
    genes = list(LEGACY_COEFFICIENTS)
    frame = expression.reindex(columns=genes).apply(pd.to_numeric, errors="coerce")
    available = frame.columns[frame.notna().any(axis=0)]
    location = frame.mean(axis=0)
    scale = frame.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    z = ((frame - location) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    coefficients = np.asarray([LEGACY_COEFFICIENTS[gene] for gene in genes])
    return z.to_numpy(float) @ coefficients, float(len(available) / len(genes))


def clinical_design(frame: pd.DataFrame) -> pd.DataFrame:
    age = pd.to_numeric(frame["AGE_AT_DIAGNOSIS"], errors="coerce")
    stage = frame["TUMOUR_STAGE"].fillna("").astype(str).str.upper().str.strip()
    grade = frame["TUMOUR_GRADE"].fillna("").astype(str).str.lower().str.strip()
    sex = frame["SEX"].fillna("").astype(str).str.lower().str.strip()
    early_stage = {"IA", "IB", "IIA", "T3N0M0"}
    advanced_stage = {"IIB", "III", "IV"}
    stage_known = stage.isin(early_stage | advanced_stage)
    favorable_grade = grade.isin(["well differentiated", "moderately differentiated"])
    poor_grade = grade.isin(["poorly differentiated", "undifferentiated", "g3"])
    grade_known = favorable_grade | poor_grade
    design = pd.DataFrame(index=frame.index)
    design["age_per_10y"] = (age.fillna(65.0) - 65.0) / 10.0
    design["age_missing"] = age.isna().astype(float)
    design["male"] = sex.eq("male").astype(float)
    design["stage_advanced"] = stage.isin(advanced_stage).astype(float)
    design["stage_missing"] = (~stage_known).astype(float)
    design["grade_poor"] = poor_grade.astype(float)
    design["grade_missing"] = (~grade_known).astype(float)
    return design.astype(float)


def discrimination(
    frame: pd.DataFrame,
    risk_column: str,
    train_time: np.ndarray,
    train_event: np.ndarray,
) -> dict[str, float]:
    return {
        "harrell_c": harrell_c(
            frame["OS_TIME"].to_numpy(float),
            frame["OS_EVENT"].to_numpy(bool),
            frame[risk_column].to_numpy(float),
        ),
        "uno_c": uno_c(
            train_time,
            train_event,
            frame["OS_TIME"].to_numpy(float),
            frame["OS_EVENT"].to_numpy(bool),
            frame[risk_column].to_numpy(float),
            tau=36.0,
        ),
    }


def cox_per_sd(frame: pd.DataFrame, risk_column: str = "v4_risk") -> dict[str, float]:
    use = frame[["OS_TIME", "OS_EVENT", risk_column]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()
    risk_sd = float(use[risk_column].std(ddof=0))
    if risk_sd <= 1e-12:
        raise RuntimeError("Risk has no dispersion")
    use["risk_z"] = (use[risk_column] - float(use[risk_column].mean())) / risk_sd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxPHFitter(penalizer=1e-6).fit(
            use[["OS_TIME", "OS_EVENT", "risk_z"]],
            duration_col="OS_TIME",
            event_col="OS_EVENT",
            robust=True,
        )
    coefficient = float(model.params_["risk_z"])
    se = float(model.standard_errors_["risk_z"])
    try:
        ph = proportional_hazard_test(
            model,
            use[["OS_TIME", "OS_EVENT", "risk_z"]],
            time_transform="rank",
        )
        ph_p = float(ph.summary.loc["risk_z", "p"])
    except Exception:
        ph_p = np.nan
    return {
        "n": int(len(use)),
        "events": int(use["OS_EVENT"].sum()),
        "log_hr_per_sd": coefficient,
        "log_hr_se": se,
        "hr_per_sd": float(np.exp(coefficient)),
        "hr_ci_low": float(np.exp(coefficient - 1.96 * se)),
        "hr_ci_high": float(np.exp(coefficient + 1.96 * se)),
        "hr_p": float(model.summary.loc["risk_z", "p"]),
        "ph_test_p": ph_p,
    }


def adjusted_cox(frame: pd.DataFrame) -> dict[str, float]:
    design = clinical_design(frame)
    use = pd.concat(
        [frame[["OS_TIME", "OS_EVENT", "v4_risk"]], design], axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    risk_sd = float(use["v4_risk"].std(ddof=0))
    use["risk_z"] = (use["v4_risk"] - float(use["v4_risk"].mean())) / risk_sd
    columns = ["OS_TIME", "OS_EVENT", "risk_z", *design.columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxPHFitter(penalizer=0.01).fit(
            use[columns],
            duration_col="OS_TIME",
            event_col="OS_EVENT",
            robust=True,
        )
    coefficient = float(model.params_["risk_z"])
    se = float(model.standard_errors_["risk_z"])
    return {
        "n": int(len(use)),
        "events": int(use["OS_EVENT"].sum()),
        "hr_per_sd": float(np.exp(coefficient)),
        "hr_ci_low": float(np.exp(coefficient - 1.96 * se)),
        "hr_ci_high": float(np.exp(coefficient + 1.96 * se)),
        "hr_p": float(model.summary.loc["risk_z", "p"]),
        "covariates": list(design.columns),
        "penalizer": 0.01,
    }


def crossfit_clinical_models(frame: pd.DataFrame) -> pd.DataFrame:
    design = clinical_design(frame)
    risk = frame["v4_risk"].to_numpy(float)
    risk_z = (risk - risk.mean()) / (risk.std(ddof=0) or 1.0)
    combined = design.copy()
    combined["frozen_v4_risk"] = risk_z
    event = frame["OS_EVENT"].to_numpy(bool)
    survival = make_y(frame["OS_TIME"].to_numpy(float), event)
    clinical_predictions = np.zeros((len(frame), CV_REPEATS), dtype=float)
    combined_predictions = np.zeros((len(frame), CV_REPEATS), dtype=float)
    for repeat in range(CV_REPEATS):
        splitter = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=CV_SEED + repeat,
        )
        for train_index, test_index in splitter.split(design, event.astype(int)):
            clinical_model = CoxPHSurvivalAnalysis(alpha=CV_ALPHA).fit(
                design.iloc[train_index].to_numpy(float), survival[train_index]
            )
            combined_model = CoxPHSurvivalAnalysis(alpha=CV_ALPHA).fit(
                combined.iloc[train_index].to_numpy(float), survival[train_index]
            )
            clinical_predictions[test_index, repeat] = clinical_model.predict(
                design.iloc[test_index].to_numpy(float)
            )
            combined_predictions[test_index, repeat] = combined_model.predict(
                combined.iloc[test_index].to_numpy(float)
            )
    return pd.DataFrame(
        {
            "clinical_oof_risk": clinical_predictions.mean(axis=1),
            "combined_oof_risk": combined_predictions.mean(axis=1),
        },
        index=frame.index,
    )


def time_auc_rows(
    frame: pd.DataFrame,
    risk_columns: list[str],
    train_time: np.ndarray,
    train_event: np.ndarray,
) -> list[dict]:
    y_train = make_y(train_time, train_event)
    y_test = make_y(frame["OS_TIME"].to_numpy(float), frame["OS_EVENT"].to_numpy(bool))
    valid_times = TIMES[(TIMES > frame["OS_TIME"].min()) & (TIMES < frame["OS_TIME"].max())]
    rows: list[dict] = []
    for column in risk_columns:
        auc, _ = cumulative_dynamic_auc(
            y_train, y_test, frame[column].to_numpy(float), valid_times
        )
        rows.extend(
            {
                "model": column,
                "time_months": float(time),
                "auc": float(value),
            }
            for time, value in zip(valid_times, auc, strict=True)
        )
    return rows


def km_tertile_outputs(frame: pd.DataFrame) -> tuple[dict, list[dict]]:
    low = frame[frame["risk_tier"].eq("low")]
    high = frame[frame["risk_tier"].eq("high")]
    test = logrank_test(
        high["OS_TIME"],
        low["OS_TIME"],
        event_observed_A=high["OS_EVENT"],
        event_observed_B=low["OS_EVENT"],
    )
    binary = frame[frame["risk_tier"].isin(["low", "high"])].copy()
    binary["high_tier"] = binary["risk_tier"].eq("high").astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxPHFitter(penalizer=1e-6).fit(
            binary[["OS_TIME", "OS_EVENT", "high_tier"]],
            duration_col="OS_TIME",
            event_col="OS_EVENT",
            robust=True,
        )
    coefficient = float(model.params_["high_tier"])
    se = float(model.standard_errors_["high_tier"])
    curves = []
    for tier in ["low", "middle", "high"]:
        group = frame[frame["risk_tier"].eq(tier)]
        km = KaplanMeierFitter().fit(group["OS_TIME"], group["OS_EVENT"])
        for time, survival in zip(
            km.survival_function_.index,
            km.survival_function_.iloc[:, 0],
            strict=True,
        ):
            curves.append(
                {
                    "risk_tier": tier,
                    "time_months": float(time),
                    "survival": float(survival),
                    "n": int(len(group)),
                    "events": int(group["OS_EVENT"].sum()),
                }
            )
    return (
        {
            "n_low": int(len(low)),
            "n_high": int(len(high)),
            "high_vs_low_logrank_p": float(test.p_value),
            "high_vs_low_hr": float(np.exp(coefficient)),
            "high_vs_low_hr_ci_low": float(np.exp(coefficient - 1.96 * se)),
            "high_vs_low_hr_ci_high": float(np.exp(coefficient + 1.96 * se)),
        },
        curves,
    )


def subgroup_definitions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    age = pd.to_numeric(frame["AGE_AT_DIAGNOSIS"], errors="coerce")
    stage = frame["TUMOUR_STAGE"].fillna("").astype(str).str.upper().str.strip()
    grade = frame["TUMOUR_GRADE"].fillna("").astype(str).str.lower().str.strip()
    sex = frame["SEX"].fillna("").astype(str).str.lower().str.strip()
    specimen = frame["SPECIMEN_TYPE"].fillna("").astype(str)
    reads = pd.to_numeric(frame["TOTAL_READ_COUNT"], errors="coerce")
    definitions: dict[str, pd.Series] = {
        "age_lt65": age.lt(65) & age.notna(),
        "age_ge65": age.ge(65) & age.notna(),
        "female": sex.eq("female"),
        "male": sex.eq("male"),
        "stage_early_IA_IB_IIA": stage.isin(["IA", "IB", "IIA", "T3N0M0"]),
        "stage_advanced_IIB_III_IV": stage.isin(["IIB", "III", "IV"]),
        "grade_well_or_moderate": grade.isin(
            ["well differentiated", "moderately differentiated"]
        ),
        "grade_poor_or_undifferentiated": grade.isin(
            ["poorly differentiated", "undifferentiated", "g3"]
        ),
        "specimen_laser_or_other": specimen.eq("Primary tumour - other"),
        "specimen_solid_tissue": specimen.eq("Primary tumour - solid tissue"),
        "reads_lt5m": reads.lt(5_000_000) & reads.ge(0),
        "reads_ge5m": reads.ge(5_000_000),
    }
    return definitions


def subgroup_rows(
    frame: pd.DataFrame, train_time: np.ndarray, train_event: np.ndarray
) -> list[dict]:
    rows = []
    for label, mask in subgroup_definitions(frame).items():
        group = frame[mask].copy()
        row: dict = {
            "subgroup": label,
            "n": int(len(group)),
            "events": int(group["OS_EVENT"].sum()),
        }
        if len(group) >= 15 and group["OS_EVENT"].sum() >= 8:
            try:
                row.update(discrimination(group, "v4_risk", train_time, train_event))
                row.update(cox_per_sd(group))
            except Exception as exc:
                row["error"] = str(exc)
        else:
            row["error"] = "descriptive_only_below_prespecified_n_or_event_threshold"
        rows.append(row)
    return rows


def bootstrap_rows(
    frame: pd.DataFrame, train_time: np.ndarray, train_event: np.ndarray
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED)
    risk_columns = [
        "v4_risk",
        "v2_risk",
        "legacy_risk",
        "clinical_oof_risk",
        "combined_oof_risk",
    ]
    rows = []
    for replicate in range(N_BOOT):
        index = rng.integers(0, len(frame), len(frame))
        sample = frame.iloc[index]
        if sample["OS_EVENT"].sum() == 0 or sample["OS_TIME"].nunique() < 2:
            continue
        row: dict[str, float | int] = {"replicate": replicate}
        try:
            for column in risk_columns:
                metric = discrimination(sample, column, train_time, train_event)
                row[f"{column}_harrell_c"] = metric["harrell_c"]
                row[f"{column}_uno_c"] = metric["uno_c"]
        except Exception:
            continue
        row["delta_uno_v4_vs_v2"] = row["v4_risk_uno_c"] - row["v2_risk_uno_c"]
        row["delta_harrell_v4_vs_v2"] = (
            row["v4_risk_harrell_c"] - row["v2_risk_harrell_c"]
        )
        row["delta_uno_v4_vs_legacy"] = (
            row["v4_risk_uno_c"] - row["legacy_risk_uno_c"]
        )
        row["delta_uno_combined_vs_clinical"] = (
            row["combined_oof_risk_uno_c"] - row["clinical_oof_risk_uno_c"]
        )
        row["delta_uno_combined_vs_v4"] = (
            row["combined_oof_risk_uno_c"] - row["v4_risk_uno_c"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_summary(bootstrap: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {"valid_replicates": int(len(bootstrap))}
    for column in bootstrap.columns:
        if column == "replicate":
            continue
        values = bootstrap[column].dropna().to_numpy(float)
        output[f"{column}_ci_low"] = float(np.quantile(values, 0.025))
        output[f"{column}_ci_high"] = float(np.quantile(values, 0.975))
        if column.startswith("delta_"):
            output[f"{column}_probability_gt_zero"] = float(np.mean(values > 0))
    return output


def classify_external_gate(
    primary: dict, adjusted: dict, subgroups: list[dict]
) -> tuple[str, dict[str, bool]]:
    clear_reversal = any(
        row.get("n", 0) >= 40
        and row.get("events", 0) >= 20
        and np.isfinite(row.get("hr_ci_high", np.nan))
        and row["hr_ci_high"] < 1.0
        for row in subgroups
    )
    green = {
        "Harrell C >= 0.60": primary["harrell_c"] >= 0.60,
        "Uno C at 36 months >= 0.60": primary["uno_c"] >= 0.60,
        "unadjusted HR 95% CI lower > 1": primary["hr_ci_low"] > 1.0,
        "adjusted HR 95% CI lower > 1": adjusted["hr_ci_low"] > 1.0,
        "no clear reversal in a large prespecified subgroup": not clear_reversal,
    }
    amber = {
        "Harrell C >= 0.56": primary["harrell_c"] >= 0.56,
        "Uno C at 36 months >= 0.56": primary["uno_c"] >= 0.56,
        "unadjusted HR per SD > 1": primary["hr_per_sd"] > 1.0,
        "adjusted HR per SD > 1": adjusted["hr_per_sd"] > 1.0,
        "no clear reversal in a large prespecified subgroup": not clear_reversal,
    }
    if all(green.values()):
        return "GREEN_INDEPENDENT_VALIDATION", green
    if all(amber.values()):
        return "AMBER_DIRECTIONAL_SUPPORT", {**green, **amber}
    return "RED_NOT_SUPPORTED", amber


def main() -> None:
    spec, plan = verify_locks()
    expression = pd.read_parquet(EXPRESSION)
    expression.index = expression.index.astype(str)
    clinical = pd.read_csv(CLINICAL).set_index("SAMPLE_ID", drop=False)
    clinical.index = clinical.index.astype(str)
    expression = expression.reindex(clinical.index)
    if expression.isna().all(axis=1).any():
        raise RuntimeError("An eligible PACA-CA sample has no expression")

    model = LockedMarginRankModel.load(MODEL)
    prediction = model.predict_from_expression(expression)
    if not prediction["absolute_survival_disabled"].all():
        raise RuntimeError("The v4 bundle enabled absolute survival")
    if any(column.startswith("survival_") for column in prediction.columns):
        raise RuntimeError("The v4 bundle emitted an absolute survival column")
    if prediction["input_degraded"].any() or prediction["reference_coverage_warning"].any():
        raise RuntimeError("PACA-CA input failed the locked inference coverage rule")
    if prediction["raw_risk"].nunique() < 0.95 * len(prediction):
        raise RuntimeError("Fresh v4 risk is unexpectedly coarse")

    v2 = LockedPDACModel.load(V2_MODEL)
    v2_universe = list(v2.bundle["rank_universe"])
    v2_ranks = SingleSampleRankTransformer().fit(v2_universe).transform(
        expression, features=v2_universe
    )
    v2_prediction = v2.predict_from_rank(v2_ranks)
    legacy, legacy_coverage = legacy_score(expression)

    frame = clinical.join(prediction, how="left")
    frame["v4_risk"] = frame["raw_risk"]
    frame["v2_risk"] = v2_prediction["raw_risk"].to_numpy(float)
    frame["legacy_risk"] = legacy
    crossfit = crossfit_clinical_models(frame)
    frame = frame.join(crossfit)
    train_time, train_event = pooled_development_survival()

    risk_columns = [
        "v4_risk",
        "v2_risk",
        "legacy_risk",
        "clinical_oof_risk",
        "combined_oof_risk",
    ]
    metric_rows = []
    for column in risk_columns:
        metric_rows.append(
            {
                "model": column,
                **discrimination(frame, column, train_time, train_event),
                **cox_per_sd(frame, column),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "41_v4_paca_ca_metrics.csv", index=False)
    primary = metrics[metrics["model"].eq("v4_risk")].iloc[0].to_dict()
    adjusted = adjusted_cox(frame)

    time_rows = time_auc_rows(frame, risk_columns, train_time, train_event)
    pd.DataFrame(time_rows).to_csv(
        RESULTS / "41_v4_paca_ca_time_auc.csv", index=False
    )
    tertile, km_rows = km_tertile_outputs(frame)
    pd.DataFrame(km_rows).to_csv(
        RESULTS / "41_v4_paca_ca_km_curves.csv", index=False
    )
    subgroups = subgroup_rows(frame, train_time, train_event)
    pd.DataFrame(subgroups).to_csv(
        RESULTS / "41_v4_paca_ca_subgroups.csv", index=False
    )

    bootstrap = bootstrap_rows(frame, train_time, train_event)
    if len(bootstrap) < int(0.95 * N_BOOT):
        raise RuntimeError("Fewer than 95% of locked bootstrap replicates were valid")
    bootstrap.to_csv(RESULTS / "41_v4_paca_ca_bootstrap.csv", index=False)
    boot = bootstrap_summary(bootstrap)

    external_gate, external_conditions = classify_external_gate(
        primary, adjusted, subgroups
    )
    v2_row = metrics[metrics["model"].eq("v2_risk")].iloc[0]
    clinical_row = metrics[metrics["model"].eq("clinical_oof_risk")].iloc[0]
    combined_row = metrics[metrics["model"].eq("combined_oof_risk")].iloc[0]
    readiness_conditions = {
        "fresh independent validation is green": external_gate
        == "GREEN_INDEPENDENT_VALIDATION",
        "bootstrap lower 95% bound for v4 Uno C >= 0.55": bool(
            boot["v4_risk_uno_c_ci_low"] >= 0.55
        ),
        "v4 Uno gain over frozen v2 predecessor >= 0.01": bool(
            primary["uno_c"] - float(v2_row["uno_c"]) >= 0.01
        ),
        "bootstrap probability v4 Uno > v2 Uno >= 0.80": bool(
            boot["delta_uno_v4_vs_v2_probability_gt_zero"] >= 0.80
        ),
        "combined Uno gain over cross-fitted clinical model >= 0.015": bool(
            float(combined_row["uno_c"] - clinical_row["uno_c"]) >= 0.015
        ),
        "bootstrap probability combined Uno > clinical Uno >= 0.80": bool(
            boot["delta_uno_combined_vs_clinical_probability_gt_zero"] >= 0.80
        ),
    }
    top_readiness = (
        "GREEN_COMPUTATIONAL_TOP_TIER_READY"
        if all(readiness_conditions.values())
        else "AMBER_ADDITIONAL_EVIDENCE_OR_REFRAMING_NEEDED"
    )

    prediction_columns = [
        "SAMPLE_ID",
        "PATIENT_ID",
        "OS_TIME",
        "OS_EVENT",
        "v4_risk",
        "within_batch_z",
        "risk_percentile",
        "risk_tier",
        "selected_gene_coverage",
        "anchor_gene_coverage",
        "reference_gene_coverage",
        "v2_risk",
        "legacy_risk",
        "clinical_oof_risk",
        "combined_oof_risk",
    ]
    frame[prediction_columns].to_csv(
        RESULTS / "41_v4_paca_ca_predictions.csv", index=False
    )
    summary = {
        "status": "PACA_CA_VALIDATED_EXACTLY_ONCE_NO_RETUNING",
        "external_validation_gate": external_gate,
        "external_gate_conditions": external_conditions,
        "computational_top_tier_readiness": top_readiness,
        "readiness_conditions": readiness_conditions,
        "cohort": "ICGC PACA-CA",
        "n": int(len(frame)),
        "events": int(frame["OS_EVENT"].sum()),
        "primary": primary,
        "adjusted": adjusted,
        "comparators": metrics.set_index("model").to_dict(orient="index"),
        "legacy_gene_coverage": legacy_coverage,
        "tertile_analysis": tertile,
        "time_dependent_auc": time_rows,
        "subgroups": subgroups,
        "bootstrap": boot,
        "absolute_survival_values_emitted": False,
        "paca_ca_used_for_tuning": False,
        "endpoint_or_exclusion_changed_after_scoring": False,
        "interpretation": plan["gate_interpretation"][external_gate],
        "hashes": {
            "model": sha256(MODEL),
            "v2_model": sha256(V2_MODEL),
            "analysis_script": sha256(SCRIPT),
            "analysis_lock": sha256(ANALYSIS_LOCK),
            "eligibility": sha256(ELIGIBILITY),
            "expression": sha256(EXPRESSION),
            "clinical": sha256(CLINICAL),
        },
    }
    result_path = RESULTS / "41_v4_paca_ca_validation_gate.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    completion = {
        "status": "V4_PACA_CA_VALIDATION_CONSUMED_ONCE",
        "result_gate": external_gate,
        "top_tier_readiness": top_readiness,
        "model_sha256": spec["model_sha256"],
        "analysis_script_sha256": sha256(SCRIPT),
        "result_sha256": sha256(result_path),
        "risk_prediction_performed": True,
        "validation_set_may_be_reused_for_tuning": False,
    }
    COMPLETION_LOCK.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
