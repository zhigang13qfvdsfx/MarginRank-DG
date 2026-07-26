from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from transsurv.inference import LockedPDACModel
from transsurv.marginrank import LockedMarginRankModel
from transsurv.metrics import harrell_c, uno_c
from transsurv.transforms import SingleSampleRankTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
MODEL = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"
V2_MODEL = ROOT / "models" / "predecessors" / "transsurv_pdac_v2_1_locked.joblib"
EXPRESSION = DATA / "prince_v4_exploratory_expression.parquet"
CLINICAL = DATA / "prince_v4_exploratory_clinical.csv"
ELIGIBILITY = RESULTS / "47_prince_exploratory_eligibility.json"
PLAN = CONFIG / "V4_PRINCE_EXPLORATORY_ONE_TIME_PLAN.lock"
COMPLETION = CONFIG / "V4_PRINCE_EXPLORATORY_COMPLETED.lock"

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    coefficient = np.asarray([LEGACY_COEFFICIENTS[gene] for gene in genes])
    return z.to_numpy(float) @ coefficient, float(len(available) / len(genes))


def cox_per_sd(frame: pd.DataFrame, risk_column: str) -> dict[str, float]:
    use = frame[["OS_TIME", "OS_EVENT", risk_column]].copy()
    sd = float(use[risk_column].std(ddof=0))
    use["risk_z"] = (use[risk_column] - use[risk_column].mean()) / sd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxPHFitter(penalizer=1e-5).fit(
            use[["OS_TIME", "OS_EVENT", "risk_z"]],
            duration_col="OS_TIME",
            event_col="OS_EVENT",
            robust=True,
        )
    coefficient = float(model.params_["risk_z"])
    se = float(model.standard_errors_["risk_z"])
    try:
        ph = proportional_hazard_test(
            model, use[["OS_TIME", "OS_EVENT", "risk_z"]], time_transform="rank"
        )
        ph_p = float(ph.summary.loc["risk_z", "p"])
    except Exception:
        ph_p = float("nan")
    return {
        "hr_per_sd": float(np.exp(coefficient)),
        "hr_ci_low": float(np.exp(coefficient - 1.96 * se)),
        "hr_ci_high": float(np.exp(coefficient + 1.96 * se)),
        "hr_p": float(model.summary.loc["risk_z", "p"]),
        "ph_test_p": ph_p,
    }


def adjusted_cox(frame: pd.DataFrame) -> dict[str, object]:
    use = frame[["OS_TIME", "OS_EVENT", "v4_risk"]].copy()
    use["risk_z"] = (use["v4_risk"] - use["v4_risk"].mean()) / use["v4_risk"].std(ddof=0)
    age = pd.to_numeric(frame["AGE_AT_DIAGNOSIS"], errors="coerce")
    use["age_per_10y"] = (age.fillna(age.median()) - age.median()) / 10.0
    use["male"] = frame["SEX"].fillna("").str.lower().eq("male").astype(float)
    use["stage_iv"] = frame["CLINICAL_STAGE"].fillna("").astype(str).str.upper().eq("IV").astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxPHFitter(penalizer=0.10).fit(
            use.drop(columns="v4_risk"),
            duration_col="OS_TIME",
            event_col="OS_EVENT",
            robust=True,
        )
    coefficient = float(model.params_["risk_z"])
    se = float(model.standard_errors_["risk_z"])
    return {
        "hr_per_sd": float(np.exp(coefficient)),
        "hr_ci_low": float(np.exp(coefficient - 1.96 * se)),
        "hr_ci_high": float(np.exp(coefficient + 1.96 * se)),
        "hr_p": float(model.summary.loc["risk_z", "p"]),
        "penalizer": 0.10,
        "covariates": ["age_per_10y", "male", "stage_iv"],
    }


def model_metrics(
    frame: pd.DataFrame,
    column: str,
    train_time: np.ndarray,
    train_event: np.ndarray,
    tau: float,
) -> dict[str, float]:
    return {
        "harrell_c": harrell_c(
            frame["OS_TIME"].to_numpy(float),
            frame["OS_EVENT"].to_numpy(bool),
            frame[column].to_numpy(float),
        ),
        "uno_c": uno_c(
            train_time,
            train_event,
            frame["OS_TIME"].to_numpy(float),
            frame["OS_EVENT"].to_numpy(bool),
            frame[column].to_numpy(float),
            tau=tau,
        ),
        **cox_per_sd(frame, column),
    }


def subgroup_metrics(
    frame: pd.DataFrame,
    train_time: np.ndarray,
    train_event: np.ndarray,
    tau: float,
) -> list[dict[str, object]]:
    definitions = {
        "female": frame["SEX"].fillna("").str.lower().eq("female"),
        "male": frame["SEX"].fillna("").str.lower().eq("male"),
        "stage_iv": frame["CLINICAL_STAGE"].fillna("").astype(str).str.upper().eq("IV"),
        "stage_i_to_iii": ~frame["CLINICAL_STAGE"].fillna("").astype(str).str.upper().eq("IV"),
        "liver_biopsy": frame["BIOPSY_SITE"].fillna("").str.lower().eq("liver"),
        "other_or_unknown_biopsy": ~frame["BIOPSY_SITE"].fillna("").str.lower().eq("liver"),
    }
    rows = []
    for name, mask in definitions.items():
        use = frame.loc[mask]
        if len(use) < 20 or use["OS_EVENT"].sum() < 5:
            continue
        rows.append(
            {
                "subgroup": name,
                "n": int(len(use)),
                "events": int(use["OS_EVENT"].sum()),
                **model_metrics(use, "v4_risk", train_time, train_event, tau),
            }
        )
    return rows


def main() -> None:
    if COMPLETION.exists():
        raise RuntimeError("PRINCE exploratory endpoint has already been consumed")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    eligibility = json.loads(ELIGIBILITY.read_text(encoding="utf-8"))
    if plan["status"] != "LOCKED_BEFORE_FIRST_PRINCE_RISK_SCORE":
        raise RuntimeError("Unexpected PRINCE one-time plan")
    if not eligibility["exploratory_stress_eligible"] or eligibility["risk_scores_generated"]:
        raise RuntimeError("PRINCE is not locked for outcome-free exploratory scoring")
    hashes = {
        "model_sha256": sha256(MODEL),
        "v2_model_sha256": sha256(V2_MODEL),
        "eligibility_sha256": sha256(ELIGIBILITY),
        "expression_sha256": sha256(EXPRESSION),
        "clinical_sha256": sha256(CLINICAL),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    for key, value in hashes.items():
        if plan[key].lower() != value.lower():
            raise RuntimeError(f"Locked PRINCE hash mismatch: {key}")

    expression = pd.read_parquet(EXPRESSION)
    expression.index = expression.index.astype(str)
    clinical = pd.read_csv(CLINICAL)
    clinical["SAMPLE_ID"] = clinical["SAMPLE_ID"].astype(str)
    clinical = clinical.set_index("SAMPLE_ID", drop=False).loc[expression.index].copy()
    if len(clinical) != 63 or clinical["OS_EVENT"].sum() != 15:
        raise RuntimeError("PRINCE endpoint count changed")

    v4 = LockedMarginRankModel.load(MODEL)
    prediction = v4.predict_from_expression(expression)
    if prediction["input_degraded"].any() or prediction["reference_coverage_warning"].any():
        raise RuntimeError("PRINCE input failed frozen v4 coverage rules")
    if any(column.startswith("survival_") for column in prediction):
        raise RuntimeError("Absolute survival values were unexpectedly emitted")
    v2 = LockedPDACModel.load(V2_MODEL)
    v2_universe = list(v2.bundle["rank_universe"])
    v2_rank = SingleSampleRankTransformer().fit(v2_universe).transform(
        expression, features=v2_universe
    )
    v2_prediction = v2.predict_from_rank(v2_rank)
    legacy, legacy_coverage = legacy_score(expression)

    frame = clinical.copy()
    frame["v4_risk"] = prediction["raw_risk"].to_numpy(float)
    frame["v2_risk"] = v2_prediction["raw_risk"].to_numpy(float)
    frame["legacy_risk"] = legacy
    train_time, train_event = pooled_development_survival()
    tau = float(plan["uno_tau_months"])
    metrics = {
        column: model_metrics(frame, column, train_time, train_event, tau)
        for column in ["v4_risk", "v2_risk", "legacy_risk"]
    }
    adjusted = adjusted_cox(frame)
    subgroups = subgroup_metrics(frame, train_time, train_event, tau)

    rng = np.random.default_rng(int(plan["bootstrap_seed"]))
    rows = []
    for replicate in range(int(plan["bootstrap_replicates"])):
        index = rng.integers(0, len(frame), size=len(frame))
        sample = frame.iloc[index]
        if sample["OS_EVENT"].sum() < 2 or sample["OS_TIME"].nunique() < 3:
            continue
        row: dict[str, float | int] = {"replicate": replicate}
        try:
            for column in ["v4_risk", "v2_risk", "legacy_risk"]:
                row[f"{column}_harrell"] = harrell_c(
                    sample["OS_TIME"].to_numpy(float),
                    sample["OS_EVENT"].to_numpy(bool),
                    sample[column].to_numpy(float),
                )
                row[f"{column}_uno"] = uno_c(
                    train_time,
                    train_event,
                    sample["OS_TIME"].to_numpy(float),
                    sample["OS_EVENT"].to_numpy(bool),
                    sample[column].to_numpy(float),
                    tau=tau,
                )
        except Exception:
            continue
        row["delta_uno_v4_vs_v2"] = row["v4_risk_uno"] - row["v2_risk_uno"]
        row["delta_uno_v4_vs_legacy"] = row["v4_risk_uno"] - row["legacy_risk_uno"]
        rows.append(row)
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(RESULTS / "48_v4_prince_bootstrap.csv", index=False)
    v4 = metrics["v4_risk"]
    harrell_ci = np.quantile(bootstrap["v4_risk_harrell"], [0.025, 0.975]).tolist()
    uno_ci = np.quantile(bootstrap["v4_risk_uno"], [0.025, 0.975]).tolist()
    gate = plan["exploratory_supportive_gate"]
    large_subgroups = [row for row in subgroups if row["n"] >= 25 and row["events"] >= 5]
    reversal = any(row["harrell_c"] < 0.45 for row in large_subgroups)
    conditions = {
        "harrell_c": v4["harrell_c"] >= gate["minimum_harrell_c"],
        "uno_c": v4["uno_c"] >= gate["minimum_uno_c"],
        "hr_point": v4["hr_per_sd"] >= gate["minimum_hr_per_sd"],
        "hr_lower_relaxed": v4["hr_ci_low"] >= gate["minimum_hr_ci_low"],
        "bootstrap_harrell_lower": harrell_ci[0]
        >= gate["minimum_bootstrap_harrell_lower_95"],
        "ph_assumption": bool(np.isnan(v4["ph_test_p"]) or v4["ph_test_p"] > 0.05),
        "no_large_subgroup_reversal": not reversal,
        "coverage": bool(
            prediction["selected_gene_coverage"].min() == 1.0
            and prediction["anchor_gene_coverage"].min() == 1.0
        ),
    }

    output_columns = [
        "PATIENT_ID",
        "SAMPLE_ID",
        "OS_TIME",
        "OS_EVENT",
        "AGE_AT_DIAGNOSIS",
        "SEX",
        "CLINICAL_STAGE",
        "ICI_PATHWAY",
        "ICI_RX",
        "BIOPSY_SITE",
        "v4_risk",
        "v2_risk",
        "legacy_risk",
    ]
    frame[output_columns].to_csv(RESULTS / "48_v4_prince_predictions.csv", index=False)
    pd.DataFrame([{"model": name, **value} for name, value in metrics.items()]).to_csv(
        RESULTS / "48_v4_prince_metrics.csv", index=False
    )
    pd.DataFrame(subgroups).to_csv(RESULTS / "48_v4_prince_subgroups.csv", index=False)

    summary = {
        "status": "PRINCE_EXPLORATORY_TRANSPORTABILITY_CONSUMED_ONCE",
        "exploratory_gate": "SUPPORTIVE" if all(conditions.values()) else "NOT_SUPPORTIVE",
        "conditions": conditions,
        "cohort": "PRINCE pretreatment metastatic PDAC",
        "n": int(len(frame)),
        "events": int(frame["OS_EVENT"].sum()),
        "uno_tau_months": tau,
        "primary": v4,
        "adjusted": adjusted,
        "comparators": metrics,
        "bootstrap": {
            "valid_replicates": int(len(bootstrap)),
            "harrell_ci_95": harrell_ci,
            "uno_ci_95": uno_ci,
            "delta_uno_v4_vs_v2_ci_95": np.quantile(
                bootstrap["delta_uno_v4_vs_v2"], [0.025, 0.975]
            ).tolist(),
            "probability_v4_uno_gt_v2": float(
                (bootstrap["delta_uno_v4_vs_v2"] > 0).mean()
            ),
            "delta_uno_v4_vs_legacy_ci_95": np.quantile(
                bootstrap["delta_uno_v4_vs_legacy"], [0.025, 0.975]
            ).tolist(),
            "probability_v4_uno_gt_legacy": float(
                (bootstrap["delta_uno_v4_vs_legacy"] > 0).mean()
            ),
        },
        "subgroups": subgroups,
        "legacy_gene_coverage": legacy_coverage,
        "primary_validation_claimed": False,
        "clinical_utility_claimed": False,
        "absolute_survival_values_emitted": False,
        "model_changed": False,
        "paca_ca_used_for_tuning": False,
        "interpretation": "Underpowered exploratory transportability stress test in pretreatment metastatic PDAC; it cannot replace the event-qualified PACA-CA primary validation.",
        "hashes": hashes,
    }
    result_path = RESULTS / "48_v4_prince_exploratory_gate.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    COMPLETION.write_text(
        json.dumps(
            {
                "status": "V4_PRINCE_EXPLORATORY_CONSUMED_ONCE",
                "result": summary["exploratory_gate"],
                "events": int(frame["OS_EVENT"].sum()),
                "may_be_reused_for_tuning": False,
                "may_be_claimed_as_primary_validation": False,
                "result_sha256": sha256(result_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
