from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transsurv.learners import PooledSurvivalLearner
from transsurv.metrics import uno_c
from transsurv.model import DomainData


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
LOCK = CONFIG / "V4_MARGINRANK_SELECTION.lock"
SELECTED = CONFIG / "v4_selected_configuration.json"


def load_v4_module():
    path = Path(__file__).with_name("36_v4_marginrank_nested.py")
    spec = importlib.util.spec_from_file_location("v4_marginrank_missing", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left, ddof=0) <= 1e-12 or np.std(right, ddof=0) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def reconstruct_candidate(v4, selected: dict):
    for candidate in v4.CANDIDATES:
        if candidate.name == selected["candidate"]:
            return candidate
    raise RuntimeError("Selected v4 candidate is absent from the locked grid")


def fit_fold(v4, candidate, train_names, test_name, domains, z_table):
    feature_train_names = list(train_names)
    if candidate.include_auxiliary:
        feature_train_names.append(v4.AUXILIARY)
    definition = v4.feature_definition(
        feature_train_names, candidate.feature, domains, z_table
    )
    train = {
        name: DomainData(
            x=v4.transform_features(domains[name].x, definition),
            time=domains[name].time,
            event=domains[name].event,
        )
        for name in feature_train_names
    }
    if candidate.learner != "pooled_ridge":
        raise RuntimeError("Missingness script expects the selected pooled-ridge learner")
    model = PooledSurvivalLearner("coxph", {"alpha": candidate.alpha}).fit(train)
    full_x = v4.transform_features(domains[test_name].x, definition)
    full_risk = model.predict(full_x)
    return model, definition, full_risk


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    fixed_gate = json.loads(
        (RESULTS / "37_v4_fixed_selection_gate.json").read_text(encoding="utf-8")
    )
    if fixed_gate["gate"] != "PASS" or selected["selection_gate"] != "PASS":
        raise RuntimeError("Fixed v4 configuration did not pass")
    v4 = load_v4_module()
    candidate = reconstruct_candidate(v4, selected)
    domains, genes = v4.load_domains()
    z_table = v4.precompute_domain_z(domains, genes)
    plan = lock["missingness_gate"]
    rng = np.random.default_rng(int(plan["seed"]))
    rows = []
    systematic_rows = []

    for fold, test_name in enumerate(v4.PRIMARY, start=1):
        train_names = [name for name in v4.PRIMARY if name != test_name]
        train_time = np.concatenate([domains[name].time for name in train_names])
        train_event = np.concatenate([domains[name].event for name in train_names])
        model, definition, full_risk = fit_fold(
            v4, candidate, train_names, test_name, domains, z_table
        )
        full_uno = uno_c(
            train_time,
            train_event,
            domains[test_name].time,
            domains[test_name].event,
            full_risk,
        )
        input_genes = sorted(set(definition["genes"]) | set(definition["anchors"]))
        for level in plan["levels"]:
            n_mask = max(1, int(np.ceil(float(level) * len(input_genes))))
            for replicate in range(int(plan["replicates_per_level"])):
                masked_genes = rng.choice(input_genes, size=n_mask, replace=False).tolist()
                masked_rank = domains[test_name].x.copy()
                masked_rank.loc[:, masked_genes] = np.nan
                masked_x = v4.transform_features(masked_rank, definition)
                masked_risk = model.predict(masked_x)
                masked_uno = uno_c(
                    train_time,
                    train_event,
                    domains[test_name].time,
                    domains[test_name].event,
                    masked_risk,
                )
                rows.append(
                    {
                        "cohort": test_name,
                        "level": float(level),
                        "replicate": replicate,
                        "n_input_genes": len(input_genes),
                        "n_masked_genes": n_mask,
                        "masked_genes": "|".join(masked_genes),
                        "risk_correlation": safe_correlation(full_risk, masked_risk),
                        "full_uno_c": full_uno,
                        "masked_uno_c": masked_uno,
                        "uno_drop": full_uno - masked_uno,
                    }
                )
        for gene in input_genes:
            masked_rank = domains[test_name].x.copy()
            masked_rank.loc[:, gene] = np.nan
            masked_risk = model.predict(v4.transform_features(masked_rank, definition))
            masked_uno = uno_c(
                train_time,
                train_event,
                domains[test_name].time,
                domains[test_name].event,
                masked_risk,
            )
            systematic_rows.append(
                {
                    "cohort": test_name,
                    "missing_gene": gene,
                    "risk_correlation": safe_correlation(full_risk, masked_risk),
                    "full_uno_c": full_uno,
                    "masked_uno_c": masked_uno,
                    "uno_drop": full_uno - masked_uno,
                }
            )
        print(f"[{fold}/{len(v4.PRIMARY)}] missingness complete: {test_name}", flush=True)

    replicate_frame = pd.DataFrame(rows)
    replicate_frame.to_csv(
        RESULTS / "38_v4_missingness_replicates.csv", index=False
    )
    systematic = pd.DataFrame(systematic_rows)
    systematic.to_csv(
        RESULTS / "38_v4_leave_one_gene_out.csv", index=False
    )
    level_summary = (
        replicate_frame.groupby("level")
        .agg(
            mean_risk_correlation=("risk_correlation", "mean"),
            minimum_cohort_mean_correlation=(
                "risk_correlation",
                lambda values: float(
                    replicate_frame.loc[values.index]
                    .groupby("cohort")["risk_correlation"]
                    .mean()
                    .min()
                ),
            ),
            mean_uno_drop=("uno_drop", "mean"),
            maximum_cohort_mean_uno_drop=(
                "uno_drop",
                lambda values: float(
                    replicate_frame.loc[values.index]
                    .groupby("cohort")["uno_drop"]
                    .mean()
                    .max()
                ),
            ),
            q95_uno_drop=("uno_drop", lambda values: float(np.quantile(values, 0.95))),
        )
        .reset_index()
    )
    conditions = {}
    for row in level_summary.itertuples(index=False):
        key = f"{row.level:.2f}"
        requirement = plan["requirements"][key]
        conditions[f"{key}_correlation"] = (
            row.mean_risk_correlation >= requirement["minimum_mean_risk_correlation"]
        )
        conditions[f"{key}_uno_drop"] = (
            row.mean_uno_drop <= requirement["maximum_mean_uno_drop"]
        )
    level_summary.to_csv(RESULTS / "38_v4_missingness_summary.csv", index=False)
    worst_genes = (
        systematic.groupby("missing_gene")
        .agg(
            mean_risk_correlation=("risk_correlation", "mean"),
            mean_uno_drop=("uno_drop", "mean"),
            maximum_uno_drop=("uno_drop", "max"),
        )
        .sort_values("mean_uno_drop", ascending=False)
        .head(10)
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "status": "V4_MISSINGNESS_STRESS_COMPLETE",
        "gate": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "selected_candidate": candidate.name,
        "replicates_per_level": int(plan["replicates_per_level"]),
        "levels": level_summary.to_dict(orient="records"),
        "worst_single_missing_genes": worst_genes,
        "gse224564_used": False,
        "fresh_candidate_scores_generated": False,
    }
    (RESULTS / "38_v4_missingness_gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
