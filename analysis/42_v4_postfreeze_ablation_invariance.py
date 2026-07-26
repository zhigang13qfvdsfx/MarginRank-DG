from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

from transsurv.marginrank import LockedMarginRankModel


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
DATA = ROOT / "data" / "processed"
LOCK_PATH = CONFIG / "V4_POSTFREEZE_ABLATION_INVARIANCE.lock"
MODEL_PATH = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"

PRIMARY = [
    "TCGA_PAAD",
    "CPTAC_PDACC",
    "GSE62452",
    "GSE85916",
    "E-MTAB-6134",
    "GSE183795",
    "GSE57495",
    "GSE79668",
    "PACA-AU",
    "GSE71729",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_path(domain: str) -> Path:
    stem = domain.lower().replace("-", "_")
    return DATA / f"{stem}_v4_single_sample_rank.parquet"


def exact_metrics(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values))), float(np.sqrt(np.mean(values**2)))


def paired_summary(
    metrics: pd.DataFrame,
    selected: str,
    comparator_name: str,
    comparator: str,
) -> dict[str, object]:
    left = metrics.loc[metrics["candidate"] == selected, ["cohort", "uno_c"]].rename(
        columns={"uno_c": "selected_uno"}
    )
    right = metrics.loc[
        metrics["candidate"] == comparator, ["cohort", "uno_c"]
    ].rename(columns={"uno_c": "comparator_uno"})
    paired = left.merge(right, on="cohort", validate="one_to_one")
    paired["delta_uno"] = paired["selected_uno"] - paired["comparator_uno"]
    nonzero = paired.loc[np.abs(paired["delta_uno"]) > 1e-15, "delta_uno"]
    if len(nonzero):
        sign_p = float(
            binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5, alternative="two-sided").pvalue
        )
        try:
            signed_rank_p = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        except ValueError:
            signed_rank_p = float("nan")
    else:
        sign_p = 1.0
        signed_rank_p = 1.0
    return {
        "comparator_label": comparator_name,
        "comparator": comparator,
        "mean_delta_uno": float(paired["delta_uno"].mean()),
        "median_delta_uno": float(paired["delta_uno"].median()),
        "minimum_delta_uno": float(paired["delta_uno"].min()),
        "maximum_delta_uno": float(paired["delta_uno"].max()),
        "domains_selected_higher": int((paired["delta_uno"] > 0).sum()),
        "domains_tied": int((np.abs(paired["delta_uno"]) <= 1e-15).sum()),
        "domains": int(len(paired)),
        "two_sided_sign_p_descriptive": sign_p,
        "two_sided_wilcoxon_p_descriptive": signed_rank_p,
    }


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["status"] != "LOCKED_POSTFREEZE_EXPLANATORY_ANALYSIS_BEFORE_INVARIANCE_SCORING":
        raise RuntimeError("Unexpected ablation/invariance lock status")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    frozen_spec = json.loads(
        (CONFIG / "frozen_model_v4_marginrank_spec.json").read_text(encoding="utf-8")
    )
    if sha256(MODEL_PATH) != frozen_spec["model_sha256"]:
        raise RuntimeError("Frozen model hash mismatch")
    completion = json.loads(
        (CONFIG / "V4_PACA_CA_VALIDATION_COMPLETED.lock").read_text(encoding="utf-8")
    )
    if completion["status"] != "V4_PACA_CA_VALIDATION_CONSUMED_ONCE":
        raise RuntimeError("PACA-CA completion lock is absent or unexpected")

    ranking = pd.read_csv(RESULTS / "37_v4_fixed_candidate_ranking.csv")
    metrics = pd.read_csv(RESULTS / "37_v4_fixed_candidate_loso_metrics.csv")
    selected = lock["selected_candidate"]
    candidate_names = {selected, *lock["comparators"].values()}
    absent = sorted(candidate_names - set(ranking["candidate"].astype(str)))
    if absent:
        raise RuntimeError(f"Locked comparators absent: {absent}")
    architecture = ranking.loc[ranking["candidate"].isin(candidate_names)].copy()
    architecture["role"] = architecture["candidate"].map(
        {selected: "selected", **{value: key for key, value in lock["comparators"].items()}}
    )
    architecture = architecture[
        [
            "role",
            "candidate",
            "mean_uno",
            "worst_uno",
            "sd_uno",
            "mean_harrell",
            "valid_folds",
            "selection_score",
        ]
    ].sort_values("role")
    architecture.to_csv(RESULTS / "42_v4_architecture_ablation.csv", index=False)

    paired_rows = [
        paired_summary(metrics, selected, label, candidate)
        for label, candidate in lock["comparators"].items()
    ]
    pd.DataFrame(paired_rows).to_csv(
        RESULTS / "42_v4_architecture_paired_domains.csv", index=False
    )

    lookup = architecture.set_index("role")
    s = lookup.loc["selected"]
    no_anchor = lookup.loc["without_anchor"]
    no_aux = lookup.loc["without_auxiliary_domain"]
    expert = lookup.loc["expert_ensemble"]
    g20 = lookup.loc["twenty_genes"]
    g80 = lookup.loc["eighty_genes"]
    thresholds = lock["explanatory_gate"]

    leave_one = pd.read_csv(RESULTS / "38_v4_leave_one_gene_out.csv")
    leave_one_summary = (
        leave_one.groupby("missing_gene", as_index=False)
        .agg(
            mean_risk_correlation=("risk_correlation", "mean"),
            minimum_risk_correlation=("risk_correlation", "min"),
            mean_uno_drop=("uno_drop", "mean"),
            maximum_uno_drop=("uno_drop", "max"),
            domains_observed=("cohort", "nunique"),
        )
        .sort_values("mean_uno_drop", ascending=False)
    )
    leave_one_summary.to_csv(RESULTS / "42_v4_leave_one_gene_summary.csv", index=False)

    explanatory_conditions = {
        "anchor_noninferior_mean": float(s["mean_uno"] - no_anchor["mean_uno"])
        >= thresholds["anchor_mean_uno_delta_minimum"],
        "anchor_noninferior_worst_domain": float(s["worst_uno"] - no_anchor["worst_uno"])
        >= thresholds["anchor_worst_uno_delta_minimum"],
        "auxiliary_improves_worst_domain": float(s["worst_uno"] - no_aux["worst_uno"])
        >= thresholds["auxiliary_worst_uno_delta_minimum"],
        "auxiliary_reduces_dispersion": float(no_aux["sd_uno"] - s["sd_uno"])
        >= thresholds["auxiliary_sd_uno_reduction_minimum"],
        "forty_gene_mean_above_size_controls": bool(
            s["mean_uno"] > g20["mean_uno"] and s["mean_uno"] > g80["mean_uno"]
        ),
        "forty_gene_worst_above_twenty": float(s["worst_uno"] - g20["worst_uno"])
        >= thresholds["forty_gene_worst_uno_delta_vs_twenty_minimum"],
        "forty_gene_worst_above_eighty": float(s["worst_uno"] - g80["worst_uno"])
        >= thresholds["forty_gene_worst_uno_delta_vs_eighty_minimum"],
        "pooled_improves_worst_vs_expert": float(s["worst_uno"] - expert["worst_uno"])
        >= thresholds["pooled_worst_uno_delta_vs_expert_minimum"],
        "pooled_reduces_dispersion_vs_expert": float(expert["sd_uno"] - s["sd_uno"])
        >= thresholds["pooled_sd_uno_reduction_vs_expert_minimum"],
        "single_gene_mean_drop_bounded": float(leave_one_summary["mean_uno_drop"].max())
        <= thresholds["maximum_single_gene_mean_uno_drop"],
        "single_gene_mean_correlation_preserved": float(
            leave_one_summary["mean_risk_correlation"].min()
        )
        >= thresholds["minimum_single_gene_mean_risk_correlation"],
    }

    locked_model = LockedMarginRankModel.load(MODEL_PATH)
    rng = np.random.default_rng(20260722)
    native_scores: dict[str, pd.Series] = {}
    sampled_frames = []
    selected_ids = []
    for domain in PRIMARY:
        rank = pd.read_parquet(rank_path(domain))
        rank.index = rank.index.astype(str)
        n_take = min(15, len(rank))
        ids = rng.choice(rank.index.to_numpy(), size=n_take, replace=False).tolist()
        selected_ids.extend(ids)
        native_scores[domain] = locked_model.predict_from_rank(rank)["raw_risk"]
        part = rank.loc[ids].copy()
        part.index = [f"{domain}::{item}" for item in part.index]
        sampled_frames.append(part)

    mixed = pd.concat(sampled_frames, axis=0)
    mixed_score = locked_model.predict_from_rank(mixed)["raw_risk"]
    permuted_columns = list(mixed.columns)
    rng.shuffle(permuted_columns)
    permuted_score = locked_model.predict_from_rank(mixed.loc[:, permuted_columns])["raw_risk"]

    invariance_rows = []
    for domain, frame in zip(PRIMARY, sampled_frames):
        original_ids = [item.split("::", 1)[1] for item in frame.index]
        native = native_scores[domain].loc[original_ids].to_numpy(float)
        individual = np.array(
            [
                locked_model.predict_from_rank(frame.loc[[tag]])["raw_risk"].iloc[0]
                for tag in frame.index
            ],
            dtype=float,
        )
        mixed_values = mixed_score.loc[frame.index].to_numpy(float)
        permuted_values = permuted_score.loc[frame.index].to_numpy(float)
        for tag, base, alone, mix, reordered in zip(
            frame.index, native, individual, mixed_values, permuted_values
        ):
            invariance_rows.append(
                {
                    "sample_id": tag,
                    "cohort": domain,
                    "native_raw_risk": base,
                    "alone_raw_risk": alone,
                    "mixed_batch_raw_risk": mix,
                    "permuted_columns_raw_risk": reordered,
                    "alone_difference": alone - base,
                    "mixed_batch_difference": mix - base,
                    "permuted_columns_difference": reordered - base,
                }
            )

    invariance = pd.DataFrame(invariance_rows)
    expression = pd.read_parquet(DATA / "tcga_paad_expression.parquet")
    expression.index = expression.index.astype(str)
    expression = expression.iloc[: min(30, len(expression))].copy()
    base_expression_score = locked_model.predict_from_expression(expression)["raw_risk"]
    slopes = np.linspace(0.7, 2.3, len(expression))
    offsets = np.linspace(-11.0, 13.0, len(expression))
    transformed_expression = expression.mul(slopes, axis=0).add(offsets, axis=0)
    transformed_expression_score = locked_model.predict_from_expression(
        transformed_expression
    )["raw_risk"]
    affine_difference = (
        transformed_expression_score.to_numpy(float)
        - base_expression_score.to_numpy(float)
    )
    affine = pd.DataFrame(
        {
            "sample_id": expression.index,
            "base_raw_risk": base_expression_score.to_numpy(float),
            "affine_raw_risk": transformed_expression_score.to_numpy(float),
            "difference": affine_difference,
        }
    )
    affine.to_csv(RESULTS / "42_v4_affine_rank_invariance.csv", index=False)
    invariance.to_csv(RESULTS / "42_v4_batch_composition_invariance.csv", index=False)

    diff_columns = [
        "alone_difference",
        "mixed_batch_difference",
        "permuted_columns_difference",
    ]
    all_differences = np.concatenate(
        [invariance[column].to_numpy(float) for column in diff_columns]
        + [affine_difference]
    )
    max_abs, rmse = exact_metrics(all_differences)
    correlations = {
        "native_vs_alone": float(
            spearmanr(invariance["native_raw_risk"], invariance["alone_raw_risk"]).statistic
        ),
        "native_vs_mixed": float(
            spearmanr(
                invariance["native_raw_risk"], invariance["mixed_batch_raw_risk"]
            ).statistic
        ),
        "native_vs_permuted": float(
            spearmanr(
                invariance["native_raw_risk"],
                invariance["permuted_columns_raw_risk"],
            ).statistic
        ),
        "expression_vs_affine": float(
            spearmanr(base_expression_score, transformed_expression_score).statistic
        ),
    }
    inv_thresholds = lock["invariance_gate"]
    invariance_conditions = {
        "raw_risk_numerically_identical": max_abs
        <= inv_thresholds["maximum_absolute_raw_risk_difference"],
        "rank_order_identical": min(correlations.values())
        >= inv_thresholds["minimum_raw_risk_correlation"],
        "all_ten_domains_in_mixed_batch": len(PRIMARY)
        >= inv_thresholds["minimum_development_domains_in_mixed_batch"],
        "sample_count_sufficient": len(invariance)
        >= inv_thresholds["minimum_samples_checked"],
    }

    summary = {
        "status": "V4_POSTFREEZE_ABLATION_AND_INVARIANCE_COMPLETE",
        "overall_gate": "PASS"
        if all(explanatory_conditions.values()) and all(invariance_conditions.values())
        else "FAIL",
        "explanatory_gate": "PASS" if all(explanatory_conditions.values()) else "FAIL",
        "explanatory_conditions": explanatory_conditions,
        "invariance_gate": "PASS" if all(invariance_conditions.values()) else "FAIL",
        "invariance_conditions": invariance_conditions,
        "architecture_effects": {
            "anchor_mean_uno_delta": float(s["mean_uno"] - no_anchor["mean_uno"]),
            "anchor_worst_uno_delta": float(s["worst_uno"] - no_anchor["worst_uno"]),
            "auxiliary_mean_uno_delta": float(s["mean_uno"] - no_aux["mean_uno"]),
            "auxiliary_worst_uno_delta": float(s["worst_uno"] - no_aux["worst_uno"]),
            "auxiliary_sd_uno_reduction": float(no_aux["sd_uno"] - s["sd_uno"]),
            "pooled_worst_uno_delta_vs_expert": float(s["worst_uno"] - expert["worst_uno"]),
            "pooled_sd_uno_reduction_vs_expert": float(expert["sd_uno"] - s["sd_uno"]),
            "forty_worst_uno_delta_vs_twenty": float(s["worst_uno"] - g20["worst_uno"]),
            "forty_worst_uno_delta_vs_eighty": float(s["worst_uno"] - g80["worst_uno"]),
        },
        "single_gene_robustness": {
            "maximum_mean_uno_drop": float(leave_one_summary["mean_uno_drop"].max()),
            "minimum_mean_risk_correlation": float(
                leave_one_summary["mean_risk_correlation"].min()
            ),
        },
        "invariance": {
            "samples_checked": int(len(invariance)),
            "development_domains": len(PRIMARY),
            "affine_expression_samples": int(len(affine)),
            "maximum_absolute_raw_risk_difference": max_abs,
            "rmse_raw_risk_difference": rmse,
            "spearman_correlations": correlations,
            "note": "Within-batch percentiles and tiers are intentionally cohort-relative; only raw risk is composition invariant.",
        },
        "formal_superiority_claimed": False,
        "model_changed": False,
        "paca_ca_used_for_tuning": False,
        "hashes": {
            "lock": sha256(LOCK_PATH),
            "model": sha256(MODEL_PATH),
            "candidate_ranking": sha256(RESULTS / "37_v4_fixed_candidate_ranking.csv"),
            "candidate_loso": sha256(RESULTS / "37_v4_fixed_candidate_loso_metrics.csv"),
            "leave_one_gene_out": sha256(RESULTS / "38_v4_leave_one_gene_out.csv"),
            "script": sha256(Path(__file__)),
        },
    }
    (RESULTS / "42_v4_postfreeze_ablation_invariance_gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
