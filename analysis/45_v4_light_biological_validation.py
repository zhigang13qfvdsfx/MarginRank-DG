from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
MANIFESTS = ROOT / "data" / "manifests"
SC_DIR = RAW / "GSE263733"
COUNT_PATH = SC_DIR / "GSE263733_Raw_counts.txt.gz"
ANNOTATION_PATH = SC_DIR / "GSE263733_Cell_annotation.txt.gz"
PLAN_PATH = CONFIG / "V4_LIGHT_BIOLOGICAL_VALIDATION_PLAN.lock"
COEFFICIENT_PATH = RESULTS / "39_v4_final_feature_coefficients.csv"
UNIVERSE_PATH = MANIFESTS / "v4_strict_common_gene_universe.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_counts(
    wanted: set[str], annotation: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, int]:
    with gzip.open(COUNT_PATH, "rb") as handle:
        header = handle.readline().rstrip(b"\r\n").decode("utf-8").split("\t")
        if len(header) != len(annotation):
            raise RuntimeError(
                f"Count header has {len(header)} cells but annotation has {len(annotation)}"
            )
        if header != annotation.index.astype(str).tolist():
            raise RuntimeError("Count-matrix cells do not exactly match annotation order")
        library_size = np.zeros(len(header), dtype=np.int64)
        selected: dict[str, np.ndarray] = {}
        parsed_genes = 0
        for line in handle:
            gene_bytes, separator, values_bytes = line.rstrip(b"\r\n").partition(b"\t")
            if not separator:
                raise RuntimeError(f"Malformed count row at gene index {parsed_genes}")
            gene = gene_bytes.decode("utf-8")
            values = np.fromstring(values_bytes, dtype=np.int64, sep="\t")
            if len(values) != len(header):
                raise RuntimeError(
                    f"{gene}: {len(values)} counts for {len(header)} cells"
                )
            library_size += values
            if gene in wanted:
                if gene in selected:
                    selected[gene] = selected[gene] + values
                else:
                    selected[gene] = values.copy()
            parsed_genes += 1
            if parsed_genes % 2500 == 0:
                print(
                    f"Parsed {parsed_genes:,} genes; retained {len(selected)}/{len(wanted)} targets",
                    flush=True,
                )
    if np.any(library_size <= 0):
        raise RuntimeError("At least one annotated cell has zero library size")
    frame = pd.DataFrame(selected, index=annotation.index).sort_index(axis=1)
    return frame, library_size, parsed_genes


def gprofiler_query(
    label: str, genes: list[str], background: list[str], sources: list[str]
) -> tuple[pd.DataFrame, Path]:
    payload = {
        "organism": "hsapiens",
        "query": genes,
        "sources": sources,
        "user_threshold": 0.05,
        "significance_threshold_method": "g_SCS",
        "domain_scope": "custom",
        "background": background,
        "no_evidences": False,
    }
    response = requests.post(
        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    raw = response.json()
    target_dir = RAW / "gprofiler"
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_path = target_dir / f"v4_{label}_gost_raw.json"
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    result = pd.DataFrame(raw.get("result", []))
    if result.empty:
        return result, raw_path
    keep = [
        "source",
        "native",
        "name",
        "p_value",
        "significant",
        "term_size",
        "query_size",
        "intersection_size",
        "effective_domain_size",
        "precision",
        "recall",
        "intersection",
    ]
    keep = [column for column in keep if column in result.columns]
    return result[keep].copy(), raw_path


def coefficient_weighted_score(
    logcpm: pd.DataFrame, coefficients: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    genes = [gene for gene in coefficients.index if gene in logcpm.columns]
    values = logcpm[genes]
    scale = values.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = values.sub(values.mean(axis=0), axis=1).div(scale, axis=1).fillna(0.0)
    weights = coefficients.loc[genes].astype(float)
    weights = weights / weights.abs().sum()
    return z.mul(weights, axis=1).sum(axis=1), z


def mean_marker_score(logcpm: pd.DataFrame, markers: list[str]) -> pd.Series:
    available = [gene for gene in markers if gene in logcpm.columns]
    values = logcpm[available]
    scale = values.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = values.sub(values.mean(axis=0), axis=1).div(scale, axis=1).fillna(0.0)
    return z.mean(axis=1)


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["status"] != "LOCKED_BEFORE_GSE263733_COUNT_EXTRACTION_AND_PATHWAY_QUERY":
        raise RuntimeError("Unexpected light-biological-validation lock")
    if COUNT_PATH.stat().st_size != 210530159:
        raise RuntimeError("GSE263733 compressed count size differs from official file")

    coefficients = pd.read_csv(COEFFICIENT_PATH)
    coefficients = coefficients.drop_duplicates("gene").set_index("gene")["coefficient"]
    if len(coefficients) != 40:
        raise RuntimeError("Frozen coefficient table does not contain exactly 40 genes")
    selected_genes = coefficients.index.astype(str).tolist()
    basal = plan["single_cell_dataset"]["basal_markers"]
    classical = plan["single_cell_dataset"]["classical_markers"]
    wanted = set(selected_genes) | set(basal) | set(classical)

    annotation = pd.read_csv(ANNOTATION_PATH, sep="\t")
    if len(annotation) != plan["single_cell_dataset"]["official_annotation_cells"]:
        raise RuntimeError("Unexpected GSE263733 annotation cell count")
    annotation["Barcodes"] = annotation["Barcodes"].astype(str)
    annotation = annotation.set_index("Barcodes", drop=False)
    count_cache = RESULTS / "45_gse263733_target_gene_cell_counts.parquet"
    qc_cache = RESULTS / "45_gse263733_cell_qc.parquet"
    if count_cache.exists() and qc_cache.exists():
        counts = pd.read_parquet(count_cache)
        cell_qc = pd.read_parquet(qc_cache)
        counts.index = counts.index.astype(str)
        cell_qc.index = cell_qc.index.astype(str)
        if counts.index.tolist() != annotation.index.tolist():
            raise RuntimeError("Cached target counts do not align to official annotation")
        if cell_qc.index.tolist() != annotation.index.tolist():
            raise RuntimeError("Cached library sizes do not align to official annotation")
        if not set(counts.columns).issubset(wanted) or np.any(cell_qc["library_size"] <= 0):
            raise RuntimeError("Cached extraction content failed validation")
        library_size = cell_qc["library_size"].to_numpy(np.int64)
        parsed_genes = 33694
        print("Reused validated target-count and library-size caches", flush=True)
    else:
        counts, library_size, parsed_genes = extract_counts(wanted, annotation)
        counts.index.name = "Barcodes"
        counts.to_parquet(count_cache, compression="zstd")
        cell_qc = annotation.copy()
        cell_qc["library_size"] = library_size
        cell_qc.to_parquet(qc_cache, compression="zstd")

    detected_selected = [gene for gene in selected_genes if gene in counts.columns]
    detected_basal = [gene for gene in basal if gene in counts.columns]
    detected_classical = [gene for gene in classical if gene in counts.columns]
    group_columns = ["Patients", "Origins", "Celltypes"]
    group_key = annotation[group_columns].astype(str).agg("||".join, axis=1)
    group_counts = counts.groupby(group_key, sort=False).sum()
    group_library = pd.Series(library_size, index=annotation.index).groupby(
        group_key, sort=False
    ).sum()
    group_n = group_key.value_counts(sort=False)
    group_metadata = group_counts.index.to_series().str.split(
        "||", expand=True, regex=False
    )
    group_metadata.columns = group_columns
    group_metadata.index = group_counts.index
    group_metadata["n_cells"] = group_n.reindex(group_counts.index).to_numpy(int)
    group_metadata["library_size"] = group_library.reindex(group_counts.index).to_numpy(float)
    minimum_cells = int(plan["single_cell_dataset"]["minimum_cells_per_pseudobulk"])
    eligible = group_metadata["n_cells"] >= minimum_cells
    logcpm = np.log1p(
        group_counts.loc[eligible].div(group_library.loc[eligible], axis=0) * 1_000_000.0
    )
    risk_program, selected_z = coefficient_weighted_score(logcpm, coefficients)
    positive = coefficients[coefficients > 0].index.intersection(logcpm.columns)
    negative = coefficients[coefficients < 0].index.intersection(logcpm.columns)
    unweighted = selected_z[positive].mean(axis=1) - selected_z[negative].mean(axis=1)

    pseudobulk = group_metadata.loc[eligible].copy()
    pseudobulk["risk_program"] = risk_program
    pseudobulk["unweighted_positive_minus_negative"] = unweighted
    pseudobulk.index.name = "pseudobulk_id"
    pseudobulk.reset_index().to_csv(RESULTS / "45_gse263733_pseudobulk_scores.csv", index=False)
    logcpm.index.name = "pseudobulk_id"
    logcpm.reset_index().to_parquet(
        RESULTS / "45_gse263733_pseudobulk_logcpm.parquet", compression="zstd"
    )

    cell_logcp10k = np.log1p(
        counts.div(pd.Series(library_size, index=counts.index), axis=0) * 10_000.0
    )
    dot_rows = []
    for cell_type, ids in annotation.groupby("Celltypes", sort=False).groups.items():
        values = cell_logcp10k.loc[ids]
        raw = counts.loc[ids]
        for gene in counts.columns:
            dot_rows.append(
                {
                    "cell_type": cell_type,
                    "gene": gene,
                    "mean_logcp10k": float(values[gene].mean()),
                    "fraction_expressing": float((raw[gene] > 0).mean()),
                    "n_cells": int(len(ids)),
                    "selected_gene": gene in selected_genes,
                    "coefficient": float(coefficients.get(gene, np.nan)),
                }
            )
    pd.DataFrame(dot_rows).to_csv(RESULTS / "45_gse263733_celltype_dotplot.csv", index=False)

    eligible_types = [
        group["risk_program"].to_numpy(float)
        for _, group in pseudobulk.groupby("Celltypes")
        if len(group) >= 3
    ]
    kw = kruskal(*eligible_types)
    celltype_summary = (
        pseudobulk.groupby("Celltypes", as_index=False)
        .agg(
            median_risk_program=("risk_program", "median"),
            mean_risk_program=("risk_program", "mean"),
            q25=("risk_program", lambda x: float(np.quantile(x, 0.25))),
            q75=("risk_program", lambda x: float(np.quantile(x, 0.75))),
            pseudobulks=("risk_program", "size"),
            patients=("Patients", "nunique"),
            cells=("n_cells", "sum"),
        )
        .sort_values("median_risk_program", ascending=False)
    )
    celltype_summary.to_csv(RESULTS / "45_gse263733_celltype_summary.csv", index=False)

    ductal = pseudobulk[pseudobulk["Celltypes"].eq("Ductal cell")].copy()
    paired = ductal[ductal["Origins"].isin(["Pm1", "Lm"])].pivot_table(
        index="Patients", columns="Origins", values="risk_program", aggfunc="first"
    ).dropna(subset=["Pm1", "Lm"])
    paired["delta_lm_minus_pm1"] = paired["Lm"] - paired["Pm1"]
    if len(paired) >= 2 and np.any(np.abs(paired["delta_lm_minus_pm1"]) > 1e-12):
        paired_test = wilcoxon(
            paired["Lm"], paired["Pm1"], alternative="greater", zero_method="wilcox"
        )
        paired_p = float(paired_test.pvalue)
    else:
        paired_p = float("nan")
    paired.reset_index().to_csv(RESULTS / "45_gse263733_matched_ductal.csv", index=False)

    primary_pm1 = ductal.loc[ductal["Origins"].eq("Pm1"), "risk_program"]
    primary_pm0 = ductal.loc[ductal["Origins"].eq("Pm0"), "risk_program"]
    if len(primary_pm1) and len(primary_pm0):
        primary_test = mannwhitneyu(primary_pm1, primary_pm0, alternative="greater")
        pm1_pm0_p = float(primary_test.pvalue)
        pm1_pm0_delta = float(np.median(primary_pm1) - np.median(primary_pm0))
    else:
        pm1_pm0_p = float("nan")
        pm1_pm0_delta = float("nan")

    ductal_logcpm = logcpm.loc[ductal.index]
    ductal["basal_score"] = mean_marker_score(ductal_logcpm, basal)
    ductal["classical_score"] = mean_marker_score(ductal_logcpm, classical)
    ductal["basal_minus_classical"] = ductal["basal_score"] - ductal["classical_score"]
    basal_alignment = spearmanr(
        ductal["risk_program"], ductal["basal_minus_classical"]
    )
    ductal.reset_index().to_csv(RESULTS / "45_gse263733_ductal_subtype_scores.csv", index=False)

    leave_one_directions = []
    for patient in paired.index:
        remaining = paired.drop(index=patient)
        leave_one_directions.append(float(np.median(remaining["delta_lm_minus_pm1"])))
    sensitivity = {
        "unweighted_weighted_spearman": float(
            spearmanr(pseudobulk["risk_program"], pseudobulk["unweighted_positive_minus_negative"]).statistic
        ),
        "matched_leave_one_patient_out_positive_fraction": float(
            np.mean(np.asarray(leave_one_directions) > 0)
        )
        if leave_one_directions
        else float("nan"),
        "matched_leave_one_patient_out_median_deltas": leave_one_directions,
        "eligible_pseudobulks_at_50_cells": int((group_metadata["n_cells"] >= 50).sum()),
    }

    universe = pd.read_csv(UNIVERSE_PATH)["gene"].astype(str).tolist()
    sources = plan["pathway_analysis"]["sources"]
    enrichment_frames = []
    enrichment_raw = []
    queries = {
        "all_selected": selected_genes,
        "risk_positive": coefficients[coefficients > 0].index.astype(str).tolist(),
        "risk_negative": coefficients[coefficients < 0].index.astype(str).tolist(),
    }
    for label, genes in queries.items():
        frame, raw_path = gprofiler_query(label, genes, universe, sources)
        enrichment_raw.append(raw_path)
        if not frame.empty:
            frame.insert(0, "query", label)
            enrichment_frames.append(frame)
    enrichment = pd.concat(enrichment_frames, ignore_index=True) if enrichment_frames else pd.DataFrame()
    enrichment.to_csv(RESULTS / "45_v4_gprofiler_enrichment.csv", index=False)
    significant_all = (
        enrichment[
            enrichment["query"].eq("all_selected")
            & enrichment["significant"].astype(bool)
            & (enrichment["p_value"] < 0.05)
        ]
        if not enrichment.empty
        else enrichment
    )

    technical_condition = bool(
        len(detected_selected)
        >= plan["single_cell_dataset"]["minimum_selected_genes_detected"]
        and annotation["Celltypes"].nunique() == len(plan["single_cell_dataset"]["cell_types"])
        and annotation["Patients"].nunique() == plan["single_cell_dataset"]["patients"]
    )
    single_conditions = {
        "technical_coverage": technical_condition,
        "cell_type_localization": float(kw.pvalue) < 0.05,
        "matched_metastasis_direction": bool(
            len(paired)
            >= plan["single_cell_dataset"]["minimum_matched_primary_metastasis_patients"]
            and np.median(paired["delta_lm_minus_pm1"]) > 0
            and paired_p < 0.10
        ),
        "basal_alignment": bool(
            basal_alignment.statistic >= 0.30 and basal_alignment.pvalue < 0.05
        ),
    }
    supportive_count = int(sum(single_conditions.values()))
    single_gate = bool(
        single_conditions["technical_coverage"]
        and supportive_count
        >= plan["single_cell_evidence_gate"]["minimum_conditions_for_supportive_gate"]
    )
    pathway_condition = int(len(significant_all)) >= plan["pathway_analysis"][
        "minimum_significant_terms_all_genes"
    ]
    summary = {
        "status": "V4_LIGHT_BIOLOGICAL_VALIDATION_COMPLETE",
        "overall_supportive_gate": "PASS" if single_gate and pathway_condition else "FAIL",
        "pathway_gate": "PASS" if pathway_condition else "FAIL",
        "single_cell_gate": "PASS" if single_gate else "FAIL",
        "single_cell_conditions": single_conditions,
        "technical": {
            "matrix_genes_parsed": parsed_genes,
            "annotation_cells": int(len(annotation)),
            "patients": int(annotation["Patients"].nunique()),
            "cell_types": int(annotation["Celltypes"].nunique()),
            "selected_genes_detected": len(detected_selected),
            "selected_genes_missing": sorted(set(selected_genes) - set(detected_selected)),
            "basal_markers_detected": detected_basal,
            "classical_markers_detected": detected_classical,
            "eligible_pseudobulks": int(eligible.sum()),
        },
        "pathway": {
            "significant_all_selected_terms": int(len(significant_all)),
            "top_terms": significant_all.head(15)[
                ["source", "native", "name", "p_value", "intersection_size"]
            ].to_dict(orient="records")
            if not significant_all.empty
            else [],
        },
        "cell_type_localization": {
            "kruskal_wallis_statistic": float(kw.statistic),
            "kruskal_wallis_p": float(kw.pvalue),
            "highest_median_cell_type": str(celltype_summary.iloc[0]["Celltypes"]),
            "cell_type_medians": celltype_summary.set_index("Celltypes")[
                "median_risk_program"
            ].to_dict(),
        },
        "matched_metastasis": {
            "paired_patients": int(len(paired)),
            "median_delta_lm_minus_pm1": float(np.median(paired["delta_lm_minus_pm1"]))
            if len(paired)
            else float("nan"),
            "positive_pairs": int((paired["delta_lm_minus_pm1"] > 0).sum()),
            "one_sided_wilcoxon_p": paired_p,
        },
        "primary_metastatic_context": {
            "median_pm1_minus_pm0_ductal": pm1_pm0_delta,
            "one_sided_mann_whitney_p": pm1_pm0_p,
        },
        "basal_classical_alignment": {
            "ductal_pseudobulks": int(len(ductal)),
            "spearman_rho": float(basal_alignment.statistic),
            "spearman_p": float(basal_alignment.pvalue),
        },
        "sensitivity": sensitivity,
        "model_changed": False,
        "single_cell_used_for_tuning": False,
        "causal_or_wet_lab_validation_claimed": False,
        "hashes": {
            "plan": sha256(PLAN_PATH),
            "counts": sha256(COUNT_PATH),
            "annotation": sha256(ANNOTATION_PATH),
            "coefficients": sha256(COEFFICIENT_PATH),
            "universe": sha256(UNIVERSE_PATH),
            "script": sha256(Path(__file__)),
            **{f"gprofiler_{path.stem}": sha256(path) for path in enrichment_raw},
        },
    }
    (RESULTS / "45_v4_light_biological_validation_gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
