from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "config"
PLAN_PATH = CONFIG / "V4_BIOLOGICAL_CONTEXT_ESCALATION.lock"
UNIVERSE_PATH = ROOT / "data" / "manifests" / "v4_strict_common_gene_universe.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v4_module():
    path = Path(__file__).with_name("36_v4_marginrank_nested.py")
    spec = importlib.util.spec_from_file_location("v4_ranked_biology", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ordered_query(
    label: str,
    genes: list[str],
    background: list[str],
    sources: list[str],
) -> tuple[pd.DataFrame, Path, dict]:
    payload = {
        "organism": "hsapiens",
        "query": genes,
        "ordered": True,
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
        timeout=180,
    )
    response.raise_for_status()
    raw = response.json()
    target = RAW / "gprofiler" / f"v4_ranked_{label}_gost_raw.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    rows = []
    for result in raw.get("result", []):
        intersections = result.get("intersections") or []
        drivers = [
            gene
            for gene, membership in zip(genes, intersections)
            if membership
        ]
        rows.append(
            {
                "direction": label,
                "source": result.get("source"),
                "native": result.get("native"),
                "name": result.get("name"),
                "p_value": result.get("p_value"),
                "significant": result.get("significant"),
                "term_size": result.get("term_size"),
                "query_size": result.get("query_size"),
                "intersection_size": result.get("intersection_size"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "driver_genes": "|".join(drivers),
            }
        )
    return pd.DataFrame(rows), target, raw.get("meta", {})


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["status"] != "LOCKED_AFTER_40_GENE_ORA_FAILURE_BEFORE_FULL_RANKED_QUERY":
        raise RuntimeError("Unexpected ranked biological-context lock")
    v4 = load_v4_module()
    domains, universe = v4.load_domains()
    if universe != pd.read_csv(UNIVERSE_PATH)["gene"].astype(str).tolist():
        raise RuntimeError("v4 universe mismatch")
    z_table = v4.precompute_domain_z(domains, universe)
    primary_screen = v4.screening_table(v4.PRIMARY, z_table).copy()
    primary_screen["signed_stability"] = np.sign(primary_screen["meta_z"]) * primary_screen[
        "stability"
    ]
    with_aux = v4.screening_table([*v4.PRIMARY, v4.AUXILIARY], z_table).copy()
    with_aux = with_aux.rename(
        columns={
            "meta_z": "meta_z_with_aux",
            "dispersion": "dispersion_with_aux",
            "directional_consistency": "directional_consistency_with_aux",
            "stability": "stability_with_aux",
        }
    )
    rank = primary_screen.join(with_aux, how="left")
    rank.index.name = "gene"
    for domain in z_table.index:
        rank[f"z_{domain}"] = z_table.loc[domain].reindex(rank.index).to_numpy(float)
    rank = rank.sort_values("signed_stability", ascending=False)
    rank.reset_index().to_csv(RESULTS / "46_v4_full_gene_survival_rank.csv", index=False)

    positive = rank.loc[rank["signed_stability"] > 0].index.astype(str).tolist()
    negative = (
        rank.loc[rank["signed_stability"] < 0]
        .sort_values("signed_stability", ascending=True)
        .index.astype(str)
        .tolist()
    )
    sources = plan["enrichment"]["sources"]
    positive_frame, positive_raw, positive_meta = ordered_query(
        "adverse", positive, universe, sources
    )
    negative_frame, negative_raw, negative_meta = ordered_query(
        "favourable", negative, universe, sources
    )
    enrichment = pd.concat([positive_frame, negative_frame], ignore_index=True)

    lower, upper = plan["enrichment"]["term_size_range"]
    minimum_drivers = int(plan["enrichment"]["minimum_driver_genes"])
    enrichment["filtered_significant"] = (
        enrichment["significant"].astype(bool)
        & (enrichment["p_value"] < plan["enrichment"]["significance"])
        & enrichment["term_size"].between(lower, upper)
        & (enrichment["intersection_size"] >= minimum_drivers)
    )

    domain_columns = [f"z_{domain}" for domain in v4.PRIMARY]
    consistency_values = []
    driver_meta_values = []
    for row in enrichment.itertuples(index=False):
        drivers = [gene for gene in str(row.driver_genes).split("|") if gene in rank.index]
        if drivers:
            per_domain = rank.loc[drivers, domain_columns].mean(axis=0).to_numpy(float)
            expected = 1.0 if row.direction == "adverse" else -1.0
            fraction = float(np.mean(expected * per_domain > 0))
            driver_meta = float(rank.loc[drivers, "meta_z"].mean())
        else:
            fraction = float("nan")
            driver_meta = float("nan")
        consistency_values.append(fraction)
        driver_meta_values.append(driver_meta)
    enrichment["domain_direction_fraction"] = consistency_values
    enrichment["mean_driver_meta_z"] = driver_meta_values
    minimum_fraction = plan["supportive_gate"][
        "robust_term_minimum_domain_direction_fraction"
    ]
    enrichment["cross_domain_robust"] = (
        enrichment["filtered_significant"]
        & (enrichment["domain_direction_fraction"] >= minimum_fraction)
    )
    enrichment = enrichment.sort_values(
        ["filtered_significant", "cross_domain_robust", "p_value"],
        ascending=[False, False, True],
    )
    enrichment.to_csv(RESULTS / "46_v4_ranked_pathway_enrichment.csv", index=False)

    filtered = enrichment[enrichment["filtered_significant"]]
    robust = enrichment[enrichment["cross_domain_robust"]]
    gate = plan["supportive_gate"]
    conditions = {
        "minimum_filtered_terms": len(filtered)
        >= gate["minimum_filtered_significant_terms"],
        "minimum_sources": filtered["source"].nunique()
        >= gate["minimum_distinct_sources"],
        "minimum_robust_terms": len(robust)
        >= gate["minimum_cross_domain_robust_terms"],
    }
    meta = positive_meta or negative_meta
    summary = {
        "status": "V4_FULL_RANKED_BIOLOGICAL_CONTEXT_COMPLETE",
        "supportive_gate": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "genes_ranked": len(rank),
        "adverse_query_genes": len(positive),
        "favourable_query_genes": len(negative),
        "filtered_significant_terms": int(len(filtered)),
        "distinct_sources": int(filtered["source"].nunique()),
        "cross_domain_robust_terms": int(len(robust)),
        "top_adverse_terms": robust[robust["direction"].eq("adverse")].head(12)[
            [
                "source",
                "native",
                "name",
                "p_value",
                "intersection_size",
                "domain_direction_fraction",
                "driver_genes",
            ]
        ].to_dict(orient="records"),
        "top_favourable_terms": robust[robust["direction"].eq("favourable")].head(12)[
            [
                "source",
                "native",
                "name",
                "p_value",
                "intersection_size",
                "domain_direction_fraction",
                "driver_genes",
            ]
        ].to_dict(orient="records"),
        "gprofiler_version": meta.get("version"),
        "development_outcomes_used": True,
        "external_outcomes_used": False,
        "model_changed": False,
        "pathways_used_for_tuning": False,
        "claim_scope": plan["interpretation"],
        "hashes": {
            "plan": sha256(PLAN_PATH),
            "universe": sha256(UNIVERSE_PATH),
            "positive_raw": sha256(positive_raw),
            "negative_raw": sha256(negative_raw),
            "script": sha256(Path(__file__)),
        },
    }
    (RESULTS / "46_v4_ranked_biological_context_gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
