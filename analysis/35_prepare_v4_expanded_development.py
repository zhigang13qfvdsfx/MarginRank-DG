from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import pyarrow.parquet as pq

from transsurv.transforms import SingleSampleRankTransformer


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MANIFESTS = ROOT / "data" / "manifests"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
MIGRATION = CONFIG / "V4_DEVELOPMENT_MIGRATION.lock"
V3_COMPLETION = CONFIG / "V3_3_CONFIRMATION_COMPLETED.lock"

PRIMARY = {
    "TCGA_PAAD": ("tcga_paad_expression.parquet", "tcga_paad_clinical.csv"),
    "CPTAC_PDACC": ("cptac_pdacc_expression.parquet", "cptac_pdacc_clinical.csv"),
    "GSE62452": ("gse62452_expression.parquet", "gse62452_clinical.csv"),
    "GSE85916": ("gse85916_expression.parquet", "gse85916_clinical.csv"),
    "E-MTAB-6134": ("emtab6134_expression.parquet", "emtab6134_clinical.csv"),
    "GSE183795": ("gse183795_v4_full_expression.parquet", "gse183795_v3_development_clinical.csv"),
    "GSE57495": ("gse57495_external_expression.parquet", "gse57495_external_clinical.csv"),
    "GSE79668": ("gse79668_external_expression.parquet", "gse79668_external_clinical.csv"),
    "PACA-AU": ("paca_au_v4_full_expression.parquet", "paca_au_v2_confirmation_clinical.csv"),
    "GSE71729": ("gse71729_v4_full_expression.parquet", "gse71729_v2_confirmation_clinical.csv"),
}
AUXILIARY = {
    "GSE205154_MET": ("gse205154_met_v4_full_expression.parquet", "gse205154_met_v4_clinical.csv")
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hgnc_aliases() -> tuple[dict[str, str], set[str]]:
    table = pd.read_csv(
        RAW / "hgnc_complete_set_20260721.txt", sep="\t", dtype=str, low_memory=False
    )
    table = table[table["status"].eq("Approved")].copy()
    approved = set(table["symbol"].dropna().str.upper())
    candidates: dict[str, set[str]] = {}
    for row in table[["symbol", "alias_symbol", "prev_symbol"]].itertuples(index=False):
        current = str(row.symbol).upper()
        for field in (row.alias_symbol, row.prev_symbol):
            if pd.isna(field):
                continue
            for alias in str(field).split("|"):
                alias = alias.strip().upper()
                if alias and alias not in approved:
                    candidates.setdefault(alias, set()).add(current)
    aliases = {symbol: symbol for symbol in approved}
    aliases.update(
        {alias: next(iter(values)) for alias, values in candidates.items() if len(values) == 1}
    )
    return aliases, approved


def clean_gene(value: object, aliases: dict[str, str]) -> str:
    symbol = str(value).strip().strip('"').upper()
    symbol = re.split(r"\s*///\s*|\s*//\s*|;|,", symbol)[0].strip()
    return aliases.get(symbol, symbol)


def canonicalize_expression(
    expression: pd.DataFrame, aliases: dict[str, str], approved: set[str]
) -> pd.DataFrame:
    expression = expression.copy()
    expression.columns = [clean_gene(column, aliases) for column in expression.columns]
    expression = expression.loc[:, [column in approved for column in expression.columns]]
    if expression.columns.duplicated().any():
        expression = expression.T.groupby(level=0, sort=False).median().T
    expression.index = expression.index.astype(str)
    expression.index.name = "SAMPLE_ID"
    return expression


def series_matrix(path: Path) -> pd.DataFrame:
    table_lines: list[str] = []
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if stripped == "!series_matrix_table_begin":
                in_table = True
                continue
            if stripped == "!series_matrix_table_end":
                break
            if in_table:
                table_lines.append(stripped)
    frame = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", low_memory=False)
    frame.columns = [str(column).strip().strip('"') for column in frame.columns]
    return frame


def normal_key(value: object) -> str:
    value = re.sub(r"\.CEL$", "", str(value), flags=re.I)
    return re.sub(r"[^A-Za-z0-9]", "", value).lower()


def prepare_gse183795(aliases: dict[str, str], approved: set[str]) -> None:
    target = PROCESSED / PRIMARY["GSE183795"][0]
    if target.exists():
        return
    clinical = pd.read_csv(PROCESSED / PRIMARY["GSE183795"][1], dtype=str)
    eligible = set(clinical["SAMPLE_ID"].astype(str))
    title_to_sample = dict(zip(clinical["TITLE"].map(normal_key), clinical["SAMPLE_ID"]))
    path = RAW / "GSE183795_normalized_matrix.txt.gz"
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    expression_columns = header[4:]
    column_to_sample = {
        column: title_to_sample[normal_key(column)]
        for column in expression_columns
        if normal_key(column) in title_to_sample
    }
    selected = [column for column, sample in column_to_sample.items() if sample in eligible]
    table = pd.read_csv(path, sep="\t", usecols=[header[0], *selected], low_memory=False)
    gene_column = table.columns[0]
    table[gene_column] = table[gene_column].map(lambda value: clean_gene(value, aliases))
    table[selected] = table[selected].apply(pd.to_numeric, errors="coerce")
    matrix = table.groupby(gene_column, sort=False)[selected].median().T
    matrix.index = [column_to_sample[column] for column in matrix.index]
    matrix = canonicalize_expression(matrix, aliases, approved).reindex(clinical["SAMPLE_ID"])
    matrix.to_parquet(target, compression="zstd")


def prepare_paca_au(aliases: dict[str, str], approved: set[str]) -> None:
    target = PROCESSED / PRIMARY["PACA-AU"][0]
    if target.exists():
        return
    clinical = pd.read_csv(PROCESSED / PRIMARY["PACA-AU"][1], dtype=str)
    table = pd.read_csv(RAW / "PACA_AU_expression.txt", sep="\t", low_memory=False)
    gene_column = "Hugo_Symbol"
    sample_columns = [column for column in clinical["SAMPLE_ID"] if column in table.columns]
    table[gene_column] = table[gene_column].map(lambda value: clean_gene(value, aliases))
    table[sample_columns] = table[sample_columns].apply(pd.to_numeric, errors="coerce")
    matrix = table.groupby(gene_column, sort=False)[sample_columns].median().T
    matrix = canonicalize_expression(matrix, aliases, approved).reindex(clinical["SAMPLE_ID"])
    matrix.to_parquet(target, compression="zstd")


def prepare_gse71729(aliases: dict[str, str], approved: set[str]) -> None:
    target = PROCESSED / PRIMARY["GSE71729"][0]
    if target.exists():
        return
    clinical = pd.read_csv(PROCESSED / PRIMARY["GSE71729"][1], dtype=str)
    table = series_matrix(RAW / "GSE71729_series_matrix.txt.gz")
    gene_column = table.columns[0]
    sample_columns = [column for column in clinical["SAMPLE_ID"] if column in table.columns]
    table[gene_column] = table[gene_column].map(lambda value: clean_gene(value, aliases))
    table[sample_columns] = table[sample_columns].apply(pd.to_numeric, errors="coerce")
    matrix = table.groupby(gene_column, sort=False)[sample_columns].median().T
    matrix = canonicalize_expression(matrix, aliases, approved).reindex(clinical["SAMPLE_ID"])
    matrix.to_parquet(target, compression="zstd")


def prepare_gse205154(aliases: dict[str, str], approved: set[str]) -> None:
    expression_target = PROCESSED / AUXILIARY["GSE205154_MET"][0]
    clinical_target = PROCESSED / AUXILIARY["GSE205154_MET"][1]
    if expression_target.exists() and clinical_target.exists():
        return
    rds_path = next(
        (RAW / "precog_annotations").rglob("clinical_PDAC_GSE205154_metastatic.rds")
    )
    clinical = next(iter(pyreadr.read_r(rds_path).values())).copy()
    clinical["OS_TIME"] = pd.to_numeric(clinical["OS_Days"], errors="coerce") / 30.4375
    clinical["OS_EVENT"] = pd.to_numeric(clinical["OS_Censor"], errors="coerce")
    clinical = clinical[
        clinical["OS_TIME"].gt(0) & clinical["OS_EVENT"].isin([0, 1])
    ].copy()
    clinical["SAMPLE_ID"] = clinical["Patient_ID"].astype(str)
    clinical["PATIENT_ID"] = clinical["supp_id"].astype(str)
    clinical["SOURCE_COHORT"] = "GSE205154_MET"
    clinical["TISSUE_CONTEXT"] = clinical["Tissue"].astype(str)
    clinical = clinical[
        [
            "SAMPLE_ID",
            "PATIENT_ID",
            "geo_accession",
            "TISSUE_CONTEXT",
            "Primary_Metastatic",
            "OS_TIME",
            "OS_EVENT",
        ]
    ]
    clinical.to_csv(clinical_target, index=False)

    path = RAW / "GSE205154_Gene_Level_TMM_Estimates.txt.gz"
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    sample_columns = [sample for sample in clinical["SAMPLE_ID"] if sample in header]
    table = pd.read_csv(
        path,
        sep="\t",
        usecols=["hgnc_symbol", *sample_columns],
        low_memory=False,
    )
    table["hgnc_symbol"] = table["hgnc_symbol"].map(
        lambda value: clean_gene(value, aliases)
    )
    table[sample_columns] = table[sample_columns].apply(pd.to_numeric, errors="coerce")
    matrix = table.groupby("hgnc_symbol", sort=False)[sample_columns].median().T
    matrix = canonicalize_expression(matrix, aliases, approved).reindex(clinical["SAMPLE_ID"])
    matrix.to_parquet(expression_target, compression="zstd")


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names) - {"__index_level_0__", "SAMPLE_ID"}


def clinical_ids(frame: pd.DataFrame) -> list[str]:
    for column in ("SAMPLE_ID", "PATIENT_ID"):
        if column in frame:
            return frame[column].astype(str).tolist()
    raise RuntimeError("Clinical file lacks a sample/patient identifier")


def main() -> None:
    migration = json.loads(MIGRATION.read_text(encoding="utf-8"))
    completion = json.loads(V3_COMPLETION.read_text(encoding="utf-8"))
    if migration["status"] != "LOCKED_BEFORE_V4_MODEL_DEVELOPMENT":
        raise RuntimeError("Unexpected v4 migration lock")
    if completion["result_gate"] != "RED_NOT_SUPPORTED":
        raise RuntimeError("v4 expansion is only permitted after the locked v3.3 red result")
    aliases, approved = hgnc_aliases()
    prepare_gse183795(aliases, approved)
    prepare_paca_au(aliases, approved)
    prepare_gse71729(aliases, approved)
    prepare_gse205154(aliases, approved)

    all_domains = {**PRIMARY, **AUXILIARY}
    gene_sets = []
    for expression_name, _ in all_domains.values():
        path = PROCESSED / expression_name
        if not path.exists():
            raise FileNotFoundError(path)
        gene_sets.append(parquet_columns(path) & approved)
    universe = sorted(set.intersection(*gene_sets))
    if len(universe) < 8000:
        raise RuntimeError(f"Unexpectedly small strict v4 gene intersection: {len(universe)}")
    pd.DataFrame({"gene": universe}).to_csv(
        MANIFESTS / "v4_strict_common_gene_universe.csv", index=False
    )
    transformer = SingleSampleRankTransformer().fit(universe)

    audit_rows = []
    hash_rows = []
    for domain, (expression_name, clinical_name) in all_domains.items():
        expression_path = PROCESSED / expression_name
        clinical_path = PROCESSED / clinical_name
        clinical = pd.read_csv(clinical_path)
        ids = clinical_ids(clinical)
        expression = pd.read_parquet(expression_path, columns=universe)
        expression.index = expression.index.astype(str)
        expression = expression.reindex(ids)
        if expression.isna().all(axis=1).any():
            raise RuntimeError(f"{domain} has an unmatched clinical sample")
        rank = transformer.transform(expression, features=universe)
        rank.index = expression.index
        rank.index.name = "SAMPLE_ID"
        rank_path = PROCESSED / f"{domain.lower().replace('-', '_')}_v4_single_sample_rank.parquet"
        rank.to_parquet(rank_path, compression="zstd")
        audit_rows.append(
            {
                "domain": domain,
                "role": "primary_development" if domain in PRIMARY else "auxiliary_metastatic_development",
                "n": int(len(clinical)),
                "events": int(pd.to_numeric(clinical["OS_EVENT"], errors="coerce").sum()),
                "genes_before_intersection": int(len(parquet_columns(expression_path))),
                "v4_universe_genes": int(len(universe)),
                "rank_missing_fraction": float(rank.isna().mean().mean()),
                "clinical_expression_aligned": bool(ids == rank.index.astype(str).tolist()),
            }
        )
        for artifact in (expression_path, clinical_path, rank_path):
            hash_rows.append(
                {
                    "domain": domain,
                    "artifact": str(artifact.relative_to(ROOT)),
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256(artifact),
                }
            )
        del expression, rank
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(RESULTS / "35_v4_development_data_audit.csv", index=False)
    pd.DataFrame(hash_rows).to_csv(
        MANIFESTS / "v4_development_file_hashes.csv", index=False
    )
    summary = {
        "status": "V4_EXPANDED_DEVELOPMENT_READY",
        "primary_domains": list(PRIMARY),
        "auxiliary_domains": list(AUXILIARY),
        "primary_total_n": int(audit.loc[audit["role"] == "primary_development", "n"].sum()),
        "primary_total_events": int(audit.loc[audit["role"] == "primary_development", "events"].sum()),
        "auxiliary_total_n": int(audit.loc[audit["role"].str.startswith("auxiliary"), "n"].sum()),
        "auxiliary_total_events": int(audit.loc[audit["role"].str.startswith("auxiliary"), "events"].sum()),
        "strict_common_gene_universe": len(universe),
        "gse224564_used": False,
        "fresh_candidate_scores_generated": False,
        "migration_lock_sha256": sha256(MIGRATION),
    }
    (RESULTS / "35_v4_development_data_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
