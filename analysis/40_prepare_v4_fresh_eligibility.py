from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MANIFESTS = ROOT / "data" / "manifests"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"

FROZEN_SPEC = CONFIG / "frozen_model_v4_marginrank_spec.json"
MODEL = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"
FRESH_LOCK = CONFIG / "V4_FRESH_CANDIDATES.lock"
PACA_CA_LOCK = CONFIG / "V4_FRESH_CANDIDATES_AMENDMENT_PACA_CA.lock"
GSE77435_OVERLAP_LOCK = CONFIG / "V4_GSE77435_SOURCE_OVERLAP_AMENDMENT.lock"
UNIVERSE_FILE = MANIFESTS / "v4_strict_common_gene_universe.csv"
HGNC_FILE = RAW / "hgnc_complete_set_20260721.txt"

ICGC_ENDPOINT = "https://object.genomeinformatics.org"
ICGC_BUCKET = "icgc25k-open"
ICGC_PREFIX = "release_28/data/PACA-CA/"
ICGC_RAW = RAW / "icgc_release28"
CBIO_API = "https://www.cbioportal.org/api"
PRINCE_STUDY = "paad_iatlas_prince_2022"
DEVELOPMENT_CLINICAL_FILES = [
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
    "gse205154_met_v4_clinical.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_governance() -> dict:
    spec = json.loads(FROZEN_SPEC.read_text(encoding="utf-8"))
    lock = json.loads(FRESH_LOCK.read_text(encoding="utf-8"))
    amendment = json.loads(PACA_CA_LOCK.read_text(encoding="utf-8"))
    overlap_amendment = json.loads(GSE77435_OVERLAP_LOCK.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_V4_FRESH_RISK_PREDICTION":
        raise RuntimeError("Unexpected v4 frozen status")
    if sha256(MODEL) != spec["model_sha256"]:
        raise RuntimeError("Frozen v4 model hash mismatch")
    if lock["status"] != "CANDIDATES_RESERVED_BEFORE_V4_MODEL_SELECTION":
        raise RuntimeError("Unexpected original fresh-candidate lock")
    if amendment["status"] != "LOCKED_BEFORE_ANY_V4_FRESH_RISK_SCORE":
        raise RuntimeError("Unexpected PACA-CA amendment lock")
    if amendment["governance"]["model_hash_at_lock"] != spec["model_sha256"]:
        raise RuntimeError("PACA-CA amendment does not bind the frozen model")
    if (
        overlap_amendment["status"] != "LOCKED_BEFORE_ANY_V4_FRESH_RISK_SCORE"
        or overlap_amendment["independent_validation_role"]
        != "PERMANENTLY_INELIGIBLE_SOURCE_COHORT_OVERLAP"
    ):
        raise RuntimeError("Unexpected GSE77435 overlap amendment")
    return spec


def hgnc_maps() -> tuple[dict[str, str], dict[str, str], set[str]]:
    table = pd.read_csv(HGNC_FILE, sep="\t", dtype=str, low_memory=False)
    table = table[table["status"].eq("Approved")].copy()
    approved = set(table["symbol"].dropna().str.upper())
    alias_candidates: dict[str, set[str]] = {}
    for row in table[["symbol", "alias_symbol", "prev_symbol"]].itertuples(index=False):
        current = str(row.symbol).upper()
        for field in (row.alias_symbol, row.prev_symbol):
            if pd.isna(field):
                continue
            for alias in str(field).split("|"):
                alias = alias.strip().upper()
                if alias and alias not in approved:
                    alias_candidates.setdefault(alias, set()).add(current)
    aliases = {symbol: symbol for symbol in approved}
    aliases.update(
        {
            alias: next(iter(symbols))
            for alias, symbols in alias_candidates.items()
            if len(symbols) == 1
        }
    )
    ensembl: dict[str, str] = {}
    for row in table[["symbol", "ensembl_gene_id"]].dropna().itertuples(index=False):
        for identifier in str(row.ensembl_gene_id).split("|"):
            identifier = identifier.strip().split(".")[0]
            if identifier:
                ensembl.setdefault(identifier, str(row.symbol).upper())
    return aliases, ensembl, approved


def clean_symbol(value: object, aliases: dict[str, str]) -> str:
    symbol = str(value).strip().strip('"').upper()
    symbol = re.split(r"\s*///\s*|\s*//\s*|;|,", symbol)[0].strip()
    return aliases.get(symbol, symbol)


def development_identifiers() -> set[str]:
    identifiers: set[str] = set()
    for filename in DEVELOPMENT_CLINICAL_FILES:
        path = PROCESSED / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        for column in frame.columns:
            upper = column.upper()
            if upper.endswith("_ID") or upper in {"SAMPLE", "PATIENT"}:
                identifiers.update(frame[column].dropna().astype(str).str.strip())
    identifiers.discard("")
    return identifiers


def cbio_get(session: requests.Session, path: str, **params: object) -> object:
    response = session.get(f"{CBIO_API}{path}", params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def cbio_post(
    session: requests.Session, path: str, body: dict, **params: object
) -> object:
    response = session.post(
        f"{CBIO_API}{path}", params=params, json=body, timeout=120
    )
    response.raise_for_status()
    return response.json()


def prepare_prince(dev_ids: set[str]) -> tuple[dict, list[Path]]:
    session = requests.Session()
    study = cbio_get(session, f"/studies/{PRINCE_STUDY}", projection="DETAILED")
    samples = cbio_get(
        session,
        f"/studies/{PRINCE_STUDY}/samples",
        projection="DETAILED",
        pageSize=1000,
    )
    sample_ids = [row["sampleId"] for row in samples]
    patient_ids = sorted({row["patientId"] for row in samples})
    sample_attributes = [
        "SAMPLE_TREATMENT",
        "BIOPSY_SITE",
        "CANCER_TISSUE",
        "SPECIMEN_PRESERVATION_TYPE",
    ]
    patient_attributes = [
        "OS_MONTHS",
        "OS_STATUS",
        "AGE_AT_DIAGNOSIS",
        "SEX",
        "ICI_PATHWAY",
        "ICI_RX",
        "CLINICAL_STAGE",
    ]
    sample_clinical = cbio_post(
        session,
        f"/studies/{PRINCE_STUDY}/clinical-data/fetch",
        {"ids": sample_ids, "attributeIds": sample_attributes},
        clinicalDataType="SAMPLE",
        projection="DETAILED",
    )
    patient_clinical = cbio_post(
        session,
        f"/studies/{PRINCE_STUDY}/clinical-data/fetch",
        {"ids": patient_ids, "attributeIds": patient_attributes},
        clinicalDataType="PATIENT",
        projection="DETAILED",
    )
    prince_raw = RAW / "prince_v4_fresh_eligibility"
    prince_raw.mkdir(parents=True, exist_ok=True)
    raw_paths = [
        prince_raw / "study.json",
        prince_raw / "samples.json",
        prince_raw / "sample_clinical.json",
        prince_raw / "patient_clinical.json",
    ]
    for path, value in zip(
        raw_paths, [study, samples, sample_clinical, patient_clinical], strict=True
    ):
        write_json(path, value)

    sample_map = pd.DataFrame(samples).set_index("sampleId")[["patientId"]]
    sample_long = pd.DataFrame(sample_clinical)
    sample_wide = sample_long.pivot(
        index="sampleId", columns="clinicalAttributeId", values="value"
    ).join(sample_map)
    pre = sample_wide[sample_wide["SAMPLE_TREATMENT"].eq("Pre")].copy()
    pre = (
        pre.reset_index()
        .sort_values(["patientId", "sampleId"])
        .drop_duplicates("patientId", keep="first")
        .set_index("patientId")
    )
    patient_wide = pd.DataFrame(patient_clinical).pivot(
        index="patientId", columns="clinicalAttributeId", values="value"
    )
    eligible = pre.join(patient_wide, how="left")
    eligible["OS_TIME"] = pd.to_numeric(eligible["OS_MONTHS"], errors="coerce")
    eligible["OS_EVENT"] = np.where(
        eligible["OS_STATUS"].astype(str).str.startswith("1"), 1, 0
    )
    eligible = eligible[
        eligible["OS_TIME"].gt(0) & eligible["OS_STATUS"].notna()
    ].copy()
    eligible = eligible.reset_index().rename(
        columns={"patientId": "PATIENT_ID", "sampleId": "SAMPLE_ID"}
    )
    overlap = sorted(
        (set(eligible["PATIENT_ID"]) | set(eligible["SAMPLE_ID"])) & dev_ids
    )
    clinical_path = PROCESSED / "prince_v4_fresh_eligibility_clinical.csv"
    eligible.to_csv(clinical_path, index=False)
    summary = {
        "role": "reserved_metastatic_transportability_stress",
        "study": PRINCE_STUDY,
        "all_samples": int(len(samples)),
        "all_patients": int(len(patient_ids)),
        "pretreatment_samples": int(len(pre)),
        "pretreatment_patients_with_valid_os": int(len(eligible)),
        "pretreatment_deaths": int(eligible["OS_EVENT"].sum()),
        "minimum_required_deaths": 20,
        "identifier_overlap_count": len(overlap),
        "identifier_overlap": overlap,
        "primary_eligibility": "FAIL_MINIMUM_EVENTS",
        "risk_scores_generated": False,
        "expression_downloaded": False,
    }
    return summary, [*raw_paths, clinical_path]


def parse_series_matrix(path: Path) -> tuple[dict[str, list[list[str]]], pd.DataFrame]:
    metadata: dict[str, list[list[str]]] = {}
    table_lines: list[str] = []
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if stripped.startswith("!") and "\t" in stripped:
                key, values = stripped.split("\t", 1)
                parsed = [item.strip().strip('"') for item in values.split("\t")]
                metadata.setdefault(key, []).append(parsed)
            if stripped == "!series_matrix_table_begin":
                in_table = True
                continue
            if stripped == "!series_matrix_table_end":
                break
            if in_table:
                table_lines.append(stripped)
    table = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", low_memory=False)
    table.columns = [str(column).strip().strip('"') for column in table.columns]
    return metadata, table


def prepare_gse77435(
    universe: list[str],
    selected_genes: list[str],
    anchors: list[str],
    aliases: dict[str, str],
    approved: set[str],
    dev_ids: set[str],
) -> tuple[dict, list[Path]]:
    matrix_path = RAW / "GSE77435_series_matrix.txt.gz"
    nonnormalized_path = RAW / "GSE77435_non-normalized.txt.gz"
    if not matrix_path.exists() or not nonnormalized_path.exists():
        raise FileNotFoundError("GSE77435 official files were not downloaded")
    metadata, table = parse_series_matrix(matrix_path)
    accessions = metadata["!Sample_geo_accession"][0]
    titles = metadata["!Sample_title"][0]
    if len(accessions) != 24 or len(titles) != 24:
        raise RuntimeError("Unexpected GSE77435 sample count")
    clinical_rows = []
    for accession, title in zip(accessions, titles, strict=True):
        if "Long-term" in title:
            group = "long"
            short_label = 0
        elif "Short survival" in title:
            group = "short"
            short_label = 1
        else:
            raise RuntimeError(f"Unmapped GSE77435 title: {title}")
        clinical_rows.append(
            {
                "SAMPLE_ID": accession,
                "PATIENT_ID": accession,
                "TITLE": title,
                "SURVIVAL_GROUP": group,
                "SHORT_SURVIVAL_LABEL": short_label,
                "TISSUE_CONTEXT": "resected primary PDAC",
            }
        )
    clinical = pd.DataFrame(clinical_rows)
    gene_column = table.columns[0]
    table[gene_column] = table[gene_column].map(lambda value: clean_symbol(value, aliases))
    table = table[table[gene_column].isin(approved)].copy()
    sample_columns = [sample for sample in accessions if sample in table.columns]
    if sample_columns != accessions:
        raise RuntimeError("GSE77435 series-matrix samples do not align to metadata")
    table[sample_columns] = table[sample_columns].apply(pd.to_numeric, errors="coerce")
    expression = table.groupby(gene_column, sort=False)[sample_columns].median().T
    expression = expression.reindex(index=accessions, columns=universe)
    expression.index.name = "SAMPLE_ID"
    per_sample_reference = expression.notna().mean(axis=1)
    per_sample_selected = expression.reindex(columns=selected_genes).notna().mean(axis=1)
    per_sample_anchor = expression.reindex(columns=anchors).notna().mean(axis=1)
    overlap = sorted((set(accessions) | set(clinical["PATIENT_ID"])) & dev_ids)
    coverage_pass = bool(
        per_sample_reference.ge(0.80).all()
        and per_sample_selected.ge(0.80).all()
        and per_sample_anchor.ge(0.50).all()
    )
    expression_path = PROCESSED / "gse77435_v4_fresh_expression.parquet"
    clinical_path = PROCESSED / "gse77435_v4_fresh_clinical.csv"
    expression.to_parquet(expression_path, compression="zstd")
    clinical.to_csv(clinical_path, index=False)
    summary = {
        "role": "technical_cross_platform_reproducibility_only",
        "n": int(len(clinical)),
        "short_survivors": int(clinical["SHORT_SURVIVAL_LABEL"].sum()),
        "long_survivors": int((1 - clinical["SHORT_SURVIVAL_LABEL"]).sum()),
        "minimum_reference_coverage": float(per_sample_reference.min()),
        "minimum_selected_gene_coverage": float(per_sample_selected.min()),
        "minimum_anchor_gene_coverage": float(per_sample_anchor.min()),
        "coverage_pass": coverage_pass,
        "identifier_overlap_count": len(overlap),
        "identifier_overlap": overlap,
        "source_cohort_overlap": "GSE62452 development domain",
        "eligibility": "INELIGIBLE_SOURCE_COHORT_OVERLAP",
        "risk_scores_generated": False,
    }
    return summary, [matrix_path, nonnormalized_path, expression_path, clinical_path]


def list_s3_objects(prefix: str) -> pd.DataFrame:
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    rows: list[dict] = []
    token: str | None = None
    while True:
        params: dict[str, object] = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": 1000,
        }
        if token:
            params["continuation-token"] = token
        response = requests.get(
            f"{ICGC_ENDPOINT}/{ICGC_BUCKET}", params=params, timeout=120
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for content in root.findall("s3:Contents", namespace):
            rows.append(
                {
                    "key": content.findtext("s3:Key", namespaces=namespace),
                    "last_modified": content.findtext(
                        "s3:LastModified", namespaces=namespace
                    ),
                    "etag": content.findtext("s3:ETag", namespaces=namespace).strip('"'),
                    "bytes": int(content.findtext("s3:Size", namespaces=namespace)),
                }
            )
        truncated = root.findtext("s3:IsTruncated", namespaces=namespace) == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=namespace)
        if not token:
            raise RuntimeError("Truncated ICGC object listing omitted continuation token")
    frame = pd.DataFrame(rows).drop_duplicates("key").sort_values("key")
    return frame.reset_index(drop=True)


def icgc_local_path(key: str) -> Path:
    if not key.startswith("release_28/") or ".." in Path(key).parts:
        raise RuntimeError(f"Unsafe ICGC object key: {key}")
    return ICGC_RAW.joinpath(*Path(key).parts[1:])


def download_icgc_object(row: dict) -> Path:
    key = str(row["key"])
    expected = int(row["bytes"])
    target = icgc_local_path(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected:
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    url = f"{ICGC_ENDPOINT}/{ICGC_BUCKET}/{key}"
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"Size mismatch for {key}: {temporary.stat().st_size} != {expected}"
                )
            temporary.replace(target)
            return target
        except Exception:
            if temporary.exists():
                temporary.unlink()
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def download_many(manifest: pd.DataFrame, workers: int = 12) -> list[Path]:
    rows = manifest.to_dict("records")
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_icgc_object, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            paths.append(future.result())
            if index % 100 == 0 or index == len(rows):
                print(f"Downloaded/verified {index}/{len(rows)} ICGC objects", flush=True)
    return sorted(paths)


def fetch_icgc_headers(types: list[str]) -> tuple[dict[str, list[str]], list[Path]]:
    headers: dict[str, list[str]] = {}
    paths: list[Path] = []
    for data_type in types:
        key = f"release_28/headers/{data_type}.tsv.gz"
        response = requests.get(
            f"{ICGC_ENDPOINT}/{ICGC_BUCKET}/{key}", timeout=120
        )
        response.raise_for_status()
        target = icgc_local_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != response.content:
            target.write_bytes(response.content)
        header = gzip.decompress(response.content).decode("utf-8").splitlines()[0]
        headers[data_type] = header.split("\t")
        paths.append(target)
    return headers, paths


def read_icgc_parts(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(
            path,
            sep="\t",
            names=columns,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=False,
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def data_type_from_key(key: str) -> str | None:
    match = re.search(r"/(donor|specimen|sample|exp_seq)/", key)
    return match.group(1) if match else None


def prepare_paca_ca(
    universe: list[str],
    selected_genes: list[str],
    anchors: list[str],
    ensembl_to_symbol: dict[str, str],
    dev_ids: set[str],
) -> tuple[dict, list[Path]]:
    object_manifest = list_s3_objects(ICGC_PREFIX)
    object_manifest["data_type"] = object_manifest["key"].map(data_type_from_key)
    manifest_path = MANIFESTS / "paca_ca_release28_object_manifest.csv"
    object_manifest.to_csv(manifest_path, index=False)
    counts = object_manifest["data_type"].value_counts(dropna=False).to_dict()
    print(f"PACA-CA object listing: {len(object_manifest)} objects; {counts}", flush=True)

    headers, header_paths = fetch_icgc_headers(
        ["donor", "specimen", "sample", "exp_seq"]
    )
    metadata_manifest = object_manifest[
        object_manifest["data_type"].isin(["donor", "specimen", "sample"])
    ].copy()
    metadata_paths = download_many(metadata_manifest)
    type_paths: dict[str, list[Path]] = {name: [] for name in headers}
    key_by_local = {
        str(icgc_local_path(key)): key for key in object_manifest["key"].astype(str)
    }
    for path in metadata_paths:
        key = key_by_local[str(path)]
        data_type = data_type_from_key(key)
        if data_type:
            type_paths[data_type].append(path)
    donors = read_icgc_parts(type_paths["donor"], headers["donor"])
    specimens = read_icgc_parts(type_paths["specimen"], headers["specimen"])
    samples = read_icgc_parts(type_paths["sample"], headers["sample"])

    donors = donors[donors["project_code"].eq("PACA-CA")].copy()
    donors["survival_days"] = pd.to_numeric(
        donors["donor_survival_time"], errors="coerce"
    )
    donors["vital"] = donors["donor_vital_status"].str.lower().str.strip()
    donors = donors[
        donors["survival_days"].gt(30)
        & donors["vital"].isin(["alive", "deceased"])
    ].copy()
    specimens["specimen_type_norm"] = specimens["specimen_type"].str.strip()
    specimens["treatment_norm"] = (
        specimens["specimen_donor_treatment_type"].str.lower().str.strip()
    )
    specimens = specimens[
        specimens["project_code"].eq("PACA-CA")
        & specimens["specimen_type_norm"].str.startswith("Primary tumour")
        & specimens["treatment_norm"].isin(["", "no treatment"])
    ].copy()
    sample_meta = samples.merge(
        specimens,
        on=["project_code", "icgc_donor_id", "icgc_specimen_id"],
        how="inner",
        suffixes=("_sample", "_specimen"),
    ).merge(
        donors,
        on=["project_code", "icgc_donor_id"],
        how="inner",
        suffixes=("", "_donor"),
    )
    candidate_samples = set(sample_meta["icgc_sample_id"].astype(str))
    candidate_donors = set(sample_meta["icgc_donor_id"].astype(str))
    exp_manifest = object_manifest[
        object_manifest["data_type"].eq("exp_seq")
        & object_manifest["key"].str.split("/").str[3].isin(candidate_donors)
    ].copy()
    exp_paths = download_many(exp_manifest)

    universe_set = set(universe)
    profile_parts: dict[tuple[str, str, str, str], list[pd.Series]] = {}
    technical: dict[tuple[str, str, str, str], float] = {}
    for index, path in enumerate(exp_paths, start=1):
        frame = pd.read_csv(
            path,
            sep="\t",
            names=headers["exp_seq"],
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=False,
        )
        frame = frame[
            frame["project_code"].eq("PACA-CA")
            & frame["icgc_sample_id"].isin(candidate_samples)
        ].copy()
        if frame.empty:
            continue
        frame["symbol"] = (
            frame["gene_id"].str.split(".").str[0].map(ensembl_to_symbol)
        )
        frame = frame[frame["symbol"].isin(universe_set)].copy()
        frame["value"] = pd.to_numeric(
            frame["normalized_read_count"], errors="coerce"
        )
        frame["total_reads"] = pd.to_numeric(
            frame["total_read_count"], errors="coerce"
        )
        for group_key, group in frame.groupby(
            ["icgc_donor_id", "icgc_specimen_id", "icgc_sample_id", "analysis_id"],
            sort=False,
        ):
            profile = group.groupby("symbol", sort=False)["value"].median()
            profile_parts.setdefault(tuple(group_key), []).append(profile)
            depth = float(group["total_reads"].max()) if group["total_reads"].notna().any() else -1.0
            technical[tuple(group_key)] = max(technical.get(tuple(group_key), -1.0), depth)
        if index % 50 == 0 or index == len(exp_paths):
            print(f"Parsed {index}/{len(exp_paths)} PACA-CA expression objects", flush=True)

    sample_lookup = sample_meta.drop_duplicates("icgc_sample_id").set_index("icgc_sample_id")
    candidate_profiles: list[dict] = []
    for key, parts in profile_parts.items():
        donor_id, specimen_id, sample_id, analysis_id = key
        merged = pd.concat(parts, axis=1).median(axis=1, skipna=True).reindex(universe)
        reference_coverage = float(merged.notna().mean())
        selected_coverage = float(merged.reindex(selected_genes).notna().mean())
        anchor_coverage = float(merged.reindex(anchors).notna().mean())
        candidate_profiles.append(
            {
                "icgc_donor_id": donor_id,
                "icgc_specimen_id": specimen_id,
                "icgc_sample_id": sample_id,
                "analysis_id": analysis_id,
                "reference_coverage": reference_coverage,
                "selected_gene_coverage": selected_coverage,
                "anchor_gene_coverage": anchor_coverage,
                "total_read_count": technical[key],
                "profile": merged,
            }
        )
    if not candidate_profiles:
        raise RuntimeError("No PACA-CA candidate expression profiles were parsed")
    candidate_table = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "profile"} for row in candidate_profiles]
    )
    passing = candidate_table[
        candidate_table["reference_coverage"].ge(0.80)
        & candidate_table["selected_gene_coverage"].ge(0.80)
        & candidate_table["anchor_gene_coverage"].ge(0.50)
    ].copy()
    passing = passing.sort_values(
        [
            "icgc_donor_id",
            "reference_coverage",
            "total_read_count",
            "icgc_sample_id",
            "analysis_id",
        ],
        ascending=[True, False, False, True, True],
    ).drop_duplicates("icgc_donor_id", keep="first")
    chosen_keys = set(
        zip(
            passing["icgc_donor_id"],
            passing["icgc_specimen_id"],
            passing["icgc_sample_id"],
            passing["analysis_id"],
            strict=True,
        )
    )
    profile_by_key = {
        (
            row["icgc_donor_id"],
            row["icgc_specimen_id"],
            row["icgc_sample_id"],
            row["analysis_id"],
        ): row["profile"]
        for row in candidate_profiles
    }
    selected_rows = passing.sort_values("icgc_donor_id")
    expression = pd.DataFrame(
        [profile_by_key[key] for key in chosen_keys],
        index=[key[2] for key in chosen_keys],
    )
    # Reorder deterministically after constructing from a set.
    selected_sample_order = selected_rows["icgc_sample_id"].astype(str).tolist()
    expression = expression.reindex(index=selected_sample_order, columns=universe)
    expression.index.name = "SAMPLE_ID"

    selected_meta = selected_rows.merge(
        sample_lookup.reset_index(), on="icgc_sample_id", how="left", suffixes=("", "_meta")
    )
    selected_meta["OS_TIME"] = selected_meta["survival_days"] / 30.4375
    selected_meta["OS_EVENT"] = selected_meta["vital"].eq("deceased").astype(int)
    clinical = pd.DataFrame(
        {
            "SAMPLE_ID": selected_meta["icgc_sample_id"],
            "PATIENT_ID": selected_meta["icgc_donor_id"],
            "SUBMITTED_PATIENT_ID": selected_meta["submitted_donor_id"],
            "SPECIMEN_ID": selected_meta["icgc_specimen_id"],
            "SUBMITTED_SPECIMEN_ID": selected_meta["submitted_specimen_id_specimen"],
            "ANALYSIS_ID": selected_meta["analysis_id"],
            "SPECIMEN_TYPE": selected_meta["specimen_type"],
            "SPECIMEN_TREATMENT": selected_meta["specimen_donor_treatment_type"],
            "HISTOLOGICAL_TYPE": selected_meta["tumour_histological_type"],
            "TUMOUR_STAGE": selected_meta["tumour_stage"],
            "TUMOUR_GRADE": selected_meta["tumour_grade"],
            "AGE_AT_DIAGNOSIS": selected_meta["donor_age_at_diagnosis"],
            "SEX": selected_meta["donor_sex"],
            "OS_TIME": selected_meta["OS_TIME"],
            "OS_EVENT": selected_meta["OS_EVENT"],
            "REFERENCE_COVERAGE": selected_meta["reference_coverage"],
            "SELECTED_GENE_COVERAGE": selected_meta["selected_gene_coverage"],
            "ANCHOR_GENE_COVERAGE": selected_meta["anchor_gene_coverage"],
            "TOTAL_READ_COUNT": selected_meta["total_read_count"],
            "SOURCE_PROJECT": "PACA-CA",
        }
    )
    clinical = clinical.sort_values("PATIENT_ID").reset_index(drop=True)
    expression = expression.reindex(clinical["SAMPLE_ID"].astype(str))
    candidate_table_path = RESULTS / "40_paca_ca_profile_eligibility.csv"
    candidate_table.to_csv(candidate_table_path, index=False)
    clinical_path = PROCESSED / "paca_ca_v4_fresh_clinical.csv"
    expression_path = PROCESSED / "paca_ca_v4_fresh_expression.parquet"
    clinical.to_csv(clinical_path, index=False)
    expression.to_parquet(expression_path, compression="zstd")

    candidate_ids: set[str] = set()
    for column in [
        "SAMPLE_ID",
        "PATIENT_ID",
        "SUBMITTED_PATIENT_ID",
        "SPECIMEN_ID",
        "SUBMITTED_SPECIMEN_ID",
    ]:
        candidate_ids.update(clinical[column].dropna().astype(str).str.strip())
    overlaps = sorted(candidate_ids & dev_ids)
    n = int(len(clinical))
    events = int(clinical["OS_EVENT"].sum())
    one_per_donor = bool(clinical["PATIENT_ID"].is_unique)
    aligned = bool(clinical["SAMPLE_ID"].astype(str).tolist() == expression.index.astype(str).tolist())
    eligibility_pass = bool(
        n >= 100
        and events >= 50
        and one_per_donor
        and aligned
        and not overlaps
        and clinical["REFERENCE_COVERAGE"].ge(0.80).all()
        and clinical["SELECTED_GENE_COVERAGE"].ge(0.80).all()
        and clinical["ANCHOR_GENE_COVERAGE"].ge(0.50).all()
    )
    summary = {
        "role": "fresh_primary_tumour_time_to_event_validation",
        "source": "ICGC Release 28 relocated official open bucket",
        "project": "PACA-CA",
        "listed_objects": int(len(object_manifest)),
        "donor_records_with_valid_os": int(len(donors)),
        "primary_untreated_candidate_donors": int(len(candidate_donors)),
        "parsed_expression_profiles": int(len(candidate_table)),
        "eligible_one_profile_per_donor": n,
        "eligible_deaths": events,
        "minimum_reference_coverage": float(clinical["REFERENCE_COVERAGE"].min()),
        "minimum_selected_gene_coverage": float(clinical["SELECTED_GENE_COVERAGE"].min()),
        "minimum_anchor_gene_coverage": float(clinical["ANCHOR_GENE_COVERAGE"].min()),
        "one_profile_per_donor": one_per_donor,
        "clinical_expression_aligned": aligned,
        "identifier_overlap_count": len(overlaps),
        "identifier_overlap": overlaps,
        "eligibility": "ELIGIBLE_PRIMARY_FRESH" if eligibility_pass else "INELIGIBLE",
        "risk_scores_generated": False,
    }
    artifacts = [
        manifest_path,
        candidate_table_path,
        clinical_path,
        expression_path,
        *header_paths,
        *metadata_paths,
        *exp_paths,
    ]
    return summary, artifacts


def main() -> None:
    spec = validate_governance()
    universe = pd.read_csv(UNIVERSE_FILE)["gene"].astype(str).tolist()
    selected_genes = list(spec["selected_genes"])
    anchors = list(spec["anchor_genes"])
    aliases, ensembl_to_symbol, approved = hgnc_maps()
    dev_ids = development_identifiers()

    print("Auditing PRINCE eligibility without generating risk scores", flush=True)
    prince, prince_artifacts = prepare_prince(dev_ids)
    print(json.dumps(prince, indent=2), flush=True)

    print("Preparing GSE77435 outcome labels and frozen-gene coverage", flush=True)
    gse77435, gse_artifacts = prepare_gse77435(
        universe, selected_genes, anchors, aliases, approved, dev_ids
    )
    print(json.dumps(gse77435, indent=2), flush=True)

    print("Preparing PACA-CA eligibility from the official ICGC open bucket", flush=True)
    paca_ca, paca_artifacts = prepare_paca_ca(
        universe, selected_genes, anchors, ensembl_to_symbol, dev_ids
    )
    print(json.dumps(paca_ca, indent=2), flush=True)

    all_artifacts = sorted(
        {path.resolve() for path in prince_artifacts + gse_artifacts + paca_artifacts if path.exists()}
    )
    hash_rows = []
    for path in all_artifacts:
        hash_rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    hash_path = MANIFESTS / "v4_fresh_eligibility_artifact_hashes.csv"
    pd.DataFrame(hash_rows).to_csv(hash_path, index=False)

    summary = {
        "status": "V4_FRESH_ELIGIBILITY_COMPLETE_NO_RISK_SCORES",
        "frozen_model_sha256": spec["model_sha256"],
        "paca_ca": paca_ca,
        "gse77435": gse77435,
        "prince": prince,
        "any_fresh_risk_score_generated": False,
        "next_action": "freeze and hash the one-time analysis plan and scripts before loading the frozen model",
        "candidate_amendment_sha256": sha256(PACA_CA_LOCK),
        "gse77435_overlap_amendment_sha256": sha256(GSE77435_OVERLAP_LOCK),
        "artifact_manifest": str(hash_path.relative_to(ROOT)),
    }
    result_json = RESULTS / "40_v4_fresh_eligibility.json"
    write_json(result_json, summary)
    rows = []
    for name, result in [("PACA-CA", paca_ca), ("GSE77435", gse77435), ("PRINCE", prince)]:
        rows.append(
            {
                "candidate": name,
                "role": result["role"],
                "eligibility": result.get("eligibility", result.get("primary_eligibility")),
                "n": result.get("eligible_one_profile_per_donor", result.get("n", result.get("pretreatment_patients_with_valid_os"))),
                "events_or_short": result.get("eligible_deaths", result.get("short_survivors", result.get("pretreatment_deaths"))),
                "risk_scores_generated": False,
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "40_v4_fresh_eligibility.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
