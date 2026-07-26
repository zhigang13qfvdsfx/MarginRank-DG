from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data" / "raw" / "paad_iatlas_prince_2022"
ARCHIVE = ROOT / "data" / "raw" / "paad_iatlas_prince_2022.tar.gz"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
UNIVERSE = ROOT / "data" / "manifests" / "v4_strict_common_gene_universe.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cbio(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


def development_ids() -> set[str]:
    values: set[str] = set()
    for path in PROCESSED.glob("*_clinical.csv"):
        if path.name.startswith(("prince_", "paca_ca_", "gse77435_")):
            continue
        frame = pd.read_csv(path, dtype=str)
        for column in ("SAMPLE_ID", "PATIENT_ID"):
            if column in frame:
                values.update(frame[column].dropna().astype(str))
    return values


def main() -> None:
    meta = (STUDY / "meta_study.txt").read_text(encoding="utf-8")
    if "paad_iatlas_prince_2022" not in meta or "pmid: 35662283" not in meta:
        raise RuntimeError("Unexpected PRINCE DataHub study metadata")
    patient = read_cbio(STUDY / "data_clinical_patient.txt")
    sample = read_cbio(STUDY / "data_clinical_sample.txt")
    expression_path = STUDY / "data_mrna_seq_expression.txt"
    expression = pd.read_csv(expression_path, sep="\t", low_memory=False)
    if expression.columns[0] != "Hugo_Symbol":
        raise RuntimeError("Unexpected PRINCE expression gene column")
    expression["Hugo_Symbol"] = expression["Hugo_Symbol"].astype(str)
    expression = expression.groupby("Hugo_Symbol", sort=False).median(numeric_only=True)

    clinical = sample[sample["SAMPLE_TREATMENT"].eq("Pre")].merge(
        patient, on="PATIENT_ID", how="left", validate="one_to_one"
    )
    clinical["OS_TIME"] = pd.to_numeric(clinical["OS_MONTHS"], errors="coerce")
    clinical["OS_EVENT"] = clinical["OS_STATUS"].astype(str).str.startswith("1:").astype(int)
    clinical = clinical[
        clinical["OS_TIME"].gt(0)
        & clinical["OS_STATUS"].astype(str).str.match(r"^[01]:")
        & clinical["SAMPLE_ID"].isin(expression.columns)
    ].copy()
    if clinical["PATIENT_ID"].duplicated().any() or clinical["SAMPLE_ID"].duplicated().any():
        raise RuntimeError("Pretreatment PRINCE set is not one sample per patient")
    clinical = clinical.sort_values("PATIENT_ID").reset_index(drop=True)

    universe = pd.read_csv(UNIVERSE)["gene"].astype(str).tolist()
    prepared = expression.reindex(index=universe, columns=clinical["SAMPLE_ID"]).T
    prepared.index = clinical["SAMPLE_ID"].astype(str)
    prepared.index.name = "SAMPLE_ID"
    spec = json.loads(
        (CONFIG / "frozen_model_v4_marginrank_spec.json").read_text(encoding="utf-8")
    )
    selected = spec["selected_genes"]
    anchors = spec["anchor_genes"]
    reference_coverage = prepared.notna().mean(axis=1)
    selected_coverage = prepared[selected].notna().mean(axis=1)
    anchor_coverage = prepared[anchors].notna().mean(axis=1)
    overlap = sorted(
        (set(clinical["SAMPLE_ID"].astype(str)) | set(clinical["PATIENT_ID"].astype(str)))
        & development_ids()
    )
    prepared_path = PROCESSED / "prince_v4_exploratory_expression.parquet"
    clinical_path = PROCESSED / "prince_v4_exploratory_clinical.csv"
    prepared.to_parquet(prepared_path, compression="zstd")
    keep = [
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
        "SPECIMEN_PRESERVATION_TYPE",
        "CLINICAL_BENEFIT",
        "RESPONSE",
        "SAMPLE_TREATMENT",
    ]
    clinical[keep].to_csv(clinical_path, index=False)

    summary = {
        "status": "PRINCE_EXPLORATORY_ELIGIBILITY_COMPLETE_NO_RISK_SCORES",
        "study": "paad_iatlas_prince_2022",
        "source": "official cBioPortal DataHub archive",
        "pmid": 35662283,
        "role": "exploratory metastatic transportability stress test",
        "pretreatment_samples_with_valid_os": int(len(clinical)),
        "pretreatment_patients": int(clinical["PATIENT_ID"].nunique()),
        "deaths": int(clinical["OS_EVENT"].sum()),
        "one_sample_per_patient": True,
        "minimum_reference_coverage": float(reference_coverage.min()),
        "minimum_selected_gene_coverage": float(selected_coverage.min()),
        "minimum_anchor_gene_coverage": float(anchor_coverage.min()),
        "identifier_overlap_count": len(overlap),
        "identifier_overlap": overlap,
        "primary_validation_eligible": False,
        "primary_ineligibility_reason": "15 deaths, below the prespecified 20-event minimum",
        "exploratory_stress_eligible": bool(
            len(clinical) >= 50
            and clinical["OS_EVENT"].sum() >= 10
            and reference_coverage.min() >= 0.8
            and selected_coverage.min() >= 0.8
            and anchor_coverage.min() >= 0.5
            and not overlap
        ),
        "risk_scores_generated": False,
        "hashes": {
            "archive": sha256(ARCHIVE),
            "expression_source": sha256(expression_path),
            "patient_source": sha256(STUDY / "data_clinical_patient.txt"),
            "sample_source": sha256(STUDY / "data_clinical_sample.txt"),
            "prepared_expression": sha256(prepared_path),
            "prepared_clinical": sha256(clinical_path),
            "script": sha256(Path(__file__)),
        },
    }
    (RESULTS / "47_prince_exploratory_eligibility.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
