from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT
WHEEL = PACKAGE / "dist" / "transsurv_pdac-0.5.0-py3-none-any.whl"
MODEL = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"
CARD = ROOT / "src" / "transsurv" / "resources" / "model_card.json"
PYTEST_XML = ROOT / "qa" / "transsurv_pytest.xml"
COVERAGE_XML = ROOT / "qa" / "transsurv_coverage.xml"
SMOKE_SCORES = ROOT / "qa" / "transsurv_cli_smoke_050" / "scores.tsv"
OUTPUT = ROOT / "results" / "49_software_release_gate.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    card = json.loads(CARD.read_text(encoding="utf-8"))
    pytest_root = ElementTree.parse(PYTEST_XML).getroot()
    suite = pytest_root if pytest_root.tag == "testsuite" else pytest_root.find("testsuite")
    if suite is None:
        raise RuntimeError("Could not read pytest suite")
    tests = int(suite.attrib["tests"])
    failures = int(suite.attrib.get("failures", 0))
    errors = int(suite.attrib.get("errors", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    coverage_root = ElementTree.parse(COVERAGE_XML).getroot()
    line_rate = float(coverage_root.attrib["line-rate"])
    scores = pd.read_csv(SMOKE_SCORES, sep="\t")
    with zipfile.ZipFile(WHEEL) as archive:
        members = set(archive.namelist())
    wheel_has_model = any(
        name.endswith("models/transsurv_pdac_v4_marginrank_locked.joblib")
        for name in members
    )
    wheel_has_card = any(name.endswith("resources/model_card.json") for name in members)
    conditions = {
        "all_tests_pass": failures == 0 and errors == 0 and tests >= 28,
        "test_coverage_at_least_80_percent": line_rate >= 0.80,
        "wheel_contains_frozen_model": wheel_has_model,
        "wheel_contains_model_card": wheel_has_card,
        "model_hash_matches_card": sha256(MODEL) == card["model_sha256"],
        "clean_cli_scores_all_samples": bool(
            len(scores) == 40 and scores["raw_risk"].notna().all()
        ),
        "clean_cli_emits_batch_relative_outputs": bool(
            scores["risk_percentile"].notna().all()
        ),
        "clean_cli_has_no_degraded_inputs": bool(not scores["input_degraded"].any()),
        "clean_cli_has_no_reference_warnings": bool(
            not scores["reference_coverage_warning"].any()
        ),
        "absolute_survival_is_disabled": bool(
            scores["absolute_survival_disabled"].all()
            and not any(column.startswith("survival_") for column in scores)
        ),
    }
    report = {
        "status": "SOFTWARE_RELEASE_GATE_COMPLETE",
        "overall_gate": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "package": "transsurv-pdac",
        "package_version": card["package_version"],
        "model_version": card["model_version"],
        "tests": {
            "collected": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "line_coverage": line_rate,
        },
        "clean_install_smoke": {
            "samples": int(len(scores)),
            "degraded": int(scores["input_degraded"].sum()),
            "reference_warnings": int(scores["reference_coverage_warning"].sum()),
            "relative_outputs_available": bool(scores["risk_percentile"].notna().all()),
            "absolute_survival_columns": [
                column for column in scores if column.startswith("survival_")
            ],
        },
        "hashes": {
            "wheel": sha256(WHEEL),
            "packaged_model": sha256(MODEL),
            "model_card": sha256(CARD),
            "readme": sha256(PACKAGE / "README.md"),
        },
        "release_boundary": card["intended_population"],
        "negative_stress_test_retained": True,
        "interpretation": (
            "The frozen model is installable and reproducible as a research package. "
            "This engineering gate does not imply clinical validity or guarantee journal acceptance."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["overall_gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
