from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from transsurv.learners import PooledSurvivalLearner
from transsurv.marginrank import LockedMarginRankModel
from transsurv.model import DomainData


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
TARGET = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"
SPEC = CONFIG / "frozen_model_v4_marginrank_spec.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v4_module():
    path = Path(__file__).with_name("36_v4_marginrank_nested.py")
    spec = importlib.util.spec_from_file_location("v4_marginrank_freeze", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if TARGET.exists() or SPEC.exists():
        raise RuntimeError("The v4 MarginRank model has already been frozen")
    for result in [
        "36_v4_marginrank_gate.json",
        "37_v4_fixed_selection_gate.json",
        "38_v4_missingness_gate.json",
    ]:
        payload = json.loads((RESULTS / result).read_text(encoding="utf-8"))
        if payload["gate"] != "PASS":
            raise RuntimeError(f"Required v4 gate failed: {result}")
    selected = json.loads(
        (CONFIG / "v4_selected_configuration.json").read_text(encoding="utf-8")
    )
    v4 = load_v4_module()
    candidate = next(
        item for item in v4.CANDIDATES if item.name == selected["candidate"]
    )
    if candidate.learner != "pooled_ridge" or not candidate.include_auxiliary:
        raise RuntimeError("Unexpected selected final candidate")
    domains, universe = v4.load_domains()
    z_table = v4.precompute_domain_z(domains, universe)
    train_names = [*v4.PRIMARY, v4.AUXILIARY]
    definition = v4.feature_definition(
        train_names, candidate.feature, domains, z_table
    )
    transformed = {
        name: DomainData(
            x=v4.transform_features(domains[name].x, definition),
            time=domains[name].time,
            event=domains[name].event,
        )
        for name in train_names
    }
    model = PooledSurvivalLearner(
        "coxph", {"alpha": candidate.alpha}
    ).fit(transformed)
    bundle = {
        "name": "TransSurv-PDAC MarginRank-DG",
        "version": "4.0.0-marginrank-locked",
        "algorithm": (
            "continuous single-sample rank margins to a six-gene anchor median, "
            "balanced cross-domain survival screening, pooled ridge Cox model"
        ),
        "absolute_survival_enabled": False,
        "rank_universe": universe,
        "feature_definition": definition,
        "model": model,
        "relative_risk": {
            "minimum_batch_n": 40,
            "z_score_ddof": 0,
            "percentile_method": "average empirical rank divided by batch size",
            "tiers": ["low <1/3", "middle 1/3 to <2/3", "high >=2/3"],
            "uses_outcomes_at_inference": False,
        },
        "minimum_reference_gene_coverage": 0.80,
        "minimum_selected_gene_coverage": 0.80,
        "minimum_anchor_gene_coverage": 0.50,
        "development_primary_domains": v4.PRIMARY,
        "development_auxiliary_domain": v4.AUXILIARY,
        "development_total_n": int(sum(len(domains[name].time) for name in train_names)),
        "development_total_events": int(
            sum(domains[name].event.sum() for name in train_names)
        ),
        "confirmation_status": "pending reserved fresh PRINCE/GSE77435 stress tests",
        "prohibited_use": (
            "No absolute survival, individual survival time, clinical threshold, "
            "treatment selection, or medical-device use"
        ),
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, TARGET, compress=3)
    model_hash = sha256(TARGET)

    smoke = LockedMarginRankModel.load(TARGET).predict_from_rank(
        domains["TCGA_PAAD"].x
    )
    if any(column.startswith("survival_") for column in smoke):
        raise RuntimeError("Frozen MarginRank bundle emitted survival output")
    if smoke["raw_risk"].nunique() < 40:
        raise RuntimeError("Frozen MarginRank bundle failed continuity smoke test")
    if not smoke["risk_percentile"].notna().all():
        raise RuntimeError("Frozen MarginRank bundle failed batch-relative smoke test")

    coefficients = np.asarray(model.estimator_.coef_, dtype=float).reshape(-1)
    coefficient_frame = pd.DataFrame(
        {"feature": model.features_, "coefficient": coefficients}
    )
    coefficient_frame["gene"] = coefficient_frame["feature"].str.replace(
        "anchor::", "", regex=False
    )
    screen = v4.screening_table(train_names, z_table).reset_index()
    coefficient_frame = coefficient_frame.merge(screen, on="gene", how="left")
    coefficient_frame["absolute_coefficient"] = coefficient_frame["coefficient"].abs()
    coefficient_frame.sort_values("absolute_coefficient", ascending=False).to_csv(
        RESULTS / "39_v4_final_feature_coefficients.csv", index=False
    )

    sources = {
        "migration_lock": CONFIG / "V4_DEVELOPMENT_MIGRATION.lock",
        "fresh_candidates_lock": CONFIG / "V4_FRESH_CANDIDATES.lock",
        "selection_lock": CONFIG / "V4_MARGINRANK_SELECTION.lock",
        "selected_configuration": CONFIG / "v4_selected_configuration.json",
        "dynamic_gate": RESULTS / "36_v4_marginrank_gate.json",
        "fixed_gate": RESULTS / "37_v4_fixed_selection_gate.json",
        "missingness_gate": RESULTS / "38_v4_missingness_gate.json",
        "universe": ROOT / "data" / "manifests" / "v4_strict_common_gene_universe.csv",
        "marginrank_source": SOFTWARE / "src" / "transsurv" / "marginrank.py",
        "core_experiment": Path(__file__).with_name("36_v4_marginrank_nested.py"),
        "freeze_script": Path(__file__).resolve(),
        "model_bundle": TARGET,
    }
    spec_payload = {
        "status": "FROZEN_BEFORE_V4_FRESH_RISK_PREDICTION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": bundle["name"],
        "model_version": bundle["version"],
        "selected_candidate": candidate.name,
        "selected_genes": definition["genes"],
        "anchor_genes": definition["anchors"],
        "feature_count": len(model.features_),
        "rank_universe_size": len(universe),
        "absolute_survival_enabled": False,
        "model_bundle": str(TARGET.relative_to(ROOT)),
        "model_sha256": model_hash,
        "development_primary_domains": v4.PRIMARY,
        "development_auxiliary_domain": v4.AUXILIARY,
        "permanently_excluded": ["GSE224564_UMD2", "GSE224564_UMD3"],
        "reserved_fresh_candidates": [
            "paad_iatlas_prince_2022",
            "GSE77435",
            "conditional GSE277386",
        ],
        "source_hashes": {name: sha256(path) for name, path in sources.items()},
    }
    SPEC.write_text(json.dumps(spec_payload, indent=2), encoding="utf-8")
    print(json.dumps(spec_payload, indent=2))


if __name__ == "__main__":
    main()
