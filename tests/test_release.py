import hashlib
import json

import numpy as np
import pandas as pd

from transsurv import __version__, load_default_model, load_model_card
from transsurv.cli import main, read_expression


def test_packaged_model_and_card_are_consistent():
    model = load_default_model()
    card = load_model_card()
    assert __version__ == card["package_version"] == "0.5.0"
    assert model.bundle["version"] == card["model_version"]
    assert model.bundle["absolute_survival_enabled"] is False
    assert card["governance"]["failed_external_tests_retained"] is True


def test_packaged_model_hash_matches_card():
    from importlib.resources import as_file, files

    resource = files("transsurv").joinpath(
        "models/transsurv_pdac_v4_marginrank_locked.joblib"
    )
    with as_file(resource) as path:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == load_model_card()["model_sha256"]


def test_cli_model_info(tmp_path):
    output = tmp_path / "model_info.json"
    assert main(["model-info", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"]["absolute_survival_enabled"] is False


def test_cli_score_emits_relative_risk_only(tmp_path):
    model = load_default_model()
    genes = sorted(model.required_input_genes)
    rng = np.random.default_rng(42)
    input_path = tmp_path / "expression.tsv"
    output_path = tmp_path / "scores.tsv"
    frame = pd.DataFrame(rng.normal(size=(3, len(genes))), columns=genes)
    frame.insert(0, "sample_id", ["S1", "S2", "S3"])
    frame.to_csv(input_path, sep="\t", index=False)
    assert main(["score", str(input_path), str(output_path)]) == 0
    scores = pd.read_csv(output_path, sep="\t", index_col=0)
    assert np.isfinite(scores["raw_risk"]).all()
    assert scores["risk_percentile"].isna().all()
    assert scores["absolute_survival_disabled"].all()
    assert scores["reference_coverage_warning"].all()
    assert not any(column.startswith("survival_") for column in scores)


def test_read_expression_supports_genes_by_samples(tmp_path):
    path = tmp_path / "transposed.csv"
    pd.DataFrame(
        {
            "gene": ["a", "b", "h1"],
            "S1": [1.0, 2.0, 3.0],
            "S2": [4.0, 5.0, 6.0],
        }
    ).to_csv(path, index=False)
    result = read_expression(path, orientation="genes-by-samples")
    assert result.index.tolist() == ["S1", "S2"]
    assert result.columns.tolist() == ["A", "B", "H1"]


def test_frozen_raw_risk_is_sample_composition_and_affine_invariant():
    model = load_default_model()
    universe = model.bundle["rank_universe"]
    rng = np.random.default_rng(20260722)
    expression = pd.DataFrame(
        rng.normal(size=(40, len(universe))),
        index=[f"S{i:02d}" for i in range(40)],
        columns=universe,
    )
    mixed = model.predict_from_expression(expression)
    alone = model.predict_from_expression(expression.iloc[[0]])
    transformed = expression.iloc[[0]] * 3.7 + 11.2
    affine = model.predict_from_expression(transformed)
    assert np.isclose(mixed.loc["S00", "raw_risk"], alone.loc["S00", "raw_risk"], atol=1e-12)
    assert np.isclose(alone.loc["S00", "raw_risk"], affine.loc["S00", "raw_risk"], atol=1e-12)
    assert mixed["risk_percentile"].notna().all()
    assert alone["risk_percentile"].isna().all()
    assert not mixed["reference_coverage_warning"].any()
