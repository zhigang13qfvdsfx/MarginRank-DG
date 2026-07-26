import numpy as np
import pandas as pd

from transsurv.marginrank import LockedMarginRankModel, anchor_rank_features


class MeanFeatureModel:
    def predict(self, x):
        return x.fillna(0.0).mean(axis=1).to_numpy(float)


def make_bundle(minimum_batch_n=2):
    return {
        "absolute_survival_enabled": False,
        "feature_definition": {
            "kind": "anchor",
            "genes": ["A", "B"],
            "anchors": ["H1", "H2"],
            "pairs": [],
        },
        "rank_universe": ["A", "B", "H1", "H2", "X"],
        "model": MeanFeatureModel(),
        "relative_risk": {"minimum_batch_n": minimum_batch_n},
        "minimum_selected_gene_coverage": 0.5,
        "minimum_anchor_gene_coverage": 0.5,
        "minimum_reference_gene_coverage": 0.8,
    }


def test_anchor_rank_features_are_continuous_margins():
    ranks = pd.DataFrame(
        {"A": [0.8, 0.2], "B": [0.4, 0.9], "H1": [0.3, 0.4], "H2": [0.5, 0.6]}
    )
    features = anchor_rank_features(ranks, ["A", "B"], ["H1", "H2"])
    assert np.allclose(features["anchor::A"], [0.4, -0.3])
    assert np.allclose(features["anchor::B"], [0.0, 0.4])


def test_rank_only_prediction_has_percentiles_and_no_survival_output():
    ranks = pd.DataFrame(
        {
            "A": [0.8, 0.2, 0.7],
            "B": [0.4, 0.9, 0.3],
            "H1": [0.3, 0.4, 0.2],
            "H2": [0.5, 0.6, 0.4],
        }
    )
    output = LockedMarginRankModel(make_bundle()).predict_from_rank(ranks)
    assert output["risk_percentile"].notna().all()
    assert output["absolute_survival_disabled"].all()
    assert not any(column.startswith("survival_") for column in output)


def test_missing_selected_gene_is_reported_without_failure():
    ranks = pd.DataFrame({"A": [0.8], "H1": [0.3], "H2": [0.5]})
    output = LockedMarginRankModel(make_bundle(minimum_batch_n=40)).predict_from_rank(ranks)
    assert output.loc["0", "selected_gene_coverage"] == 0.5
    assert not output.loc["0", "input_degraded"]
    assert np.isnan(output.loc["0", "risk_percentile"])


def test_expression_path_reports_reference_coverage():
    expression = pd.DataFrame(
        {"A": [3.0, 1.0], "B": [2.0, 4.0], "H1": [1.0, 2.0], "H2": [2.5, 3.0], "X": [0.5, 0.2]}
    )
    output = LockedMarginRankModel(make_bundle()).predict_from_expression(expression)
    assert (output["reference_gene_coverage"] == 1.0).all()
    assert not output["reference_coverage_warning"].any()


def test_reference_coverage_is_computed_per_sample():
    expression = pd.DataFrame(
        {
            "A": [3.0, 1.0],
            "B": [2.0, 4.0],
            "H1": [1.0, 2.0],
            "H2": [2.5, 3.0],
            "X": [0.5, np.nan],
        }
    )
    output = LockedMarginRankModel(make_bundle()).predict_from_expression(expression)
    assert np.allclose(output["reference_gene_coverage"], [1.0, 0.8])
    assert not output["reference_coverage_warning"].any()
