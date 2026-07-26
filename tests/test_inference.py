import numpy as np
import pandas as pd

from transsurv.inference import LockedPDACModel, selected_pair_features


class PairModel:
    def predict(self, x):
        return x.iloc[:, 0].fillna(0.5).to_numpy(float)


def test_selected_pairs_preserve_missing_endpoints():
    ranks = pd.DataFrame({"A": [0.8, 0.2], "B": [0.1, 0.7], "C": [0.3, 0.4]})
    result = selected_pair_features(ranks, ["A|B", "A|D"], available_genes={"A", "B", "C"})
    assert result["A|B"].tolist() == [1.0, 0.0]
    assert result["A|D"].isna().all()


def test_selected_pairs_preserve_sample_level_missingness():
    ranks = pd.DataFrame({"A": [0.8, np.nan], "B": [0.1, 0.7]})
    result = selected_pair_features(ranks, ["A|B"])
    assert result.loc[0, "A|B"] == 1.0
    assert np.isnan(result.loc[1, "A|B"])


def make_batchscale_bundle(minimum_batch_n=2):
    components = [
        {
            "branch": branch,
            "pairs": ["A|B"],
            "model": PairModel(),
            "location": 0.0,
            "scale": 1.0,
        }
        for branch in ("multi_learner", "ipcw")
    ]
    return {
        "components": components,
        "ipcw_weight": 0.0,
        "batchscale": {"minimum_batch_n": minimum_batch_n, "coefficient": 0.4},
        "baseline_times_months": np.array([12.0, 24.0]),
        "baseline_cumulative_hazard": np.array([0.2, 0.5]),
    }


def test_batchscale_has_locked_dispersion_for_eligible_batch():
    ranks = pd.DataFrame({"A": [0.9, 0.1, 0.8], "B": [0.2, 0.7, 0.3]})
    result = LockedPDACModel(make_batchscale_bundle()).predict_from_rank(ranks)
    assert result["batchscale_applied"].all()
    assert not result["calibration_warning"].any()
    assert np.isclose(result["risk"].std(ddof=0), 0.4)
    assert result["relative_outputs_available"].all()
    assert np.isclose(result["within_batch_z"].std(ddof=0), 1.0)
    assert result["risk_percentile"].between(0, 1).all()
    assert result[["survival_12m", "survival_24m"]].notna().all().all()


def test_small_batch_returns_relative_risk_with_calibration_warning():
    ranks = pd.DataFrame({"A": [0.9], "B": [0.2]})
    result = LockedPDACModel(make_batchscale_bundle(minimum_batch_n=40)).predict_from_rank(ranks)
    assert not result["batchscale_applied"].any()
    assert not result["relative_outputs_available"].any()
    assert result["risk_percentile"].isna().all()
    assert result["risk_tier"].isna().all()
    assert result["calibration_warning"].all()
    assert result[["survival_12m", "survival_24m"]].isna().all().all()


def test_rank_only_bundle_never_emits_absolute_survival():
    bundle = make_batchscale_bundle()
    bundle["absolute_survival_enabled"] = False
    bundle["relative_risk"] = {"minimum_batch_n": 2}
    ranks = pd.DataFrame({"A": [0.9, 0.1, 0.8], "B": [0.2, 0.7, 0.3]})
    result = LockedPDACModel(bundle).predict_from_rank(ranks)
    assert result["relative_outputs_available"].all()
    assert result["risk_percentile"].notna().all()
    assert set(result["risk_tier"]) <= {"low", "middle", "high"}
    assert result["absolute_survival_disabled"].all()
    assert result[["survival_12m", "survival_24m"]].isna().all().all()
