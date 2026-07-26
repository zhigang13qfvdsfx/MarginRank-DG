import numpy as np
import pandas as pd

from transsurv.missingness import RiskDistiller, available_pair_features


def test_available_pair_features_removes_missing_gene():
    columns = ["a|b", "a|c", "b|c"]
    assert available_pair_features(columns, ["a"]) == ["b|c"]


def test_distiller_recovers_linear_teacher():
    rng = np.random.default_rng(44)
    x = pd.DataFrame(rng.normal(size=(100, 3)), columns=["x1", "x2", "x3"])
    teacher = 0.8 * x["x1"].to_numpy() - 0.4 * x["x2"].to_numpy()
    model = RiskDistiller(alpha=1e-6).fit(x, teacher)
    prediction = model.predict(x)
    assert np.corrcoef(teacher, prediction)[0, 1] > 0.999
