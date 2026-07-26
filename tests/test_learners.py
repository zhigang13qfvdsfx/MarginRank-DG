import numpy as np
import pandas as pd

from transsurv.learners import PooledSurvivalLearner
from transsurv.metrics import harrell_c
from transsurv.model import DomainData


def test_pooled_coxph_predicts_risk():
    rng = np.random.default_rng(22)
    domains = {}
    for name in ("a", "b"):
        x = rng.normal(size=(80, 3))
        event_time = rng.exponential(scale=np.exp(-x[:, 0]))
        censor = rng.exponential(scale=2.0, size=80)
        domains[name] = DomainData(
            pd.DataFrame(x, columns=["x1", "x2", "x3"]),
            np.minimum(event_time, censor) + 0.01,
            event_time <= censor,
        )
    model = PooledSurvivalLearner("coxph", {"alpha": 0.1}).fit(domains)
    risk = model.predict(domains["a"].x)
    assert harrell_c(domains["a"].time, domains["a"].event, risk) > 0.6
