import numpy as np
import pandas as pd

from transsurv.metrics import harrell_c
from transsurv.model import DomainData
from transsurv.ranking import IPCWPairwiseRanker, _pair_loss_gradient


def _domains(seed=31):
    rng = np.random.default_rng(seed)
    output = {}
    for name in ("a", "b"):
        x = rng.normal(size=(90, 3))
        event_time = rng.exponential(scale=np.exp(-0.9 * x[:, 0]))
        censor = rng.exponential(scale=2.0, size=90)
        output[name] = DomainData(
            pd.DataFrame(x, columns=["x1", "x2", "x3"]),
            np.minimum(event_time, censor) + 0.01,
            event_time <= censor,
        )
    return output


def test_pair_loss_gradient():
    differences = np.array([[1.0, -0.5], [0.2, 0.3], [-0.4, 0.7]])
    weights = np.array([1.0, 2.0, 0.5])
    beta = np.array([0.2, -0.1])
    _, gradient = _pair_loss_gradient(beta, differences, weights)
    numeric = []
    eps = 1e-6
    for j in range(2):
        step = np.zeros(2)
        step[j] = eps
        plus = _pair_loss_gradient(beta + step, differences, weights)[0]
        minus = _pair_loss_gradient(beta - step, differences, weights)[0]
        numeric.append((plus - minus) / (2 * eps))
    assert np.allclose(gradient, numeric, atol=1e-6)


def test_ranker_learns_risk_direction():
    domains = _domains()
    model = IPCWPairwiseRanker(l1=0.0001, l2=0.01).fit(domains)
    assert model.optimization_["success"]
    assert harrell_c(
        domains["a"].time,
        domains["a"].event,
        model.predict(domains["a"].x),
    ) > 0.6
