import numpy as np
import pandas as pd

from transsurv.model import DomainData, MultiDomainCox, _cox_loss_gradient
from transsurv.metrics import harrell_c


def _domain(seed: int, n: int = 90) -> DomainData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    beta = np.array([0.8, -0.6, 0.0, 0.0])
    event_time = rng.exponential(scale=np.exp(-(x @ beta)))
    censor = rng.exponential(scale=2.0, size=n)
    time = np.minimum(event_time, censor) + 0.01
    event = event_time <= censor
    return DomainData(pd.DataFrame(x, columns=list("abcd")), time, event)


def test_gradient_matches_finite_difference():
    domain = _domain(1, 50)
    beta = np.array([0.2, -0.1, 0.05, 0.0])
    value, gradient = _cox_loss_gradient(
        beta, domain.x.to_numpy(), domain.time, domain.event
    )
    eps = 1e-6
    numeric = []
    for j in range(len(beta)):
        step = np.zeros_like(beta)
        step[j] = eps
        plus = _cox_loss_gradient(
            beta + step, domain.x.to_numpy(), domain.time, domain.event
        )[0]
        minus = _cox_loss_gradient(
            beta - step, domain.x.to_numpy(), domain.time, domain.event
        )[0]
        numeric.append((plus - minus) / (2 * eps))
    assert np.isfinite(value)
    assert np.allclose(gradient, numeric, atol=2e-5, rtol=2e-5)


def test_model_learns_direction():
    domains = {"a": _domain(2), "b": _domain(3)}
    model = MultiDomainCox(robustness=3.0, l1=0.0001, l2=0.01).fit(domains)
    assert model.optimization_["success"]
    for domain in domains.values():
        assert harrell_c(domain.time, domain.event, model.predict(domain.x)) > 0.6


def test_shared_private_model_exposes_consensus_predictor():
    domains = {"a": _domain(4), "b": _domain(5)}
    model = MultiDomainCox(
        robustness=2.0, l1=0.0001, l2=0.01, domain_l2=1.0
    ).fit(domains)
    assert model.optimization_["success"]
    assert set(model.domain_offsets_) == set(domains)
    assert model.coef_.shape == (4,)
