"""Uniform-weight importance sampling must reproduce the empirical
estimators exactly -- the weighted estimators share the ecosystem's
inverted-CDF VaR / average-quantile TVaR convention."""
import numpy as np
import pytest

from extremeloss.estimation.importance_sampling import (
    estimate_mean_is,
    estimate_tail_probability_is,
    estimate_tvar_is,
    estimate_var_is,
)
from extremeloss.estimation.metrics import empirical_tvar, empirical_var


def test_uniform_weights_reduce_to_empirical_estimators():
    rng = np.random.default_rng(42)
    x = rng.lognormal(3.0, 1.0, size=5001)
    w = np.ones_like(x)
    assert estimate_mean_is(x, w).estimate == pytest.approx(float(x.mean()), rel=1e-12)
    assert estimate_tail_probability_is(x, w, 60.0).estimate == pytest.approx(
        float(np.mean(x > 60.0)), rel=1e-12
    )
    for q in (0.5, 0.9, 0.95, 0.99):
        assert estimate_var_is(x, w, q).estimate == pytest.approx(
            empirical_var(x, q), rel=1e-12
        )
        assert estimate_tvar_is(x, w, q).estimate == pytest.approx(
            empirical_tvar(x, q), rel=1e-9
        )


def test_weighted_tvar_weights_the_atom_at_var_correctly():
    # Same pinned atoms vector as the ecosystem conformance tests:
    # TVaR_.5 = (1*0.3 + 10*0.2) / 0.5 = 4.6, not the conditional mean 2.8.
    x = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
    w = np.ones(5)
    assert estimate_var_is(x, w, 0.5).estimate == pytest.approx(1.0)
    assert estimate_tvar_is(x, w, 0.5).estimate == pytest.approx(4.6)


def test_weighted_tvar_dominates_weighted_var():
    rng = np.random.default_rng(7)
    x = rng.gamma(2.0, 5000.0, size=2000)
    w = rng.random(2000) + 0.1
    for q in (0.5, 0.9, 0.99):
        assert (
            estimate_tvar_is(x, w, q).estimate
            >= estimate_var_is(x, w, q).estimate - 1e-9
        )
