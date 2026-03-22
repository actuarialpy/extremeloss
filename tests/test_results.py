from __future__ import annotations

import numpy as np

from extremeloss.results import BootstrapResult, GEVFit, GPDFit, TailEstimateResult, ThresholdScan


def test_tail_estimate_result_summary_includes_optional_fields():
    result = TailEstimateResult(
        estimate=0.1,
        method="empirical",
        stderr=0.01,
        ci=(0.08, 0.12),
        n=100,
        effective_n=90.0,
        threshold=10.0,
        quantile=0.99,
        diagnostics={"n_exceedances": 5},
    )
    summary = result.summary()

    assert summary["estimate"] == 0.1
    assert summary["method"] == "empirical"
    assert summary["stderr"] == 0.01
    assert summary["ci"] == (0.08, 0.12)
    assert summary["n"] == 100
    assert summary["effective_n"] == 90.0
    assert summary["threshold"] == 10.0
    assert summary["quantile"] == 0.99
    assert summary["diagnostics"] == {"n_exceedances": 5}


def test_gpd_fit_summary_and_return_level():
    fit = GPDFit(
        threshold=10.0,
        xi=0.2,
        beta=5.0,
        exceedance_fraction=0.1,
        n_exceedances=20,
    )
    summary = fit.summary()
    level = fit.return_level(20.0)

    assert summary["threshold"] == 10.0
    assert summary["xi"] == 0.2
    assert summary["beta"] == 5.0
    assert summary["exceedance_fraction"] == 0.1
    assert summary["n_exceedances"] == 20
    assert level > fit.threshold


def test_gev_fit_summary_and_return_level():
    fit = GEVFit(xi=0.1, loc=10.0, scale=2.0, n_blocks=15, block_size=5)
    summary = fit.summary()
    assert summary["n_blocks"] == 15
    assert summary["block_size"] == 5
    assert fit.return_level(20.0) >= fit.return_level(10.0)


def test_bootstrap_result_summary_reports_bootstrap_count():
    result = BootstrapResult(
        estimate=1.5,
        bootstrap_estimates=np.array([1.0, 1.5, 2.0]),
        stderr=0.2,
        ci=(1.1, 1.9),
        alpha=0.05,
    )
    summary = result.summary()
    assert summary["n_bootstrap"] == 3
    assert summary["ci"] == (1.1, 1.9)


def test_threshold_scan_to_dict_preserves_arrays():
    scan = ThresholdScan(
        thresholds=np.array([1.0, 2.0]),
        mean_excess=np.array([3.0, 4.0]),
        xi=np.array([0.1, 0.2]),
        beta=np.array([1.0, 2.0]),
        n_exceedances=np.array([10, 5]),
    )
    out = scan.to_dict()
    assert out["thresholds"].shape == (2,)
    assert out["n_exceedances"].dtype.kind in {"i", "u"}
