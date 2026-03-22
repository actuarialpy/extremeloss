from __future__ import annotations

import extremeloss


def test_public_api_exports_expected_symbols():
    assert hasattr(extremeloss, 'fit_gpd')
    assert hasattr(extremeloss, 'fit_gev')
    assert hasattr(extremeloss, 'estimate_tail_probability')
    assert hasattr(extremeloss, 'estimate_tail_probability_cmc')
    assert hasattr(extremeloss, 'bootstrap_var')
    assert hasattr(extremeloss, 'extreme_loss_summary')
