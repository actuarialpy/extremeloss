from __future__ import annotations

import numpy as np

from extremeloss.protocols import SupportsLosses, SupportsSample, SupportsSimulationResult


class DummyModel:
    def sample(self, size: int = 1) -> np.ndarray:
        return np.zeros(size)


class DummyResult:
    gross_losses = np.array([1.0, 2.0])
    retained_losses = np.array([0.5, 1.5])
    ceded_losses = np.array([0.5, 0.5])

    @property
    def losses(self) -> np.ndarray:
        return self.retained_losses


def test_supports_sample_protocol_runtime_checkable():
    assert isinstance(DummyModel(), SupportsSample)


def test_supports_losses_and_simulation_result_protocols_runtime_checkable():
    result = DummyResult()
    assert isinstance(result, SupportsLosses)
    assert isinstance(result, SupportsSimulationResult)
