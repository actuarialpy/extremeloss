"""Peaks-over-threshold example using a simulated heavy-tailed sample."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extremeloss import (  # noqa: E402
    extract_exceedances,
    fit_pot,
    mean_excess,
    return_level,
    return_period,
    threshold_diagnostic_table,
)


def main() -> None:
    rng = np.random.default_rng(2026)
    losses = rng.pareto(a=2.8, size=50_000) * 40.0

    thresholds = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 75.0])
    threshold = 40.0

    me = mean_excess(losses, thresholds)
    scan = threshold_diagnostic_table(losses, thresholds)
    exceedances = extract_exceedances(losses, threshold)
    fit = fit_pot(losses, threshold=threshold)

    print("Mean excess scan")
    for u, value, count in zip(me["thresholds"], me["mean_excess"], me["n_exceedances"], strict=True):
        print(f"  u={u:6.1f}  mean excess={value:10.4f}  n_exceedances={int(count):6d}")

    print()
    print("Threshold diagnostic scan")
    scan_dict = scan.to_dict()
    for u, xi, beta, count in zip(
        scan_dict["thresholds"],
        scan_dict["xi"],
        scan_dict["beta"],
        scan_dict["n_exceedances"],
        strict=True,
    ):
        print(f"  u={u:6.1f}  xi={xi:8.4f}  beta={beta:10.4f}  n_exceedances={int(count):6d}")

    print()
    print(f"Threshold: {threshold:.1f}")
    print(f"Number of exceedances: {exceedances.size}")
    print("Fitted POT model")
    print(fit.summary())
    print(f"EVT VaR(0.995): {fit.var(0.995):.4f}")
    print(f"EVT TVaR(0.995): {fit.tvar(0.995):.4f}")

    period = 200.0
    level = return_level(period, fit)
    print(f"Return level for period {period:.1f}: {level:.4f}")
    print(f"Return period for exceedance probability 0.01: {return_period(0.01):.2f}")


if __name__ == "__main__":
    main()
