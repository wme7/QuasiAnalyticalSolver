"""Initial conditions for quasi-analytical solver tests."""

from __future__ import annotations

import numpy as np


def rectangular_pulse(a: float, b: float, x: np.ndarray) -> np.ndarray:
    return np.heaviside(x - a, 0.5) - np.heaviside(x - b, 0.5)


def common_ic(x: np.ndarray, ic_case: int) -> np.ndarray:
    """Return u0 on grid x for the selected IC case (1-11)."""
    lx = x[-1] - x[0]
    xmid = 0.5 * (x[-1] + x[0])

    if ic_case == 1:
        return np.exp(-20 * (x - xmid) ** 2)

    if ic_case == 2:
        mu = 0.01
        return np.exp(-(x - xmid) ** 2 / (4 * mu))

    if ic_case == 3:
        return np.sin(x)

    if ic_case == 4:
        return 0.5 - np.sin(x)

    if ic_case == 5:
        mu = 0.02
        return 0.5 * (1 - np.tanh(x / (4 * mu)))

    if ic_case == 6:
        u0 = np.ones_like(x)
        u0[x <= xmid] = 2
        return u0

    if ic_case == 7:
        a, b = x[0], x[-1]
        xi = (4 - (-4)) / (b - a) * (x - a) - 4
        return 0.5 * (np.tanh(-4 * xi) + 1)

    if ic_case == 8:
        return rectangular_pulse(xmid - 0.1 * lx, xmid + 0.1 * lx, x) + 1

    if ic_case == 9:
        xmid = -0.25
        return rectangular_pulse(xmid - 0.125 * lx, xmid + 0.125 * lx, x)

    if ic_case == 10:
        return np.exp(x) * rectangular_pulse(xmid - 0.1 * lx, xmid + 0.1 * lx, x) * np.exp(0.1)

    if ic_case == 11:
        return np.sin(x) * np.exp(-(x**2) / 50) * (np.abs(x) < 4 * np.pi)

    raise ValueError(f"IC case {ic_case} not in the list")
