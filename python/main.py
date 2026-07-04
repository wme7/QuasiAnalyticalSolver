"""Driver script mirroring matlab/RunCoulouvratSolver.m."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common_ic import common_ic
from quasi_analytical_solver import quasi_analytical_solver


def main() -> None:
    t_end = 5.5
    ic_case = 11
    flux_name = "buckley"
    debug = True

    dx = 1 / 100
    x = np.arange(-4 * np.pi, 4 * np.pi + dx / 2, dx)
    u0 = common_ic(x, ic_case)

    x_exact, u_exact, _ = quasi_analytical_solver(x, u0, t_end, flux_name, debug=debug)

    plt.figure(figsize=(8, 4))
    plt.plot(x, u0, "-.k", label="Initial Condition")
    plt.plot(x_exact, u_exact, "-r", label="Quasi-analytical")
    plt.title(f"t={t_end:g} [-]")
    plt.ylabel("u(x,t)")
    plt.xlabel("x")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
