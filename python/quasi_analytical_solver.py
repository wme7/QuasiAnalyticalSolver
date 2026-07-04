"""Quasi-analytical solver for scalar nonlinear conservation laws.

Implements Coulouvrat (2009), Wave Motion 46(2), 97-107.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


# ------------------------ Get Flux Model ------------------------
@dataclass(frozen=True)
class FluxModel:
    name: str
    f: Callable[[np.ndarray], np.ndarray]
    df: Callable[[np.ndarray], np.ndarray]
    Df: Callable[[np.ndarray], np.ndarray]
    I: Callable[[np.ndarray], np.ndarray]


def _lambdify(u: sp.Symbol, expr: sp.Expr) -> Callable[[np.ndarray], np.ndarray]:
    return sp.lambdify(u, expr, modules="numpy")


def _from_symbolic(
    name: str,
    f_expr: sp.Expr,
    i_expr: sp.Expr,
    *,
    parameters: dict[str, float] | None = None,
) -> FluxModel:
    u = sp.Symbol("u")
    params = parameters or {}
    f_expr = f_expr.subs(params)
    i_expr = i_expr.subs(params)
    df_expr = sp.diff(f_expr, u)
    df_expr = sp.simplify(df_expr)
    Df_expr = sp.simplify(u * df_expr - f_expr)

    return FluxModel(
        name=name,
        f=_lambdify(u, f_expr),
        df=_lambdify(u, df_expr),
        Df=_lambdify(u, Df_expr),
        I=_lambdify(u, i_expr),
    )


def _buckley_entropy(u: sp.Symbol, a: sp.Expr) -> sp.Expr:
    w = (a + 1) * u / sp.sqrt(a) - sp.sqrt(a)
    return (sp.sqrt(a) / (1 + a) ** 2) * (
        w + (a - 1) * sp.atan(w) + sp.sqrt(a) * sp.log(1 + w**2)
    )


def get_flux(flux_name: str) -> FluxModel:
    u = sp.Symbol("u")

    models: dict[str, FluxModel] = {
        "burgers": _from_symbolic("burgers", -u**2 / 2, -u**3 / 3),
        "burgers+": _from_symbolic("burgers+", u**2 / 2, u**3 / 3),
        "cubic": _from_symbolic("cubic", -u**3 / 3, -u**4 / 4),
        "cubic+": _from_symbolic("cubic+", u**3 / 3, u**4 / 4),
    }

    if flux_name == "buckley":
        a = sp.Rational(1, 2)
        f_expr = u**2 / (u**2 + a * (1 - u) ** 2)
        return _from_symbolic(
            "buckley",
            f_expr,
            _buckley_entropy(u, a),
            parameters={},
        )

    if flux_name in models:
        return models[flux_name]

    raise ValueError(f"Unknown flux function: {flux_name!r}")


# ------------------------ Build Solution ------------------------
@dataclass
class Branch:
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    u: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))
    x_sh: np.ndarray = field(default_factory=lambda: np.array([]))
    up_sh: np.ndarray = field(default_factory=lambda: np.array([]))
    un_sh: np.ndarray = field(default_factory=lambda: np.array([]))
    ec: np.ndarray = field(default_factory=lambda: np.array([]))


def _cumulative_integral(x: np.ndarray, u0: np.ndarray) -> np.ndarray:
    dx = np.max(np.diff(x))
    return np.cumsum(dx * u0)


def _mark_tvd_violations(y: np.ndarray) -> np.ndarray:
    y_marked = y.copy()
    tvd = np.diff(y_marked) < 0
    y_marked[:-1][tvd] = np.nan
    return y_marked


def _identify_branches(y: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray]:
    nx = y.size
    y_work = _mark_tvd_violations(y)

    isnan_prev = np.isnan(y_work[:-1])
    idx = np.flatnonzero(np.diff(isnan_prev.astype(int)))

    n_shocks = idx.size // 2
    n_branches = n_shocks + 1
    idx_branches = np.concatenate(([0], idx, [nx]))
    branches = idx_branches.reshape((2, n_branches), order="F")
    branches[0, :] += 1

    return branches, n_shocks, n_branches, y_work


def _largest_finite_segment(
    y: np.ndarray,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    y_slice = y[start - 1 : end]
    best: tuple[int, int] | None = None
    i = 0
    while i < y_slice.size:
        if np.isfinite(y_slice[i]):
            j = i
            while j < y_slice.size and np.isfinite(y_slice[j]):
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1

    if best is None:
        return None

    return start + best[0], start + best[1] - 1


def _interp_branch(
    x: np.ndarray,
    y: np.ndarray,
    u0: np.ndarray,
    phi: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment = _largest_finite_segment(y, start, end)
    if segment is None:
        return np.array([]), np.array([]), np.array([])

    seg_start, seg_end = segment
    y_slice = y[seg_start - 1 : seg_end]
    u_slice = u0[seg_start - 1 : seg_end]
    phi_slice = phi[seg_start - 1 : seg_end]

    if y_slice.size < 2:
        return np.array([]), np.array([]), np.array([])

    y_lo = min(y_slice[0], y_slice[-1])
    y_hi = max(y_slice[0], y_slice[-1])
    mask = (x >= y_lo) & (x <= y_hi)

    branch_x = x[mask]
    branch_u = np.interp(branch_x, y_slice, u_slice)
    branch_phi = np.interp(branch_x, y_slice, phi_slice)
    return branch_x, branch_u, branch_phi


def _finite_branch_samples(
    y: np.ndarray,
    u0: np.ndarray,
    phi: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment = _largest_finite_segment(y, start, end)
    if segment is None:
        return np.array([]), np.array([]), np.array([])

    seg_start, seg_end = segment
    return (
        y[seg_start - 1 : seg_end],
        u0[seg_start - 1 : seg_end],
        phi[seg_start - 1 : seg_end],
    )


def _find_intersections(
    branches: list[Branch],
    y: np.ndarray,
    u0: np.ndarray,
    phi: np.ndarray,
    branch_idx: np.ndarray,
    n_branches: int,
    dx: float,
    flux: FluxModel,
    debug: bool,
    ax_u: Any | None,
    ax_phi: Any | None,
) -> None:
    for j in range(1, n_branches):
        intersection = np.full(n_branches, np.nan)
        up_sh = np.full(n_branches, np.nan)
        un_sh = np.full(n_branches, np.nan)
        entropy_condition = np.zeros(n_branches, dtype=bool)

        for k in range(j + 1, n_branches + 1):
            common_x, j_idx, k_idx = np.intersect1d(
                branches[j - 1].x,
                branches[k - 1].x,
                return_indices=True,
            )

            if common_x.size > 0:
                x1 = common_x[:-1]
                phi1 = branches[j - 1].phi[j_idx[:-1]]
                phi1s = branches[k - 1].phi[k_idx[:-1]]
                x2 = common_x[1:]
                phi2 = branches[j - 1].phi[j_idx[1:]]
                phi2s = branches[k - 1].phi[k_idx[1:]]

                denom = (phi2 - phi1) - (phi2s - phi1s)
                with np.errstate(divide="ignore", invalid="ignore"):
                    x_cross = x1 + (phi1s - phi1) * (x2 - x1) / denom

                valid = (x_cross > x1) & (x_cross < x2) & np.isfinite(x_cross)
                if np.any(valid):
                    x_hit = x_cross[valid][0]
                    intersection[k - 1] = x_hit

                    y_k, u_k, phi_k = _finite_branch_samples(
                        y, u0, phi, branch_idx[0, k - 1], branch_idx[1, k - 1]
                    )
                    y_j, u_j, _ = _finite_branch_samples(
                        y, u0, phi, branch_idx[0, j - 1], branch_idx[1, j - 1]
                    )

                    up = np.interp(x_hit, y_k, u_k)
                    un = np.interp(x_hit, y_j, u_j)
                    phi_sh = np.interp(x_hit, y_k, phi_k)

                    up_sh[k - 1] = up
                    un_sh[k - 1] = un
                    entropy_condition[k - 1] = (flux.I(up) - flux.I(un)) > (
                        (flux.f(up) - flux.f(un)) * (up + un) / 2
                    )

                    if debug and ax_u is not None and ax_phi is not None:
                        ax_u.plot([x_hit, x_hit], [up, un], ".-k")
                        ax_phi.plot(x_hit, phi_sh, ".k")

            if (
                k == j + 1
                and branches[j - 1].x.size > 0
                and branches[k - 1].x.size > 0
                and branches[j - 1].x[-1] - branches[k - 1].x[0] < dx
            ):
                x_hit = 0.5 * (branches[j - 1].x[-1] + branches[k - 1].x[0])
                intersection[k - 1] = x_hit

                y_k, u_k, phi_k = _finite_branch_samples(
                    y, u0, phi, branch_idx[0, k - 1], branch_idx[1, k - 1]
                )
                y_j, u_j, _ = _finite_branch_samples(
                    y, u0, phi, branch_idx[0, j - 1], branch_idx[1, j - 1]
                )

                up = np.interp(x_hit, y_k, u_k)
                un = np.interp(x_hit, y_j, u_j)
                phi_sh = np.interp(x_hit, y_k, phi_k)

                up_sh[k - 1] = up
                un_sh[k - 1] = un
                entropy_condition[k - 1] = (flux.I(up) - flux.I(un)) > (
                    (flux.f(up) - flux.f(un)) * (up + un) / 2
                )

                if debug and ax_u is not None and ax_phi is not None:
                    ax_u.plot([x_hit, x_hit], [up, un], ".-k")
                    ax_phi.plot(x_hit, phi_sh, ".k")

        branches[j - 1].x_sh = intersection
        branches[j - 1].up_sh = up_sh
        branches[j - 1].un_sh = un_sh
        branches[j - 1].ec = entropy_condition


def _select_shock_path(
    branches: list[Branch],
    n_branches: int,
    n_shocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_shocks == 0:
        return np.array([1], dtype=int), np.array([])

    path = [tuple(range(1, n_branches + 1))]
    shocks = np.array(
        [[branches[from_b - 1].x_sh[to_b - 1] for from_b, to_b in zip(path[0], path[0][1:])]],
        dtype=float,
    )
    shocks_flat = shocks[0] if shocks.size else np.array([])

    if shocks_flat.size and np.all(np.isfinite(shocks_flat)) and np.all(np.diff(shocks_flat) > 0):
        return np.array(path[0], dtype=int), shocks_flat

    i = 0
    idx: list[int] = []
    while not idx:
        i += 1
        path = list(combinations(range(1, n_branches + 1), n_branches - i))
        shocks = np.full((len(path), n_shocks - i), np.nan)
        index = np.ones(len(path), dtype=bool)

        for p, candidate in enumerate(path):
            if candidate[0] != 1 or candidate[-1] != n_branches:
                index[p] = False
                continue
            for b in range(n_shocks - i):
                from_b = candidate[b]
                to_b = candidate[b + 1]
                shocks[p, b] = branches[from_b - 1].x_sh[to_b - 1]

        for b in range(max(0, n_shocks - 1 - i)):
            index &= np.isfinite(shocks[:, b]) & np.isfinite(shocks[:, b + 1])
            index &= shocks[:, b] < shocks[:, b + 1]

        idx = np.flatnonzero(index).tolist()

    pick = idx[0]
    return np.array(path[pick], dtype=int), shocks[pick]


def _assemble_solution(
    branches: list[Branch],
    n_branches: int,
    solution_path: np.ndarray,
    intersections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_branches == 1:
        return branches[0].x, branches[0].u, branches[0].phi

    x_parts: list[np.ndarray] = []
    u_parts: list[np.ndarray] = []
    phi_parts: list[np.ndarray] = []

    for i in range(1, solution_path.size + 1):
        if i == 1:
            mask = branches[0].x <= intersections[0]
            x_parts.append(branches[0].x[mask])
            u_parts.append(branches[0].u[mask])
            phi_parts.append(branches[0].phi[mask])
        elif i == solution_path.size:
            mask = branches[n_branches - 1].x >= intersections[i - 2]
            x_parts.append(branches[n_branches - 1].x[mask])
            u_parts.append(branches[n_branches - 1].u[mask])
            phi_parts.append(branches[n_branches - 1].phi[mask])
        else:
            branch_id = solution_path[i - 1]
            branch = branches[branch_id - 1]
            mask = (branch.x > intersections[i - 2]) & (branch.x < intersections[i - 1])
            x_parts.append(branch.x[mask])
            u_parts.append(branch.u[mask])
            phi_parts.append(branch.phi[mask])

    return np.concatenate(x_parts), np.concatenate(u_parts), np.concatenate(phi_parts)


def quasi_analytical_solver(
    x: np.ndarray | None = None,
    u0: np.ndarray | None = None,
    t: float | None = None,
    flux_func: str = "burgers+",
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve u_t - f(u)_x = 0 using Coulouvrat's quasi-analytical method."""
    if x is None and u0 is None and t is None:
        dx = np.pi / 100
        x = np.arange(-5 * np.pi, 5 * np.pi + dx / 2, dx)
        u0 = np.sin(x) * np.exp(-(x**2) / 50) * (np.abs(x) < 4 * np.pi)
        t = 2.4
        debug = True

    if x is None or u0 is None or t is None:
        raise ValueError("Provide x, u0, and t, or call with no arguments for the demo.")

    x = np.asarray(x, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    flux = get_flux(flux_func)
    nx = x.size
    dx = np.max(np.diff(x))

    phi0 = _cumulative_integral(x, u0)

    if debug:
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        ax_u0, ax_phi0, ax_u_map, ax_phi_map, ax_u_sol, ax_phi_sol = axes.ravel()
        ax_u0.plot(x, u0, ".k")
        ax_u0.set_ylabel("u(x,t)")
        ax_phi0.plot(x, phi0, ".k")
        ax_phi0.set_ylabel(r"$\phi(x,t)$")
    else:
        ax_u_map = ax_phi_map = ax_u_sol = ax_phi_sol = None

    y = x - t * flux.df(u0)
    phi = phi0 - t * flux.Df(u0)

    if debug and ax_u_map is not None and ax_phi_map is not None:
        ax_u_map.plot(y, u0, ":k")
        ax_phi_map.plot(y, phi, ":k")

    branch_idx, n_shocks, n_branches, y = _identify_branches(y)

    if debug and ax_u_map is not None and ax_phi_map is not None:
        starts = branch_idx[0, :] - 1
        ends = branch_idx[1, :] - 1
        ax_u_map.plot(y[starts], u0[starts], "o")
        ax_u_map.plot(y[ends], u0[ends], "o")
        ax_phi_map.plot(y[starts], phi[starts], "o")
        ax_phi_map.plot(y[ends], phi[ends], "o")

    branches: list[Branch] = []
    for j in range(1, n_branches + 1):
        start = branch_idx[0, j - 1]
        end = branch_idx[1, j - 1]
        branch_x, branch_u, branch_phi = _interp_branch(x, y, u0, phi, start, end)
        branches.append(Branch(x=branch_x, u=branch_u, phi=branch_phi))

        if debug and ax_u_map is not None and ax_phi_map is not None:
            ax_u_map.plot(branch_x, branch_u)
            ax_phi_map.plot(branch_x, branch_phi)

    _find_intersections(
        branches,
        y,
        u0,
        phi,
        branch_idx,
        n_branches,
        dx,
        flux,
        debug,
        ax_u_map,
        ax_phi_map,
    )

    if debug and ax_u_map is not None and ax_phi_map is not None:
        ax_u_map.set_ylabel("u(x,t)")
        ax_phi_map.set_ylabel(r"$\phi(x,t)$")

    solution_path, intersections = _select_shock_path(branches, n_branches, n_shocks)
    x_out, u_out, phi_out = _assemble_solution(
        branches,
        n_branches,
        solution_path,
        intersections,
    )

    if debug and ax_u_sol is not None and ax_phi_sol is not None:
        ax_u_sol.plot(x_out, u_out, "-k")
        ax_u_sol.set_ylabel("u(x,t)")
        ax_u_sol.set_xlabel("x")
        ax_phi_sol.plot(x_out, phi_out, "-k")
        ax_phi_sol.set_ylabel(r"$\phi(x,t)$")
        ax_phi_sol.set_xlabel("x")
        fig.tight_layout()
        plt.show()

    return x_out, u_out, phi_out
