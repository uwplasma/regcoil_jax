from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
import numpy as np

from .constants import mu0, pi
from .dipoles import dipole_bnormal, dipole_array_from_surface_offset
from .optimize import minimize_adam, OptimizeResult


@dataclass(frozen=True)
class PermanentMagnetSolveResult:
    dipole_positions: np.ndarray  # (M,3)
    dipole_moments: np.ndarray  # (M,3)
    cg_info: int


def _cg_info_to_int(info) -> int:
    # jax.scipy.sparse.linalg.cg has returned different info types across JAX versions:
    #   - None
    #   - an int
    #   - a namedtuple/struct with a `converged` boolean
    if info is None:
        return 0
    if isinstance(info, (int, np.integer)):
        return int(info)
    if hasattr(info, "converged"):
        return 0 if bool(getattr(info, "converged")) else -1
    return -1


def place_dipoles_on_winding_surface(
    *,
    surface_points: Any,
    surface_normals_unit: Any,
    offset: float,
    stride: int,
    moment_magnitude: float = 0.0,
    normal_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a dipole lattice on an offset-from-surface set of points.

    This is a convenience wrapper around :func:`regcoil_jax.dipoles.dipole_array_from_surface_offset`
    that optionally initializes moments to a fixed magnitude (either normal-only or free-vector).
    """
    pos, m0 = dipole_array_from_surface_offset(
        surface_points=np.asarray(surface_points, dtype=float),
        surface_normals_unit=np.asarray(surface_normals_unit, dtype=float),
        offset=float(offset),
        stride=int(stride),
    )
    if float(moment_magnitude) <= 0.0:
        return pos, m0
    if normal_only:
        # Initialize along surface normal direction.
        nh = np.asarray(surface_normals_unit, dtype=float)[:: int(stride)]
        nh = nh.reshape((-1, 3))
        nh = nh / np.linalg.norm(nh, axis=1, keepdims=True)
        m0 = float(moment_magnitude) * nh
    else:
        m0 = float(moment_magnitude) * np.asarray(m0, dtype=float)
    return pos, m0


def solve_dipole_moments_ridge_cg(
    *,
    points: Any,
    normals_unit: Any,
    dipole_positions: Any,
    target_bnormal: Any,
    l2_moment: float = 1e-8,
    batch: int = 4096,
    tol: float = 1e-10,
    maxiter: int = 200,
) -> PermanentMagnetSolveResult:
    """Solve for dipole moments that match a target B·n using ridge-regularized least squares.

    The problem is:
      minimize_m  || A m - b ||^2 + l2_moment * ||m||^2

    where m concatenates the 3-vector moment for each dipole and A is the linear map
    from moments to B·n on the target surface points.

    This routine uses conjugate gradients on the normal equations:
      (Aᵀ A + l2 I) m = Aᵀ b

    without explicitly forming A, by using JAX's reverse-mode autodiff (VJP) to apply Aᵀ.
    """
    pts = jnp.asarray(points, dtype=jnp.float64)
    nhat = jnp.asarray(normals_unit, dtype=jnp.float64)
    pos = jnp.asarray(dipole_positions, dtype=jnp.float64)
    targ = jnp.asarray(target_bnormal, dtype=jnp.float64).reshape((-1,))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if nhat.shape != pts.shape:
        raise ValueError("normals_unit must have same shape as points")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("dipole_positions must be (M,3)")
    if int(targ.shape[0]) != int(pts.shape[0]):
        raise ValueError("target_bnormal must have length equal to number of points")

    M = int(pos.shape[0])
    x0 = jnp.zeros((M * 3,), dtype=jnp.float64)

    # For small problems, build the dense linear map explicitly and solve in one shot.
    # This is robust and helps keep unit tests deterministic across JAX versions.
    N = int(pts.shape[0])
    if (N * M) <= 20000:
        coeff = mu0 / (4.0 * pi)
        eps2 = 1e-9 * 1e-9
        r = pts[:, None, :] - pos[None, :, :]  # (N,M,3)
        r2 = jnp.sum(r * r, axis=-1) + eps2  # (N,M)
        inv_r = 1.0 / jnp.sqrt(r2)
        inv_r3 = inv_r / r2
        inv_r5 = inv_r3 / r2
        n_dot_r = jnp.sum(nhat[:, None, :] * r, axis=-1)  # (N,M)
        K = coeff * (3.0 * r * (n_dot_r * inv_r5)[:, :, None] - nhat[:, None, :] * inv_r3[:, :, None])  # (N,M,3)
        A = K.reshape((N, M * 3))
        alpha = float(l2_moment)
        # Solve ridge regression using an SVD (more stable than normal equations).
        # A = U diag(s) Vᵀ => x = V diag(s/(s^2+α)) Uᵀ b
        U, s, VT = jnp.linalg.svd(A, full_matrices=False)
        UTb = U.T @ targ
        filt = s / (s * s + alpha)
        x_sol = VT.T @ (filt * UTb)
        return PermanentMagnetSolveResult(
            dipole_positions=np.asarray(pos, dtype=float),
            dipole_moments=np.asarray(x_sol.reshape((M, 3)), dtype=float),
            cg_info=0,
        )

    def Ax(x: jnp.ndarray) -> jnp.ndarray:
        m = x.reshape((M, 3))
        return dipole_bnormal(points=pts, normals_unit=nhat, positions=pos, moments=m, batch=int(batch))

    def At(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # A is linear in x, but we compute Aᵀ with a VJP for clarity.
        _, vjp = jax.vjp(Ax, x)
        return vjp(y)[0]

    rhs = At(targ, x0)
    alpha = float(l2_moment)

    def matvec(x: jnp.ndarray) -> jnp.ndarray:
        y = Ax(x)
        return At(y, x0) + alpha * x

    x_sol, info = jsla.cg(matvec, rhs, x0=x0, tol=float(tol), maxiter=int(maxiter))
    m_sol = x_sol.reshape((M, 3))
    return PermanentMagnetSolveResult(
        dipole_positions=np.asarray(pos, dtype=float),
        dipole_moments=np.asarray(m_sol, dtype=float),
        cg_info=_cg_info_to_int(info),
    )


@dataclass(frozen=True)
class FixedMagnitudeMagnetOptResult:
    dipole_positions: np.ndarray  # (M,3)
    dipole_moments: np.ndarray  # (M,3) fixed magnitude
    loss_history: np.ndarray  # (steps,)


def optimize_dipole_orientations_fixed_magnitude(
    *,
    points: Any,
    normals_unit: Any,
    dipole_positions: Any,
    target_bnormal: Any,
    moment_magnitude: float,
    v0: Any | None = None,
    steps: int = 400,
    lr: float = 2e-2,
    l2_orientation: float = 0.0,
    batch: int = 4096,
    eps: float = 1e-20,
) -> FixedMagnitudeMagnetOptResult:
    """Optimize dipole *orientations* with fixed moment magnitude (smooth relaxation).

    This is a convenient proxy for "fixed-strength magnets" when a full discrete (±) or
    integer programming approach is not desired.

    We parameterize each dipole's moment as:
    .. math::

       \\mathbf{m}_i = m_0\\,\\frac{\\mathbf{v}_i}{\\lVert\\mathbf{v}_i\\rVert}

    with an optional L2 regularization on v_i (to avoid drift).
    """
    pts = jnp.asarray(points, dtype=jnp.float64)
    nhat = jnp.asarray(normals_unit, dtype=jnp.float64)
    pos = jnp.asarray(dipole_positions, dtype=jnp.float64)
    targ = jnp.asarray(target_bnormal, dtype=jnp.float64).reshape((-1,))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if nhat.shape != pts.shape:
        raise ValueError("normals_unit must have same shape as points")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("dipole_positions must be (M,3)")
    if int(targ.shape[0]) != int(pts.shape[0]):
        raise ValueError("target_bnormal must have length equal to number of points")
    M = int(pos.shape[0])
    m0 = float(moment_magnitude)
    if m0 <= 0.0:
        raise ValueError("moment_magnitude must be > 0")

    if v0 is None:
        v_init = jnp.ones((M, 3), dtype=jnp.float64)
    else:
        v_init = jnp.asarray(v0, dtype=jnp.float64).reshape((M, 3))
    x0 = v_init.reshape((-1,))

    def moments_from_v(v: jnp.ndarray) -> jnp.ndarray:
        v = v.reshape((M, 3))
        n = jnp.linalg.norm(v, axis=1, keepdims=True)
        n = jnp.where(n == 0.0, 1.0, n)
        u = v / (n + float(eps))
        return m0 * u

    def loss_fn(x: jnp.ndarray) -> jnp.ndarray:
        v = x.reshape((M, 3))
        m = moments_from_v(v)
        bn = dipole_bnormal(points=pts, normals_unit=nhat, positions=pos, moments=m, batch=int(batch))
        err = bn - targ
        loss = jnp.mean(err * err)
        if float(l2_orientation) != 0.0:
            loss = loss + float(l2_orientation) * jnp.mean(v * v)
        return loss

    res: OptimizeResult = minimize_adam(loss_fn, x0, steps=int(steps), lr=float(lr), jit=True)
    v_opt = jnp.asarray(res.x, dtype=jnp.float64).reshape((M, 3))
    m_opt = moments_from_v(v_opt)
    return FixedMagnitudeMagnetOptResult(
        dipole_positions=np.asarray(pos, dtype=float),
        dipole_moments=np.asarray(m_opt, dtype=float),
        loss_history=np.asarray(res.loss_history, dtype=float),
    )
