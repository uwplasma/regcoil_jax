from __future__ import annotations
import jax
import jax.numpy as jnp
from .geometry_fourier import FourierSurface, eval_surface_xyz_and_derivs, unit_normal

def expand_plasma_surface_with_offset(s: FourierSurface, theta: jnp.ndarray, zeta_plasma: jnp.ndarray, separation: float):
    xyz, dth, dze = eval_surface_xyz_and_derivs(s, theta, zeta_plasma)
    n_hat = unit_normal(dth, dze)
    xyz_off = xyz + separation * n_hat
    # For scalar theta/zeta inputs, eval_surface_xyz_and_derivs returns a degenerate
    # leading dimension of length 1, e.g. (1,3). If left as-is, Newton iterations
    # in solve_zeta_plasma_for_target_toroidal_angle will inadvertently grow rank
    # via broadcasting (z becomes (1,), then (1,1), ...), contaminating downstream
    # surface derivative shapes. Flatten to a point here to keep everything scalar.
    theta = jnp.asarray(theta)
    zeta_plasma = jnp.asarray(zeta_plasma)
    if theta.ndim == 0 and zeta_plasma.ndim == 0:
        xyz_off = jnp.reshape(xyz_off, (3,))
    return xyz_off

def wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    # Map to (-pi, pi]
    two_pi = 2*jnp.pi
    x = (x + jnp.pi) % two_pi - jnp.pi
    return x

def solve_zeta_plasma_for_target_toroidal_angle(
    s: FourierSurface,
    theta: float,
    zeta_target: float,
    separation: float,
    iters: int = 12,
) -> float:
    """Differentiable fixed-iteration Newton solve for zeta_plasma such that atan2(y,x) == zeta_target,
    matching the intent of regcoil_compute_offset_surface_xyz_of_thetazeta (Fortran uses fzero).
    """
    # Keep these as scalar arrays to avoid accidental rank growth via broadcasting
    # inside the Newton loop.
    theta = jnp.asarray(theta).reshape(())
    zeta_target = jnp.asarray(zeta_target).reshape(())

    def residual(zeta_plasma):
        xyz_off = expand_plasma_surface_with_offset(s, theta, zeta_plasma, separation)
        x = xyz_off[0]
        y = xyz_off[1]
        zeta_out = jnp.arctan2(y, x)
        return wrap_to_pi(zeta_out - zeta_target)

    # Newton iterations with safe damping to avoid blowups:
    z = zeta_target  # good initial guess (scalar)
    d_residual = jax.grad(residual)
    for _ in range(iters):
        f = residual(z)
        df = d_residual(z)
        # Avoid divide-by-zero:
        df = jnp.where(jnp.abs(df) < 1e-12, jnp.sign(df)*1e-12 + 1e-12, df)
        step = f / df
        # Smooth damping: limit step magnitude to ~0.5 rad
        step = step / (1.0 + jnp.abs(step)/0.5)
        z = z - step
    return z

def offset_surface_point(s: FourierSurface, theta: float, zeta_target: float, separation: float, iters: int = 12):
    zeta_plasma = solve_zeta_plasma_for_target_toroidal_angle(s, theta, zeta_target, separation, iters=iters)
    xyz_off = expand_plasma_surface_with_offset(s, theta, zeta_plasma, separation)
    return xyz_off
