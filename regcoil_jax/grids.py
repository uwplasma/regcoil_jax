from __future__ import annotations
import jax.numpy as jnp
from .constants import twopi

def theta_grid(ntheta: int):
    i = jnp.arange(ntheta, dtype=jnp.float64)
    return twopi * i / ntheta

def zeta_grid(nzeta: int, nfp: int):
    i = jnp.arange(nzeta, dtype=jnp.float64)
    return (twopi / nfp) * i / nzeta

def dtheta(ntheta: int) -> float:
    return float(2.0 * jnp.pi / ntheta)

def dzeta(nzeta: int, nfp: int) -> float:
    return float((2.0 * jnp.pi / nfp) / nzeta)
