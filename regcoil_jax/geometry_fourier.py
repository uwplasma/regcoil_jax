from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class FourierSurface:
    nfp: int
    lasym: bool
    xm: jnp.ndarray  # (mnmax,)
    xn: jnp.ndarray  # (mnmax,) already multiplied by nfp where appropriate
    rmnc: jnp.ndarray
    zmns: jnp.ndarray
    rmns: jnp.ndarray
    zmnc: jnp.ndarray

def eval_surface_xyz_and_derivs(s: FourierSurface, theta: jnp.ndarray, zeta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Matches regcoil_expand_plasma_surface.f90 parameterization:
    # R = sum rmnc*cos(m*theta - n*zeta) + rmns*sin(...) if lasym
    # Z = sum zmns*sin(...) + zmnc*cos(...) if lasym
    # phi = zeta, x=R*cos(phi), y=R*sin(phi).
    # Returns (xyz, dxyz_dtheta, dxyz_dzeta) with shapes (...,3).
    theta = jnp.asarray(theta)
    zeta = jnp.asarray(zeta)
    # Broadcast theta and zeta to grid
    th, ze = jnp.broadcast_arrays(theta[..., None], zeta[None, ...])
    # angles shape (..., mnmax)
    ang = th[..., None] * s.xm + (-ze[..., None]) * s.xn
    cosang = jnp.cos(ang)
    sinang = jnp.sin(ang)

    R = jnp.sum(s.rmnc * cosang, axis=-1)
    Z = jnp.sum(s.zmns * sinang, axis=-1)
    dR_dtheta = jnp.sum(s.rmnc * (-sinang) * s.xm, axis=-1)
    dR_dzeta  = jnp.sum(s.rmnc * ( sinang) * s.xn, axis=-1)  # because d/dzeta of cos(m th - n ze) = +n sin(...)
    dZ_dtheta = jnp.sum(s.zmns * ( cosang) * s.xm, axis=-1)
    dZ_dzeta  = jnp.sum(s.zmns * (-cosang) * s.xn, axis=-1)

    if s.lasym:
        R = R + jnp.sum(s.rmns * sinang, axis=-1)
        Z = Z + jnp.sum(s.zmnc * cosang, axis=-1)
        dR_dtheta = dR_dtheta + jnp.sum(s.rmns * (cosang) * s.xm, axis=-1)
        dR_dzeta  = dR_dzeta  + jnp.sum(s.rmns * (-cosang) * s.xn, axis=-1)
        dZ_dtheta = dZ_dtheta + jnp.sum(s.zmnc * (-sinang) * s.xm, axis=-1)
        dZ_dzeta  = dZ_dzeta  + jnp.sum(s.zmnc * ( sinang) * s.xn, axis=-1)

    phi = ze
    cosphi = jnp.cos(phi)
    sinphi = jnp.sin(phi)

    x = R * cosphi
    y = R * sinphi
    z = Z

    # derivatives
    dphi_dzeta = 1.0
    dx_dtheta = dR_dtheta * cosphi
    dy_dtheta = dR_dtheta * sinphi
    dz_dtheta = dZ_dtheta

    dx_dzeta = dR_dzeta * cosphi + R * (-sinphi) * dphi_dzeta
    dy_dzeta = dR_dzeta * sinphi + R * ( cosphi) * dphi_dzeta
    dz_dzeta = dZ_dzeta

    xyz = jnp.stack([x,y,z], axis=-1)
    dxyz_dtheta = jnp.stack([dx_dtheta, dy_dtheta, dz_dtheta], axis=-1)
    dxyz_dzeta = jnp.stack([dx_dzeta, dy_dzeta, dz_dzeta], axis=-1)
    return xyz, dxyz_dtheta, dxyz_dzeta


def eval_surface_xyz_and_derivs2(
    s: FourierSurface, theta: jnp.ndarray, zeta: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate xyz and 1st/2nd derivatives for a FourierSurface.

    Returns:
      xyz         : (..., 3)
      dxyz_dtheta : (..., 3)
      dxyz_dzeta  : (..., 3)
      d2xyz_dtheta2    : (..., 3)
      d2xyz_dthetadzeta: (..., 3)
      d2xyz_dzeta2     : (..., 3)
    """
    theta = jnp.asarray(theta)
    zeta = jnp.asarray(zeta)
    th, ze = jnp.broadcast_arrays(theta[..., None], zeta[None, ...])

    ang = th[..., None] * s.xm + (-ze[..., None]) * s.xn
    cosang = jnp.cos(ang)
    sinang = jnp.sin(ang)

    m = s.xm
    n = s.xn
    m2 = m * m
    n2 = n * n
    mn = m * n

    # Base (symmetric) Fourier series: R from rmnc*cos, Z from zmns*sin
    R = jnp.sum(s.rmnc * cosang, axis=-1)
    Z = jnp.sum(s.zmns * sinang, axis=-1)

    dR_dtheta = jnp.sum(s.rmnc * (-sinang) * m, axis=-1)
    dR_dzeta = jnp.sum(s.rmnc * (sinang) * n, axis=-1)
    dZ_dtheta = jnp.sum(s.zmns * (cosang) * m, axis=-1)
    dZ_dzeta = jnp.sum(s.zmns * (-cosang) * n, axis=-1)

    d2R_dtheta2 = jnp.sum(s.rmnc * (-cosang) * m2, axis=-1)
    d2R_dthetadzeta = jnp.sum(s.rmnc * (cosang) * mn, axis=-1)
    d2R_dzeta2 = jnp.sum(s.rmnc * (-cosang) * n2, axis=-1)

    d2Z_dtheta2 = jnp.sum(s.zmns * (-sinang) * m2, axis=-1)
    d2Z_dthetadzeta = jnp.sum(s.zmns * (sinang) * mn, axis=-1)
    d2Z_dzeta2 = jnp.sum(s.zmns * (-sinang) * n2, axis=-1)

    if s.lasym:
        R = R + jnp.sum(s.rmns * sinang, axis=-1)
        Z = Z + jnp.sum(s.zmnc * cosang, axis=-1)

        dR_dtheta = dR_dtheta + jnp.sum(s.rmns * (cosang) * m, axis=-1)
        dR_dzeta = dR_dzeta + jnp.sum(s.rmns * (-cosang) * n, axis=-1)
        dZ_dtheta = dZ_dtheta + jnp.sum(s.zmnc * (-sinang) * m, axis=-1)
        dZ_dzeta = dZ_dzeta + jnp.sum(s.zmnc * (sinang) * n, axis=-1)

        d2R_dtheta2 = d2R_dtheta2 + jnp.sum(s.rmns * (-sinang) * m2, axis=-1)
        d2R_dthetadzeta = d2R_dthetadzeta + jnp.sum(s.rmns * (sinang) * mn, axis=-1)
        d2R_dzeta2 = d2R_dzeta2 + jnp.sum(s.rmns * (-sinang) * n2, axis=-1)

        d2Z_dtheta2 = d2Z_dtheta2 + jnp.sum(s.zmnc * (-cosang) * m2, axis=-1)
        d2Z_dthetadzeta = d2Z_dthetadzeta + jnp.sum(s.zmnc * (cosang) * mn, axis=-1)
        d2Z_dzeta2 = d2Z_dzeta2 + jnp.sum(s.zmnc * (-cosang) * n2, axis=-1)

    phi = ze
    cosphi = jnp.cos(phi)
    sinphi = jnp.sin(phi)

    x = R * cosphi
    y = R * sinphi
    z = Z

    dx_dtheta = dR_dtheta * cosphi
    dy_dtheta = dR_dtheta * sinphi
    dz_dtheta = dZ_dtheta

    dx_dzeta = dR_dzeta * cosphi - R * sinphi
    dy_dzeta = dR_dzeta * sinphi + R * cosphi
    dz_dzeta = dZ_dzeta

    d2x_dtheta2 = d2R_dtheta2 * cosphi
    d2y_dtheta2 = d2R_dtheta2 * sinphi
    d2z_dtheta2 = d2Z_dtheta2

    d2x_dthetadzeta = d2R_dthetadzeta * cosphi - dR_dtheta * sinphi
    d2y_dthetadzeta = d2R_dthetadzeta * sinphi + dR_dtheta * cosphi
    d2z_dthetadzeta = d2Z_dthetadzeta

    d2x_dzeta2 = d2R_dzeta2 * cosphi - 2.0 * dR_dzeta * sinphi - R * cosphi
    d2y_dzeta2 = d2R_dzeta2 * sinphi + 2.0 * dR_dzeta * cosphi - R * sinphi
    d2z_dzeta2 = d2Z_dzeta2

    xyz = jnp.stack([x, y, z], axis=-1)
    dxyz_dtheta = jnp.stack([dx_dtheta, dy_dtheta, dz_dtheta], axis=-1)
    dxyz_dzeta = jnp.stack([dx_dzeta, dy_dzeta, dz_dzeta], axis=-1)
    d2xyz_dtheta2 = jnp.stack([d2x_dtheta2, d2y_dtheta2, d2z_dtheta2], axis=-1)
    d2xyz_dthetadzeta = jnp.stack([d2x_dthetadzeta, d2y_dthetadzeta, d2z_dthetadzeta], axis=-1)
    d2xyz_dzeta2 = jnp.stack([d2x_dzeta2, d2y_dzeta2, d2z_dzeta2], axis=-1)

    if theta.ndim == 0 and zeta.ndim == 0:
        xyz = jnp.reshape(xyz, (3,))
        dxyz_dtheta = jnp.reshape(dxyz_dtheta, (3,))
        dxyz_dzeta = jnp.reshape(dxyz_dzeta, (3,))
        d2xyz_dtheta2 = jnp.reshape(d2xyz_dtheta2, (3,))
        d2xyz_dthetadzeta = jnp.reshape(d2xyz_dthetadzeta, (3,))
        d2xyz_dzeta2 = jnp.reshape(d2xyz_dzeta2, (3,))

    return xyz, dxyz_dtheta, dxyz_dzeta, d2xyz_dtheta2, d2xyz_dthetadzeta, d2xyz_dzeta2

def unit_normal(dxyz_dtheta: jnp.ndarray, dxyz_dzeta: jnp.ndarray) -> jnp.ndarray:
    # Normal = (dr/dzeta) x (dr/dtheta) per Fortran:
    N = jnp.cross(dxyz_dzeta, dxyz_dtheta)
    return N / jnp.linalg.norm(N, axis=-1, keepdims=True)
