from __future__ import annotations
import jax.numpy as jnp

def torus_xyz_and_derivs(theta, zeta, R0: float, a: float):
    """Analytic circular torus parameterization.

    x = (R0 + a cosθ) cosζ
    y = (R0 + a cosθ) sinζ
    z = a sinθ

    Returns:
      r      : (3,T,Z)
      r_theta: (3,T,Z)
      r_zeta : (3,T,Z)
      nunit  : (3,T,Z)
      normN  : (T,Z) where N = r_zeta × r_theta
    """
    th = theta[:, None]               # (T,1)
    ze = zeta[None, :]                # (1,Z)
    cth = jnp.cos(th); sth = jnp.sin(th)
    cze = jnp.cos(ze); sze = jnp.sin(ze)

    R = R0 + a * cth                  # (T,1)

    x = R * cze                       # (T,Z)
    y = R * sze                       # (T,Z)
    z = (a * sth) * jnp.ones_like(ze) # (T,Z)

    # derivatives w.r.t theta
    dR_dth = -a * sth                 # (T,1)
    dx_dth = dR_dth * cze             # (T,Z)
    dy_dth = dR_dth * sze             # (T,Z)
    dz_dth = (a * cth) * jnp.ones_like(ze)  # (T,Z)

    # derivatives w.r.t zeta
    dx_dze = -R * sze                 # (T,Z) = -y
    dy_dze =  R * cze                 # (T,Z) =  x
    dz_dze = jnp.zeros_like(dx_dze)

    r_theta = jnp.stack([dx_dth, dy_dth, dz_dth], axis=0)  # (3,T,Z)
    r_zeta  = jnp.stack([dx_dze, dy_dze, dz_dze], axis=0)  # (3,T,Z)
    r       = jnp.stack([x, y, z], axis=0)                 # (3,T,Z)

    # normal = r_zeta × r_theta
    nvec = jnp.cross(jnp.moveaxis(r_zeta, 0, -1), jnp.moveaxis(r_theta, 0, -1))
    nvec = jnp.moveaxis(nvec, -1, 0)  # (3,T,Z)
    normN = jnp.linalg.norm(nvec, axis=0)
    nunit = nvec / normN

    return r, r_theta, r_zeta, nunit, normN


def torus_xyz_and_derivs2(theta, zeta, R0: float, a: float):
    """Analytic circular torus parameterization with 2nd derivatives.

    Returns:
      r      : (3,T,Z)
      r_theta: (3,T,Z)
      r_zeta : (3,T,Z)
      r_tt   : (3,T,Z)
      r_tz   : (3,T,Z)
      r_zz   : (3,T,Z)
      nunit  : (3,T,Z)
      normN  : (T,Z) where N = r_zeta × r_theta
    """
    th = theta[:, None]               # (T,1)
    ze = zeta[None, :]                # (1,Z)
    cth = jnp.cos(th); sth = jnp.sin(th)
    cze = jnp.cos(ze); sze = jnp.sin(ze)

    R = R0 + a * cth                  # (T,1)
    x = R * cze                       # (T,Z)
    y = R * sze                       # (T,Z)
    z = (a * sth) * jnp.ones_like(ze) # (T,Z)

    # first derivatives
    dR_dth = -a * sth                 # (T,1)
    dx_dth = dR_dth * cze
    dy_dth = dR_dth * sze
    dz_dth = (a * cth) * jnp.ones_like(ze)

    dx_dze = -R * sze
    dy_dze =  R * cze
    dz_dze = jnp.zeros_like(dx_dze)

    # second derivatives
    d2R_dth2 = -a * cth
    d2x_dth2 = d2R_dth2 * cze
    d2y_dth2 = d2R_dth2 * sze
    d2z_dth2 = (-a * sth) * jnp.ones_like(ze)

    d2x_dthdze = -(dR_dth * sze)
    d2y_dthdze =  (dR_dth * cze)
    d2z_dthdze = jnp.zeros_like(d2x_dthdze)

    d2x_dze2 = -R * cze
    d2y_dze2 = -R * sze
    d2z_dze2 = jnp.zeros_like(d2x_dze2)

    r_theta = jnp.stack([dx_dth, dy_dth, dz_dth], axis=0)
    r_zeta = jnp.stack([dx_dze, dy_dze, dz_dze], axis=0)
    r_tt = jnp.stack([d2x_dth2, d2y_dth2, d2z_dth2], axis=0)
    r_tz = jnp.stack([d2x_dthdze, d2y_dthdze, d2z_dthdze], axis=0)
    r_zz = jnp.stack([d2x_dze2, d2y_dze2, d2z_dze2], axis=0)
    r = jnp.stack([x, y, z], axis=0)

    nvec = jnp.cross(jnp.moveaxis(r_zeta, 0, -1), jnp.moveaxis(r_theta, 0, -1))
    nvec = jnp.moveaxis(nvec, -1, 0)
    normN = jnp.linalg.norm(nvec, axis=0)
    nunit = nvec / normN

    return r, r_theta, r_zeta, r_tt, r_tz, r_zz, nunit, normN
