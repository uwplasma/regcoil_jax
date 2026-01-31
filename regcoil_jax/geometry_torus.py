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
