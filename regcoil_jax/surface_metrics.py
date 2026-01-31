from __future__ import annotations
import jax.numpy as jnp

def metrics_and_normals(r_theta, r_zeta):
    """Compute first fundamental form and unit normals.
    Inputs r_theta,r_zeta: (3,T,Z)
    Returns gtt,gtz,gzz: (T,Z), nunit: (3,T,Z), normN: (T,Z)
    where N = r_zeta × r_theta (same convention as torus module).
    """
    gtt = jnp.sum(r_theta*r_theta, axis=0)
    gzz = jnp.sum(r_zeta*r_zeta, axis=0)
    gtz = jnp.sum(r_theta*r_zeta, axis=0)
    nvec = jnp.cross(jnp.moveaxis(r_zeta,0,-1), jnp.moveaxis(r_theta,0,-1))
    nvec = jnp.moveaxis(nvec,-1,0)
    norm = jnp.linalg.norm(nvec, axis=0)
    nunit = nvec / norm
    return gtt, gtz, gzz, nunit, norm
