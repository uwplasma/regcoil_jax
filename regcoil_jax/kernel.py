from __future__ import annotations
import jax
import jax.numpy as jnp
from .constants import mu0, pi

def rotate_z(vec3, phi):
    """Rotate a 3-vector field about +z by angle phi (radians).
    vec3 shape (...,3) or (3,...). Returns same shape.
    """
    c = jnp.cos(phi); s = jnp.sin(phi)
    if vec3.shape[0] == 3:
        x, y, z = vec3[0], vec3[1], vec3[2]
        xr = c*x - s*y
        yr = s*x + c*y
        return jnp.stack([xr, yr, z], axis=0)
    else:
        x, y, z = vec3[...,0], vec3[...,1], vec3[...,2]
        xr = c*x - s*y
        yr = s*x + c*y
        return jnp.stack([xr, yr, z], axis=-1)

@jax.jit
def inductance_and_h_sum(plasma_r, plasma_n, coil_r, coil_n, coil_f, nfp: int):
    """Compute effective inductance kernel (summed over nfp images) and the h field.

    plasma_r: (P,3)
    plasma_n: (P,3) unit normals at plasma
    coil_r:   (C,3) positions for one field period (zeta in [0,2pi/nfp))
    coil_n:   (C,3) unit normals for one field period
    coil_f:   (C,3) factor_for_h vectors for one field period
    Returns:
      induct_eff: (P,C) scaled by mu0/(4pi)
      h_eff:      (P,)  UNweighted sum over coil points (still needs *dtheta*dzeta*mu0/(8pi^2) externally)
    """
    P = plasma_r.shape[0]
    C = coil_r.shape[0]

    def body(l, acc):
        ind_acc, h_acc = acc
        phi = (2.0*jnp.pi/ nfp) * l
        rC = rotate_z(coil_r.T, phi).T
        nC = rotate_z(coil_n.T, phi).T
        fC = rotate_z(coil_f.T, phi).T

        # dr = rP - rC (matches regcoil_build_matrices.f90 where dx = x_plasma - x_coil)
        # Note: the inductance kernel is even in dr, but the h kernel is odd in dr.
        dr = plasma_r[:,None,:] - rC[None,:,:]
        dr2 = jnp.sum(dr*dr, axis=-1)  # (P,C)
        inv_r = jnp.reciprocal(jnp.sqrt(dr2))
        dr2inv = jnp.reciprocal(dr2)
        inv_r3 = inv_r * dr2inv  # 1/r^3

        np_dot_nc = jnp.sum(plasma_n[:,None,:] * nC[None,:,:], axis=-1)
        np_dot_dr = jnp.sum(plasma_n[:,None,:] * dr, axis=-1)
        nc_dot_dr = jnp.sum(nC[None,:,:] * dr, axis=-1)

        ind = (np_dot_nc - 3.0*dr2inv*np_dot_dr*nc_dot_dr) * inv_r3  # (P,C)

        # h kernel: dot( cross(fC, dr), nP ) / r^3
        # cross(fC, dr): (P,C,3)
        cr = jnp.cross(fC[None,:,:], dr)
        h = jnp.sum(cr * plasma_n[:,None,:], axis=-1) * inv_r3  # (P,C)

        return (ind_acc + ind, h_acc + jnp.sum(h, axis=1))

    ind0 = jnp.zeros((P,C), dtype=plasma_r.dtype)
    h0 = jnp.zeros((P,), dtype=plasma_r.dtype)
    (ind_sum, h_sum) = jax.lax.fori_loop(0, nfp, body, (ind0, h0))
    ind_sum = ind_sum * (mu0/(4.0*pi))
    return ind_sum, h_sum
