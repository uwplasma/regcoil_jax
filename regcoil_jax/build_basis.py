from __future__ import annotations
import jax.numpy as jnp
import numpy as np

from .modes import init_fourier_modes
from .constants import twopi

def build_basis_and_f(theta_coil, zeta_coil, drdtheta_coil, drdzeta_coil,
                      g_theta_theta, g_theta_zeta, g_zeta_zeta,
                      LB_dPhi_dtheta_coeff, LB_dPhi_dzeta_coeff,
                      mpol_potential: int, ntor_potential: int, nfp: int,
                      symmetry_option: int):
    """Build basis_functions and f_x,f_y,f_z,f_LB matching regcoil_build_matrices.f90.

    Notes on JAX:
      - mpol_potential / ntor_potential / nfp / symmetry_option are treated as
        compile-time constants elsewhere. Here we keep this function **non-jitted**
        to avoid tracer concretization pitfalls during early development.
      - init_fourier_modes is pure-numpy and returns numpy arrays.
    """
    mpol_potential = int(mpol_potential)
    ntor_potential = int(ntor_potential)
    nfp = int(nfp)
    symmetry_option = int(symmetry_option)

    xm_np, xn_np = init_fourier_modes(mpol_potential, ntor_potential, include_00=False)
    xn_np = xn_np * nfp  # match Fortran convention
    xm = jnp.asarray(xm_np, dtype=jnp.int32)
    xn = jnp.asarray(xn_np, dtype=jnp.int32)
    mnmax = int(xm_np.shape[0])

    # symmetry expansion size
    if symmetry_option in (1, 2):
        nb = mnmax
    elif symmetry_option == 3:
        nb = 2 * mnmax
    else:
        raise ValueError(f"Invalid symmetry_option={symmetry_option}")

    th = theta_coil[:, None]        # (T,1)
    ze = zeta_coil[None, :]         # (1,Z)
    angle = xm[None, None, :] * th[:, :, None] - xn[None, None, :] * ze[:, :, None]  # (T,Z,mn)

    cosang = jnp.cos(angle)
    sinang = jnp.sin(angle)

    # Build basis_functions(T,Z,nb) in Fortran ordering:
    # symmetry_option 1: sin(mθ - nζ)
    # symmetry_option 2: cos(mθ - nζ)
    # symmetry_option 3: both (sin then cos blocks)
    if symmetry_option == 1:
        basis_TZB = sinang
        dPhi_dtheta_TZB = cosang * xm
        dPhi_dzeta_TZB  = -cosang * xn
    elif symmetry_option == 2:
        basis_TZB = cosang
        dPhi_dtheta_TZB = -sinang * xm
        dPhi_dzeta_TZB  = sinang * xn
    else:
        basis_TZB = jnp.concatenate([sinang, cosang], axis=-1)
        dPhi_dtheta_TZB = jnp.concatenate([cosang * xm, -sinang * xm], axis=-1)
        dPhi_dzeta_TZB  = jnp.concatenate([-cosang * xn, sinang * xn], axis=-1)

    # Flatten (T,Z,nb) -> (Ncoil, nb) with Ncoil=T*Z in the same way build_matrices expects
    basis = jnp.reshape(basis_TZB, (-1, nb))

    # f_x,f_y,f_z match regcoil_build_matrices:
    # f = dPhi_dtheta * dr/dzeta - dPhi_dzeta * dr/dtheta
    # Here drdtheta_coil and drdzeta_coil are (3,T,Z). We want (Ncoil,nb) for each component.
    rth = drdtheta_coil   # (3,T,Z)
    rze = drdzeta_coil
    # expand to (3,T,Z,1) * (T,Z,nb) -> (3,T,Z,nb)
    f3 = (rze[:, :, :, None] * dPhi_dtheta_TZB[None, :, :, :] -
          rth[:, :, :, None] * dPhi_dzeta_TZB[None, :, :, :])

    fx = jnp.reshape(f3[0], (-1, nb))
    fy = jnp.reshape(f3[1], (-1, nb))
    fz = jnp.reshape(f3[2], (-1, nb))

    # f_LB placeholder (Laplace–Beltrami regularization) for now:
    flb = jnp.zeros_like(basis)

    return xm, xn, basis, fx, fy, fz, flb
