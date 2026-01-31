from __future__ import annotations
import numpy as np

def init_fourier_modes(mpol: int, ntor: int, include_00: bool = False):
    """Match regcoil_init_Fourier_modes_mod.f90.

    xm >= 0, xn can be negative/zero/positive. When xm==0, xn>=0.
    If include_00 is False, (m,n)=(0,0) is excluded.

    Returns:
      xm, xn as numpy int32 arrays of shape (mnmax,).

    Note: this is intentionally pure-numpy so it can be called safely from
    both jitted and non-jitted contexts (and it is fast anyway).
    """
    mpol = int(mpol)
    ntor = int(ntor)
    mnmax = mpol * (2 * ntor + 1) + ntor
    if include_00:
        mnmax += 1

    xm = np.zeros((mnmax,), dtype=np.int32)
    xn = np.zeros((mnmax,), dtype=np.int32)

    index = 0
    if include_00:
        index = 1  # leave (0,0) in slot 0

    # m=0, n=1..ntor
    for jn in range(1, ntor + 1):
        xm[index] = 0
        xn[index] = jn
        index += 1

    # m=1..mpol, n=-ntor..ntor
    for jm in range(1, mpol + 1):
        for jn in range(-ntor, ntor + 1):
            xm[index] = jm
            xn[index] = jn
            index += 1

    if index != mnmax:
        raise RuntimeError(f"init_fourier_modes internal error: index={index} mnmax={mnmax}")

    return xm, xn
