from __future__ import annotations

from pathlib import Path

import numpy as np


def test_read_vmec_input_boundary_lp2021_qa():
    from regcoil_jax.io_vmec_input import read_vmec_input_boundary, vmec_input_boundary_as_fourier_surface
    from regcoil_jax.geometry_fourier import eval_surface_xyz_and_derivs

    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "3_advanced" / "stellarators" / "LP2021_QA" / "input.LandremanPaul2021_QA_lowres"
    assert path.exists(), "expected VMEC input file to exist in repo"

    b = read_vmec_input_boundary(path)
    assert b.nfp == 2
    assert b.xm.size == b.xn.size == b.rmnc.size == b.zmns.size
    assert np.isfinite(b.rmnc).all()
    assert np.isfinite(b.zmns).all()

    surf = vmec_input_boundary_as_fourier_surface(b)
    # Evaluate a small grid; just ensure no NaNs and the shape is correct.
    import jax.numpy as jnp

    th = jnp.linspace(0.0, 2.0 * jnp.pi, 9, endpoint=False)
    ze = jnp.linspace(0.0, 2.0 * jnp.pi / surf.nfp, 7, endpoint=False)
    xyz, rth, rze = eval_surface_xyz_and_derivs(surf, th, ze)
    assert xyz.shape == (9, 7, 3)
    assert rth.shape == (9, 7, 3)
    assert rze.shape == (9, 7, 3)
    assert np.isfinite(np.asarray(xyz)).all()
