from __future__ import annotations

import numpy as np

from regcoil_jax.io_nescin import NescinCurrentSurface, read_nescin_current_surface, write_nescin_current_surface


def test_nescin_current_surface_roundtrip(tmp_path):
    surf = NescinCurrentSurface(
        xm=np.array([0, 1, 2], dtype=int),
        xn=np.array([0, -1, 2], dtype=int),
        rmnc=np.array([10.0, 0.5, 0.1], dtype=float),
        zmns=np.array([0.0, 0.5, -0.1], dtype=float),
        rmns=np.array([0.0, 0.0, 0.0], dtype=float),
        zmnc=np.array([0.0, 0.0, 0.0], dtype=float),
    )
    p = tmp_path / "nescin.test"
    write_nescin_current_surface(p, surf)
    got = read_nescin_current_surface(p)

    assert np.array_equal(got.xm, surf.xm)
    assert np.array_equal(got.xn, surf.xn)
    assert np.allclose(got.rmnc, surf.rmnc)
    assert np.allclose(got.zmns, surf.zmns)
    assert np.allclose(got.rmns, surf.rmns)
    assert np.allclose(got.zmnc, surf.zmnc)

