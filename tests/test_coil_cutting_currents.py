from __future__ import annotations

import numpy as np


def test_write_makecoil_filaments_supports_per_coil_currents(tmp_path):
    from regcoil_jax.coil_cutting import write_makecoil_filaments

    f1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)
    f2 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=float)
    currents = np.array([1.25, -3.5], dtype=float)

    out = tmp_path / "coils.test"
    write_makecoil_filaments(out, filaments_xyz=[f1, f2], coil_currents=currents, nfp=1)

    txt = out.read_text(encoding="utf-8").splitlines()
    # Extract numeric lines (x y z I) ignoring header/footer and sentinel lines.
    numeric = []
    for line in txt:
        parts = line.split()
        if len(parts) == 4:
            numeric.append(parts)
    # First filament has 3 lines, second filament has 3 lines.
    assert len(numeric) == 6
    I1 = [float(p[3]) for p in numeric[:3]]
    I2 = [float(p[3]) for p in numeric[3:]]
    assert all(abs(v - currents[0]) < 1e-12 for v in I1)
    assert all(abs(v - currents[1]) < 1e-12 for v in I2)

