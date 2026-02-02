#!/usr/bin/env python3
"""LP2021 QH lowres dipole-lattice fit to REGCOIL surface current B_sv."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    vmec = here / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
    script = here.parents[1] / "dipole_fit_to_surface_current_bn.py"
    cmd = [
        sys.executable,
        str(script),
        "--vmec",
        str(vmec),
        "--case",
        "LP2021_QH_lowres",
        "--separation",
        "0.75",
        "--dipole_offset",
        "0.25",
        "--dipole_stride",
        "14",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

