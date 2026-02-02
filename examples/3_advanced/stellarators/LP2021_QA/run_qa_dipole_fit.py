#!/usr/bin/env python3
"""LP2021 QA lowres dipole-lattice fit to REGCOIL surface current B_sv."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    vmec = here / "wout_LandremanPaul2021_QA_lowres.nc"
    script = here.parents[1] / "dipole_fit_to_surface_current_bn.py"
    cmd = [
        sys.executable,
        str(script),
        "--vmec",
        str(vmec),
        "--case",
        "LP2021_QA_lowres",
        "--separation",
        "0.60",
        "--dipole_offset",
        "0.20",
        "--dipole_stride",
        "12",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

