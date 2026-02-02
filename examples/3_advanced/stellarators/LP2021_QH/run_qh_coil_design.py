#!/usr/bin/env python3
"""LP2021 QH reactor-scale lowres coil design demo (VMEC boundary → REGCOIL_JAX → cut coils → optimize currents)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    vmec = here / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
    script = here.parents[1] / "coil_design_cut_optimize.py"
    cmd = [
        sys.executable,
        str(script),
        "--vmec",
        str(vmec),
        "--case",
        "LP2021_QH_lowres",
        "--separation",
        "0.75",
        "--ntheta",
        "48",
        "--nzeta",
        "48",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

