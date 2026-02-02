#!/usr/bin/env python3
"""n3are finite-pressure lowres coil design demo (wout VMEC equilibrium)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    vmec = here / "wout_n3are_R7.75B5.7_lowres.nc"
    script = here.parents[1] / "coil_design_cut_optimize.py"
    cmd = [
        sys.executable,
        str(script),
        "--vmec",
        str(vmec),
        "--case",
        "n3are_lowres",
        "--separation",
        "0.90",
        "--ntheta",
        "48",
        "--nzeta",
        "48",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

