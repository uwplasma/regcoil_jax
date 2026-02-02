#!/usr/bin/env python3
"""Autodiff example: optimize VMEC offset-surface separation.

This script demonstrates differentiating through:
  separation -> winding surface geometry -> matrix build -> linear solve -> diagnostics.

It is intentionally lightweight and uses a small grid by default so it runs on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from regcoil_jax.winding_surface_optimization import optimize_vmec_offset_separation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wout", type=str, default=str(Path(__file__).with_name("wout_d23p4_tm.nc")))
    ap.add_argument("--separation0", type=float, default=0.5)
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--step_size", type=float, default=0.1)
    ap.add_argument("--ntheta", type=int, default=16)
    ap.add_argument("--nzeta", type=int, default=16)
    ap.add_argument("--mpol_potential", type=int, default=6)
    ap.add_argument("--ntor_potential", type=int, default=6)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0e-14)
    args = ap.parse_args()

    res = optimize_vmec_offset_separation(
        wout_filename=args.wout,
        separation0=args.separation0,
        nsteps=args.nsteps,
        step_size=args.step_size,
        ntheta=args.ntheta,
        nzeta=args.nzeta,
        mpol_potential=args.mpol_potential,
        ntor_potential=args.ntor_potential,
        lam=args.lam,
    )

    sep0 = float(res.separation_history[0])
    sepf = float(res.separation_history[-1])
    obj0 = float(res.objective_history[0])
    objf = float(res.objective_history[-1])
    print(f"separation: {sep0:.6e} -> {sepf:.6e}")
    print(f"objective:  {obj0:.6e} -> {objf:.6e}")


if __name__ == "__main__":
    main()

