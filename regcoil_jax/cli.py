from __future__ import annotations

import argparse
import os

from .run import run_regcoil


def main() -> None:
    parser = argparse.ArgumentParser(description="regcoil_jax (JAX port of REGCOIL)")
    parser.add_argument("input", type=str, help="Input namelist file (must be regcoil_in.*)")
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        choices=["cpu", "gpu"],
        help="Force JAX platform (must be set before JAX is imported)",
    )
    parser.add_argument("--no_jit", action="store_true", help="Disable jit (useful for debugging)")
    parser.add_argument("--x32", action="store_true", help="Force float32 (disables x64)")
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostics.")
    parser.add_argument("--debug_dir", type=str, default=None, help="Write debug artifacts (npz/json) to this directory.")
    args = parser.parse_args()

    # Must be set before importing jax:
    if args.platform:
        os.environ["JAX_PLATFORM_NAME"] = args.platform

    run_regcoil(
        args.input,
        verbose=bool(args.verbose),
        no_jit=bool(args.no_jit),
        x32=bool(args.x32),
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()

