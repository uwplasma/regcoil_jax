from __future__ import annotations
import argparse
import os
import json
import numpy as np

from .utils import parse_namelist, resolve_existing_path
from .io_vmec import read_wout_boundary, compute_curpol_and_G

def main():
    parser = argparse.ArgumentParser(description="regcoil_jax (JAX port, WIP)")
    parser.add_argument("input", type=str, help="Input namelist file (must be regcoil_in.*)")
    parser.add_argument("--platform", type=str, default=None, choices=["cpu", "gpu"],
                        help="Force JAX platform (must be set before JAX is imported)")
    parser.add_argument("--no_jit", action="store_true", help="Disable jit (useful for debugging)")
    parser.add_argument("--x32", action="store_true", help="Force float32 (disables x64)")
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostics.")
    parser.add_argument("--debug_dir", type=str, default=None, help="Write debug artifacts (npz/json) to this directory.")
    args = parser.parse_args()

    # Must be set before importing jax:
    if args.platform:
        os.environ["JAX_PLATFORM_NAME"] = args.platform

    # Prefer enabling x64 before importing JAX (avoids dtype truncation warnings):
    if not args.x32:
        os.environ.setdefault("JAX_ENABLE_X64", "True")

    import jax

    # Default to x64 for parity unless explicitly forced to x32:
    if not args.x32:
        jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    # Best precision for dense linear algebra:
    try:
        jax.config.update("jax_default_matmul_precision", "highest")
    except Exception:
        pass

    if args.no_jit:
        jax.config.update("jax_disable_jit", True)

    from .geometry_fourier import FourierSurface
    from .surfaces import plasma_surface_from_inputs, coil_surface_from_inputs
    from .build_matrices_jax import build_matrices
    from .solve_jax import lambda_grid, solve_for_lambdas, diagnostics, choose_lambda, auto_regularization_solve
    from .io_output import write_output_nc
    import time as _time

    def _as_fourier_surface(bound) -> FourierSurface:
        # io_vmec returns numpy arrays; convert to jnp
        return FourierSurface(
            nfp=bound.nfp,
            lasym=bound.lasym,
            xm=jnp.asarray(bound.xm, dtype=jnp.int32),
            xn=jnp.asarray(bound.xn, dtype=jnp.int32),
            rmnc=jnp.asarray(bound.rmnc, dtype=jnp.float64),
            zmns=jnp.asarray(bound.zmns, dtype=jnp.float64),
            rmns=jnp.asarray(bound.rmns, dtype=jnp.float64),
            zmnc=jnp.asarray(bound.zmnc, dtype=jnp.float64),
        )

    input_path = resolve_existing_path(args.input)
    input_path_abs = os.path.abspath(input_path)
    input_dir = os.path.dirname(input_path_abs) or '.'

    def _resolve_relpath(p: str | None) -> str | None:
        if p is None:
            return None
        if os.path.isabs(p):
            return p
        cand = os.path.join(input_dir, p)
        return cand if os.path.exists(cand) else p

    base = os.path.basename(input_path)
    if not base.startswith("regcoil_in."):
        raise SystemExit("Input file must be named regcoil_in.XXX for some extension XXX")

    inputs = parse_namelist(input_path)
    if args.verbose:
        print(f"[regcoil_jax] input: {input_path_abs}")
    print(f"[regcoil_jax] parsed keys: {sorted(list(inputs.keys()))}")

    # Resolve any filenames relative to the input directory (matching REGCOIL behavior).
    if "wout_filename" in inputs:
        inputs["wout_filename"] = _resolve_relpath(str(inputs["wout_filename"]))
    if "bnorm_filename" in inputs:
        inputs["bnorm_filename"] = _resolve_relpath(str(inputs["bnorm_filename"]))
    if "nescin_filename" in inputs:
        inputs["nescin_filename"] = _resolve_relpath(str(inputs["nescin_filename"]))
    if "shape_filename_plasma" in inputs:
        inputs["shape_filename_plasma"] = _resolve_relpath(str(inputs["shape_filename_plasma"]))
    if "efit_filename" in inputs:
        inputs["efit_filename"] = _resolve_relpath(str(inputs["efit_filename"]))

    # VMEC boundary (optional, used by plasma/coil options 2/3/4 and coil options 2/4)
    vmec_surface = None
    gpl = int(inputs.get("geometry_option_plasma", 1))
    gcl = int(inputs.get("geometry_option_coil", 1))
    if gpl in (2, 3, 4) or gcl in (2, 4) or (gpl == 0 and "wout_filename" in inputs):
        wout = inputs.get("wout_filename", None)
        if wout is None:
            raise SystemExit("VMEC-based geometry requires wout_filename")
        wout = _resolve_relpath(wout)
        radial_mode = "half" if gpl in (3, 4) else "full"
        bound = read_wout_boundary(wout, radial_mode=radial_mode)
        curpol, G, nfp, lasym = compute_curpol_and_G(bound)
        print(f"[vmec] nfp={nfp} lasym={lasym} curpol={curpol} G={G}")
        # Match regcoil_init_plasma_mod.f90 behavior: if a VMEC wout file is used
        # for the plasma surface, override the net poloidal current with the value
        # derived from VMEC's bvco (half-mesh extrapolation).
        if gpl in (2, 3, 4) and G is not None:
            if args.verbose and "net_poloidal_current_amperes" in inputs:
                print(
                    "[regcoil_jax] overriding input net_poloidal_current_amperes="
                    f"{float(inputs['net_poloidal_current_amperes']):.6e} with VMEC value {float(G):.6e}"
                )
            inputs["net_poloidal_current_amperes"] = float(G)
        if gpl in (2, 3, 4) and curpol is not None:
            inputs["curpol"] = float(curpol)
        # Match regcoil_init_plasma_mod.f90: overwrite R0_plasma with VMEC's major radius (Rmajor).
        if gpl in (2, 3, 4) and getattr(bound, "Rmajor_p", None) is not None:
            inputs["r0_plasma"] = float(bound.Rmajor_p)
        vmec_surface = _as_fourier_surface(bound)

    
    # Optional debug artifact dumping (outside JIT)
    if args.debug_dir:
        import json, numpy as _np
        os.makedirs(args.debug_dir, exist_ok=True)
        with open(os.path.join(args.debug_dir, 'inputs.json'), 'w', encoding='utf-8') as f:
            json.dump(inputs, f, indent=2, sort_keys=True, default=str)
        if vmec_surface is not None:
            # Dump the underlying VMEC boundary coefficients for reproducibility
            try:
                _np.savez(os.path.join(args.debug_dir, 'vmec_boundary_coeffs.npz'),
                         xm=_np.array(bound.xm), xn=_np.array(bound.xn),
                         rmnc=_np.array(bound.rmnc), zmns=_np.array(bound.zmns),
                         rmns=_np.array(getattr(bound, 'rmns', _np.zeros_like(bound.rmnc))),
                         zmnc=_np.array(getattr(bound, 'zmnc', _np.zeros_like(bound.zmns))),
                         nfp=int(bound.nfp), lasym=bool(bound.lasym))
            except Exception as e:
                print(f"[regcoil_jax] debug_dir: could not dump vmec boundary coeffs: {e}")
    # Build surfaces
    plasma = plasma_surface_from_inputs(inputs, vmec_surface)
    coil = coil_surface_from_inputs(inputs, plasma, vmec_surface)

    def _surface_shapes(surf):
        return dict(
            theta=tuple(surf["theta"].shape),
            zeta=tuple(surf["zeta"].shape),
            r=tuple(surf["r"].shape),
            rth=tuple(surf["rth"].shape),
            rze=tuple(surf["rze"].shape),
            nunit=tuple(surf["nunit"].shape),
            normN=tuple(surf["normN"].shape),
        )

    def _assert_surface_shapes(name: str, surf):
        # Expected REGCOIL convention: (3,ntheta,nzeta) for vector fields.
        vec_keys = ("r", "rth", "rze", "nunit")
        for k in vec_keys:
            a = surf[k]
            if a.ndim != 3 or a.shape[0] != 3:
                raise ValueError(f"{name}.{k} expected shape (3,T,Z) but got {tuple(a.shape)}")
        a = surf["normN"]
        if a.ndim != 2:
            raise ValueError(f"{name}.normN expected shape (T,Z) but got {tuple(a.shape)}")

    if args.verbose:
        print(f"[regcoil_jax] plasma shapes: {_surface_shapes(plasma)}")
        print(f"[regcoil_jax] coil   shapes: {_surface_shapes(coil)}")

    # Only assert in verbose/debug runs to keep the default path permissive during development.
    if args.verbose or args.debug_dir:
        _assert_surface_shapes("plasma", plasma)
        _assert_surface_shapes("coil", coil)

    if args.debug_dir:
        try:
            with open(os.path.join(args.debug_dir, "surface_shapes.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"plasma": _surface_shapes(plasma), "coil": _surface_shapes(coil)},
                    f,
                    indent=2,
                    sort_keys=True,
                )
        except Exception as e:
            print(f"[regcoil_jax] debug_dir: could not dump surface shapes: {e}")

    t0_total = _time.time()

    # Matrices
    print("[regcoil_jax] building matrices (this may take a bit the first time due to JIT)...")
    mats = build_matrices(inputs, plasma, coil)
    print(f"[regcoil_jax] nbasis={mats['matrix_B'].shape[0]}  Np={plasma['r'].shape[1]*plasma['r'].shape[2]}  Nc={coil['r'].shape[1]*coil['r'].shape[2]}")

    general_option = int(inputs.get("general_option", 1))
    if general_option == 5:
        lambdas, sols, chi2_B, chi2_K, max_B, max_K, idx, exit_code = auto_regularization_solve(inputs, mats)
        # For output parity: if the chosen target option is max_K_lse or lp_norm_K, also store
        # the per-lambda diagnostic arrays with the Fortran variable names.
        target_option = str(inputs.get("target_option", "max_K")).strip()
        if target_option in ("max_K_lse", "lp_norm_K"):
            from .solve_jax import target_quantity
            vals = []
            for j in range(int(len(lambdas))):
                vals.append(
                    target_quantity(
                        mats,
                        sol=sols[j],
                        chi2_B=chi2_B[j],
                        chi2_K=chi2_K[j],
                        max_B=max_B[j],
                        max_K=max_K[j],
                        target_option=target_option,
                        target_option_p=float(inputs.get("target_option_p", 4.0)),
                    )
                )
            mats[target_option] = jnp.asarray(vals)
    else:
        lambdas = lambda_grid(inputs)
        sols = solve_for_lambdas(mats, lambdas)
        chi2_B, chi2_K, max_B, max_K = diagnostics(mats, sols)
        idx = choose_lambda(inputs, lambdas, chi2_B, chi2_K, max_B, max_K)
        exit_code = 0

    print(f"[regcoil_jax] lambdas: [{float(lambdas[0]):.3e} .. {float(lambdas[-1]):.3e}]  n={len(lambdas)}")
    if general_option == 5 and exit_code != 0:
        # Match REGCOIL's conventions (see regcoil_auto_regularization_solve.f90)
        msg = { -1: "lambda search did not converge",
                -2: "target_value not achievable (too low)",
                -3: "target_value not achievable (too high)"}.get(int(exit_code), f"exit_code={int(exit_code)}")
        print(f"[regcoil_jax] general_option=5: {msg}")
    if idx is not None:
        print(f"[regcoil_jax] general_option=5 target -> picked jlambda={idx}  lambda={float(lambdas[idx]):.6e}")

    # Output filename like Fortran
    out_path = os.path.join(input_dir, "regcoil_out" + base[10:] + ".nc")
    t1_total = _time.time()
    write_output_nc(
        out_path,
        inputs,
        mats,
        lambdas,
        sols,
        chi2_B,
        chi2_K,
        max_B,
        max_K,
        exit_code=exit_code,
        total_time=float(t1_total - t0_total),
    )
    print(f"[regcoil_jax] wrote: {out_path}")

    # Also write a small human-readable summary log next to the netCDF file.
    summary_path = out_path.replace(".nc", ".log")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"input={input_path_abs}\n")
            f.write(f"nbasis={int(mats['matrix_B'].shape[0])}\n")
            f.write(f"lambdas=[{float(lambdas[0]):.6e}, {float(lambdas[-1]):.6e}] n={len(lambdas)}\n")
            if general_option == 5:
                f.write(f"target_option={str(inputs.get('target_option', 'max_K')).strip()}\n")
                f.write(f"target_value={float(inputs.get('target_value', 0.0)):.16e}\n")
                f.write(f"target_option_p={float(inputs.get('target_option_p', 4.0)):.16e}\n")
            f.write(f"exit_code={int(exit_code)}\n")
            if idx is not None:
                f.write(f"chosen_idx={idx} chosen_lambda={float(lambdas[idx]):.6e}\n")
            f.write("j lambda chi2_B chi2_K max|B| maxK\n")
            for j in range(len(lambdas)):
                f.write(f"{j} {float(lambdas[j]):.6e} {float(chi2_B[j]):.6e} {float(chi2_K[j]):.6e} {float(max_B[j]):.6e} {float(max_K[j]):.6e}\n")
        if args.verbose:
            print(f"[regcoil_jax] wrote: {summary_path}")
    except Exception as e:
        print(f"[regcoil_jax] WARNING: could not write summary log: {e}")

    # Print quick summary table
    for j in range(len(lambdas)):
        mark = "*" if (idx is not None and j==idx) else " "
        print(f"{mark} j={j:02d}  lambda={float(lambdas[j]):.3e}  chi2_B={float(chi2_B[j]):.3e}  chi2_K={float(chi2_K[j]):.3e}  max|B|={float(max_B[j]):.3e}  maxK={float(max_K[j]):.3e}")

if __name__ == "__main__":
    main()
