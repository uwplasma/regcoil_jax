from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time as _time

import numpy as np

from .utils import parse_namelist, resolve_existing_path
from .io_vmec import read_wout_boundary, compute_curpol_and_G


@dataclass(frozen=True)
class RunResult:
    input_path: str
    output_nc: str
    output_log: str
    exit_code: int
    chosen_idx: int | None


def run_regcoil(
    input_path: str,
    *,
    verbose: bool = False,
    no_jit: bool = False,
    x32: bool = False,
    debug_dir: str | None = None,
) -> RunResult:
    """Run regcoil_jax on a namelist file, writing outputs next to the input file.

    This is a programmatic entrypoint used by tests/examples. The CLI wraps this
    function (but also supports selecting the platform before importing JAX).
    """
    # Import JAX lazily so callers can configure env vars first.
    import jax

    if not x32:
        jax.config.update("jax_enable_x64", True)
        os.environ.setdefault("JAX_ENABLE_X64", "True")
    if no_jit:
        jax.config.update("jax_disable_jit", True)
    try:
        jax.config.update("jax_default_matmul_precision", "highest")
    except Exception:
        pass

    import jax.numpy as jnp

    from .geometry_fourier import FourierSurface
    from .surfaces import plasma_surface_from_inputs, coil_surface_from_inputs
    from .build_matrices_jax import build_matrices
    from .solve_jax import lambda_grid, solve_for_lambdas, diagnostics, choose_lambda, auto_regularization_solve, svd_scan
    from .io_output import write_output_nc

    input_path = resolve_existing_path(input_path)
    input_path_abs = os.path.abspath(input_path)
    input_dir = os.path.dirname(input_path_abs) or "."

    def _resolve_relpath(p: str | None) -> str | None:
        if p is None:
            return None
        if os.path.isabs(p):
            return p
        cand = os.path.join(input_dir, p)
        return cand if os.path.exists(cand) else p

    base = os.path.basename(input_path_abs)
    if not base.startswith("regcoil_in."):
        raise ValueError("Input file must be named regcoil_in.XXX for some extension XXX")

    inputs = parse_namelist(input_path_abs)

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

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    # VMEC boundary (optional, used by plasma/coil options 2/3/4 and coil options 2/4)
    vmec_surface = None
    gpl = int(inputs.get("geometry_option_plasma", 1))
    gcl = int(inputs.get("geometry_option_coil", 1))
    if gpl in (2, 3, 4) or gcl in (2, 4) or (gpl == 0 and "wout_filename" in inputs):
        wout = inputs.get("wout_filename", None)
        if wout is None:
            raise ValueError("VMEC-based geometry requires wout_filename")
        wout = _resolve_relpath(str(wout))
        radial_mode = "half" if gpl in (3, 4) else "full"
        bound = read_wout_boundary(str(wout), radial_mode=radial_mode)
        curpol, G, nfp, lasym = compute_curpol_and_G(bound)
        if verbose:
            print(f"[vmec] nfp={nfp} lasym={lasym} curpol={curpol} G={G}")
        # Match regcoil_init_plasma_mod.f90 behavior: if a VMEC wout file is used
        # for the plasma surface, override the net poloidal current.
        inputs["net_poloidal_current_amperes"] = float(G)
        inputs["curpol"] = float(curpol)
        inputs["nfp_imposed"] = int(nfp)

        def _as_fourier_surface(b) -> FourierSurface:
            return FourierSurface(
                nfp=b.nfp,
                lasym=b.lasym,
                xm=jnp.asarray(b.xm, dtype=jnp.int32),
                xn=jnp.asarray(b.xn, dtype=jnp.int32),
                rmnc=jnp.asarray(b.rmnc, dtype=jnp.float64),
                zmns=jnp.asarray(b.zmns, dtype=jnp.float64),
                rmns=jnp.asarray(b.rmns, dtype=jnp.float64),
                zmnc=jnp.asarray(b.zmnc, dtype=jnp.float64),
            )

        vmec_surface = _as_fourier_surface(bound)
        # For output parity, match R0_plasma to VMEC major radius.
        try:
            inputs["r0_plasma"] = float(getattr(bound, "Rmajor_p"))
        except Exception:
            pass

        if debug_dir is not None:
            try:
                np.savez(
                    os.path.join(debug_dir, "vmec_boundary_coeffs.npz"),
                    xm=np.asarray(bound.xm),
                    xn=np.asarray(bound.xn),
                    rmnc=np.asarray(bound.rmnc),
                    zmns=np.asarray(bound.zmns),
                    rmns=np.asarray(bound.rmns),
                    zmnc=np.asarray(bound.zmnc),
                )
            except Exception as e:
                if verbose:
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
        vec_keys = ("r", "rth", "rze", "nunit")
        for k in vec_keys:
            a = surf[k]
            if a.ndim != 3 or a.shape[0] != 3:
                raise ValueError(f"{name}.{k} expected shape (3,T,Z) but got {tuple(a.shape)}")
        a = surf["normN"]
        if a.ndim != 2:
            raise ValueError(f"{name}.normN expected shape (T,Z) but got {tuple(a.shape)}")

    if verbose:
        print(f"[regcoil_jax] input: {input_path_abs}")
        print(f"[regcoil_jax] parsed keys: {sorted(list(inputs.keys()))}")
        print(f"[regcoil_jax] plasma shapes: {_surface_shapes(plasma)}")
        print(f"[regcoil_jax] coil   shapes: {_surface_shapes(coil)}")

    if verbose or debug_dir is not None:
        _assert_surface_shapes("plasma", plasma)
        _assert_surface_shapes("coil", coil)

    if debug_dir is not None:
        try:
            with open(os.path.join(debug_dir, "surface_shapes.json"), "w", encoding="utf-8") as f:
                json.dump({"plasma": _surface_shapes(plasma), "coil": _surface_shapes(coil)}, f, indent=2, sort_keys=True)
        except Exception as e:
            if verbose:
                print(f"[regcoil_jax] debug_dir: could not dump surface shapes: {e}")

    t0_total = _time.time()

    if verbose:
        print("[regcoil_jax] building matrices (this may take a bit the first time due to JIT)...")
    mats = build_matrices(inputs, plasma, coil)

    general_option = int(inputs.get("general_option", 1))
    if general_option == 2:
        from .io_nescout import read_nescout_potentials

        nescout = inputs.get("nescout_filename", None)
        if nescout is None:
            raise ValueError("general_option=2 requires nescout_filename")
        nescout = _resolve_relpath(str(nescout))
        pots = read_nescout_potentials(
            str(nescout),
            mpol_potential=int(inputs.get("mpol_potential", mats.get("mpol_potential", 12))),
            ntor_potential=int(inputs.get("ntor_potential", mats.get("ntor_potential", 12))),
            nfp=int(mats.get("nfp", 1)),
            symmetry_option=int(inputs.get("symmetry_option", 1)),
        )
        if len(pots.solutions) == 0:
            raise ValueError(f"No Phi(m,n) blocks found in nescout file: {nescout}")
        # Match regcoil_validate_input.f90: overwrite nlambda with the number of potentials detected.
        inputs = dict(inputs)
        inputs["nlambda"] = int(len(pots.solutions))
        lambdas = lambda_grid(inputs)
        sols = jnp.asarray(np.stack(pots.solutions, axis=0), dtype=jnp.float64)
        chi2_B, chi2_K, max_B, max_K = diagnostics(mats, sols)
        idx = None
        exit_code = 0
    elif general_option == 3:
        lambdas, sols, chi2_B, chi2_K, max_B, max_K = svd_scan(mats)
        idx = None
        exit_code = 0
    elif general_option in (4, 5):
        lambdas, sols, chi2_B, chi2_K, max_B, max_K, idx, exit_code = auto_regularization_solve(inputs, mats)
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

    out_nc = os.path.join(input_dir, "regcoil_out" + base[10:] + ".nc")
    t1_total = _time.time()
    write_output_nc(
        out_nc,
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

    out_log = out_nc.replace(".nc", ".log")
    try:
        with open(out_log, "w", encoding="utf-8") as f:
            f.write(f"input={input_path_abs}\n")
            f.write(f"nbasis={int(mats['matrix_B'].shape[0])}\n")
            f.write(f"lambdas=[{float(lambdas[0]):.6e}, {float(lambdas[-1]):.6e}] n={len(lambdas)}\n")
            if general_option in (4, 5):
                f.write(f"target_option={str(inputs.get('target_option', 'max_K')).strip()}\n")
                f.write(f"target_value={float(inputs.get('target_value', 0.0)):.16e}\n")
                f.write(f"target_option_p={float(inputs.get('target_option_p', 4.0)):.16e}\n")
            f.write(f"exit_code={int(exit_code)}\n")
            if idx is not None:
                f.write(f"chosen_idx={idx} chosen_lambda={float(lambdas[idx]):.6e}\n")
            f.write("j lambda chi2_B chi2_K max|B| maxK\n")
            for j in range(len(lambdas)):
                f.write(
                    f"{j} {float(lambdas[j]):.6e} {float(chi2_B[j]):.6e} {float(chi2_K[j]):.6e} {float(max_B[j]):.6e} {float(max_K[j]):.6e}\n"
                )
    except Exception as e:
        if verbose:
            print(f"[regcoil_jax] WARNING: could not write summary log: {e}")

    if verbose:
        for j in range(len(lambdas)):
            mark = "*" if (idx is not None and j == idx) else " "
            print(
                f"{mark} j={j:02d}  lambda={float(lambdas[j]):.3e}  chi2_B={float(chi2_B[j]):.3e}  chi2_K={float(chi2_K[j]):.3e}  max|B|={float(max_B[j]):.3e}  maxK={float(max_K[j]):.3e}"
            )

    return RunResult(
        input_path=input_path_abs,
        output_nc=os.path.abspath(out_nc),
        output_log=os.path.abspath(out_log),
        exit_code=int(exit_code),
        chosen_idx=int(idx) if idx is not None else None,
    )
