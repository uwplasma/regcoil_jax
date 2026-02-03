from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time as _time

import numpy as np

from .utils import parse_namelist, resolve_existing_path
from .io_vmec import read_wout_boundary, compute_curpol_and_G
from .io_vmec_input import read_vmec_input_boundary, vmec_input_boundary_as_fourier_surface
from .fortran_verbose import FortranVerbose


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
    from .sensitivity_jax import compute_sensitivity_outputs

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
    logger = FortranVerbose(enabled=bool(verbose))

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

    if verbose:
        logger.header()
        logger.resolution_block(
            input_basename=base,
            ntheta_plasma=int(inputs.get("ntheta_plasma", 0)),
            ntheta_coil=int(inputs.get("ntheta_coil", 0)),
            nzeta_plasma=int(inputs.get("nzeta_plasma", 0)),
            nzeta_coil=int(inputs.get("nzeta_coil", 0)),
            mpol_potential=int(inputs.get("mpol_potential", 12)),
            ntor_potential=int(inputs.get("ntor_potential", 12)),
            symmetry_option=int(inputs.get("symmetry_option", 1)),
        )

        # Print the lambda list early (as in regcoil_read_input.f90 + regcoil_compute_lambda.f90).
        try:
            lambdas_preview = np.asarray(lambda_grid(inputs), dtype=float).tolist()
        except Exception:
            lambdas_preview = []
        if lambdas_preview:
            logger.lambda_list(lambdas_preview)

    # VMEC boundary (optional, used by plasma/coil options 2/3/4 and coil options 2/4)
    vmec_surface = None
    gpl = int(inputs.get("geometry_option_plasma", 1))
    gcl = int(inputs.get("geometry_option_coil", 1))
    vmec_source = None
    vmec_metadata = None
    if gpl in (2, 3, 4) or gcl in (2, 4) or (gpl == 0 and "wout_filename" in inputs):
        wout = inputs.get("wout_filename", None)
        if wout is None:
            raise ValueError("VMEC-based geometry requires wout_filename")
        wout = _resolve_relpath(str(wout))
        vmec_source = str(wout)
        if str(wout).endswith(".nc"):
            radial_mode = "half" if gpl in (3, 4) else "full"
            bound = read_wout_boundary(str(wout), radial_mode=radial_mode)
            curpol, G, nfp, lasym = compute_curpol_and_G(bound)
            vmec_metadata = dict(nfp=int(nfp), lasym=bool(lasym), curpol=curpol, G=G)
            # Match regcoil_init_plasma_mod.f90 behavior: if a VMEC wout file is used
            # for the plasma surface and VMEC profile arrays are available, override net poloidal current.
            if (curpol is not None) and (G is not None):
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
        else:
            # VMEC input boundary only (vacuum-style workflows): use RBC/ZBS (and optional asym terms).
            vin = read_vmec_input_boundary(str(wout))
            vmec_surface = vmec_input_boundary_as_fourier_surface(vin)
            inputs["nfp_imposed"] = int(vin.nfp)
            if verbose:
                logger.surface_detail(
                    f"VMEC input boundary: nfp={int(vin.nfp)} lasym={bool(vin.lasym)} mnmax={int(vin.xm.size)} file={wout}"
                )

    def _surface_area_volume(*, surf, ntheta: int, nzeta: int, nfp: int) -> tuple[float, float]:
        dth = float(2.0 * np.pi / max(int(ntheta), 1))
        dze = float((2.0 * np.pi / max(int(nfp), 1)) / max(int(nzeta), 1))
        area = float(nfp) * dth * dze * float(jnp.sum(surf["normN"]))

        r3tz = jnp.asarray(surf["r"])  # (3,T,Z) for a single field period
        R2 = r3tz[0] * r3tz[0] + r3tz[1] * r3tz[1]
        Z = r3tz[2]
        R2_half = 0.5 * (R2[:-1, :] + R2[1:, :])
        dZ = Z[1:, :] - Z[:-1, :]
        acc = jnp.sum(R2_half * dZ)
        acc = acc + jnp.sum(0.5 * (R2[0, :] + R2[-1, :]) * (Z[0, :] - Z[-1, :]))
        volume = float(jnp.abs((float(nfp) * acc) * dze / 2.0))
        return area, volume

    # Build surfaces (and print Fortran-style initialization messages in verbose mode).
    t0_init_plasma = _time.perf_counter()
    if verbose:
        logger.init_surface("plasma")
        gpl_eff = gpl
        if gpl == 0:
            gpl_eff = 2 if (vmec_surface is not None and inputs.get("wout_filename", None) is not None) else 1
        if gpl_eff == 1:
            logger.surface_detail("Building a plain circular torus.")
        elif gpl_eff in (2, 3):
            logger.surface_detail(f"Using VMEC boundary from {vmec_source}.")
        elif gpl_eff == 4:
            logger.surface_detail(f"Using VMEC boundary with straight-field-line poloidal coordinate from {vmec_source}.")
        elif gpl_eff == 5:
            logger.surface_detail(f"Using EFIT boundary from {inputs.get('efit_filename', '(unknown)')}.")
        elif gpl_eff == 6:
            logger.surface_detail(f"Using Fourier-table boundary from {inputs.get('shape_filename_plasma', '(unknown)')}.")
        elif gpl_eff == 7:
            logger.surface_detail(f"Using FOCUS boundary from {inputs.get('shape_filename_plasma', '(unknown)')}.")
        else:
            logger.surface_detail(f"Using geometry_option_plasma={gpl_eff}.")
        if vmec_metadata is not None:
            logger.surface_detail(
                f"VMEC metadata: nfp={vmec_metadata['nfp']} lasym={vmec_metadata['lasym']} curpol={vmec_metadata['curpol']} G={vmec_metadata['G']}"
            )

    plasma = plasma_surface_from_inputs(inputs, vmec_surface)
    t1_init_plasma = _time.perf_counter()

    if verbose:
        area_p, vol_p = _surface_area_volume(
            surf=plasma,
            ntheta=int(inputs.get("ntheta_plasma", 0)),
            nzeta=int(inputs.get("nzeta_plasma", 0)),
            nfp=int(plasma["nfp"]),
        )
        logger.surface_area_volume(which="plasma", area=area_p, volume=vol_p)
        if vmec_surface is None:
            logger.surface_detail(
                "No VMEC file is available, so net_poloidal_current_Amperes will be taken from the regcoil input."
            )
        logger.done_init_surface("plasma", float(t1_init_plasma - t0_init_plasma))

    t0_init_coil = _time.perf_counter()
    if verbose:
        logger.init_surface("coil")
        gcl_eff = gcl
        if gcl == 0:
            gcl_eff = 2 if (vmec_surface is not None and inputs.get("wout_filename", None) is not None) else 1
        if gcl_eff == 1:
            logger.surface_detail("Building a plain circular torus.")
        elif gcl_eff == 2:
            logger.surface_detail("Offset surface from plasma (VMEC) with Fourier refit.")
        elif gcl_eff == 4:
            logger.surface_detail("Offset surface from plasma (VMEC) with constant-arclength theta.")
        else:
            logger.surface_detail(f"Using geometry_option_coil={gcl_eff}.")

    coil = coil_surface_from_inputs(inputs, plasma, vmec_surface)
    t1_init_coil = _time.perf_counter()

    if verbose:
        logger.surface_detail(f"Evaluating coil surface & derivatives: {float(t1_init_coil - t0_init_coil):.8E} sec.")
        area_c, vol_c = _surface_area_volume(
            surf=coil,
            ntheta=int(inputs.get("ntheta_coil", 0)),
            nzeta=int(inputs.get("nzeta_coil", 0)),
            nfp=int(plasma["nfp"]),
        )
        logger.surface_area_volume(which="coil", area=area_c, volume=vol_c)
        logger.done_init_surface("coil", float(t1_init_coil - t0_init_coil))

        load_bnorm = bool(inputs.get("load_bnorm", False))
        if not load_bnorm:
            logger.bnorm_message("Not reading a bnorm file, so Bnormal_from_plasma_current arrays will all be 0.")
        else:
            if plasma.get("focus_bnorm", None) is not None:
                logger.bnorm_message("Reading Bn modes embedded in the FOCUS boundary file.")
            else:
                logger.bnorm_message(f"Reading bnorm file: {inputs.get('bnorm_filename', '(missing)')}")

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

    t0_total = _time.perf_counter()

    t0_build = _time.perf_counter()
    sync_timing = bool(int(os.environ.get("REGCOIL_JAX_SYNC_TIMING", "0")))
    mats = build_matrices(inputs, plasma, coil, verbose=bool(verbose), logger=logger, sync_timing=sync_timing)
    t1_build = _time.perf_counter()
    if verbose:
        logger.p(" Optimal LWORK: (not applicable; using jax.numpy.linalg.solve)")

    t0_solve = _time.perf_counter()
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
            import jax
            from .solve_jax import _target_quantity_from_K_distribution

            mode = 0 if target_option == "max_K_lse" else 1
            p = float(inputs.get("target_option_p", 4.0))

            fx = mats["fx"]
            fy = mats["fy"]
            fz = mats["fz"]
            dx = mats["dx"]
            dy = mats["dy"]
            dz = mats["dz"]
            normNc = mats["normNc"]
            area_coil = float(mats.get("area_coil"))
            dth_c = float(mats.get("dth_c"))
            dze_c = float(mats.get("dze_c"))
            nfp = int(mats.get("nfp"))

            mats[target_option] = jax.vmap(
                lambda sol, mK: _target_quantity_from_K_distribution(
                    fx=fx,
                    fy=fy,
                    fz=fz,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    normNc=normNc,
                    area_coil=area_coil,
                    dth_c=dth_c,
                    dze_c=dze_c,
                    nfp=nfp,
                    sol=sol,
                    max_K=mK,
                    target_option_p=p,
                    mode=mode,
                )
            )(sols, max_K)
    else:
        lambdas = lambda_grid(inputs)
        sols = solve_for_lambdas(mats, lambdas)
        chi2_B, chi2_K, max_B, max_K = diagnostics(mats, sols)
        idx = choose_lambda(inputs, lambdas, chi2_B, chi2_K, max_B, max_K)
        exit_code = 0
    t1_solve = _time.perf_counter()

    # Extra diagnostics for Fortran-style terminal output.
    nfp = int(mats.get("nfp", 1))
    dth_c = float(mats["dth_c"])
    dze_c = float(mats["dze_c"])
    normNc = mats["normNc"]
    flb = mats["flb"]
    dLB = mats["d_Laplace_Beltrami"]
    KLB = dLB[None, :] - (sols @ flb.T)  # (nlambda, Ncoil)
    LB2_times_N = (KLB * KLB) / normNc[None, :]
    chi2_LB = (float(nfp) * dth_c * dze_c) * jnp.sum(LB2_times_N, axis=1)

    area_coil = float(mats.get("area_coil", 1.0))
    rms_K = jnp.sqrt(jnp.asarray(chi2_K, dtype=jnp.float64) / area_coil)

    # Optional sensitivity outputs (winding surface Fourier-coefficient derivatives).
    sensitivity_option = int(inputs.get("sensitivity_option", 1))
    if sensitivity_option > 1:
        if verbose:
            logger.p(" Computing sensitivity outputs (autodiff; may take a while)...")
        try:
            sens = compute_sensitivity_outputs(inputs=inputs, plasma=plasma, coil=coil, lambdas=lambdas, exit_code=exit_code)
        except Exception as e:
            if verbose:
                logger.p(f" ERROR: sensitivity output computation failed: {e}")
            raise
        mats.update(sens)

    out_nc = os.path.join(input_dir, "regcoil_out" + base[10:] + ".nc")
    t1_total = _time.perf_counter()
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
            f.write(f"time_build_matrices_s={float(t1_build - t0_build):.6f}\n")
            f.write(f"time_solve_and_diagnostics_s={float(t1_solve - t0_solve):.6f}\n")
            f.write(f"time_total_s={float(t1_total - t0_total):.6f}\n")
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
            logger.p(f" WARNING: could not write summary log: {e}")

    if verbose:
        nlam = int(len(lambdas))
        for j in range(nlam):
            logger.solve_one(
                lam=float(lambdas[j]),
                j=int(j + 1),
                n=int(nlam),
                chi2_B=float(chi2_B[j]),
                chi2_K=float(chi2_K[j]),
                chi2_LB=float(chi2_LB[j]),
                max_B=float(max_B[j]),
                max_K=float(max_K[j]),
                rms_K=float(rms_K[j]),
            )
        logger.complete(sec=float(t1_total - t0_total), out_nc_basename=os.path.basename(out_nc))

    return RunResult(
        input_path=input_path_abs,
        output_nc=os.path.abspath(out_nc),
        output_log=os.path.abspath(out_log),
        exit_code=int(exit_code),
        chosen_idx=int(idx) if idx is not None else None,
    )
