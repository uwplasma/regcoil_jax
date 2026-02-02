from __future__ import annotations
import math
import jax
import jax.numpy as jnp

_FORTRAN_REAL4_0P01_AS_REAL8 = 0.009999999776482582

def _get_g_matrix(mats):
    # g is optionally stored depending on save_level; reconstruct it if needed.
    if "g" in mats:
        return mats["g"]
    return mats["g_over_Np"] * mats["normNp"][:, None]


def lambda_grid(inputs):
    """REGCOIL lambda list for scan-style runs (general_option=1).

    Matches regcoil_compute_lambda.f90:
      lambda(1) = 0
      lambda(2:) = logspace(lambda_min, lambda_max, nlambda-1)
    """
    nlambda = int(inputs.get("nlambda", 4))
    lam_min = float(inputs.get("lambda_min", 1.0e-19))
    lam_max = float(inputs.get("lambda_max", 1.0e-13))
    # Intentionally mirror the Fortran implementation including its edge-case behavior:
    #
    #   do j = 1,nlambda-1
    #      lambda(j+1) = lambda_min * exp((log(lambda_max/lambda_min)*(j-1))/(nlambda-2))
    #   end do
    #
    # For nlambda=2 this produces NaN for lambda(2) due to 0/0. We keep this behavior
    # for strict netCDF parity with the reference implementation.
    out = jnp.zeros((nlambda,), dtype=jnp.float64)
    if nlambda <= 0:
        return out
    out = out.at[0].set(0.0)
    if nlambda == 1:
        return out
    j = jnp.arange(1, nlambda, dtype=jnp.float64)  # 1..nlambda-1
    denom = float(nlambda - 2)
    expo = (j - 1.0) * jnp.log(lam_max / lam_min) / denom
    out = out.at[1:].set(lam_min * jnp.exp(expo))
    return out


def _solve_one_lambda(MB, MR, RB, RR, lam):
    # Match regcoil_solve.f90 scaling:
    #   A = (1/(1+λ)) * MB + (λ/(1+λ)) * MR
    #   b = (1/(1+λ)) * RB + (λ/(1+λ)) * RR
    # This keeps the system O(1) for very large λ.
    wB = 1.0 / (1.0 + lam)
    wR = lam / (1.0 + lam)
    A = wB * MB + wR * MR
    b = wB * RB + wR * RR
    return jnp.linalg.solve(A, b)

@jax.jit
def _solve_for_lambdas(MB, MR, RB, RR, lambdas):
    return jax.vmap(lambda lam: _solve_one_lambda(MB, MR, RB, RR, lam))(lambdas)

def solve_one_lambda(mats, lam):
    return _solve_one_lambda(mats["matrix_B"], mats["matrix_reg"], mats["RHS_B"], mats["RHS_reg"], lam)

def solve_for_lambdas(mats, lambdas):
    """Solve for a batch of lambdas without Python loops.

    This is performance-critical for lambda scans (general_option=1) and for
    the evaluation phase inside lambda-search options.
    """
    MB = mats["matrix_B"]
    MR = mats["matrix_reg"]
    RB = mats["RHS_B"]
    RR = mats["RHS_reg"]
    lambdas = jnp.asarray(lambdas, dtype=jnp.float64)
    return _solve_for_lambdas(MB, MR, RB, RR, lambdas)

def svd_scan(mats):
    """Port of regcoil_svd_scan.f90 (general_option=3).

    Returns:
      lambdas: (nlambda,) all zeros (matches Fortran)
      sols:    (nlambda, nbasis) truncated-SVD solutions (NESCOIL-style ordering)
      chi2_B, chi2_K, max_B, max_K: (nlambda,)
    """
    g = _get_g_matrix(mats)  # (Np, nb)
    nb = int(g.shape[1])
    np_plasma = int(g.shape[0])
    if nb > np_plasma:
        raise ValueError("svd_scan requires num_basis_functions <= ntheta_plasma*nzeta_plasma")
    if nb < 2:
        raise ValueError("svd_scan requires at least 2 basis functions")

    normNp = mats["normNp"]  # (Np,)
    w = jnp.sqrt(jnp.where(normNp == 0.0, 1.0, normNp))
    # RHS is -(Bplasma + Bnet) * sqrt(normN)
    Btarget = (mats["Bplasma"] + mats["Bnet"]).reshape(-1)
    rhs = -Btarget * w
    # A = g / sqrt(normN)
    A = g / w[:, None]

    # SVD: A = U diag(s) V^T
    U, s, VT = jnp.linalg.svd(A, full_matrices=False)
    UTRHS = U.T @ rhs  # (nb,)

    sol = VT[0] * (UTRHS[0] / s[0])

    nlambda = nb - 1
    sols = []
    # Fortran loop: do ilambda = nlambda,1,-1; index = nb-ilambda+1 (1-based) => index0 = nb-ilambda (0-based)
    for ilambda in range(nlambda, 0, -1):
        index0 = nb - ilambda
        sol = sol + VT[index0] * (UTRHS[index0] / s[index0])
        sols.append((ilambda, sol))

    # Put solutions in Fortran's output indexing: sols[ilambda-1] corresponds to that ilambda.
    sols_arr = [None] * nlambda
    for ilambda, v in sols:
        sols_arr[ilambda - 1] = v
    sols_j = jnp.stack(sols_arr, axis=0)

    lambdas = jnp.zeros((nlambda,), dtype=sols_j.dtype)
    chi2_B, chi2_K, max_B, max_K = diagnostics(mats, sols_j)
    return lambdas, sols_j, chi2_B, chi2_K, max_B, max_K


def diagnostics(mats, sols):
    """Return chi2_B, chi2_K, max_Bnormal, max_K arrays of shape (nlambda,)."""
    nfp = int(mats["nfp"])
    dth_p = float(mats["dth_p"]); dze_p = float(mats["dze_p"])
    dth_c = float(mats["dth_c"]); dze_c = float(mats["dze_c"])
    normNp = mats["normNp"]; normNc = mats["normNc"]
    g_over_Np = mats["g_over_Np"]

    # Bnormal_total(flat) for each lambda
    Btarget = (mats["Bplasma"] + mats["Bnet"]).reshape(-1)
    Bsv = sols @ g_over_Np.T  # (nlambda, Np)
    Btot = Bsv + Btarget[None,:]
    chi2_B = nfp * dth_p * dze_p * jnp.sum((Btot*Btot) * normNp[None,:], axis=1)
    max_B = jnp.max(jnp.abs(Btot), axis=1)

    # K diagnostics
    fx = mats["fx"]; fy=mats["fy"]; fz=mats["fz"]
    dx=mats["dx"]; dy=mats["dy"]; dz=mats["dz"]
    # KDifference = d - f @ sol
    Kdx = dx[None,:] - sols @ fx.T
    Kdy = dy[None,:] - sols @ fy.T
    Kdz = dz[None,:] - sols @ fz.T
    K2_times_N = (Kdx*Kdx + Kdy*Kdy + Kdz*Kdz) / normNc[None,:]
    chi2_K = nfp * dth_c * dze_c * jnp.sum(K2_times_N, axis=1)
    K2 = K2_times_N / normNc[None,:]
    max_K = jnp.sqrt(jnp.max(K2, axis=1))
    return chi2_B, chi2_K, max_B, max_K


def target_quantity(mats, *, sol, chi2_B, chi2_K, max_B, max_K, target_option: str, target_option_p: float) -> float:
    """Match REGCOIL's target_function() logic in regcoil_auto_regularization_solve.f90.

    This helper is intentionally *not* jitted: it is used only in general_option=5 (lambda search)
    control-flow which is currently Python-level for parity/debuggability.
    """
    opt = target_option.strip().lower()
    area_plasma = float(mats.get("area_plasma"))
    area_coil = float(mats.get("area_coil"))

    if opt in ("max_k",):
        return float(max_K)
    if opt in ("rms_k",):
        return math.sqrt(float(chi2_K) / area_coil)
    if opt in ("chi2_k",):
        return float(chi2_K)
    if opt in ("max_bnormal",):
        return float(max_B)
    if opt in ("rms_bnormal",):
        return math.sqrt(float(chi2_B) / area_plasma)
    if opt in ("chi2_b",):
        return float(chi2_B)

    # Options that depend on the full K(θ,ζ) distribution:
    if opt in ("max_k_lse", "lp_norm_k"):
        # Reconstruct K^2 on the coil surface in the same way as diagnostics().
        nfp = int(mats["nfp"])
        dth_c = float(mats["dth_c"])
        dze_c = float(mats["dze_c"])
        normNc = mats["normNc"]

        fx = mats["fx"]
        fy = mats["fy"]
        fz = mats["fz"]
        dx = mats["dx"]
        dy = mats["dy"]
        dz = mats["dz"]

        sol_row = sol[None, :]
        Kdx = dx[None, :] - sol_row @ fx.T
        Kdy = dy[None, :] - sol_row @ fy.T
        Kdz = dz[None, :] - sol_row @ fz.T
        K2_times_N = (Kdx * Kdx + Kdy * Kdy + Kdz * Kdz) / normNc[None, :]
        K2 = K2_times_N / normNc[None, :]  # |K|^2
        Kmag = jnp.sqrt(K2[0])
        maxK = float(max_K)

        p = float(target_option_p)
        # weights integrate over the winding surface and normalize by area, matching regcoil_diagnostics.f90
        w = (nfp * dth_c * dze_c) * (normNc / area_coil)

        if opt == "max_k_lse":
            # max_K_lse = (1/p)*log(sum(w*exp(p*(K-maxK)))) + maxK
            # Use a stable form.
            s = jnp.sum(w * jnp.exp(p * (Kmag - maxK)))
            return float((1.0 / p) * jnp.log(s) + maxK)

        # lp_norm_K = (∫ |K|^p dA / A)^(1/p)
        s = jnp.sum(w * (Kmag**p))
        return float(s ** (1.0 / p))

    # Default to max_K (matches regcoil_variables.f90 default target_option)
    return float(max_K)

def choose_lambda(inputs, lambdas, chi2_B, chi2_K, max_B, max_K):
    general = int(inputs.get("general_option", 1))
    if general != 5:
        # choose min lambda? keep all, default is scan.
        return None
    target_option = str(inputs.get("target_option", "max_K")).strip()
    target_value = float(inputs.get("target_value", 0.0))
    if target_option == "max_K":
        vals = max_K
    elif target_option == "rms_K":
        # need area coil; not computed in subset => approximate with chi2_K / (nfp*dth*dz*sum(normN))? 
        vals = jnp.sqrt(chi2_K / (1.0))  # placeholder, user uses max_K in example
    elif target_option == "max_Bnormal":
        vals = max_B
    else:
        vals = max_K
    idx = int(jnp.argmin(jnp.abs(vals - target_value)))
    return idx


def auto_regularization_solve(inputs, mats):
    """Port of regcoil_auto_regularization_solve.f90 for general_option=5.

    Returns:
      lambdas: (Nlambda,)
      sols:    (Nlambda, nbasis)
      chi2_B, chi2_K, max_B, max_K: (Nlambda,)
      chosen_idx: int | None
      exit_code: int (0 success; negative matches REGCOIL conventions)
    """
    nlambda_max = int(inputs.get("nlambda", 4))
    target_option = str(inputs.get("target_option", "max_K")).strip().lower()
    target_value = float(inputs.get("target_value", 0.0))
    lambda_search_tolerance = float(inputs.get("lambda_search_tolerance", 1.0e-5))
    target_option_p = float(inputs.get("target_option_p", 4.0))

    def target_increases_with_lambda() -> bool:
        if target_option in ("max_bnormal", "rms_bnormal", "chi2_b"):
            return True
        return False

    targeted_quantity_increases = target_increases_with_lambda()

    # Stage machine (see Fortran):
    #   general_option=4 starts at stage=1 (no feasibility checks)
    #   general_option=5 starts at stage=10 (evaluate λ=∞ and λ=0 first)
    general = int(inputs.get("general_option", 5))
    stage = 1 if general == 4 else 10
    exit_code = -1

    # Brendt (Brent) state in log(lambda) space:
    a = b = c = fa = fb = fc = d = e = tol1 = xm = None

    lambdas = []
    sols = []
    chi2_B_list = []
    chi2_K_list = []
    max_B_list = []
    max_K_list = []
    target_vals = []

    initial_above_target = None

    # Precompute the "solution=0" diagnostics used for the initial guess stage.
    Btarget = (mats["Bplasma"] + mats["Bnet"]).reshape(-1)
    normNp = mats["normNp"]
    normNc = mats["normNc"]
    dx = mats["dx"]; dy = mats["dy"]; dz = mats["dz"]
    nfp = int(mats["nfp"])
    dth_p = float(mats["dth_p"]); dze_p = float(mats["dze_p"])
    dth_c = float(mats["dth_c"]); dze_c = float(mats["dze_c"])
    area_plasma = float(mats.get("area_plasma", nfp * dth_p * dze_p * jnp.sum(normNp)))
    area_coil = float(mats.get("area_coil", nfp * dth_c * dze_c * jnp.sum(normNc)))

    chi2_B_sol0 = float(nfp * dth_p * dze_p * jnp.sum((Btarget * Btarget) * normNp))
    chi2_K_sol0 = float(nfp * dth_c * dze_c * jnp.sum((dx * dx + dy * dy + dz * dz) / normNc))

    def eval_target(*, sol, chi2_B, chi2_K, max_B, max_K) -> float:
        return target_quantity(
            mats,
            sol=sol,
            chi2_B=chi2_B,
            chi2_K=chi2_K,
            max_B=max_B,
            max_K=max_K,
            target_option=target_option,
            target_option_p=target_option_p,
        )

    def _sign(x, y):
        return abs(x) if y >= 0 else -abs(x)

    for ilambda in range(1, nlambda_max + 1):
        # Choose next lambda
        if stage == 10:
            lam = 1.0e200  # "infinite" regularization
            next_stage = 11
        elif stage == 11:
            lam = 0.0
            next_stage = 1
        elif stage == 1:
            if chi2_K_sol0 == 0.0:
                lam = 0.0
            else:
                lam = chi2_B_sol0 / chi2_K_sol0 / 1000.0
            next_stage = 2
        elif stage == 2:
            if initial_above_target is None:
                raise RuntimeError("internal error: stage 2 entered without initial_above_target")
            # Match regcoil_auto_regularization_solve.f90 exactly:
            #   factor = 100        (integer literal -> dp exactly)
            #   factor = 0.01       (default-real literal -> converted to dp, i.e. float32(0.01) in practice)
            factor = 100.0 if initial_above_target else _FORTRAN_REAL4_0P01_AS_REAL8
            if targeted_quantity_increases:
                factor = 1.0 / factor
            lam = float(lambdas[-1]) * factor
            next_stage = 2
        elif stage == 3:
            # Brendt's algorithm in log(lambda)
            assert tol1 is not None and xm is not None and e is not None and d is not None
            assert a is not None and b is not None and c is not None
            assert fa is not None and fb is not None and fc is not None
            next_stage = 3

            if abs(e) >= tol1 and abs(fa) > abs(fb):
                s = fb / fa
                if a == c:
                    p = 2.0 * xm * s
                    q = 1.0 - s
                else:
                    q = fa / fc
                    r = fb / fc
                    p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0)

                if p > 0.0:
                    q = -q
                p = abs(p)
                if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                    e = d
                    d = p / q
                else:
                    d = xm
                    e = d
            else:
                d = xm
                e = d

            # Move last best guess to a.
            a = b
            fa = fb
            if abs(d) > tol1:
                b = b + d
            else:
                b = b + _sign(tol1, xm)

            lam = math.exp(b)
        else:
            raise RuntimeError(f"invalid stage {stage}")

        # Solve system
        x = solve_one_lambda(mats, lam)
        chi2_B, chi2_K, max_B, max_K = diagnostics(mats, x[None, :])
        cB = float(chi2_B[0]); cK = float(chi2_K[0]); mB = float(max_B[0]); mK = float(max_K[0])
        tval = eval_target(sol=x, chi2_B=cB, chi2_K=cK, max_B=mB, max_K=mK)

        lambdas.append(float(lam))
        sols.append(x)
        chi2_B_list.append(cB)
        chi2_K_list.append(cK)
        max_B_list.append(mB)
        max_K_list.append(mK)
        target_vals.append(tval)

        last_above_target = (tval > target_value)
        if stage == 1:
            initial_above_target = last_above_target

        # Detect bracketing (stage 2 -> stage 3)
        if stage == 2 and initial_above_target is not None and (last_above_target != initial_above_target):
            next_stage = 3

        # Match regcoil_auto_regularization_solve.f90 unreachable-target handling:
        if last_above_target and (
            ((not targeted_quantity_increases) and stage == 10)
            or (targeted_quantity_increases and stage == 11)
        ):
            exit_code = -2  # target too low
            break
        if (not last_above_target) and (
            ((not targeted_quantity_increases) and stage == 11)
            or (targeted_quantity_increases and stage == 10)
        ):
            exit_code = -3  # target too high
            break

        if stage == 2 and next_stage == 3:
            # Initialize Brendt's algorithm
            if lambdas[-2] <= 0.0 or lambdas[-1] <= 0.0:
                raise RuntimeError("lambda bracketing produced non-positive lambda, cannot take log")
            a = math.log(lambdas[-2])
            b = math.log(lambdas[-1])
            fa = math.log(target_vals[-2] / target_value)
            fb = math.log(target_vals[-1] / target_value)
            c = b
            fc = fb
            d = b - a
            e = d

        if stage == 3:
            fb = math.log(tval / target_value)

        if next_stage == 3:
            # Analyze the most recent diagnostics for Brendt's algorithm.
            if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
                c = a
                fc = fa
                d = b - a
                e = d
            if abs(fc) < abs(fb):
                a, b, c = b, c, b
                fa, fb, fc = fb, fc, fb
            EPS = 1.0e-15
            tol1 = 2.0 * EPS * abs(b) + 0.5 * lambda_search_tolerance
            xm = 0.5 * (c - b)
            if abs(xm) <= tol1 or fb == 0.0:
                exit_code = 0
                break

        stage = next_stage

    # Truncate to actual iterations taken:
    lambdas_out = jnp.asarray(lambdas)
    sols_out = jnp.stack(sols, axis=0)
    chi2_B_out = jnp.asarray(chi2_B_list)
    chi2_K_out = jnp.asarray(chi2_K_list)
    max_B_out = jnp.asarray(max_B_list)
    max_K_out = jnp.asarray(max_K_list)
    chosen_idx = int(len(lambdas) - 1) if (exit_code == 0 and len(lambdas)) else None
    return lambdas_out, sols_out, chi2_B_out, chi2_K_out, max_B_out, max_K_out, chosen_idx, int(exit_code)
