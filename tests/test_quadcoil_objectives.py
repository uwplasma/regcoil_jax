from __future__ import annotations

import numpy as np


def test_quadcoil_metrics_scaling():
    import jax
    import jax.numpy as jnp

    from regcoil_jax.build_matrices_jax import build_matrices
    from regcoil_jax.solve_jax import solve_one_lambda
    from regcoil_jax.surfaces import plasma_surface_from_inputs, coil_surface_from_inputs
    from regcoil_jax.quadcoil_objectives import quadcoil_metrics_from_mats_and_solution

    jax.config.update("jax_enable_x64", True)

    inputs = dict(
        geometry_option_plasma=1,
        geometry_option_coil=1,
        ntheta_plasma=18,
        nzeta_plasma=18,
        ntheta_coil=18,
        nzeta_coil=18,
        r0_plasma=10.0,
        a_plasma=0.8,
        r0_coil=10.0,
        a_coil=1.1,
        symmetry_option=1,
        mpol_potential=6,
        ntor_potential=6,
        net_poloidal_current_amperes=1.0,
        net_toroidal_current_amperes=0.0,
        regularization_term_option="chi2_K",
    )
    plasma = plasma_surface_from_inputs(inputs, None)
    coil = coil_surface_from_inputs(inputs, plasma, None)
    mats = build_matrices(inputs, plasma, coil)

    # Use a non-degenerate coefficient vector (a REGCOIL solution can be extremely small for this toy case,
    # which makes spacing estimates dominated by eps/infs).
    rng = np.random.default_rng(0)
    sol = jnp.asarray(rng.normal(size=(int(mats["num_basis_functions"]),)), dtype=jnp.float64)

    m1 = quadcoil_metrics_from_mats_and_solution(mats, sol, coils_per_half_period=2)
    m2 = quadcoil_metrics_from_mats_and_solution(mats, 2.0 * sol, coils_per_half_period=2)

    assert np.isfinite(m1.int_gradphi2)
    assert np.isfinite(m1.coil_spacing_min)
    assert m1.delta_phi > 0
    # Scaling: |∇Φ| scales with coeffs; spacing scales inversely; ∫|∇Φ|^2 scales quadratically.
    assert np.isclose(m2.int_gradphi2, 4.0 * m1.int_gradphi2, rtol=2e-2, atol=0.0)
    assert np.isclose(m2.gradphi_rms, 2.0 * m1.gradphi_rms, rtol=2e-2, atol=0.0)
    assert np.isclose(m2.coil_spacing_rms, 0.5 * m1.coil_spacing_rms, rtol=2e-2, atol=0.0)


def test_gradphi2_regularization_smoke():
    import jax

    from regcoil_jax.build_matrices_jax import build_matrices
    from regcoil_jax.surfaces import plasma_surface_from_inputs, coil_surface_from_inputs

    jax.config.update("jax_enable_x64", True)

    inputs = dict(
        geometry_option_plasma=1,
        geometry_option_coil=1,
        ntheta_plasma=12,
        nzeta_plasma=12,
        ntheta_coil=12,
        nzeta_coil=12,
        r0_plasma=10.0,
        a_plasma=0.8,
        r0_coil=10.0,
        a_coil=1.1,
        symmetry_option=1,
        mpol_potential=5,
        ntor_potential=5,
        net_poloidal_current_amperes=1.0,
        net_toroidal_current_amperes=0.0,
        regularization_term_option="chi2_K",
        gradphi2_weight=1e-6,
    )
    plasma = plasma_surface_from_inputs(inputs, None)
    coil = coil_surface_from_inputs(inputs, plasma, None)
    mats = build_matrices(inputs, plasma, coil)
    assert "matrix_reg" in mats
