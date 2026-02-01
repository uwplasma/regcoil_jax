# 2_intermediate

JAX-first examples that highlight differentiability, gradients, and JIT.

These scripts are not necessarily “full REGCOIL solves”; some focus on individual kernels
to keep the objective functions simple and fast to run.

Included scripts:
- `jax_optimize_coil_minor_radius.py`: optimize `a_coil` for the `Bnormal_from_net_coil_currents` kernel (fastest).
- `jax_optimize_coil_radius_full_regcoil.py`: optimize `a_coil` while differentiating through matrix assembly + linear solve.
