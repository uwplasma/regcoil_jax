from __future__ import annotations

from dataclasses import dataclass


def _fmt_E3(x: float) -> str:
    # Fortran prints like 1.000E-30 / 0.000E+00.
    return f"{float(x):.3E}"


def _fmt_time_s(sec: float) -> str:
    # The Fortran code prints REAL(4) seconds with list-directed formatting.
    # We keep a stable scientific format.
    return f"{float(sec):.8E} sec."


@dataclass
class FortranVerbose:
    enabled: bool = False

    def p(self, msg: str = "") -> None:
        if self.enabled:
            print(msg)

    def header(self) -> None:
        self.p(" This is REGCOIL_JAX,")
        self.p(" a JAX port of REGCOIL (regularized least-squares coil design).")

    def resolution_block(
        self,
        *,
        input_basename: str,
        ntheta_plasma: int,
        ntheta_coil: int,
        nzeta_plasma: int,
        nzeta_coil: int,
        mpol_potential: int,
        ntor_potential: int,
        symmetry_option: int,
    ) -> None:
        self.p(f" Successfully read parameters from regcoil_nml namelist in {input_basename}.")
        self.p(" Resolution parameters:")
        self.p(f"   ntheta_plasma  = {ntheta_plasma:5d}")
        self.p(f"   ntheta_coil    = {ntheta_coil:5d}")
        self.p(f"   nzeta_plasma   = {nzeta_plasma:5d}")
        self.p(f"   nzeta_coil     = {nzeta_coil:5d}")
        self.p(f"   mpol_potential = {mpol_potential:5d}")
        self.p(f"   ntor_potential = {ntor_potential:5d}")
        if symmetry_option == 1:
            self.p(" Symmetry: sin(m*theta - n*zeta) modes only")
        elif symmetry_option == 2:
            self.p(" Symmetry: cos(m*theta - n*zeta) modes only")
        elif symmetry_option == 3:
            self.p(" Symmetry: both sin(m*theta - n*zeta) and cos(m*theta - n*zeta) modes")
        else:
            self.p(f" Symmetry: (unknown symmetry_option={symmetry_option})")

    def lambda_list(self, lambdas: list[float]) -> None:
        self.p(" We will use the following values of the regularization weight lambda:")
        self.p(" " + " ".join(_fmt_E3(x) for x in lambdas))

    def init_surface(self, which: str) -> None:
        self.p(f" Initializing {which} surface.")

    def surface_detail(self, msg: str) -> None:
        self.p(f"   {msg}")

    def surface_area_volume(self, *, which: str, area: float, volume: float) -> None:
        title = "Plasma" if which == "plasma" else "Coil"
        self.p(f" {title} surface area: {_fmt_E3(area)} m^2. Volume: {_fmt_E3(volume)} m^3.")

    def done_init_surface(self, which: str, sec: float) -> None:
        self.p(f" Done initializing {which} surface. Took {_fmt_time_s(sec)}")

    def bnorm_message(self, msg: str) -> None:
        self.p(f" {msg}")

    def phase(self, msg: str) -> None:
        self.p(f" {msg}")

    def phase_done(self, sec: float) -> None:
        self.p(f" Done. Took {_fmt_time_s(sec)}")

    def phase_timing(self, label: str, sec: float) -> None:
        self.p(f" {label}: {_fmt_time_s(sec)}")

    def solve_one(
        self,
        *,
        lam: float,
        j: int,
        n: int,
        chi2_B: float,
        chi2_K: float,
        chi2_LB: float,
        max_B: float,
        max_K: float,
        rms_K: float,
    ) -> None:
        self.p(f" Solving system for lambda= {_fmt_E3(lam)} ({j:3d} of {n:3d})")
        # Keep these lines for parity with Fortran output structure; timings are not separated in the JAX solve path.
        self.p("   Additions:    0.00000000      sec.")
        self.p("   Dense solve (jax.numpy.linalg.solve):    0.00000000      sec.")
        self.p("   Diagnostics:    0.00000000      sec.")
        self.p(f"   chi2_B: {_fmt_E3(chi2_B)},  chi2_K: {_fmt_E3(chi2_K)},  chi2_Laplace_Beltrami: {_fmt_E3(chi2_LB)}")
        self.p(f"   max(B_n): {_fmt_E3(max_B)},  max(K): {_fmt_E3(max_K)},  rms K: {_fmt_E3(rms_K)}")

    def complete(self, *, sec: float, out_nc_basename: str) -> None:
        self.p(f" REGCOIL_JAX complete. Total time= {_fmt_time_s(sec)}")
        self.p(f" You can plot results in {out_nc_basename} (netCDF).")

