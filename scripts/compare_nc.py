#!/usr/bin/env python3
"""Compare key scalar arrays between REGCOIL Fortran output and regcoil_jax output."""
import sys
import numpy as np

try:
    import netCDF4
except Exception as e:
    raise SystemExit("netCDF4 is required")

KEYS = ["lambda", "chi2_B", "chi2_K", "max_Bnormal", "max_K"]

def read(path):
    ds = netCDF4.Dataset(path, "r")
    out = {}
    for k in KEYS:
        if k in ds.variables:
            out[k] = np.array(ds.variables[k][:])
    ds.close()
    return out

def main(a,b):
    A=read(a); B=read(b)
    for k in KEYS:
        if k not in A or k not in B:
            print(f"{k}: missing in one file")
            continue
        da = np.max(np.abs(A[k]-B[k]))
        rel = da / (np.max(np.abs(A[k])) + 1e-300)
        print(f"{k}: max|Δ|={da:.6e}  rel={rel:.6e}")

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("Usage: compare_nc.py fortran.nc jax.nc")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
