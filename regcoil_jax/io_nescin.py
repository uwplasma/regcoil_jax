from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class NescinSurface:
    nfp: int
    lasym: bool
    xm: np.ndarray
    xn: np.ndarray
    rmnc: np.ndarray
    zmns: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray

def read_nescin(path: str) -> NescinSurface:
    # Minimal NESCOIL nescin reader sufficient for regcoil examples.
    # Format: header lines then mnmax and nfp etc; robust-ish parsing.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # Find the line containing mnmax (often second non-empty line)
    # We'll scan for first line that has 2+ ints.
    mnmax = None
    idx = None
    for i, ln in enumerate(lines[:20]):
        parts = ln.split()
        if len(parts) >= 2:
            try:
                a = int(parts[0]); b = int(parts[1])
                # heuristic: mnmax is usually large-ish, nfp small-ish
                if a > 0 and b > 0:
                    mnmax = a; nfp = b
                    idx = i
                    break
            except:
                continue
    if mnmax is None or idx is None:
        raise ValueError(f"Could not parse mnmax/nfp from nescin file: {path}")
    # Next, skip to mode table: find first line that starts with 2 ints and 4 floats after idx
    table_start = None
    for i in range(idx+1, len(lines)):
        parts = lines[i].split()
        if len(parts) >= 6:
            try:
                int(parts[0]); int(parts[1]); float(parts[2]); float(parts[3]); float(parts[4]); float(parts[5])
                table_start = i
                break
            except:
                continue
    if table_start is None:
        raise ValueError(f"Could not find mode table in nescin file: {path}")
    xm=[]; xn=[]; rmnc=[]; zmns=[]; rmns=[]; zmnc=[]
    for j in range(table_start, min(table_start+mnmax, len(lines))):
        parts = lines[j].split()
        xm.append(int(parts[0])); xn.append(int(parts[1]))
        rmnc.append(float(parts[2])); zmns.append(float(parts[3]))
        rmns.append(float(parts[4])); zmnc.append(float(parts[5]))
    xm=np.array(xm, dtype=int); xn=np.array(xn, dtype=int)
    rmnc=np.array(rmnc, dtype=float); zmns=np.array(zmns, dtype=float)
    rmns=np.array(rmns, dtype=float); zmnc=np.array(zmnc, dtype=float)
    # Many nescin files include asym arrays even if 0.
    lasym = bool(np.any(np.abs(rmns) > 0) or np.any(np.abs(zmnc) > 0))
    return NescinSurface(nfp=nfp, lasym=lasym, xm=xm, xn=xn, rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc)
