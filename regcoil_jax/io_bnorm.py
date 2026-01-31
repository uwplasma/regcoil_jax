from __future__ import annotations
import numpy as np

def read_bnorm_modes(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns (m, n, bf) with bf corresponding to sin(m*theta + n*nfp*zeta) convention in REGCOIL.
    m=[]; n=[]; bf=[]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            try:
                mm = int(parts[0]); nn = int(parts[1]); val = float(parts[2])
            except:
                continue
            m.append(mm); n.append(nn); bf.append(val)
    return np.array(m, dtype=int), np.array(n, dtype=int), np.array(bf, dtype=float)
