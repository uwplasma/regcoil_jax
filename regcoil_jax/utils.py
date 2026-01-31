from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

def fortran_float(s: str) -> float:
    # Accept Fortran 'd' exponent.
    return float(s.replace("D", "E").replace("d", "e"))

def parse_fortran_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in (".t.", "t", "true", ".true."):
        return True
    if s in (".f.", "f", "false", ".false."):
        return False
    raise ValueError(f"Not a Fortran boolean: {s}")

def strip_fortran_comment(line: str) -> str:
    # '!' starts a comment in the regcoil inputs.
    i = line.find("!")
    return line if i < 0 else line[:i]

def split_assignments(block: str) -> list[str]:
    # Handle multi-line namelist with commas/newlines.
    # We split on newlines and also keep '=' lines.
    lines = []
    for raw in block.splitlines():
        s = strip_fortran_comment(raw).strip()
        if not s:
            continue
        lines.append(s)
    # Join lines that are continued via trailing commas.
    joined = []
    buf = ""
    for s in lines:
        if buf:
            buf += " " + s
        else:
            buf = s
        if buf.endswith(","):
            continue
        joined.append(buf)
        buf = ""
    if buf:
        joined.append(buf)
    return joined

def parse_value(raw: str) -> Any:
    raw = raw.strip()
    # Strings
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    # Booleans
    if raw.lower() in (".t.", ".f.", "t", "f", ".true.", ".false.", "true", "false"):
        return parse_fortran_bool(raw)
    # Arrays in (/ ... /)
    if raw.startswith("(/") and raw.endswith("/)"):
        inner = raw[2:-2].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return [parse_value(p) for p in parts]
    # Numeric (int/float)
    if re.match(r"^[+-]?\d+$", raw):
        return int(raw)
    # Fortran float
    if re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([deDE][+-]?\d+)?$", raw):
        return fortran_float(raw)
    # Some inputs use 1e+200 etc:
    try:
        return fortran_float(raw)
    except Exception as e:
        raise ValueError(f"Could not parse value: {raw}") from e

def parse_namelist(path: str, namelist_name: str = "regcoil_nml") -> Dict[str, Any]:
    # Allow passing just a basename when running from repo root; search common locations.
    if not os.path.exists(path):
        base = os.path.basename(path)
        candidates = [
            os.path.join('examples', base),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples', base),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                path = cand
                break
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Extract &name ... /
    pattern = re.compile(r"&\s*" + re.escape(namelist_name) + r"\b(.*?)/\s*", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Did not find namelist &{namelist_name} in {path}")
    block = m.group(1)
    assigns = split_assignments(block)
    out: Dict[str, Any] = {}
    for a in assigns:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Remove trailing commas.
        if v.endswith(","):
            v = v[:-1].strip()
        out[k.lower()] = parse_value(v)
    return out
