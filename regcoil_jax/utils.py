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

def resolve_existing_path(path: str) -> str:
    """Resolve an input path that may be given as a basename.

    This helper exists to make it convenient to run example inputs without typing
    the full `examples/<tier>/...` path, while still ensuring outputs are written
    next to the *actual* input file.
    """
    if os.path.exists(path):
        return path

    base = os.path.basename(path)
    # Search common locations inside the repo.
    candidates = [
        os.path.join("examples", base),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", base),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    # Also search recursively under examples/ (tiered examples layout).
    search_roots = [
        os.path.join("examples"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples"),
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if base in filenames:
                return os.path.join(dirpath, base)

    return path

def parse_namelist(path: str, namelist_name: str = "regcoil_nml") -> Dict[str, Any]:
    path = resolve_existing_path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    # Extract &name ... / in a way that is robust to paths containing '/' inside strings.
    # REGCOIL inputs conventionally terminate the namelist with a standalone '/' line.
    start_re = re.compile(r"^\s*&\s*" + re.escape(namelist_name) + r"\b", re.IGNORECASE)
    end_re = re.compile(r"^\s*/\s*$", re.IGNORECASE)
    end2_re = re.compile(r"^\s*&\s*end\s*$", re.IGNORECASE)

    in_nml = False
    block_lines: list[str] = []
    for raw in lines:
        if not in_nml:
            if start_re.search(raw):
                in_nml = True
            continue

        # Only treat '/' as the terminator if it is the first non-comment token on the line.
        s = strip_fortran_comment(raw).strip()
        if end_re.match(s) or end2_re.match(s):
            break
        block_lines.append(raw)

    if not in_nml:
        raise ValueError(f"Did not find namelist &{namelist_name} in {path}")

    block = "\n".join(block_lines)
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
