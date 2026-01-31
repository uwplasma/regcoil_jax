import sys
from regcoil_jax.utils import parse_namelist

if __name__ == "__main__":
    cfg = parse_namelist(sys.argv[1], "regcoil_nml")
    for k in sorted(cfg.keys()):
        print(k, "=", cfg[k])
