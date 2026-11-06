"""Write a small pair of synthetic Wannier models with opposite magnetization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_synthetic_mixed_model import build_model, write_tb


def main():
    lattice, vectors, hamiltonian, position = build_model()
    destination = Path(__file__).resolve().parent / "data"
    for name, sign in (("plus", 1), ("minus", -1)):
        magnetic = hamiltonian.copy()
        magnetic[1, 2, 2] += sign * 0.09j
        magnetic[2, 2, 2] -= sign * 0.09j
        path = destination / f"{name}_tb.dat"
        write_tb(path, lattice, vectors, magnetic, position)
        print(path)


if __name__ == "__main__":
    main()
