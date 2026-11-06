#!/usr/bin/env python3
"""Write a small deterministic Wannier model for the HopTB mixed-SHG test."""

from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs" / "spectra" / "synthetic_mixed_tb.dat"


def _hermitian(matrix):
    return (matrix + matrix.conj().T) / 2.0


def build_model():
    lattice = np.diag([3.2, 4.1, 18.0])
    r_vectors = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        dtype=int,
    )

    onsite = np.diag([-1.65, -0.52, 0.68, 1.82]).astype(complex)
    onsite += _hermitian(
        np.array(
            [
                [0, 0.08, -0.03, 0.02],
                [0, 0, 0.06, -0.04],
                [0, 0, 0, 0.05],
                [0, 0, 0, 0],
            ],
            dtype=complex,
        )
    )
    hop_x = np.array(
        [
            [0.12, 0.05, -0.02, 0.01],
            [-0.035, -0.08, 0.045, 0.025],
            [0.018, -0.03, 0.09, -0.055],
            [0.012, 0.022, 0.04, -0.06],
        ],
        dtype=complex,
    )
    hop_y = np.array(
        [
            [-0.07, 0.028, 0.035, -0.018],
            [0.052, 0.10, -0.026, 0.038],
            [-0.012, 0.044, -0.075, 0.031],
            [0.027, -0.016, -0.048, 0.055],
        ],
        dtype=complex,
    )
    hamiltonian = np.array(
        [onsite, hop_x, hop_x.conj().T, hop_y, hop_y.conj().T]
    )

    position = np.zeros((len(r_vectors), 4, 4, 3), dtype=complex)
    position[0, :, :, 0] = _hermitian(
        np.array(
            [
                [0.18, 0.07, -0.025, 0.012],
                [0, 1.03, 0.055, -0.032],
                [0, 0, 2.08, 0.046],
                [0, 0, 0, 2.74],
            ],
            dtype=complex,
        )
    )
    position[0, :, :, 1] = _hermitian(
        np.array(
            [
                [0.24, -0.045, 0.038, 0.021],
                [0, 1.36, 0.063, -0.027],
                [0, 0, 2.47, 0.052],
                [0, 0, 0, 3.31],
            ],
            dtype=complex,
        )
    )
    position[0, :, :, 2] = _hermitian(
        np.array(
            [
                [0.0, 0.015, -0.008, 0.004],
                [0, 0.0, 0.011, -0.006],
                [0, 0, 0.0, 0.009],
                [0, 0, 0, 0.0],
            ],
            dtype=complex,
        )
    )

    pos_x = np.array(
        [
            [0.012, -0.008, 0.004, 0.002],
            [0.006, -0.010, 0.007, -0.003],
            [-0.005, 0.002, 0.009, 0.006],
            [0.003, -0.004, -0.007, 0.005],
        ],
        dtype=complex,
    )
    pos_y = np.array(
        [
            [-0.007, 0.005, 0.003, -0.004],
            [-0.002, 0.011, -0.006, 0.008],
            [0.004, 0.007, -0.005, -0.003],
            [0.006, -0.002, 0.005, 0.009],
        ],
        dtype=complex,
    )
    for axis in range(3):
        position[1, :, :, axis] = (axis + 1) * pos_x
        position[2, :, :, axis] = position[1, :, :, axis].conj().T
        position[3, :, :, axis] = (axis + 1) * pos_y
        position[4, :, :, axis] = position[3, :, :, axis].conj().T

    return lattice, r_vectors, hamiltonian, position


def write_tb(path, lattice, r_vectors, hamiltonian, position):
    path.parent.mkdir(parents=True, exist_ok=True)
    norbits = hamiltonian.shape[1]
    with path.open("w") as handle:
        handle.write("Synthetic four-orbital mixed-SHG validation model\n")
        for vector in lattice:
            handle.write(" ".join(f"{value:.16e}" for value in vector) + "\n")
        handle.write(f"{norbits}\n{len(r_vectors)}\n")
        handle.write(" ".join("1" for _ in r_vectors) + "\n\n")

        # HopTB's reader consumes matrix entries in (column, row) order and
        # ignores the printed indices. Printing n,m here makes both readers
        # reconstruct the same matrix without changing either implementation.
        for vector, matrix in zip(r_vectors, hamiltonian):
            handle.write(" ".join(str(value) for value in vector) + "\n")
            for column in range(norbits):
                for row in range(norbits):
                    value = matrix[row, column]
                    handle.write(
                        f"{row + 1:5d}{column + 1:5d} "
                        f"{value.real: .16e} {value.imag: .16e}\n"
                    )
            handle.write("\n")

        for vector, matrix in zip(r_vectors, position):
            handle.write(" ".join(str(value) for value in vector) + "\n")
            for column in range(norbits):
                for row in range(norbits):
                    values = matrix[row, column]
                    fields = " ".join(
                        f"{value.real: .16e} {value.imag: .16e}"
                        for value in values
                    )
                    handle.write(f"{row + 1:5d}{column + 1:5d} {fields}\n")
            handle.write("\n")


def main():
    lattice, r_vectors, hamiltonian, position = build_model()
    np.testing.assert_allclose(hamiltonian[0], hamiltonian[0].conj().T)
    np.testing.assert_allclose(hamiltonian[2], hamiltonian[1].conj().T)
    np.testing.assert_allclose(hamiltonian[4], hamiltonian[3].conj().T)
    write_tb(OUTPUT, lattice, r_vectors, hamiltonian, position)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
