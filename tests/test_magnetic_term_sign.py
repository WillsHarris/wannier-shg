"""Check Chen's unrearranged magnetic SHG term against the two printed signs."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_synthetic_mixed_model import build_model  # noqa: E402
from wann_tb import at_k  # noqa: E402


class MagneticTermSignTest(unittest.TestCase):
    def test_unrearranged_chen_expression_favors_main_text_sign(self):
        lattice, vectors, hamiltonian, position = build_model()
        hamiltonian = hamiltonian.copy()
        hamiltonian[1, 2, 2] += 0.09j
        hamiltonian[2, 2, 2] -= 0.09j
        tb = dict(lat=lattice, nw=hamiltonian.shape[1], R=vectors,
                  deg=np.ones(len(vectors)), H=hamiltonian,
                  r=position.copy(), Rcart=vectors @ lattice)
        photon, width, nk = 0.1, 1e-7, 24
        for zero_position in (False, True):
            if zero_position:
                tb["r"][:] = 0
            totals = np.zeros((2, 3), complex)
            for ix in range(nk):
                for iy in range(nk):
                    q = at_k(tb, (ix / nk, iy / nk, 0.0),
                             with_generalized_derivative=True)
                    energy, dipole, derivative, velocity = (
                        q[key] for key in ("E", "r", "rd", "v")
                    )
                    f = (energy < 0.06).astype(float)
                    fnm = f[:, None] - f[None, :]
                    wmn = energy[None, :] - energy[:, None]
                    wmn = np.where(np.eye(len(energy), dtype=bool), 1.0, wmn)
                    denominator = wmn - (photon + 1j * width)
                    for a in (0, 1):
                        r = dipole[a]
                        d = derivative[a, a]
                        diagonal_v = velocity[a].diagonal().real
                        delta = diagonal_v[None, :] - diagonal_v[:, None]
                        deriv = 2 * r.T * d
                        metric = 2 * r * r.T
                        eq14 = -1j / (4 * photon) * fnm * (
                            deriv / denominator
                            - delta * metric / (2 * photon * denominator))
                        eq16 = -1j / 4 * fnm * (
                            deriv / (wmn * denominator)
                            - delta * metric / (2 * wmn**2 * denominator))
                        appendix = -1j / 4 * fnm * (
                            deriv / (wmn * denominator)
                            + delta * metric / (2 * wmn**2 * denominator))
                        totals[a] += (eq14.sum(), eq16.sum(), appendix.sum())
            totals /= nk**2
            for eq14, eq16, appendix in totals:
                mismatch = abs(eq14 - appendix)
                self.assertGreater(mismatch, 1e-4)
                self.assertLess(abs(eq14 - eq16), 1e-3 * mismatch)


if __name__ == "__main__":
    unittest.main()
