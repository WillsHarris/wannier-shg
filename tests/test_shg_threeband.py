import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from shg import (  # noqa: E402
    bulk_chi_to_sheet_si,
    bz_mean_to_bulk_chi_si,
    chi_k,
    chi_mixed_k,
    chi_total_k,
    gauge_invariance_report,
    intrinsic_permutation_residual,
    transform_band_gauge,
)
from wann_tb import at_k  # noqa: E402


def direct_threeband(E, r, f, photon_energy, eta, energy_tol=1e-6):
    chi = np.zeros((3, 3, 3), dtype=complex)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for n in range(len(E)):
                    for m in range(len(E)):
                        for ell in range(len(E)):
                            denominator = (
                                (E[ell] - E[n]) - (E[m] - E[ell])
                            )
                            if abs(denominator) < energy_tol:
                                continue
                            weight = (
                                2 * (f[n] - f[m])
                                / (E[m] - E[n] - 2 * photon_energy - 2j * eta)
                                + (f[m] - f[ell])
                                / (E[m] - E[ell] - photon_energy - 1j * eta)
                                + (f[ell] - f[n])
                                / (E[ell] - E[n] - photon_energy - 1j * eta)
                            )
                            numerator = r[a, n, m] * (
                                r[b, m, ell] * r[c, ell, n]
                                + r[c, m, ell] * r[b, ell, n]
                            )
                            chi[a, b, c] += numerator * weight / (
                                2 * denominator
                            )
    return chi


def gpaw_trs_reduced_threeband(E, r, f, photon_energy, eta):
    """GPAW length-gauge three-band sum, without its physical prefactor."""
    chi = np.zeros((3, 3, 3), dtype=complex)
    frequencies = (photon_energy + 1j * eta, -photon_energy - 1j * eta)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for z in frequencies:
                    for n in range(len(E)):
                        for m in range(n + 1, len(E)):
                            for ell in range(len(E)):
                                E_mn = E[m] - E[n]
                                E_ln = E[ell] - E[n]
                                E_ml = E[m] - E[ell]
                                denominator = E_ln - E_ml
                                if abs(denominator) < 1e-6:
                                    continue
                                numerator = np.real(
                                    r[a, n, m]
                                    * (
                                        r[b, m, ell] * r[c, ell, n]
                                        + r[c, m, ell] * r[b, ell, n]
                                    )
                                )
                                weight = (
                                    2 * (f[n] - f[m]) / (2 * z - E_mn)
                                    - (f[n] - f[ell]) / (z - E_ln)
                                    + (f[m] - f[ell]) / (z - E_ml)
                                )
                                chi[a, b, c] += (
                                    numerator * weight / (2 * denominator)
                                )
    return chi


def direct_mixed(E, r, rd, v, f, photon_energy, eta, energy_tol=1e-6):
    """Literal band-loop transcription of Xiao et al. Appendix-C Eq. (C3)."""
    chi = np.zeros((3, 3, 3), dtype=complex)
    z = photon_energy + 1j * eta
    velocity = np.real(np.diagonal(v, axis1=1, axis2=2))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for n in range(len(E)):
                    for m in range(len(E)):
                        w_mn = E[m] - E[n]
                        if abs(w_mn) < energy_tol:
                            continue
                        delta = velocity[:, m] - velocity[:, n]
                        value = (
                            2
                            * r[a, n, m]
                            * (rd[b, c, m, n] + rd[c, b, m, n])
                            / (w_mn * (w_mn - 2 * z))
                            + (
                                rd[a, c, n, m] * r[b, m, n]
                                + rd[a, b, n, m] * r[c, m, n]
                            )
                            / (w_mn * (w_mn - z))
                            + r[a, n, m]
                            * (
                                r[b, m, n] * delta[c]
                                + r[c, m, n] * delta[b]
                            )
                            / w_mn**2
                            * (1 / (w_mn - z) - 4 / (w_mn - 2 * z))
                            - (
                                rd[b, a, n, m] * r[c, m, n]
                                + rd[c, a, n, m] * r[b, m, n]
                            )
                            / (2 * w_mn * (w_mn - z))
                            - delta[a]
                            * np.real(r[b, n, m] * r[c, m, n])
                            / (2 * w_mn**2 * (w_mn - z))
                        )
                        chi[a, b, c] += 0.5j * (f[n] - f[m]) * value
    return chi


class ThreeBandKernelTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.E = np.array([-1.8, -0.35, 0.8, 2.35])
        raw = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
        self.r = (raw + raw.swapaxes(1, 2).conj()) / 2
        for axis in range(3):
            np.fill_diagonal(self.r[axis], 0.0)
        self.f = np.array([1.0, 1.0, 0.0, 0.0])
        self.q = {
            "E": self.E,
            "r": self.r,
            "w": self.E[:, None] - self.E[None, :],
            "v": np.zeros_like(self.r),
        }
        raw_rd = rng.normal(size=(3, 3, 4, 4)) + 1j * rng.normal(
            size=(3, 3, 4, 4)
        )
        self.rd = (raw_rd + raw_rd.swapaxes(2, 3).conj()) / 2
        for a in range(3):
            for b in range(3):
                np.fill_diagonal(self.rd[a, b], 0.0)
        self.q["rd"] = self.rd
        self.q["v"] = np.array(
            [
                np.diag([-0.7, -0.1, 0.35, 0.9]),
                np.diag([0.2, -0.3, 0.5, -0.4]),
                np.diag([0.05, 0.1, -0.2, 0.3]),
            ],
            dtype=complex,
        )

    def test_matches_direct_band_sum(self):
        expected = direct_threeband(self.E, self.r, self.f, 0.73, 0.05)
        actual = chi_k(self.q, self.f, 0.73, 0.05)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_diagonal_dipoles_do_not_contribute(self):
        q = dict(self.q)
        q["r"] = self.r.copy()
        for axis in range(3):
            np.fill_diagonal(q["r"][axis], 10 + 3j)
        np.testing.assert_allclose(
            chi_k(q, self.f, 0.73, 0.05),
            chi_k(self.q, self.f, 0.73, 0.05),
        )

    def test_reduces_to_gpaw_trs_kernel_up_to_global_sign(self):
        real_r = self.r.real
        q = dict(self.q)
        q["r"] = real_r
        expected = gpaw_trs_reduced_threeband(
            self.E, real_r, self.f, 0.73, 0.05
        )
        # The unresolved physical prefactor contains the electron-charge sign.
        np.testing.assert_allclose(
            chi_k(q, self.f, 0.73, 0.05),
            -expected,
            rtol=2e-13,
            atol=2e-13,
        )

    def test_phase_permutation_and_intrinsic_symmetry(self):
        report = gauge_invariance_report(self.q, self.f, 0.73, 0.05, seed=11)
        self.assertLess(report["phase_residual"], 1e-12)
        self.assertLess(report["permutation_residual"], 1e-12)
        self.assertLess(
            intrinsic_permutation_residual(chi_k(self.q, self.f, 0.73, 0.05)),
            1e-12,
        )

    def test_mixed_matches_appendix_c_band_sum(self):
        expected = direct_mixed(
            self.E,
            self.r,
            self.rd,
            self.q["v"],
            self.f,
            0.73,
            0.05,
        )
        actual = chi_mixed_k(self.q, self.f, 0.73, 0.05)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    def test_mixed_phase_permutation_and_intrinsic_symmetry(self):
        reference = chi_mixed_k(self.q, self.f, 0.73, 0.05)
        phases = np.array([0.2, -0.7, 1.1, 2.4])
        phase_result = chi_mixed_k(
            transform_band_gauge(self.q, phases=phases),
            self.f,
            0.73,
            0.05,
        )
        permutation = np.array([2, 0, 3, 1])
        permutation_result = chi_mixed_k(
            transform_band_gauge(self.q, permutation=permutation),
            self.f[permutation],
            0.73,
            0.05,
        )
        np.testing.assert_allclose(phase_result, reference, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(
            permutation_result, reference, rtol=2e-13, atol=2e-13
        )
        self.assertLess(intrinsic_permutation_residual(reference), 1e-12)

    def test_total_is_interband_plus_mixed(self):
        np.testing.assert_allclose(
            chi_total_k(self.q, self.f, 0.73, 0.05),
            chi_k(self.q, self.f, 0.73, 0.05)
            + chi_mixed_k(self.q, self.f, 0.73, 0.05),
        )

    def test_mixed_requires_generalized_derivative(self):
        q = dict(self.q)
        del q["rd"]
        with self.assertRaisesRegex(ValueError, "generalized_derivative"):
            chi_mixed_k(q, self.f, 0.73, 0.05)

    def test_wannier_transform_returns_only_interband_dipoles(self):
        position = np.zeros((1, 2, 2, 3), dtype=complex)
        position[0, 0, 0] = [4.0, 5.0, 6.0]
        position[0, 1, 1] = [7.0, 8.0, 9.0]
        position[0, 0, 1] = [0.2 + 0.1j, 0.3, -0.4j]
        position[0, 1, 0] = position[0, 0, 1].conj()
        tb = {
            "R": np.zeros((1, 3), dtype=int),
            "deg": np.ones(1),
            "Rcart": np.zeros((1, 3)),
            "H": np.array([np.diag([-0.5, 0.7])], dtype=complex),
            "r": position,
        }
        transformed = at_k(
            tb, np.zeros(3), with_generalized_derivative=True
        )
        for axis in range(3):
            np.testing.assert_allclose(np.diag(transformed["r"][axis]), 0.0)
            np.testing.assert_allclose(
                transformed["r"][axis],
                transformed["r"][axis].conj().T,
            )
        self.assertEqual(transformed["rd"].shape, (3, 3, 2, 2))
        expected_rd = np.zeros_like(transformed["rd"])
        for a in range(3):
            for b in range(3):
                diagonal_b = np.diag(transformed["Abar"][b])
                expected_rd[a, b] = (
                    1j
                    * transformed["Abar"][a]
                    * (diagonal_b[None, :] - diagonal_b[:, None])
                )
                np.fill_diagonal(expected_rd[a, b], 0.0)
        np.testing.assert_allclose(transformed["rd"], expected_rd)

    def test_diagonal_connection_uses_parallel_transport_gauge(self):
        position = np.zeros((1, 2, 2, 3), dtype=complex)
        position[0, 0, 0] = [0.4, 0.7, 0.2]
        position[0, 1, 1] = [1.1, 1.6, 0.5]
        tb = {
            "R": np.zeros((1, 3), dtype=int),
            "deg": np.ones(1),
            "Rcart": np.zeros((1, 3)),
            "H": np.array([np.diag([-0.5, 0.7])], dtype=complex),
            "r": position,
        }
        transformed = at_k(tb, np.zeros(3), with_generalized_derivative=True)
        diagonal = np.arange(2)
        np.testing.assert_allclose(
            transformed["Abar"][:, diagonal, diagonal]
            + 1j * transformed["D"][:, diagonal, diagonal],
            0.0,
        )
        # r is interband-only and remains zero; the diagonal connection is
        # consumed internally when differentiating it.
        np.testing.assert_allclose(transformed["r"], 0.0)

    def test_direct_generalized_derivative_matches_finite_difference(self):
        rng = np.random.default_rng(19)
        nbands = 4
        hopping = 0.12 * (
            rng.normal(size=(nbands, nbands))
            + 1j * rng.normal(size=(nbands, nbands))
        )
        tb = {
            "R": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
            "deg": np.ones(3),
            "Rcart": np.array([[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]),
            "H": np.array(
                [
                    hopping.conj().T,
                    np.diag([-1.7, -0.3, 0.8, 2.2]),
                    hopping,
                ]
            ),
            "r": np.zeros((3, nbands, nbands, 3), dtype=complex),
        }
        k = np.array([0.173, 0.0, 0.0])
        center = at_k(tb, k, with_generalized_derivative=True)

        def aligned_r(point):
            overlap = np.diag(center["U"].conj().T @ point["U"])
            factors = np.exp(-1j * np.angle(overlap))
            return (
                factors.conj()[None, :, None]
                * point["r"]
                * factors[None, None, :]
            )

        step = 1.0e-4
        k_plus = k.copy()
        k_minus = k.copy()
        k_plus[0] += step
        k_minus[0] -= step
        finite_difference = (
            aligned_r(at_k(tb, k_plus)) - aligned_r(at_k(tb, k_minus))
        ) / (2 * step * 2 * np.pi)
        interband = np.abs(center["w"]) > 1.0e-6
        np.testing.assert_allclose(
            center["rd"][0, 0][interband],
            finite_difference[0][interband],
            rtol=2.0e-6,
            atol=2.0e-8,
        )

    def test_bulk_and_sheet_unit_conversion(self):
        volume_A3 = 278.772718201
        height_A = 19.995020429
        bulk_pm_per_V = (
            bz_mean_to_bulk_chi_si(1.0, volume_A3) * 1.0e12
        )
        sheet_nm2_per_V = (
            bulk_chi_to_sheet_si(bulk_pm_per_V * 1.0e-12, height_A)
            * 1.0e18
        )
        self.assertAlmostEqual(float(bulk_pm_per_V), 64.909967864, places=8)
        self.assertAlmostEqual(float(sheet_nm2_per_V), 0.129787613, places=8)

    def test_sheet_conversion_is_vacuum_independent(self):
        area_A2 = 13.942107196
        values = []
        for height_A in (10.0, 20.0, 35.0):
            bulk = bz_mean_to_bulk_chi_si(1.0, area_A2 * height_A)
            values.append(bulk_chi_to_sheet_si(bulk, height_A))
        np.testing.assert_allclose(values, values[0], rtol=1e-14)

    def test_unit_conversion_rejects_invalid_cell_dimensions(self):
        with self.assertRaises(ValueError):
            bz_mean_to_bulk_chi_si(1.0, 0.0)
        with self.assertRaises(ValueError):
            bulk_chi_to_sheet_si(1.0, -1.0)


if __name__ == "__main__":
    unittest.main()
