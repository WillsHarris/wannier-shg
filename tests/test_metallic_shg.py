"""Checks for the independent metallic length-gauge prototype."""

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from metallic_shg import (  # noqa: E402
    band_hessian,
    fermi_data,
    kinetic_chi_from_tb,
    kinetic_chi_k,
)
from generate_synthetic_mixed_model import build_model  # noqa: E402
from shg import chi_total_k, transform_band_gauge  # noqa: E402
from wann_tb import at_k  # noqa: E402


class MetallicShgTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lattice, r_vectors, hamiltonian, position = build_model()
        cls.tb = {
            "lat": lattice,
            "nw": hamiltonian.shape[1],
            "R": r_vectors,
            "deg": np.ones(len(r_vectors)),
            "H": hamiltonian,
            "r": position,
            "Rcart": r_vectors @ lattice,
        }

    def test_band_hessian_against_finite_difference(self):
        kf = np.array([0.213, 0.173, 0.0])
        q = at_k(self.tb, kf, with_generalized_derivative=True)
        hessian = band_hessian(self.tb, kf, q)
        dk = 1e-3  # inverse Angstrom, Cartesian x
        shift = np.array([dk, 0.0, 0.0]) @ self.tb["lat"].T / (2 * np.pi)
        plus = at_k(self.tb, kf + shift)["E"]
        minus = at_k(self.tb, kf - shift)["E"]
        finite_difference = (plus - 2 * q["E"] + minus) / dk**2
        np.testing.assert_allclose(hessian[0, 0], finite_difference, rtol=1e-4, atol=1e-5)

    def test_one_band_drude_matches_bhalla_eq26(self):
        energy = np.array([0.0])
        velocity = np.array([[[1.2]], [[0.0]], [[0.0]]], complex)
        q = {
            "E": energy,
            "r": np.zeros((3, 1, 1), complex),
            "rd": np.zeros((3, 3, 1, 1), complex),
            "v": velocity,
        }
        hessian = np.zeros((3, 3, 1))
        hessian[0, 0, 0] = -0.4
        w, eta = 0.35, 0.07
        full, frozen, fermi = kinetic_chi_k(
            q, hessian, 0.03, w, eta, eta, 0.025,
            components=((0, 0, 0),),
        )
        _, fp, fpp = fermi_data(energy, 0.03, 0.025)
        d2f = fpp[0] * 1.2**2 + fp[0] * (-0.4)
        expected_current = -1.2 * d2f / ((eta - 1j * w) * (eta - 2j * w))
        expected = -expected_current / (-2j * w)
        self.assertAlmostEqual(abs(full[0, 0, 0] - expected), 0.0, places=11)
        self.assertAlmostEqual(abs(fermi[0, 0, 0] - expected), 0.0, places=11)
        self.assertEqual(frozen[0, 0, 0], 0.0)

    def test_hipolito_a11_drude_integrated_by_parts(self):
        # Hipolito et al. Eq. A11 contains integral v * d_k^2 f.  On a
        # periodic BZ its independently integrated form is integral f * d_k^2 v.
        # A chiral one-band dispersion breaks k -> -k, so this check does not
        # rely on time reversal or a vanishing second-order tensor.
        nmesh = 4096
        length = 3.1
        theta = 2 * np.pi * (np.arange(nmesh) + 0.5) / nmesh
        energy = (0.4 * np.cos(theta) + 0.13 * np.sin(theta)
                  + 0.2 * np.cos(2 * theta) + 0.05 * np.sin(2 * theta))
        velocity = length * (-0.4 * np.sin(theta) + 0.13 * np.cos(theta)
                             - 0.4 * np.sin(2 * theta) + 0.1 * np.cos(2 * theta))
        hessian = length**2 * (-0.4 * np.cos(theta) - 0.13 * np.sin(theta)
                               - 0.8 * np.cos(2 * theta) - 0.2 * np.sin(2 * theta))
        d2_velocity = length**3 * (0.4 * np.sin(theta) - 0.13 * np.cos(theta)
                                    + 1.6 * np.sin(2 * theta) - 0.4 * np.cos(2 * theta))
        ef, kbt, w, gamma = 0.14, 0.03, 0.4, 0.05
        f, fp, fpp = fermi_data(energy, ef, kbt)
        a11 = np.mean(velocity * (fpp * velocity**2 + fp * hessian))
        integrated_by_parts = np.mean(f * d2_velocity)
        self.assertGreater(abs(a11), 1e-3)
        np.testing.assert_allclose(a11, integrated_by_parts, rtol=1e-10)
        direct = 0j
        for e, v, h in zip(energy, velocity, hessian):
            q = dict(E=np.array([e]), r=np.zeros((3, 1, 1), complex),
                     rd=np.zeros((3, 3, 1, 1), complex),
                     v=np.array([[[v]], [[0]], [[0]]], complex))
            curvature = np.zeros((3, 3, 1))
            curvature[0, 0, 0] = h
            direct += kinetic_chi_k(q, curvature, ef, w, gamma, gamma, kbt,
                                    components=((0, 0, 0),))[0][0, 0, 0]
        direct /= nmesh
        expected = -(-integrated_by_parts /
                     ((gamma - 1j * w) * (gamma - 2j * w))) / (-2j * w)
        np.testing.assert_allclose(direct, expected, rtol=1e-10)

    def test_phase_gauge_invariance_and_insulating_limit(self):
        kf = np.array([0.213, 0.173, 0.0])
        q = at_k(self.tb, kf, with_generalized_derivative=True)
        hessian = band_hessian(self.tb, kf, q)
        args = (0.67, 0.45, 0.03, 0.025, 0.015)
        reference = kinetic_chi_k(q, hessian, *args)
        phased = transform_band_gauge(q, phases=[0.2, -0.4, 1.1, 0.7])
        result = kinetic_chi_k(phased, hessian, *args)
        for a, b in zip(reference, result):
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)
        insulator = kinetic_chi_k(q, hessian, 0.0, 0.1, 0.01, 0.01, 0.005)
        self.assertLess(np.max(np.abs(insulator[2])), 1e-8)

    def test_covariant_density_derivative_against_wannier_finite_difference(self):
        # Independently differentiate the first-order density matrix in the
        # fixed Wannier gauge, then transform back to the energy basis.
        kf = np.array([0.213, 0.173, 0.0])
        ef, w, eta_intra, eta_inter, kbt = 0.67, 0.45, 0.03, 0.025, 0.015
        q = at_k(self.tb, kf, with_generalized_derivative=True)
        hessian = band_hessian(self.tb, kf, q)
        analytic = kinetic_chi_k(
            q, hessian, ef, w, eta_intra, eta_inter, kbt,
            components=((0, 0, 0),),
        )[0][0, 0, 0]

        def first_order_wannier(point):
            state = at_k(self.tb, point)
            e, u = state["E"], state["U"]
            f, fp, _ = fermi_data(e, ef, kbt)
            diff = e[:, None] - e[None, :]
            gamma = np.full_like(diff, eta_inter)
            np.fill_diagonal(gamma, eta_intra)
            g1 = 1.0 / (gamma - 1j * (w - diff))
            rho = 1j * state["r"][0] * (f[:, None] - f[None, :]) * g1
            rho[np.diag_indices(len(e))] = (
                fp * state["v"][0].diagonal().real * g1.diagonal()
            )
            return u @ rho @ u.conj().T

        dk = 1e-5  # inverse Angstrom
        shift = np.array([dk, 0.0, 0.0]) @ self.tb["lat"].T / (2 * np.pi)
        rho_w = first_order_wannier(kf)
        derivative_w = (
            first_order_wannier(kf + shift)
            - first_order_wannier(kf - shift)
        ) / (2 * dk)
        u = q["U"]
        connection_w = u @ q["Abar"][0] @ u.conj().T
        covariant_w = derivative_w - 1j * (
            connection_w @ rho_w - rho_w @ connection_w
        )
        covariant_h = u.conj().T @ covariant_w @ u
        energy_diff = q["E"][:, None] - q["E"][None, :]
        gamma2 = np.full_like(energy_diff, eta_inter)
        np.fill_diagonal(gamma2, eta_intra)
        rho2 = covariant_h / (gamma2 - 1j * (2 * w - energy_diff))
        current_vertex = 1j * energy_diff * q["r"][0]
        current_vertex[np.diag_indices(len(q["E"]))] = (
            q["v"][0].diagonal().real
        )
        finite_difference = np.trace(current_vertex @ rho2) / (-2j * w)
        self.assertLess(abs(analytic - finite_difference) / abs(analytic), 1e-6)

    def test_full_metallic_response_magnetic_wannier_finite_difference(self):
        # Break time reversal with imaginary same-orbital hopping. This checks
        # the *full* metallic response via a separately differentiated
        # Wannier-gauge density matrix on a magnetic model, including f' and f''.
        tb = dict(self.tb)
        tb["H"] = self.tb["H"].copy()
        tb["H"][1, 2, 2] += 0.09j
        tb["H"][2, 2, 2] -= 0.09j
        kf = np.array([0.213, 0.173, 0.0])
        ef, w, eta, kbt = 0.67, 0.45, 0.025, 0.02
        q = at_k(tb, kf, with_generalized_derivative=True)
        hessian = band_hessian(tb, kf, q)
        analytic = kinetic_chi_k(q, hessian, ef, w, eta, eta, kbt,
                                 components=((0, 0, 0),))[0][0, 0, 0]

        def first_order(point):
            state = at_k(tb, point)
            e, u = state["E"], state["U"]
            f, fp, _ = fermi_data(e, ef, kbt)
            difference = e[:, None] - e[None, :]
            propagator = 1 / (eta - 1j * (w - difference))
            rho = 1j * state["r"][0] * (f[:, None] - f[None, :]) * propagator
            rho[np.diag_indices(len(e))] = (
                fp * state["v"][0].diagonal().real * propagator.diagonal()
            )
            return u @ rho @ u.conj().T

        dk = 1e-5
        shift = np.array([dk, 0.0, 0.0]) @ tb["lat"].T / (2 * np.pi)
        rho = first_order(kf)
        derivative = (first_order(kf + shift) - first_order(kf - shift)) / (2 * dk)
        u = q["U"]
        connection = u @ q["Abar"][0] @ u.conj().T
        covariant = derivative - 1j * (connection @ rho - rho @ connection)
        rho2 = u.conj().T @ covariant @ u
        energy_diff = q["E"][:, None] - q["E"][None, :]
        rho2 /= eta - 1j * (2 * w - energy_diff)
        current_vertex = 1j * energy_diff * q["r"][0]
        current_vertex[np.diag_indices(len(q["E"]))] = q["v"][0].diagonal().real
        reference = np.trace(current_vertex @ rho2) / (-2j * w)
        self.assertGreater(abs(analytic), 1e-5)
        self.assertLess(abs(analytic - reference) / abs(analytic), 1e-5)

    def test_adiabatic_width_matches_xiao_insulating_bz(self):
        # With output width 2*eta and complex-frequency current-to-polarization
        # conversion, the current and polarization kernels agree exactly after
        # BZ integration, including their imaginary parts at finite eta.
        nk = 12
        for w in (0.1, 0.25, 0.45):
            old = 0j
            frozen = 0j
            for ix in range(nk):
                for iy in range(nk):
                    kf = np.array([(ix + 0.5) / nk, (iy + 0.5) / nk, 0.0])
                    q = at_k(self.tb, kf, with_generalized_derivative=True)
                    occ = (q["E"] < 0.0).astype(float)
                    old += chi_total_k(q, occ, w, 0.02)[1, 1, 1]
                    frozen += kinetic_chi_from_tb(
                        self.tb, kf, 0.0, w, 0.02, 0.02, 0.005,
                        components=((1, 1, 1),),
                        broadening_scheme="adiabatic",
                    )[1][1, 1, 1]
            self.assertLess(abs(old - frozen) / abs(old), 1e-10)


if __name__ == "__main__":
    unittest.main()
