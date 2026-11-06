"""Independent length-gauge second-order current for a Wannier model.

Implements the density-matrix recursion of Bhalla et al., PRB 107, 165131
(2023), Eqs. (11), (16)-(24), for degenerate-input SHG.  The difference
between the full result and the same recursion with derivatives of the Fermi
occupation suppressed isolates the Fermi-surface contribution.  All outputs
are *raw per-k* susceptibilities in the Angstrom^3/eV^2 convention of shg.py;
this is a validation kernel, not yet wired into the production spectrum.

The ``adiabatic`` width convention exactly matches the existing Xiao kernel:
each input photon carries +i*eta, so the 2-omega output carries +2i*eta.
The ``relaxation`` convention instead uses independent, constant intraband
and interband widths at each order. No spin degeneracy is assumed: a spinor
Wannier model already contains both spin sectors.
"""

import numpy as np

from wann_tb import at_k


def fermi_data(energies, ef, kbt):
    """Return f, df/dE, and d2f/dE2 for a positive thermal energy (eV)."""
    if kbt <= 0:
        raise ValueError("kbt must be positive; zero-T Fermi derivatives are distributions")
    x = np.clip((np.asarray(energies) - ef) / kbt, -700.0, 700.0)
    f = 1.0 / (1.0 + np.exp(x))
    df = -f * (1.0 - f) / kbt
    d2f = f * (1.0 - f) * (1.0 - 2.0 * f) / kbt**2
    return f, df, d2f


def band_hessian(tb, kf, q, energy_tol_eV=1e-6):
    """Return d2E_n/dk_a dk_b (eV Angstrom^2), via Hellmann-Feynman."""
    phase = np.exp(2j * np.pi * (tb["R"] @ kf)) / tb["deg"]
    d2h = np.einsum(
        "r,ra,rb,rmn->abmn",
        phase,
        1j * tb["Rcart"],
        1j * tb["Rcart"],
        tb["H"],
        optimize=True,
    )
    u = q["U"]
    d2h_band = np.einsum("in,abij,jn->abn", u.conj(), d2h, u, optimize=True).real
    energies = q["E"]
    denom = energies[:, None] - energies[None, :]
    inverse = np.zeros_like(denom)
    valid = np.abs(denom) > energy_tol_eV
    np.fill_diagonal(valid, False)
    np.divide(1.0, denom, out=inverse, where=valid)
    h = np.empty_like(d2h_band)
    for a in range(3):
        for b in range(3):
            coupling = q["v"][a] * q["v"][b].T
            h[a, b] = d2h_band[a, b] + 2.0 * np.sum(
                (coupling * inverse).real, axis=1
            )
    return 0.5 * (h + h.swapaxes(0, 1))


def kinetic_chi_k(
    q,
    hessian,
    ef,
    photon_energy,
    eta_intra,
    eta_inter,
    kbt,
    components=((0, 0, 0), (1, 1, 1)),
    broadening_scheme="relaxation",
):
    """Return (full, frozen-occupation, Fermi-surface) raw SHG at one k.

    The frozen-occupation result suppresses k-derivatives of f while retaining
    f_n-f_m.  It is a *diagnostic*, not automatically identical to the Xiao
    C2+C3 implementation; that equivalence must be checked before using the
    Fermi-surface difference as an additive production correction.
    """
    if photon_energy <= 0 or eta_intra <= 0 or eta_inter <= 0:
        raise ValueError("photon energy and both widths must be positive")
    if broadening_scheme not in ("relaxation", "adiabatic"):
        raise ValueError("broadening_scheme must be 'relaxation' or 'adiabatic'")
    if broadening_scheme == "adiabatic" and not np.isclose(eta_intra, eta_inter):
        raise ValueError("adiabatic switching requires one common input width")
    energies = np.asarray(q["E"], dtype=float)
    nb = len(energies)
    r = np.asarray(q["r"], dtype=complex)
    rd = np.asarray(q["rd"], dtype=complex)
    v = np.asarray(q["v"], dtype=complex)
    hessian = np.asarray(hessian, dtype=float)
    if r.shape != (3, nb, nb) or v.shape != r.shape:
        raise ValueError("r and v must have shape (3, nbands, nbands)")
    if rd.shape != (3, 3, nb, nb) or hessian.shape != (3, 3, nb):
        raise ValueError("rd or band hessian has the wrong shape")
    f, fp, fpp = fermi_data(energies, ef, kbt)
    band_v = np.real(np.diagonal(v, axis1=1, axis2=2))
    df = fp[None, :] * band_v
    d2f = (
        fpp[None, None, :] * band_v[:, None, :] * band_v[None, :, :]
        + fp[None, None, :] * hessian
    )
    f_nm = f[:, None] - f[None, :]
    en_nm = energies[:, None] - energies[None, :]
    # q['v'] is U^dagger (dH_W/dk) U.  Its diagonal is the band velocity,
    # but its off-diagonal omits the Wannier position-matrix contribution.
    # The physical current vertex is v_nm = i(E_n-E_m) r_nm, n != m.
    current_v = 1j * en_nm[None, :, :] * r
    current_v[:, np.arange(nb), np.arange(nb)] = band_v
    diagonal = np.eye(nb, dtype=bool)
    gamma = np.where(diagonal, eta_intra, eta_inter)
    g1 = 1.0 / (gamma - 1j * (photon_energy - en_nm))
    output_gamma = (2.0 if broadening_scheme == "adiabatic" else 1.0) * gamma
    g2 = 1.0 / (output_gamma - 1j * (2.0 * photon_energy - en_nm))
    output_energy = photon_energy + (
        1j * eta_inter if broadening_scheme == "adiabatic" else 0.0
    )

    full = np.zeros((3, 3, 3), dtype=complex)
    frozen = np.zeros_like(full)
    components = tuple(tuple(c) for c in components)
    if any(len(c) != 3 or any(a not in (0, 1, 2) for a in c) for c in components):
        raise ValueError("components must be Cartesian index triples")
    input_axes = {c for _, _, c in components} | {b for _, b, _ in components}

    rho1_frozen = {}
    rho1_full = {}
    for c in input_axes:
        base = 1j * r[c] * f_nm * g1
        rho1_frozen[c] = base
        rho1_full[c] = base + np.diag(df[c] * g1.diagonal())

    def ordered_response(a, b, c, with_fermi):
        rho1 = rho1_full[c] if with_fermi else rho1_frozen[c]
        # Differentiate the off-diagonal first-order density matrix using
        # r^c_nm;b, which includes the diagonal Berry-connection commutator.
        delta_v = band_v[b, :, None] - band_v[b, None, :]
        dg1 = -1j * delta_v * g1**2
        derivative = 1j * f_nm * (rd[c, b] * g1 + r[c] * dg1)
        if with_fermi:
            df_nm = df[b, :, None] - df[b, None, :]
            derivative += 1j * r[c] * df_nm * g1
            np.fill_diagonal(derivative, 0.0)
            derivative[np.diag_indices(nb)] = d2f[b, c] * g1.diagonal()
        else:
            np.fill_diagonal(derivative, 0.0)
        # The off-diagonal Berry connection supplies the rest of [D_b,rho1].
        cov_derivative = derivative - 1j * (r[b] @ rho1 - rho1 @ r[b])
        rho2 = g2 * cov_derivative
        raw_current = -np.trace(current_v[a] @ rho2)
        # The Appendix-C polarization kernel and this current kernel have
        # opposite global charge-sign conventions.  Calibrate to Appendix C;
        # in the adiabatic scheme J = -i*2*(w+i*eta)*P.
        return -raw_current / (-2j * output_energy)

    for a, b, c in components:
        full[a, b, c] = 0.5 * (
            ordered_response(a, b, c, True)
            + ordered_response(a, c, b, True)
        )
        frozen[a, b, c] = 0.5 * (
            ordered_response(a, b, c, False)
            + ordered_response(a, c, b, False)
        )
    return full, frozen, full - frozen


def kinetic_chi_from_tb(
    tb, kf, ef, photon_energy, eta_intra, eta_inter, kbt,
    components=((0, 0, 0), (1, 1, 1)),
    broadening_scheme="relaxation",
):
    """Evaluate the independent response from an existing Wannier _tb.dat model."""
    q = at_k(tb, np.asarray(kf), with_generalized_derivative=True)
    hessian = band_hessian(tb, np.asarray(kf), q)
    return kinetic_chi_k(
        q, hessian, ef, photon_energy, eta_intra, eta_inter, kbt,
        components=components,
        broadening_scheme=broadening_scheme,
    )
