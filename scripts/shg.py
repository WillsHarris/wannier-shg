#!/usr/bin/env python
"""Length-gauge SHG helpers for the Wannier model.

The pure-interband and mixed/two-band kernels implement Xiao et al.,
arXiv:2211.04354v1, Appendix C, Eqs. (C2) and (C3).  Explicit metallic
Fermi-surface/Drude terms are not included in that convention and remain a
separate extension.
"""
from itertools import product

import numpy as np
from wann_tb import read_tb, at_k


# CODATA constants. Wannier90 _tb.dat uses Angstrom and eV.
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12
_ANGSTROM_M = 1.0e-10
_EV_J = _ELEMENTARY_CHARGE_C


def bz_mean_to_bulk_chi_si(chi_raw, cell_volume_A3):
    """Convert a full-BZ mean from Angstrom^3/eV^2 to bulk SI m/V.

    ``chi_raw`` must already include the band sum and the uniform full-BZ
    average, but not the reciprocal-BZ volume or the physical prefactor. The
    positive elementary-charge convention matches GPAW; this function does
    not otherwise alter the kernel's global sign.
    """
    if cell_volume_A3 <= 0.0:
        raise ValueError("cell_volume_A3 must be positive")
    volume_m3 = cell_volume_A3 * _ANGSTROM_M**3
    prefactor = (
        _ELEMENTARY_CHARGE_C**3
        / _VACUUM_PERMITTIVITY_F_PER_M
        * _ANGSTROM_M**3
        / _EV_J**2
        / volume_m3
    )
    return np.asarray(chi_raw) * prefactor


def bulk_chi_to_sheet_si(chi_bulk, cell_height_A):
    """Convert bulk-equivalent m/V to vacuum-independent sheet m^2/V."""
    if cell_height_A <= 0.0:
        raise ValueError("cell_height_A must be positive")
    return np.asarray(chi_bulk) * cell_height_A * _ANGSTROM_M


_EVEN_22 = (
    ((1, 1, 1), 1.0),
    ((1, 0, 0), -1.0),
    ((0, 1, 0), -1.0),
    ((0, 0, 1), -1.0),
)
_ODD_11 = (
    ((0, 0, 0), 1.0),
    ((0, 1, 1), -1.0),
    ((1, 0, 1), -1.0),
    ((1, 1, 0), -1.0),
)


def rotation_z(angle_deg):
    """Return the passive Cartesian basis rotation about z by ``angle_deg``."""
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_rank3_tensor(chi, rotation):
    """Express a Cartesian rank-three tensor in a rotated orthonormal basis."""
    chi = np.asarray(chi, dtype=complex)
    rotation = np.asarray(rotation, dtype=float)
    if chi.shape != (3, 3, 3):
        raise ValueError("chi must have shape (3, 3, 3)")
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    return np.einsum("ai,bj,ck,ijk->abc", rotation, rotation, rotation, chi)


def magnetization_even_odd(chi_plus, chi_minus):
    """Return parts even and odd under reversal of the magnetization."""
    chi_plus = np.asarray(chi_plus, dtype=complex)
    chi_minus = np.asarray(chi_minus, dtype=complex)
    if chi_plus.shape != (3, 3, 3) or chi_minus.shape != (3, 3, 3):
        raise ValueError("chi_plus and chi_minus must have shape (3, 3, 3)")
    return 0.5 * (chi_plus + chi_minus), 0.5 * (chi_plus - chi_minus)


def relative_residual(delta, reference):
    """Return ||delta||/||reference||, with a stable zero-reference limit."""
    denominator = np.linalg.norm(reference)
    numerator = np.linalg.norm(delta)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def intrinsic_permutation_residual(chi):
    """Measure violation of the degenerate-SHG identity chi_abc = chi_acb."""
    chi = np.asarray(chi, dtype=complex)
    return relative_residual(chi - chi.swapaxes(1, 2), chi)


def _family_report(chi, family):
    in_plane = np.asarray(chi, dtype=complex)[:2, :2, :2]
    signed_values = np.array([sign * chi[index] for index, sign in family])
    amplitude = np.mean(signed_values)

    family_indices = {index for index, _ in family}
    forbidden = np.array(
        [
            chi[a, b, c]
            for a in range(2)
            for b in range(2)
            for c in range(2)
            if (a, b, c) not in family_indices
        ]
    )
    family_norm = np.linalg.norm(signed_values)
    in_plane_norm = np.linalg.norm(in_plane)
    relation_residual = (
        float(np.linalg.norm(signed_values - amplitude) / family_norm)
        if family_norm > 0.0
        else 0.0
    )
    forbidden_residual = (
        float(np.linalg.norm(forbidden) / in_plane_norm)
        if in_plane_norm > 0.0
        else 0.0
    )
    return {
        "amplitude": amplitude,
        "relation_residual": relation_residual,
        "forbidden_residual": forbidden_residual,
        "in_plane_norm": float(in_plane_norm),
    }


def tensor_symmetry_report(chi_even, chi_odd):
    """Diagnose the paper's even-22 and odd-11 in-plane tensor relations."""
    return {
        "even_22": _family_report(chi_even, _EVEN_22),
        "odd_11": _family_report(chi_odd, _ODD_11),
        "even_permutation_residual": intrinsic_permutation_residual(chi_even),
        "odd_permutation_residual": intrinsic_permutation_residual(chi_odd),
    }


def _symmetry_score(report):
    residuals = []
    for name in ("even_22", "odd_11"):
        if report[name]["in_plane_norm"] > 0.0:
            residuals.extend(
                [
                    report[name]["relation_residual"],
                    report[name]["forbidden_residual"],
                ]
            )
    return float(np.linalg.norm(residuals)) if residuals else 0.0


def find_inplane_axis(chi_even, chi_odd, step_deg=0.25):
    """Find the best paper-axis alignment in one symmetry-equivalent 60-degree sector."""
    if step_deg <= 0.0 or step_deg > 60.0:
        raise ValueError("step_deg must be in (0, 60]")

    best = None
    for angle_deg in np.arange(0.0, 60.0, step_deg):
        rotation = rotation_z(angle_deg)
        even_rotated = rotate_rank3_tensor(chi_even, rotation)
        odd_rotated = rotate_rank3_tensor(chi_odd, rotation)
        report = tensor_symmetry_report(even_rotated, odd_rotated)
        candidate = {
            "angle_deg": float(angle_deg),
            "score": _symmetry_score(report),
            "report": report,
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    return best


def _transform_band_operator(operator, phases=None, permutation=None):
    transformed = np.array(operator, dtype=complex, copy=True)
    if phases is not None:
        factors = np.exp(1j * np.asarray(phases))
        transformed = (
            factors.conj()[None, :, None]
            * transformed
            * factors[None, None, :]
        )
    if permutation is not None:
        permutation = np.asarray(permutation)
        transformed = transformed[:, permutation, :][:, :, permutation]
    return transformed


def transform_band_gauge(q, phases=None, permutation=None):
    """Apply band U(1) phases and/or relabel bands in an ``at_k`` result."""
    transformed = dict(q)
    transformed["r"] = _transform_band_operator(q["r"], phases, permutation)
    transformed["v"] = _transform_band_operator(q["v"], phases, permutation)
    if "rd" in q:
        transformed["rd"] = np.array(
            [
                _transform_band_operator(q["rd"][a], phases, permutation)
                for a in range(3)
            ]
        )
    if permutation is None:
        transformed["E"] = np.array(q["E"], copy=True)
    else:
        transformed["E"] = np.asarray(q["E"])[permutation]
    transformed["w"] = transformed["E"][:, None] - transformed["E"][None, :]
    return transformed

def gen_deriv(r, w, v, energy_tol_eV=1e-6):
    """Evaluate the Aversa-Sipe sum rule for ``r^a_nm;b``.

    ``w[n,m]`` is ``E_n-E_m``.  Prefer the direct real-space derivative
    returned by ``at_k(..., with_generalized_derivative=True)`` for production
    calculations; this sum rule is retained as a compact diagnostic.
    """
    N=r.shape[1]
    valid=np.abs(w)>=energy_tol_eV
    np.fill_diagonal(valid,False)
    wr=np.where(valid,w,1.0)
    D=np.array([v[a].diagonal().real for a in range(3)])   # (3,N) band velocity
    Delta=D[:,:,None]-D[:,None,:]                          # (3,N,N) Delta^a_nm
    rd=np.zeros((3,3,N,N),complex)
    # D_mn = v_mm-v_nn = -Delta_nm in the Appendix-C convention.
    for a in range(3):
        for b in range(3):
            rd[a,b]=-(r[a]*Delta[b]+r[b]*Delta[a])/wr
    for a in range(3):
        for b in range(3):
            S=np.einsum('pm,np,pm->nm', w, r[a], r[b]) - np.einsum('np,np,pm->nm', w, r[b], r[a])
            rd[a,b]+=1j*S/wr
            rd[a,b][~valid]=0.0
    return rd

def chi_k(q, occ, w_ph, eta, energy_tol_eV=1e-6):
    """Return the unnormalized, interband three-band SHG tensor at one k.

    Energies, photon energy, broadening, and tolerance are in eV; dipoles are
    in Angstrom. This routine deliberately omits the two-band/mixed term,
    metallic Fermi-surface terms, BZ normalization, and physical prefactors.
    """
    E=np.asarray(q["E"],dtype=float)
    r=np.array(q["r"],dtype=complex,copy=True)
    f=np.asarray(occ,dtype=float)
    if r.shape!=(3,len(E),len(E)):
        raise ValueError("q['r'] must have shape (3, nbands, nbands)")
    if f.shape!=(len(E),):
        raise ValueError("occupations must have one value per band")
    if eta<=0 or energy_tol_eV<=0:
        raise ValueError("eta and energy_tol_eV must be positive")

    # The pure three-band term contains only interband dipoles. Suppress
    # diagonal and unresolved near-degenerate matrix elements defensively.
    dE=E[:,None]-E[None,:]
    interband=np.abs(dE)>=energy_tol_eV
    np.fill_diagonal(interband,False)
    r*=interband[None,:,:]

    w_mn=E[None,:,None]-E[:,None,None]
    w_ml=E[None,:,None]-E[None,None,:]
    w_ln=E[None,None,:]-E[:,None,None]
    f_nm=f[:,None,None]-f[None,:,None]
    f_ml=f[None,:,None]-f[None,None,:]
    f_ln=f[None,None,:]-f[:,None,None]

    with np.errstate(divide="ignore",invalid="ignore"):
        frequency_weight=(
            2*f_nm/(w_mn-2*w_ph-2j*eta)
            +f_ml/(w_ml-w_ph-1j*eta)
            +f_ln/(w_ln-w_ph-1j*eta)
        )

    intermediate_denominator=w_ln-w_ml
    inverse_intermediate=np.zeros_like(intermediate_denominator)
    np.divide(
        1.0,
        intermediate_denominator,
        out=inverse_intermediate,
        where=np.abs(intermediate_denominator)>=energy_tol_eV,
    )
    kernel=0.5*frequency_weight*inverse_intermediate

    chi=np.zeros((3,3,3),complex)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                numerator=(
                    np.einsum("nm,ml,ln->nml",r[a],r[b],r[c],optimize=True)
                    +np.einsum("nm,ml,ln->nml",r[a],r[c],r[b],optimize=True)
                )
                chi[a,b,c]=np.sum(numerator*kernel)
    return chi


def _component_indices(components):
    """Return validated Cartesian tensor indices, or all indices by default."""
    if components is None:
        return tuple(product(range(3), repeat=3))
    normalized = []
    for component in components:
        index = tuple(component)
        if len(index) != 3 or any(axis not in (0, 1, 2) for axis in index):
            raise ValueError("components must contain Cartesian index triples")
        if index not in normalized:
            normalized.append(index)
    if not normalized:
        raise ValueError("components must not be empty")
    return tuple(normalized)


def prepare_chi_k_spectrum(q, occ, energy_tol_eV=1e-6):
    """Prepare frequency-independent three-band data for one k-point."""
    E = np.asarray(q["E"], dtype=float)
    r = np.array(q["r"], dtype=complex, copy=True)
    f = np.asarray(occ, dtype=float)
    if r.shape != (3, len(E), len(E)):
        raise ValueError("q['r'] must have shape (3, nbands, nbands)")
    if f.shape != (len(E),):
        raise ValueError("occupations must have one value per band")
    if energy_tol_eV <= 0:
        raise ValueError("energy_tol_eV must be positive")

    dE = E[:, None] - E[None, :]
    interband = np.abs(dE) >= energy_tol_eV
    np.fill_diagonal(interband, False)
    r *= interband[None, :, :]

    w_mn = E[None, :, None] - E[:, None, None]
    w_ml = E[None, :, None] - E[None, None, :]
    w_ln = E[None, None, :] - E[:, None, None]
    intermediate_denominator = w_ln - w_ml
    inverse_intermediate = np.zeros_like(intermediate_denominator)
    np.divide(
        1.0,
        intermediate_denominator,
        out=inverse_intermediate,
        where=np.abs(intermediate_denominator) >= energy_tol_eV,
    )

    velocity_pair = None
    if "v" in q:
        velocity = np.asarray(q["v"])
        if velocity.shape != (3, len(E), len(E)):
            raise ValueError("q['v'] must have shape (3, nbands, nbands)")
        velocity_diagonal = np.real(
            np.diagonal(velocity, axis1=1, axis2=2)
        )
        velocity_pair = np.sqrt(
            (
                (
                    velocity_diagonal[:, :, None]
                    - velocity_diagonal[:, None, :]
                )
                ** 2
            ).sum(axis=0)
        )

    return {
        "r": r,
        "w_mn": w_mn,
        "w_ml": w_ml,
        "w_ln": w_ln,
        "f_nm": f[:, None, None] - f[None, :, None],
        "f_ml": f[None, :, None] - f[None, None, :],
        "f_ln": f[None, None, :] - f[:, None, None],
        "inverse_intermediate": inverse_intermediate,
        "velocity_pair": velocity_pair,
    }


def chi_k_spectrum_prepared(
    prepared,
    photon_energies,
    eta,
    components=None,
    omega_chunk_size=8,
    adaptive_dk=0.0,
    adaptive_a=1.0,
    eta_floor=1.0e-3,
):
    """Evaluate a three-band spectrum from prepared one-k-point data."""
    r = prepared["r"]
    w_mn = prepared["w_mn"]
    w_ml = prepared["w_ml"]
    w_ln = prepared["w_ln"]
    f_nm = prepared["f_nm"]
    f_ml = prepared["f_ml"]
    f_ln = prepared["f_ln"]
    inverse_intermediate = prepared["inverse_intermediate"]
    omega = np.atleast_1d(np.asarray(photon_energies, dtype=float))
    if omega.ndim != 1 or len(omega) == 0:
        raise ValueError("photon_energies must be a nonempty one-dimensional array")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if omega_chunk_size <= 0:
        raise ValueError("omega_chunk_size must be positive")

    selected = _component_indices(components)
    if adaptive_dk > 0.0:
        velocity_pair = prepared["velocity_pair"]
        if velocity_pair is None:
            raise ValueError("adaptive broadening requires q['v']")
        base = np.maximum(
            adaptive_a * adaptive_dk * velocity_pair,
            eta_floor,
        )
        e_mn = base[None, :, :, None]
        e_ml = base[None, None, :, :]
        e_ln = base[None, :, None, :]
    else:
        e_mn = e_ml = e_ln = eta

    numerators = None
    if components is not None:
        numerators = {}
        for a, b, c in selected:
            numerators[(a, b, c)] = np.einsum(
                "nm,ml,ln->nml",
                r[a],
                r[b],
                r[c],
                optimize=True,
            ) + np.einsum(
                "nm,ml,ln->nml",
                r[a],
                r[c],
                r[b],
                optimize=True,
            )

    spectrum = np.zeros((len(omega), 3, 3, 3), dtype=complex)
    for start in range(0, len(omega), omega_chunk_size):
        stop = min(start + omega_chunk_size, len(omega))
        photon = omega[start:stop, None, None, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            frequency_weight = (
                2.0
                * f_nm[None, :, :, :]
                / (w_mn[None, :, :, :] - 2.0 * photon - 2j * e_mn)
                + f_ml[None, :, :, :]
                / (w_ml[None, :, :, :] - photon - 1j * e_ml)
                + f_ln[None, :, :, :]
                / (w_ln[None, :, :, :] - photon - 1j * e_ln)
            )
        kernel = 0.5 * frequency_weight * inverse_intermediate[None, :, :, :]

        if components is None:
            contraction = np.einsum(
                "anm,bml,cln,wnml->wabc",
                r,
                r,
                r,
                kernel,
                optimize=True,
            )
            spectrum[start:stop] = contraction + contraction.swapaxes(2, 3)
        else:
            for a, b, c in selected:
                spectrum[start:stop, a, b, c] = np.einsum(
                    "nml,wnml->w",
                    numerators[(a, b, c)],
                    kernel,
                    optimize=True,
                )
    return spectrum


def chi_k_spectrum(
    q,
    occ,
    photon_energies,
    eta,
    energy_tol_eV=1e-6,
    components=None,
    omega_chunk_size=8,
    adaptive_dk=0.0,
    adaptive_a=1.0,
    eta_floor=1.0e-3,
):
    """Batched equivalent of :func:`chi_k` for one or more photon energies.

    Frequency-independent band data are formed once per k-point.  Photon
    energies are processed in bounded chunks to avoid a large
    ``(n_omega, nband, nband, nband)`` temporary.  Supplying ``components``
    computes only those Cartesian triples; all other output entries are zero.
    """
    prepared = prepare_chi_k_spectrum(q, occ, energy_tol_eV)
    return chi_k_spectrum_prepared(
        prepared,
        photon_energies,
        eta,
        components=components,
        omega_chunk_size=omega_chunk_size,
        adaptive_dk=adaptive_dk,
        adaptive_a=adaptive_a,
        eta_floor=eta_floor,
    )


def chi_mixed_k(
    q,
    occ,
    w_ph,
    eta,
    energy_tol_eV=1e-6,
    include_final_delta_term=True,
    components=None,
    adaptive_dk=0.0,
    adaptive_a=1.0,
    eta_floor=1.0e-3,
):
    """Return the Appendix-C mixed/two-band SHG tensor at one k.

    This follows Eq. (C3) of Xiao et al., arXiv:2211.04354v1, except that
    the final band-velocity-difference term uses the sign of Chen et al.'s
    Eqs. (14)-(16), which is opposite to Chen/Xiao Appendix C.  See
    docs/magnetic_term_sign.md.  ``q`` must contain the direct generalized
    derivative ``rd[a,b,n,m] = r^a_nm;b`` generated by
    ``at_k(..., with_generalized_derivative=True)``.

    The result uses the same unnormalized Angstrom/eV convention and the same
    degenerate-input-field factor of one half as ``chi_k``.  It is not an
    explicit Fermi-surface or Drude contribution.  The final term can be
    disabled only to cross-check the first four terms against HopTB 1.0.0,
    whose mixed kernel omits it.
    """
    E=np.asarray(q["E"],dtype=float)
    r=np.array(q["r"],dtype=complex,copy=True)
    v=np.asarray(q["v"],dtype=complex)
    f=np.asarray(occ,dtype=float)
    if "rd" not in q:
        raise ValueError(
            "mixed-band SHG requires at_k(..., "
            "with_generalized_derivative=True)"
        )
    rd=np.array(q["rd"],dtype=complex,copy=True)
    nbands=len(E)
    if r.shape!=(3,nbands,nbands) or v.shape!=(3,nbands,nbands):
        raise ValueError("q['r'] and q['v'] must have shape (3, nbands, nbands)")
    if rd.shape!=(3,3,nbands,nbands):
        raise ValueError("q['rd'] must have shape (3, 3, nbands, nbands)")
    if f.shape!=(nbands,):
        raise ValueError("occupations must have one value per band")
    if eta<=0 or energy_tol_eV<=0:
        raise ValueError("eta and energy_tol_eV must be positive")

    w_mn=E[None,:]-E[:,None]  # E_m-E_n, indexed [n,m]
    valid=np.abs(w_mn)>=energy_tol_eV
    np.fill_diagonal(valid,False)
    r*=valid[None,:,:]
    rd*=valid[None,None,:,:]
    inv_w=np.zeros_like(w_mn)
    np.divide(1.0,w_mn,out=inv_w,where=valid)
    inv_w2=inv_w**2
    band_velocity=np.real(np.diagonal(v,axis1=1,axis2=2))
    delta_mn=(
        band_velocity[:,None,:]-band_velocity[:,:,None]
    )  # v_mm-v_nn, [a,n,m]
    if adaptive_dk>0.0:
        eta_mn=np.maximum(
            adaptive_a*adaptive_dk*np.sqrt((delta_mn**2).sum(0)), eta_floor
        )  # per-transition broadening [n,m]
    else:
        eta_mn=eta
    z=w_ph+1j*eta_mn
    inv_one=1.0/(w_mn-z)
    inv_two=1.0/(w_mn-2.0*z)
    f_nm=f[:,None]-f[None,:]

    chi=np.zeros((3,3,3),complex)
    for a, b, c in _component_indices(components):
        term1=(
            2.0*r[a]*(rd[b,c].T+rd[c,b].T)
            *inv_w*inv_two
        )
        term2=(
            rd[a,c]*r[b].T+rd[a,b]*r[c].T
        )*inv_w*inv_one
        term3=(
            r[a]
            *(r[b].T*delta_mn[c]+r[c].T*delta_mn[b])
            *inv_w2
            *(inv_one-4.0*inv_two)
        )
        term4=-(
            rd[b,a]*r[c].T+rd[c,a]*r[b].T
        )*0.5*inv_w*inv_one
        term5=0.0
        if include_final_delta_term:
            term5=(
                delta_mn[a]*np.real(r[b]*r[c].T)
            )*0.5*inv_w2*inv_one
        chi[a,b,c]=0.5j*np.sum(
            f_nm*(term1+term2+term3+term4+term5)*valid
        )
    return chi


def prepare_chi_mixed_k_spectrum(q, occ, energy_tol_eV=1e-6):
    """Prepare frequency-independent mixed/two-band data at one k-point."""
    E = np.asarray(q["E"], dtype=float)
    r = np.array(q["r"], dtype=complex, copy=True)
    v = np.asarray(q["v"], dtype=complex)
    f = np.asarray(occ, dtype=float)
    if "rd" not in q:
        raise ValueError(
            "mixed-band SHG requires at_k(..., "
            "with_generalized_derivative=True)"
        )
    rd = np.array(q["rd"], dtype=complex, copy=True)
    nbands = len(E)
    if r.shape != (3, nbands, nbands) or v.shape != (3, nbands, nbands):
        raise ValueError("q['r'] and q['v'] must have shape (3, nbands, nbands)")
    if rd.shape != (3, 3, nbands, nbands):
        raise ValueError("q['rd'] must have shape (3, 3, nbands, nbands)")
    if f.shape != (nbands,):
        raise ValueError("occupations must have one value per band")
    if energy_tol_eV <= 0:
        raise ValueError("energy_tol_eV must be positive")

    w_mn = E[None, :] - E[:, None]
    valid = np.abs(w_mn) >= energy_tol_eV
    np.fill_diagonal(valid, False)
    r *= valid[None, :, :]
    rd *= valid[None, None, :, :]
    inv_w = np.zeros_like(w_mn)
    np.divide(1.0, w_mn, out=inv_w, where=valid)
    band_velocity = np.real(np.diagonal(v, axis1=1, axis2=2))
    delta_mn = band_velocity[:, None, :] - band_velocity[:, :, None]
    return {
        "r": r,
        "rd": rd,
        "w_mn": w_mn,
        "valid": valid,
        "inv_w": inv_w,
        "inv_w2": inv_w**2,
        "delta_mn": delta_mn,
        "f_nm": f[:, None] - f[None, :],
    }


def chi_mixed_k_spectrum_prepared(
    prepared,
    photon_energies,
    eta,
    include_final_delta_term=True,
    components=None,
    omega_chunk_size=32,
    adaptive_dk=0.0,
    adaptive_a=1.0,
    eta_floor=1.0e-3,
):
    """Evaluate a mixed/two-band spectrum from prepared one-k-point data."""
    omega = np.atleast_1d(np.asarray(photon_energies, dtype=float))
    if omega.ndim != 1 or len(omega) == 0:
        raise ValueError("photon_energies must be a nonempty one-dimensional array")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if omega_chunk_size <= 0:
        raise ValueError("omega_chunk_size must be positive")

    r = prepared["r"]
    rd = prepared["rd"]
    w_mn = prepared["w_mn"]
    valid = prepared["valid"]
    inv_w = prepared["inv_w"]
    inv_w2 = prepared["inv_w2"]
    delta_mn = prepared["delta_mn"]
    f_nm = prepared["f_nm"]
    if adaptive_dk > 0.0:
        eta_mn = np.maximum(
            adaptive_a
            * adaptive_dk
            * np.sqrt((delta_mn**2).sum(axis=0)),
            eta_floor,
        )
    else:
        eta_mn = eta

    spectrum = np.zeros((len(omega), 3, 3, 3), dtype=complex)
    for a, b, c in _component_indices(components):
        static1 = (
            2.0 * r[a] * (rd[b, c].T + rd[c, b].T) * inv_w
        )
        static2 = (
            rd[a, c] * r[b].T + rd[a, b] * r[c].T
        ) * inv_w
        static3 = (
            r[a]
            * (r[b].T * delta_mn[c] + r[c].T * delta_mn[b])
            * inv_w2
        )
        static4 = -(
            rd[b, a] * r[c].T + rd[c, a] * r[b].T
        ) * 0.5 * inv_w
        static5 = 0.0
        if include_final_delta_term:
            static5 = (
                delta_mn[a] * np.real(r[b] * r[c].T)
            ) * 0.5 * inv_w2

        for start in range(0, len(omega), omega_chunk_size):
            stop = min(start + omega_chunk_size, len(omega))
            photon = omega[start:stop, None, None]
            z = photon + 1j * eta_mn
            inv_one = 1.0 / (w_mn[None, :, :] - z)
            inv_two = 1.0 / (w_mn[None, :, :] - 2.0 * z)
            terms = (
                static1[None, :, :] * inv_two
                + static2[None, :, :] * inv_one
                + static3[None, :, :] * (inv_one - 4.0 * inv_two)
                + static4[None, :, :] * inv_one
                + static5 * inv_one
            )
            spectrum[start:stop, a, b, c] = 0.5j * np.sum(
                f_nm[None, :, :] * terms * valid[None, :, :],
                axis=(1, 2),
            )
    return spectrum


def chi_mixed_k_spectrum(
    q,
    occ,
    photon_energies,
    eta,
    energy_tol_eV=1e-6,
    include_final_delta_term=True,
    components=None,
    omega_chunk_size=32,
    adaptive_dk=0.0,
    adaptive_a=1.0,
    eta_floor=1.0e-3,
):
    """Batched mixed/two-band spectrum with one-time k-point preparation."""
    prepared = prepare_chi_mixed_k_spectrum(q, occ, energy_tol_eV)
    return chi_mixed_k_spectrum_prepared(
        prepared,
        photon_energies,
        eta,
        include_final_delta_term=include_final_delta_term,
        components=components,
        omega_chunk_size=omega_chunk_size,
        adaptive_dk=adaptive_dk,
        adaptive_a=adaptive_a,
        eta_floor=eta_floor,
    )


def prepare_chi_total_k_spectrum(q, occ, energy_tol_eV=1e-6):
    """Prepare all frequency-independent three- and mixed-band data."""
    return {
        "three_band": prepare_chi_k_spectrum(q, occ, energy_tol_eV),
        "mixed_band": prepare_chi_mixed_k_spectrum(q, occ, energy_tol_eV),
    }


def chi_total_k_spectrum_prepared(
    prepared,
    photon_energies,
    eta,
    components=None,
    omega_chunk_size=8,
    adaptive_dk=0.0,
    adaptive_a=1.0,
    eta_floor=1.0e-3,
):
    """Evaluate the total SHG spectrum from prepared one-k-point data."""
    return chi_k_spectrum_prepared(
        prepared["three_band"],
        photon_energies,
        eta,
        components=components,
        omega_chunk_size=omega_chunk_size,
        adaptive_dk=adaptive_dk,
        adaptive_a=adaptive_a,
        eta_floor=eta_floor,
    ) + chi_mixed_k_spectrum_prepared(
        prepared["mixed_band"],
        photon_energies,
        eta,
        components=components,
        omega_chunk_size=omega_chunk_size,
        adaptive_dk=adaptive_dk,
        adaptive_a=adaptive_a,
        eta_floor=eta_floor,
    )


def chi_total_k_spectrum(
    q,
    occ,
    photon_energies,
    eta,
    energy_tol_eV=1e-6,
    components=None,
    omega_chunk_size=8,
    adaptive_dk=0.0,
    adaptive_a=1.0,
    eta_floor=1.0e-3,
):
    """Batched Appendix-C three-band plus mixed-band response."""
    prepared = prepare_chi_total_k_spectrum(q, occ, energy_tol_eV)
    return chi_total_k_spectrum_prepared(
        prepared,
        photon_energies,
        eta,
        components=components,
        omega_chunk_size=omega_chunk_size,
        adaptive_dk=adaptive_dk,
        adaptive_a=adaptive_a,
        eta_floor=eta_floor,
    )


def chi_total_k(q, occ, w_ph, eta, energy_tol_eV=1e-6):
    """Return the Appendix-C pure-interband plus mixed-band response."""
    return chi_k(q,occ,w_ph,eta,energy_tol_eV)+chi_mixed_k(
        q,occ,w_ph,eta,energy_tol_eV
    )


def gauge_invariance_report(q, occ, w_ph, eta, seed=0):
    """Test one-k response invariance to band phases and band relabeling."""
    occ = np.asarray(occ)
    rng = np.random.default_rng(seed)
    reference = chi_k(q, occ, w_ph, eta)

    phases = rng.uniform(0.0, 2.0 * np.pi, len(q["E"]))
    phase_result = chi_k(
        transform_band_gauge(q, phases=phases), occ, w_ph, eta
    )

    permutation = rng.permutation(len(q["E"]))
    permutation_result = chi_k(
        transform_band_gauge(q, permutation=permutation),
        occ[permutation],
        w_ph,
        eta,
    )
    return {
        "phase_residual": relative_residual(phase_result - reference, reference),
        "permutation_residual": relative_residual(
            permutation_result - reference, reference
        ),
    }

if __name__=="__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit("usage: python scripts/shg.py MODEL_tb.dat EF_eV [NK]")
    tbfile=sys.argv[1]
    EF=float(sys.argv[2])
    nkq=int(sys.argv[3]) if len(sys.argv)>3 else 6
    tb=read_tb(tbfile); eta=0.05
    ws=np.array([1.0])  # single test photon energy (eV)
    ks=[(i/nkq,j/nkq,0) for i in range(nkq) for j in range(nkq)]
    chi=np.zeros((3,3,3),complex)
    for kf in ks:
        q=at_k(tb,np.array(kf)); occ=(q['E']<EF).astype(float)
        chi+=chi_k(q,occ,ws[0],eta)
    chi/=len(ks)
    print(f"test chi at w={ws[0]} eV, {nkq}x{nkq} k: |chi_yyy|={abs(chi[1,1,1]):.4e}, |chi_xxx|={abs(chi[0,0,0]):.4e}")
    print("ran OK (structural test)")
