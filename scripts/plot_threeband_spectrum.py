#!/usr/bin/env python3
"""Plot Appendix-C SHG for paired +/-M or a single +M Wannier model."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shg import (
    at_k,
    bulk_chi_to_sheet_si,
    bz_mean_to_bulk_chi_si,
    chi_k,
    chi_k_spectrum,
    chi_total_k,
    chi_total_k_spectrum,
    read_tb,
    rotation_z,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plus_tb", help="+M Wannier90 _tb.dat")
    parser.add_argument("minus_tb", nargs="?", help="-M Wannier90 _tb.dat")
    parser.add_argument("--ef-plus", type=float, required=True, help="+M Fermi energy (eV)")
    parser.add_argument("--ef-minus", type=float, help="-M Fermi energy (eV)")
    parser.add_argument("--omega-min", type=float, default=0.1, help="Minimum photon energy (eV)")
    parser.add_argument("--omega-max", type=float, default=4.0, help="Maximum photon energy (eV)")
    parser.add_argument("--n-omega", type=int, default=80, help="Number of photon energies")
    parser.add_argument("--eta", type=float, default=0.05, help="Broadening (eV)")
    parser.add_argument("--nk", type=int, default=6, help="Uniform in-plane k mesh")
    parser.add_argument(
        "--plus-only",
        action="store_true",
        help="Calculate +M only; do not form magnetization-even/odd tensors",
    )
    parser.add_argument(
        "--components",
        help="Comma-separated Cartesian components to calculate, e.g. xxx,yyy",
    )
    parser.add_argument(
        "--omega-chunk-size",
        type=int,
        default=8,
        help="Photon energies evaluated together by the optimized kernel",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of deterministic k-point worker threads",
    )
    parser.add_argument(
        "--legacy-kernel",
        action="store_true",
        help="Use the original one-frequency-at-a-time calculation",
    )
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help=(
            "Add the Appendix-C mixed/two-band term. This requires direct "
            "generalized derivatives and is separate from metallic Drude terms."
        ),
    )
    parser.add_argument(
        "--cell-volume-a3",
        type=float,
        help=(
            "Cell volume in Angstrom^3. When supplied, save bulk SI tensors "
            "and plot in pm/V instead of raw units."
        ),
    )
    parser.add_argument(
        "--cell-height-a",
        type=float,
        help=(
            "Cell height in Angstrom. With --cell-volume-a3, also save the "
            "vacuum-independent sheet susceptibility."
        ),
    )
    parser.add_argument(
        "--plot-units",
        choices=("raw", "bulk", "sheet"),
        help=(
            "Plot raw, bulk pm/V, or sheet nm^2/V values. Defaults to bulk "
            "when cell dimensions are supplied and raw otherwise."
        ),
    )
    parser.add_argument(
        "--axis-angle-deg",
        type=float,
        default=0.0,
        help="Fixed passive rotation into reporting axes; default is calculation axes",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/spectra/fgt_shg_threeband_spectrum",
        help="Output path prefix",
    )
    return parser.parse_args()


def parse_components(value):
    if value is None:
        return None
    axis = {"x": 0, "y": 1, "z": 2}
    components = []
    for label in value.split(","):
        label = label.strip().lower()
        if len(label) != 3 or any(character not in axis for character in label):
            raise ValueError("--components entries must be labels such as xxx or yyy")
        component = tuple(axis[character] for character in label)
        if component not in components:
            components.append(component)
    if not components:
        raise ValueError("--components must not be empty")
    return tuple(components)


def component_labels(components):
    if components is None:
        return "all"
    axis = "xyz"
    return ",".join("".join(axis[index] for index in component) for component in components)


def prepare_mesh(tb, ef, nk, include_mixed=False):
    mesh = []
    for i in range(nk):
        for j in range(nk):
            q = at_k(
                tb,
                np.array([i / nk, j / nk, 0.0]),
                with_generalized_derivative=include_mixed,
            )
            mesh.append((q, (q["E"] < ef).astype(float)))
    return mesh


def calculate_tensor(mesh, omega, eta, include_mixed=False):
    tensor = np.zeros((3, 3, 3), dtype=complex)
    kernel = chi_total_k if include_mixed else chi_k
    for q, occupation in mesh:
        tensor += kernel(q, occupation, omega, eta)
    return tensor / len(mesh)


def calculate_legacy_spectrum(mesh, omega, eta, include_mixed=False):
    """Original frequency-outer evaluation retained as a reference path."""
    return np.array(
        [
            calculate_tensor(mesh, photon_energy, eta, include_mixed)
            for photon_energy in omega
        ]
    )


def calculate_tb_spectrum(
    tb,
    ef,
    nk,
    omega,
    eta,
    include_mixed=False,
    components=None,
    omega_chunk_size=8,
    workers=1,
):
    """Stream k-points through the batched kernel without retaining the mesh."""
    kpoints = [
        np.array([i / nk, j / nk, 0.0])
        for i in range(nk)
        for j in range(nk)
    ]
    kernel = chi_total_k_spectrum if include_mixed else chi_k_spectrum

    def evaluate(kpoint):
        q = at_k(tb, kpoint, with_generalized_derivative=include_mixed)
        occupation = (q["E"] < ef).astype(float)
        return kernel(
            q,
            occupation,
            omega,
            eta,
            components=components,
            omega_chunk_size=omega_chunk_size,
        )

    spectrum = np.zeros((len(omega), 3, 3, 3), dtype=complex)
    if workers == 1:
        results = map(evaluate, kpoints)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(evaluate, kpoints)
    try:
        for result in results:
            spectrum += result
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return spectrum / len(kpoints)


def rotate_spectrum(spectrum, rotation):
    return np.einsum(
        "ai,bj,ck,wijk->wabc",
        rotation,
        rotation,
        rotation,
        spectrum,
        optimize=True,
    )


def plot_spectrum(omega, even_yyy, odd_xxx, output, unit_label, note):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True)
    axes[0].plot(omega, even_yyy.real, color="black", lw=1.5)
    axes[1].plot(omega, odd_xxx.real, color="#b72e3b", lw=1.5)

    axes[0].set_ylabel(
        rf"$\mathrm{{Re}}\,\chi^{{(2)}}_{{\mathrm{{even}},yyy}}$ ({unit_label})"
    )
    axes[1].set_ylabel(
        rf"$\mathrm{{Re}}\,\chi^{{(2)}}_{{\mathrm{{odd}},xxx}}$ ({unit_label})"
    )
    axes[1].set_xlabel(r"Photon energy $\hbar\omega$ (eV)")
    axes[0].text(
        0.98,
        0.92,
        note,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    for axis in axes:
        axis.axhline(0.0, color="0.65", lw=0.7)
        axis.margins(x=0)

    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def plot_plus_spectrum(omega, plus_yyy, plus_xxx, output, unit_label, note):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True)
    axes[0].plot(omega, plus_yyy.real, color="black", lw=1.5)
    axes[1].plot(omega, plus_xxx.real, color="#555555", lw=1.5)
    axes[0].set_ylabel(rf"$\mathrm{{Re}}\,\chi^{{(2)}}_{{+M,yyy}}$ ({unit_label})")
    axes[1].set_ylabel(rf"$\mathrm{{Re}}\,\chi^{{(2)}}_{{+M,xxx}}$ ({unit_label})")
    axes[1].set_xlabel(r"Photon energy $\hbar\omega$ (eV)")
    axes[0].text(
        0.98,
        0.92,
        note,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    for axis in axes:
        axis.axhline(0.0, color="0.65", lw=0.7)
        axis.margins(x=0)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main():
    args = parse_args()
    if args.nk <= 0 or args.n_omega <= 0:
        raise ValueError("--nk and --n-omega must be positive")
    if args.n_omega > 1 and args.omega_max <= args.omega_min:
        raise ValueError("--omega-max must exceed --omega-min")
    if args.workers <= 0 or args.omega_chunk_size <= 0:
        raise ValueError("--workers and --omega-chunk-size must be positive")
    if args.plus_only and args.minus_tb is not None:
        raise ValueError("omit minus_tb when using --plus-only")
    if not args.plus_only and args.minus_tb is None:
        raise ValueError("minus_tb is required unless --plus-only is used")
    if args.cell_height_a is not None and args.cell_volume_a3 is None:
        raise ValueError("--cell-height-a requires --cell-volume-a3")
    plot_units = args.plot_units
    if plot_units is None:
        plot_units = "bulk" if args.cell_volume_a3 is not None else "raw"
    if plot_units == "bulk" and args.cell_volume_a3 is None:
        raise ValueError("bulk plotting requires --cell-volume-a3")
    if plot_units == "sheet" and (
        args.cell_volume_a3 is None or args.cell_height_a is None
    ):
        raise ValueError(
            "sheet plotting requires --cell-volume-a3 and --cell-height-a"
        )

    components = parse_components(args.components)
    if args.legacy_kernel and components is not None:
        raise ValueError("--components is available only with the optimized kernel")
    required_plot_components = {(0, 0, 0), (1, 1, 1)}
    if components is not None and not required_plot_components.issubset(components):
        raise ValueError("--components must include xxx and yyy for the standard plot")
    if components is not None and args.axis_angle_deg != 0.0:
        raise ValueError("component filtering requires --axis-angle-deg 0")

    rotation = rotation_z(args.axis_angle_deg)
    if args.n_omega == 1:
        omega = np.array([args.omega_min])
    else:
        omega = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    plus_tb = read_tb(args.plus_tb)
    if args.legacy_kernel:
        plus_mesh = prepare_mesh(
            plus_tb, args.ef_plus, args.nk, args.include_mixed
        )
        chi_plus_spectrum = calculate_legacy_spectrum(
            plus_mesh, omega, args.eta, args.include_mixed
        )
    else:
        chi_plus_spectrum = calculate_tb_spectrum(
            plus_tb,
            args.ef_plus,
            args.nk,
            omega,
            args.eta,
            include_mixed=args.include_mixed,
            components=components,
            omega_chunk_size=args.omega_chunk_size,
            workers=args.workers,
        )
    chi_plus_spectrum = rotate_spectrum(chi_plus_spectrum, rotation)
    print(f"completed +M on {args.nk}x{args.nk} k mesh")

    chi_minus_spectrum = None
    chi_even_spectrum = None
    chi_odd_spectrum = None
    if not args.plus_only:
        ef_minus = args.ef_plus if args.ef_minus is None else args.ef_minus
        minus_tb = read_tb(args.minus_tb)
        if args.legacy_kernel:
            minus_mesh = prepare_mesh(
                minus_tb, ef_minus, args.nk, args.include_mixed
            )
            chi_minus_spectrum = calculate_legacy_spectrum(
                minus_mesh, omega, args.eta, args.include_mixed
            )
        else:
            chi_minus_spectrum = calculate_tb_spectrum(
                minus_tb,
                ef_minus,
                args.nk,
                omega,
                args.eta,
                include_mixed=args.include_mixed,
                components=components,
                omega_chunk_size=args.omega_chunk_size,
                workers=args.workers,
            )
        chi_minus_spectrum = rotate_spectrum(chi_minus_spectrum, rotation)
        chi_even_spectrum = 0.5 * (chi_plus_spectrum + chi_minus_spectrum)
        chi_odd_spectrum = 0.5 * (chi_plus_spectrum - chi_minus_spectrum)
        print(f"completed -M on {args.nk}x{args.nk} k mesh")

    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        omega_eV=omega,
        chi_plus=chi_plus_spectrum,
        chi_plus_raw_A3_per_eV2=chi_plus_spectrum,
        eta_eV=args.eta,
        nk=args.nk,
        include_mixed=args.include_mixed,
        axis_angle_deg=args.axis_angle_deg,
        plus_only=args.plus_only,
        components=component_labels(components),
        optimized_kernel=not args.legacy_kernel,
        omega_chunk_size=args.omega_chunk_size,
        workers=args.workers,
        warning=(
            "Appendix-C interband"
            + (" plus mixed-band" if args.include_mixed else " only")
            + " response; explicit metallic Fermi-surface/Drude terms, input "
            "provenance, optical-window convergence, and the reference paper's "
            "monolayer normalization convention remain unresolved."
        ),
    )
    responses = {"plus": chi_plus_spectrum}
    if not args.plus_only:
        payload.update(
            chi_minus=chi_minus_spectrum,
            chi_even=chi_even_spectrum,
            chi_odd=chi_odd_spectrum,
            chi_minus_raw_A3_per_eV2=chi_minus_spectrum,
            chi_even_raw_A3_per_eV2=chi_even_spectrum,
            chi_odd_raw_A3_per_eV2=chi_odd_spectrum,
        )
        responses.update(
            minus=chi_minus_spectrum,
            even=chi_even_spectrum,
            odd=chi_odd_spectrum,
        )

    if args.plus_only:
        plot_top = chi_plus_spectrum[:, 1, 1, 1]
        plot_bottom = chi_plus_spectrum[:, 0, 0, 0]
    else:
        plot_top = chi_even_spectrum[:, 1, 1, 1]
        plot_bottom = chi_odd_spectrum[:, 0, 0, 0]
    unit_label = "a.u."
    response_note = "interband + mixed" if args.include_mixed else "three-band only"
    plot_note = f"{response_note}; raw"
    if args.cell_volume_a3 is not None:
        bulk = {
            name: bz_mean_to_bulk_chi_si(value, args.cell_volume_a3)
            for name, value in responses.items()
        }
        payload.update(
            {f"chi_{name}_bulk_m_per_V": value for name, value in bulk.items()}
        )
        payload["cell_volume_A3"] = args.cell_volume_a3
        if plot_units == "bulk":
            top_name = "plus" if args.plus_only else "even"
            bottom_name = "plus" if args.plus_only else "odd"
            plot_top = bulk[top_name][:, 1, 1, 1] * 1.0e12
            plot_bottom = bulk[bottom_name][:, 0, 0, 0] * 1.0e12
            unit_label = "pm/V"
            plot_note = f"{response_note}; supercell norm."

        if args.cell_height_a is not None:
            sheet = {
                name: bulk_chi_to_sheet_si(value, args.cell_height_a)
                for name, value in bulk.items()
            }
            payload.update(
                {
                    f"chi_{name}_sheet_m2_per_V": value
                    for name, value in sheet.items()
                }
            )
            payload["cell_height_A"] = args.cell_height_a
            if plot_units == "sheet":
                top_name = "plus" if args.plus_only else "even"
                bottom_name = "plus" if args.plus_only else "odd"
                plot_top = sheet[top_name][:, 1, 1, 1] * 1.0e18
                plot_bottom = sheet[bottom_name][:, 0, 0, 0] * 1.0e18
                unit_label = r"nm$^2$/V"
                plot_note = f"{response_note}; sheet norm."

    np.savez_compressed(output.with_suffix(".npz"), **payload)
    plotter = plot_plus_spectrum if args.plus_only else plot_spectrum
    plotter(omega, plot_top, plot_bottom, output, unit_label, plot_note)
    if args.cell_volume_a3 is not None:
        bulk_factor = float(bz_mean_to_bulk_chi_si(1.0, args.cell_volume_a3))
        print(f"bulk conversion: {bulk_factor * 1e12:.9f} pm/V per raw unit")
        if args.cell_height_a is not None:
            sheet_factor = float(
                bulk_chi_to_sheet_si(bulk_factor, args.cell_height_a)
            )
            print(
                "sheet conversion: "
                f"{sheet_factor * 1e18:.9f} nm^2/V per raw unit"
            )
    print(f"saved {output.with_suffix('.npz')}")
    print(f"saved {output.with_suffix('.png')}")
    print(f"saved {output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
