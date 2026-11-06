#!/usr/bin/env python3
"""Checkpointed full-BZ metallic SHG for paired Wannier models.

This is a validation runner for metallic_shg.py, not a replacement for the
published Xiao C2+C3 spectrum. Use the adiabatic prescription to compare with
that spectrum; use relaxation widths for a separate scattering-rate study.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metallic_shg import band_hessian, kinetic_chi_k
from shg import bulk_chi_to_sheet_si, bz_mean_to_bulk_chi_si
from wann_tb import at_k, read_tb


def parse_components(value):
    result = []
    for label in value.lower().split(","):
        label = label.strip()
        if len(label) != 3 or any(axis not in "xyz" for axis in label):
            raise ValueError("components must be comma-separated labels, e.g. xxx,yyy")
        component = tuple("xyz".index(axis) for axis in label)
        if component not in result:
            result.append(component)
    return tuple(result)


def save_checkpoint(path, settings, completed, accumulated):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, settings=json.dumps(settings, sort_keys=True),
                 completed=completed, accumulated=accumulated)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_mesh(tb_path, ef, nk, omega, eta_intra, eta_inter, kbt, components,
             scheme, checkpoint, checkpoint_every=32):
    """Return BZ averages [frequency, response_kind, a, b, c]."""
    source = Path(tb_path).resolve()
    stat = source.stat()
    settings = dict(source=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    ef=ef, nk=nk, omega=list(map(float, omega)),
                    eta_intra=eta_intra, eta_inter=eta_inter, kbt=kbt,
                    components=[list(c) for c in components], scheme=scheme)
    accumulated = np.zeros((len(omega), 3, 3, 3, 3), complex)
    completed = 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["settings"]) != json.dumps(settings, sort_keys=True):
                raise ValueError(f"checkpoint settings/source changed: {checkpoint}")
            completed = int(saved["completed"])
            accumulated = np.array(saved["accumulated"], copy=True)
        if not 0 <= completed <= nk * nk or accumulated.shape != (len(omega), 3, 3, 3, 3):
            raise ValueError(f"corrupt checkpoint: {checkpoint}")
    tb = read_tb(source)
    for index in range(completed, nk * nk):
        # The midpoint mesh avoids pathological sampling exactly on degeneracies.
        kf = np.array([(index // nk + 0.5) / nk,
                       (index % nk + 0.5) / nk, 0.0])
        q = at_k(tb, kf, with_generalized_derivative=True)
        hessian = band_hessian(tb, kf, q)
        for iw, w in enumerate(omega):
            accumulated[iw] += np.stack(kinetic_chi_k(
                q, hessian, ef, float(w), eta_intra, eta_inter, kbt,
                components=components, broadening_scheme=scheme,
            ))
        if (index + 1) % checkpoint_every == 0 or index + 1 == nk * nk:
            save_checkpoint(checkpoint, settings, index + 1, accumulated)
            print(f"{source.name}: {index + 1}/{nk * nk} k points", flush=True)
    return accumulated / (nk * nk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plus_tb")
    parser.add_argument("minus_tb")
    parser.add_argument("--ef-plus", type=float, required=True)
    parser.add_argument("--ef-minus", type=float, required=True)
    parser.add_argument("--nk", type=int, default=24)
    parser.add_argument("--omega-min", type=float, default=0.1)
    parser.add_argument("--omega-max", type=float, default=1.9)
    parser.add_argument("--n-omega", type=int, default=181)
    parser.add_argument("--eta-intra", type=float, required=True)
    parser.add_argument("--eta-inter", type=float, default=0.05)
    parser.add_argument("--kbt", type=float, required=True,
                        help="thermal energy k_B T in eV")
    parser.add_argument("--scheme", choices=("adiabatic", "relaxation"),
                        default="adiabatic")
    parser.add_argument("--components", default="xxx,yyy")
    parser.add_argument("--cell-volume-a3", type=float, required=True)
    parser.add_argument("--cell-height-a", type=float, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.nk < 1 or args.n_omega < 1 or args.omega_min <= 0 or args.omega_max < args.omega_min:
        parser.error("invalid mesh or photon-energy range")
    if args.kbt <= 0 or args.eta_intra <= 0 or args.eta_inter <= 0:
        parser.error("kbt and widths must be positive")
    if args.checkpoint_every < 1:
        parser.error("checkpoint-every must be positive")
    if args.scheme == "adiabatic" and not np.isclose(args.eta_intra, args.eta_inter):
        parser.error("adiabatic switching requires eta-intra = eta-inter")
    components = parse_components(args.components)
    omega = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    plus = run_mesh(args.plus_tb, args.ef_plus, args.nk, omega,
                    args.eta_intra, args.eta_inter, args.kbt, components,
                    args.scheme, out.with_name(out.name + "_plus.npz"),
                    args.checkpoint_every)
    minus = run_mesh(args.minus_tb, args.ef_minus, args.nk, omega,
                     args.eta_intra, args.eta_inter, args.kbt, components,
                     args.scheme, out.with_name(out.name + "_minus.npz"),
                     args.checkpoint_every)

    def sheet(raw):
        return 1e18 * bulk_chi_to_sheet_si(
            bz_mean_to_bulk_chi_si(raw, args.cell_volume_a3), args.cell_height_a)

    responses = dict(plus=sheet(plus), minus=sheet(minus),
                     even=sheet((plus + minus) / 2),
                     odd=sheet((plus - minus) / 2))
    output = Path(str(out) + ".csv")
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        columns = ["omega_eV"]
        for tag in responses:
            for kind in ("full", "frozen", "fermi_difference"):
                for component in components:
                    label = "".join("xyz"[i] for i in component)
                    columns.extend((f"Re_{tag}_{kind}_{label}_nm2V",
                                    f"Im_{tag}_{kind}_{label}_nm2V"))
        writer.writerow(columns)
        for iw, w in enumerate(omega):
            row = [float(w)]
            for response in responses.values():
                for ikind in range(3):
                    for component in components:
                        value = response[(iw, ikind) + component]
                        row.extend((value.real, value.imag))
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    metadata = dict(plus_tb=str(Path(args.plus_tb).resolve()),
                    minus_tb=str(Path(args.minus_tb).resolve()),
                    ef_plus=args.ef_plus, ef_minus=args.ef_minus,
                    nk=args.nk, kbt=args.kbt, eta_intra=args.eta_intra,
                    eta_inter=args.eta_inter, scheme=args.scheme,
                    components=args.components, cell_volume_a3=args.cell_volume_a3,
                    cell_height_a=args.cell_height_a, seconds=time.time() - started)
    Path(str(out) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"DONE {output}", flush=True)


if __name__ == "__main__":
    main()
