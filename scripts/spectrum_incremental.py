#!/usr/bin/env python3
"""Even/odd SHG spectrum with restartable k-point checkpointing.

Builds the +/-M k-meshes once and evaluates the complete photon-energy grid for
each k-point. This reuses all frequency-independent band quantities and dipole
products. Partial tensor sums are checkpointed by k-point batch so interrupted
runs can resume without repeating the expensive response calculation.

Usage:
  spectrum_incremental.py PLUS_TB MINUS_TB --ef EF --nk 96 \
     --omega-min 0.1 --omega-max 1.7 --n-omega 161 --eta 0.05 \
     --cell-volume-a3 V --cell-height-a H --out PREFIX
"""
import argparse, csv, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shg import (
    at_k,
    bulk_chi_to_sheet_si,
    bz_mean_to_bulk_chi_si,
    chi_total_k_spectrum,
    magnetization_even_odd,
    read_tb,
)

CART = "xyz"


def parse_components(value):
    """Parse comma-separated Cartesian tensor labels such as ``xxx,yyy``."""
    if value is None:
        return tuple(
            (i, j, k)
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        )
    axis = {label: index for index, label in enumerate(CART)}
    components = []
    for label in value.split(","):
        label = label.strip().lower()
        if len(label) != 3 or any(character not in axis for character in label):
            raise ValueError(
                "--components entries must be three-letter labels such as xxx or yyy"
            )
        component = tuple(axis[character] for character in label)
        if component not in components:
            components.append(component)
    if not components:
        raise ValueError("--components must not be empty")
    return tuple(components)


def build_mesh_threaded(tb, ef, nk, workers):
    """Same mesh as prepare_mesh but at_k is evaluated over k-points in parallel
    (the serial version is ~10x slower and was the real bottleneck)."""
    kpts = [np.array([i / nk, j / nk, 0.0]) for i in range(nk) for j in range(nk)]

    def one(k):
        q = at_k(tb, k, with_generalized_derivative=True)
        occupation = (q["E"] < ef).astype(float)
        return (q, occupation)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, kpts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plus_tb"); p.add_argument("minus_tb")
    p.add_argument("--ef", type=float, required=True)
    p.add_argument("--ef-minus", type=float)
    p.add_argument("--nk", type=int, default=96)
    p.add_argument("--omega-min", type=float, default=0.1)
    p.add_argument("--omega-max", type=float, default=1.7)
    p.add_argument("--n-omega", type=int, default=161)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--cell-volume-a3", type=float, required=True)
    p.add_argument("--cell-height-a", type=float, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--checkpoint-kpoints", type=int, default=64)
    p.add_argument(
        "--components",
        help=(
            "comma-separated tensor components to calculate, e.g. xxx,yyy; "
            "default: all eight in-plane components"
        ),
    )
    p.add_argument("--out", required=True, help="output prefix")
    a = p.parse_args()
    ef_m = a.ef if a.ef_minus is None else a.ef_minus
    omega = np.linspace(a.omega_min, a.omega_max, a.n_omega)

    def sheet(ten):  # raw BZ-mean tensor -> sheet nm^2/V
        return bulk_chi_to_sheet_si(bz_mean_to_bulk_chi_si(ten, a.cell_volume_a3),
                                    a.cell_height_a) * 1e18

    t0 = time.time()
    print(f"building +M mesh nk={a.nk} (threaded, {a.workers} workers) ...", flush=True)
    mesh_p = build_mesh_threaded(read_tb(a.plus_tb), a.ef, a.nk, a.workers)
    print(f"  +M mesh done ({time.time()-t0:.0f}s); building -M mesh ...", flush=True)
    mesh_m = build_mesh_threaded(read_tb(a.minus_tb), ef_m, a.nk, a.workers)
    print(f"  -M mesh done ({time.time()-t0:.0f}s). Evaluating spectra ...", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    comps = [(a_, b_, c_) for a_ in range(3) for b_ in range(3) for c_ in range(3)]
    header = ["omega_eV"]
    for tag in ("even", "odd"):
        for (i, j, k) in comps:
            cc = CART[i] + CART[j] + CART[k]
            header += [f"Re_{tag}_{cc}_nm2V", f"Im_{tag}_{cc}_nm2V"]

    components = parse_components(a.components)
    component_labels = ",".join("".join(CART[i] for i in item) for item in components)
    print(f"tensor components: {component_labels}", flush=True)

    CHUNK = 8
    if a.checkpoint_kpoints <= 0:
        raise ValueError("--checkpoint-kpoints must be positive")

    def save_checkpoint(path, completed, accumulator, source):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez(
                handle,
                completed=completed,
                accumulator=accumulator,
                omega=omega,
                eta=a.eta,
                nk=a.nk,
                source=str(source),
                components=component_labels,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def load_checkpoint(path, source):
        if not path.exists():
            return 0, np.zeros((len(omega), 3, 3, 3), dtype=complex)
        with np.load(path, allow_pickle=False) as saved:
            if (
                int(saved["nk"]) != a.nk
                or float(saved["eta"]) != a.eta
                or str(saved["source"]) != str(source)
                or "components" not in saved.files
                or str(saved["components"]) != component_labels
                or not np.array_equal(saved["omega"], omega)
            ):
                raise ValueError(f"checkpoint settings do not match: {path}")
            completed = int(saved["completed"])
            accumulator = np.array(saved["accumulator"], copy=True)
        if completed < 0 or completed > a.nk * a.nk:
            raise ValueError(f"invalid completed k-point count in {path}")
        return completed, accumulator

    def spectrum(mesh, source, label):
        checkpoint = out.with_name(out.name + f"_{label}_checkpoint.npz")
        completed, accumulator = load_checkpoint(checkpoint, source)
        if completed:
            print(
                f"  {label}: resuming at k-point {completed}/{len(mesh)}",
                flush=True,
            )

        def evaluate(qo):
            return chi_total_k_spectrum(
                qo[0],
                qo[1],
                omega,
                a.eta,
                components=components,
                omega_chunk_size=CHUNK,
            )

        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            parts = executor.map(evaluate, mesh[completed:])
            for index, part in enumerate(parts, start=completed + 1):
                accumulator += part
                if index % a.checkpoint_kpoints == 0 or index == len(mesh):
                    save_checkpoint(checkpoint, index, accumulator, source)
                    print(
                        f"  {label}: {index}/{len(mesh)} k-points "
                        f"({time.time()-t0:.0f}s)",
                        flush=True,
                    )
        return accumulator / len(mesh)

    chi_plus = spectrum(mesh_p, a.plus_tb, "plus")
    chi_minus = spectrum(mesh_m, a.minus_tb, "minus")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index, wv in enumerate(omega):
            even, odd = magnetization_even_odd(
                chi_plus[index], chi_minus[index]
            )
            even_s, odd_s = sheet(even), sheet(odd)
            row = [f"{wv:.5f}"]
            for tensor in (even_s, odd_s):
                for i, j, k in comps:
                    value = tensor[i, j, k]
                    row += [f"{value.real:.8e}", f"{value.imag:.8e}"]
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    print(f"DONE -> {csv_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
