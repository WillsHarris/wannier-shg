# Wannier SHG

Calculate the complex second harmonic susceptibility from a Wannier90 tight binding model. The input is a `seedname_tb.dat` file containing both the Hamiltonian and position matrix. The code evaluates the full Brillouin zone and can combine separate (+M) and (-M) models into magnetization even and odd responses.

The established calculation implements the three band and mixed two band length gauge terms in Appendix C of [Xiao *et al.*](https://arxiv.org/abs/2211.04354v1). A separate metallic response runner includes Fermi occupation derivatives following [Bhalla *et al.*](https://doi.org/10.1103/PhysRevB.107.165131). That extension is still being checked against an independent full multiband formula; its output should be treated as a validation result.

## Run it

Tested with Python 3.9.6 and the versions in `requirements.txt`. The scripts are run from this directory; no installation step is required.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
python examples/make_magnetic_pair.py
python scripts/metallic_shg_spectrum.py \
  examples/data/plus_tb.dat examples/data/minus_tb.dat \
  --ef-plus 0.67 --ef-minus 0.67 --nk 16 \
  --omega-min 0.4 --omega-max 0.5 --n-omega 11 \
  --eta-intra 0.05 --eta-inter 0.05 --kbt 0.02 \
  --cell-volume-a3 236.16 --cell-height-a 18 \
  --out examples/data/demo
```

The example writes a small synthetic magnetic pair, then a CSV of complex sheet susceptibilities in nm²/V. It is a code check, not a material prediction. The `full` columns are the metallic runner's complete result. Its `frozen` and `fermi_difference` columns are diagnostics; the difference should not be added to an existing spectrum from the Appendix C runner.

For a Wannier90 model, pass its `_tb.dat` path and the Fermi energy in eV. Use `scripts/spectrum_incremental.py --help` for the established Appendix C response, or `scripts/metallic_shg_spectrum.py --help` for the metallic validation runner. Both accept paired magnetization models; `--components xxx,yyy` limits the tensor components evaluated. The cell volume and height determine the conversion to sheet units.

The core is in `scripts/wann_tb.py` (Wannier interpolation), `scripts/shg.py` (Appendix C response), and `scripts/metallic_shg.py` (metallic response). The spectrum scripts handle mesh integration and output; `scripts/plot_threeband_spectrum.py` also plots an Appendix C spectrum.

The Wannier basis rotations use optimized NumPy contractions. On a synthetic 56-band model, this reduced the interpolation time per k point from about 0.56 s to 0.005 s on one Mac with one BLAS thread; the calculated response agreed to floating point precision. Timings for a real model will depend on its size and hardware. Set `OPENBLAS_NUM_THREADS=1` for a reproducible single-thread benchmark.

## Scope and checks

- The Appendix C calculation includes the three band and mixed two band terms, with a full zone sum and no time reversal reduction. Its occupations are step functions and its broadening is constant.
- The metallic runner uses finite temperature occupations. Its `adiabatic` width choice matches the Appendix C complex frequency prescription in the insulating limit; `relaxation` allows separate intraband and interband widths. The scattering width is an input parameter.
- The tests check direct band sums, generalized derivatives, gauge changes, unit conversion, the insulating limit, and the metallic response on a synthetic magnetic model. The tests do not establish convergence for a particular DFT calculation.

Before using the metallic output quantitatively, compare a short run with the same inputs before and after the interpolation change, then check k-mesh, temperature, and intraband-width sensitivity for both magnetizations. An independent comparison of the complete multiband metallic response with the formulas of Hipolito *et al.* is still needed. Keep the `adiabatic` and `relaxation` width prescriptions distinct.

For a trilayer versus monolayer comparison, use matched DFT and optical settings wherever possible. Document any necessary changes to the vacuum, band count, or Wannier basis. If the trilayer needs a different exchange-correlation or vdW treatment, recalculate the monolayer comparator with that treatment too.

Wannier90 documents the [`_tb.dat` format](https://wannier90.readthedocs.io/en/stable/user_guide/wannier90/files/). For the metallic length gauge formulation and the physical nonlinear Drude terms, see [Hipolito *et al.*](https://doi.org/10.1103/PhysRevB.98.205420) and [Bhalla *et al.*](https://doi.org/10.1103/PhysRevB.107.165131).

The code is available under the [MIT License](LICENSE).
