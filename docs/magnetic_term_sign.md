# Sign of the magnetic mixed SHG term

The last term of the mixed length-gauge susceptibility in [Chen et al.'s
arXiv v1](https://arxiv.org/html/2110.13415v1) has opposite signs in its
main-text Eq. (16) and Appendix C. [Xiao et al.'s Appendix C](https://arxiv.org/pdf/2211.04354v1)
uses Chen's appendix sign. Earlier versions of this code implemented that
printed appendix sign.

For SHG, setting the two input-frequency ratios to 1/2 in Chen Eq. (15)
gives a **positive** fifth contribution proportional to

\[
\frac{i e^3}{4\hbar^2}\sum_{nm}\int[d\mathbf{k}]\,
f_{nm}\frac{\Delta^a_{mn}g^{bc}_{nm}}
{\omega_{mn}^2(\omega_{mn}-\omega)},
\qquad
g^{bc}_{nm}=\operatorname{Re}(r^b_{nm}r^c_{mn}).
\]

Chen Eq. (16) has this sign; its Appendix C prints the negative. Evaluating
Chen's unrearranged Eq. (14) on a gapped four-band toy model agrees with
Eq. (16) in the narrow-width limit, both with and without the Wannier
position matrix. A separate velocity-gauge calculation using only the toy
Hamiltonian and its k derivatives agrees with the sign in Eq. (16) for all
six tested in-plane tensor components. These checks favor the main-text sign
and motivate the correction in both the single-frequency and batched kernels.

This audit concerns the fifth magnetic mixed term. It does not validate a
metallic Fermi-surface contribution, a particular DFT/Wannier model, or every
finite-broadening prescription. The published PRB version of Chen's paper was
not available for this audit; it may differ from the arXiv version. Existing
spectra computed with the old code should be regenerated before comparison.
