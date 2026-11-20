Toroidal and parallel current densities in TokaMaker and IMAS     {#doc_tokamaker_current_conventions}
================

TokaMaker's `jphi-*` profiles and IMAS (FUSE, IMAS.jl, FRESCO) use different, equally valid
definitions of "toroidal current density". This note derives the exact conversions between them
and from the flux-surface-averaged parallel current \f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle \f$, as used by the
bootstrap solver (`solve_bootstrap` / `solve_with_bootstrap`), and the exact plasma current of a profile.

## Setting
Axisymmetric equilibrium, \f$ \mathbf{B} = F(\psi)\nabla\phi + \nabla\psi\times\nabla\phi \f$, with \f$ \psi \f$ the poloidal
flux per radian. The sign of \f$ \psi \f$ is chosen so that the Grad-Shafranov equation reads
\f$ \Delta^*\psi = -\mu_0 R^2 p' - FF' \f$ (\f$ ' = d/d\psi \f$), so that

\f[ j_\phi = R p' + \frac{FF'}{\mu_0 R} \qquad \text{(A1)} \f]

which is TokaMaker's convention (`util.get_jphi_from_GS`). COCOS 11 (IMAS) uses \f$ \psi \f$ per full turn
and a different sign: that changes only the factors \f$ 2\pi \f$ and \f$ \pm 1 \f$ in what follows, never the geometry.
Flux-surface average: \f$ \langle A \rangle = \oint A\, dl/B_p \big/ \oint dl/B_p \f$, so
\f$ V' = dV/d\psi = 2\pi \oint dl/B_p \f$. \f$ F \f$, \f$ F' \f$ and \f$ p' \f$ are flux functions.

## Parallel current
Ampere's law gives \f$ \mathbf{J} = (F'/\mu_0)\nabla\psi\times\nabla\phi + j_\phi R\nabla\phi \f$, so
\f$ \mathbf{J}\cdot\mathbf{B} = (F/R)\, j_\phi + (F'/\mu_0) B_p^2 \f$. With (A1):

\f[ \mathbf{J}\cdot\mathbf{B} = F p' + \frac{F' B^2}{\mu_0}, \quad \text{hence} \quad
    \langle \mathbf{J}\cdot\mathbf{B} \rangle = F p' + \frac{F' \langle B^2 \rangle}{\mu_0} \qquad \text{(A2)} \f]

## The two toroidal conventions
Averaging (A1):

\f[ \text{TokaMaker:} \quad J_{TM} \equiv \langle j_\phi \rangle = \langle R \rangle p' + \langle 1/R \rangle \frac{FF'}{\mu_0} \qquad \text{(A3)} \f]
\f[ \text{IMAS:} \quad J_{IMAS} \equiv \frac{\langle j_\phi/R \rangle}{\langle 1/R \rangle}
    = \frac{p' + \langle 1/R^2 \rangle FF'/\mu_0}{\langle 1/R \rangle} \qquad \text{(A4)} \f]

TokaMaker's jphi \f$ \rightarrow FF' \f$ map (`jphi_update`, `jphi_bs_update`) inverts (A3) exactly. IMAS.jl
`J_tor` and FRESCO `PressureJt` use (A4). On the \f$ FF' \f$ part the two differ by
\f$ \langle 1/R^2 \rangle / \langle 1/R \rangle^2 = 1 + O(\epsilon^2) \f$, largest at the edge.
\f$ J_{IMAS} \f$ is also the current per unit cross-section area, \f$ dI/dA \f$ (A9c).

## IMAS \f$ \leftrightarrow \f$ TokaMaker
Eliminating \f$ FF'/\mu_0 \f$ between (A3) and (A4):

\f[ J_{TM} = \langle R \rangle p' + \langle 1/R \rangle \frac{J_{IMAS}\langle 1/R \rangle - p'}{\langle 1/R^2 \rangle},
    \qquad
    J_{IMAS} = \frac{1}{\langle 1/R \rangle}\left[ p' + \langle 1/R^2 \rangle \frac{J_{TM} - \langle R \rangle p'}{\langle 1/R \rangle} \right]
    \qquad \text{(A5)} \f]

## Parallel \f$ \rightarrow \f$ toroidal
From (A2), \f$ FF'/\mu_0 = F(\langle \mathbf{J}\cdot\mathbf{B} \rangle - F p')/\langle B^2 \rangle \f$. Substituting into (A4):

\f[ J_{IMAS} = \frac{F \langle 1/R^2 \rangle \langle \mathbf{J}\cdot\mathbf{B} \rangle}{\langle B^2 \rangle \langle 1/R \rangle}
    + \frac{p'}{\langle 1/R \rangle}\left(1 - \frac{F^2 \langle 1/R^2 \rangle}{\langle B^2 \rangle}\right) \qquad \text{(A6)} \f]

which is IMAS.jl `JparB_2_JtoR` (with `includes_bootstrap=true`; COCOS 11 adds \f$ -2\pi \f$ to \f$ p' \f$), a check on
the derivation. Substituting into (A3) instead:

\f[ J_{TM} = \frac{F \langle 1/R \rangle \langle \mathbf{J}\cdot\mathbf{B} \rangle}{\langle B^2 \rangle} + p' G, \qquad
    G = \langle R \rangle - \frac{F^2 \langle 1/R \rangle}{\langle B^2 \rangle} \qquad \text{(A7)} \f]

## Components
(A6) and (A7) are exact for the total current with the equilibrium's own \f$ F \f$ and averages, and at
fixed geometry they are linear in \f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle \f$. So for a split
\f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle = \sum_k \langle \mathbf{J}\cdot\mathbf{B} \rangle_k \f$ (ohmic, bootstrap,
current drive, ...) each component converts with the first term alone, and the single pressure term
is assigned to one component; following IMAS.jl (`includes_bootstrap=true`), to the bootstrap.
Building \f$ J_{TM} \f$ this way guarantees

\f[ \langle \mathbf{J}\cdot\mathbf{B} \rangle_{\text{equilibrium}} = F p' + \frac{F' \langle B^2 \rangle}{\mu_0}
    = \sum_k \langle \mathbf{J}\cdot\mathbf{B} \rangle_k \qquad \text{(A8)} \f]

The pressure term is the pressure-driven, non-field-aligned current (diamagnetic plus
Pfirsch-Schlueter, the latter with \f$ \langle \mathbf{j}_{PS}\cdot\mathbf{B} \rangle = 0 \f$). Neoclassical bootstrap
and current-drive models give the field-aligned part as \f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle \f$ (Hirshman & Sigmar 1981;
Sauter et al. 1999; Redl et al. 2021), which is why it converts with the first term only. For
\f$ B_p \to 0 \f$, \f$ G \to \langle R \rangle - \langle 1/R \rangle / \langle 1/R^2 \rangle \ge 0 \f$.

Earlier bootstrap conversions used \f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle \langle R \rangle / F \f$, then
\f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle / \langle |B| \rangle \f$. For \f$ B_p \to 0 \f$ these exceed the first term of (A7) by
\f$ \langle R \rangle \langle 1/R^2 \rangle / \langle 1/R \rangle \f$ and \f$ \langle 1/R^2 \rangle / \langle 1/R \rangle^2 \f$
respectively, and both omit \f$ p'G \f$.

## Plasma current
The cross-section element between neighbouring surfaces is \f$ dA = dl\, d\psi/|\nabla\psi| = dl\, d\psi/(R B_p) \f$,
so for any flux function \f$ f \f$

\f[ \int f\, dA = \int d\psi\, f \oint \frac{dl}{R B_p} = \int d\psi\, \frac{V'}{2\pi} f \langle 1/R \rangle \qquad \text{(A9a)} \f]

and for the toroidal current

\f[ I_p = \int j_\phi\, dA = \int d\psi \oint \frac{j_\phi}{R} \frac{dl}{B_p} = \int d\psi\, \frac{V'}{2\pi} \langle j_\phi/R \rangle \qquad \text{(A9b)} \f]

By (A4), \f$ \langle j_\phi/R \rangle = \langle 1/R \rangle J_{IMAS} \f$, so comparing with (A9a)

\f[ I_p = \int J_{IMAS}\, dA \quad \text{(exact; } J_{IMAS} \text{ from } J_{TM} \text{ by (A5))} \qquad \text{(A9c)} \f]
\f[ I_p = \int d\psi\, \frac{V'}{2\pi} \left[ J_{TM} \frac{\langle 1/R^2 \rangle}{\langle 1/R \rangle}
    + p'\left(1 - \frac{\langle R \rangle \langle 1/R^2 \rangle}{\langle 1/R \rangle}\right) \right] \qquad \text{(A9d)} \f]

TokaMaker evaluates (A9c) as `gs_flux_int` of `eval_dIdA`, which is (A9a) on the plasma region.
The integral covers only the region inside the LCFS. The inductive scale \f$ \alpha \f$ of `jphi_bs_update` is set
by (A9c) alone (converged \f$ I_p \f$ within a few \f$ 10^{-4} \f$ of the target), and so is the jphi scale of
`jphi_update` (`jphi-linterp`) at every iteration (converged \f$ I_p \f$ within a few \f$ 10^{-6} \f$ of the target on the
ITER example). To get a current from a jphi profile, integrate \f$ dI/dA = J_{IMAS} \f$ (`util.get_dIdA_from_jphi`),
not \f$ J_{TM} \f$.

## Bootstrap profile names
`solve_bootstrap`, `get_boot_profs` and `solve_with_bootstrap` return \f$ J_{TM} \f$ profiles on `psi_n` (0 = axis,
1 = LCFS) named `jphi_<part>` (as it enters `jphi_total`) or `jphi_<part>_raw` (before edge isolation,
scaling and taper). Parts: `bs` (bootstrap, the field-aligned term of (A7)), `pd` (its pressure-driven \f$ p'G \f$
term), `ind` (inductive), `fixed`, `saw`: `jphi_total` = `jphi_ind` + `jphi_bs` + `jphi_pd` + `jphi_fixed` +
`jphi_saw`. The edge isolation and `scale_jbs` act on `bs` only. The Python solve (`use_python_solve=True`)
returns the GS output as `jphi_total`. `jdotB_bs_raw` is the bootstrap \f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle \f$
behind `jphi_bs_raw`. Options use `jbs` (`scale_jbs`, `isolate_edge_jbs`, `taper_edge_jphi`, `djbs_tol`).

## Bootstrap model of the internal solver
At each nonlinear iteration `solve_bootstrap` evaluates the Redl et al. (2021) fits (or those of Sauter et al.
(1999), `bs_model` below) in the `jboot1` form of Sauter et al. (1999) on the profile nodes (Fortran `bootstrap_eval`). The trapped fraction,
\f$ q \f$, the inverse aspect ratio \f$ \epsilon = (R_{max}-R_{min})/(R_{max}+R_{min}) \f$ and the major radius
\f$ (R_{max}+R_{min})/2 \f$ of each surface come from one contour pass (`sauter_fc`); the collisionalities use \f$ |q| \f$, and
by default \f$ \nu_{i*} \f$ the charge factor \f$ Z^4 = (n_e/n_i) Z_{eff} \f$ (`ion_charge_model` below). The kinetic gradients are taken on the nodes without the
LCFS, and the values on the LCFS (and on the axis) are extrapolated. `eval_bootstrap` runs the same code on the
current equilibrium at given surfaces and kinetic profiles, without solving and without changing the equilibrium,
and returns `jdotB_bs_raw`, `jphi_bs_raw`, `jphi_pd_raw`, \f$ f_t \f$, \f$ \nu_{e*} \f$ and \f$ \nu_{i*} \f$, and the
geometry of its flux-surface pass (\f$ \epsilon \f$, \f$ (R_{max}+R_{min})/2 \f$, \f$ q \f$, \f$ \langle R \rangle \f$,
\f$ \langle 1/R \rangle \f$, \f$ \langle 1/R^2 \rangle \f$, \f$ \langle |B| \rangle \f$, \f$ \langle |B|^2 \rangle \f$), equal to
what `calc_sauter_fc` returns on the same surfaces, so a caller that needs both makes one pass. Four options
(`set_boot_ops`, `solve_bootstrap`, `eval_bootstrap`) act on both:
- `bs_model`: the fits of the four coefficients \f$ L_{31} \f$, \f$ L_{32} \f$, \f$ L_{34} \f$ and \f$ \alpha \f$; everything
  else (collisionalities, Coulomb logarithms, geometry, the `jboot1` assembly, the conversion to \f$ j_\phi \f$) is
  shared. 1 (default): Redl et al. (2021) eqs. 10-21 with \f$ L_{34} = L_{31} \f$. 2: Sauter et al. (1999) eqs.
  14-17 with the corrections of the 2002 erratum: the \f$ 0.315\,\nu_{i*}^2 f_t^6 \f$ term of \f$ \alpha \f$ (eq. 17b) is
  positive, and the ion-temperature term of the bootstrap formula is
  \f$ L_{34}\,\alpha\,(1-R_{pe})/R_{pe}\; d\ln T_i/d\psi \f$, as in the shared assembly. Redl et al. fitted the same
  forms to NEO drift-kinetic calculations and present them as the more accurate set, mainly at the finite
  collisionality of the edge; Sauter et al. (1999) is the earlier set that many codes and published results use,
  for comparison with them. In the banana regime the two \f$ L_{31} \f$ differ by up to about 0.08 (of at most 1);
  at finite collisionality and \f$ Z_{eff} > 1 \f$ the fits differ more, most in \f$ \alpha \f$ (Sauter's has no
  \f$ Z_{eff} \f$ dependence; as \f$ \nu_{i*} \to \infty \f$, \f$ \alpha \to 2.1 \f$ (Sauter) and \f$ -0.5 \f$ (Redl)).
  With 2 only the coefficient fits change; the collisionalities keep the solver's conventions above. The published
  model of Sauter et al. (1999) with its erratum is `bs_model`, `ln_lambda_model` and `ion_charge_model` all 2.
  Sauter's own Coulomb logarithms (eqs. 18d and 18e) are `ln_lambda_model` = 2; against the default NRL electron
  logarithm they change \f$ \langle j\cdot B \rangle \f$ by about \f$ 10^{-3} \f$ of its peak on the ITER test case
  and by up to 3 % in collisional cases.
- `ln_lambda_model`: 1 (default) uses the NRL formulary electron logarithm and the ion logarithm of Sauter et al.
  (1999) eq. 18e, both floored at 10; 2 uses Sauter et al. (1999) eqs. 18d and 18e, without a floor. The ion
  logarithm takes the \f$ Z \f$ of `ion_charge_model`.
- `ion_charge_model`: the ion charge \f$ Z \f$ in \f$ \nu_{i*} \f$ (as \f$ Z^4 \f$, eq. 18c) and in \f$ \ln\Lambda_{ii} \f$
  (eq. 18e). 1 (default): the Koh multi-species form, \f$ Z^4 = (n_e/n_i) Z_{eff} \f$ and \f$ Z = \max(n_e/n_i, 1) \f$.
  2: the main-ion charge \f$ Z_i \f$ in both, as the 2002 erratum prescribes (\f$ Z_{eff} \f$ stays in \f$ \nu_{e*} \f$ and
  in the coefficients). The solver's main ions have charge 1 (as `imp_pressure` assumes), so 2 sets \f$ Z = 1 \f$;
  with \f$ Z_{eff} = 1 \f$ and \f$ n_i = n_e \f$ the two are the same. On the ITER test case with carbon,
  2 changes \f$ \langle j\cdot B \rangle \f$ by 0.9 % (\f$ Z_{eff} = 1.5 \f$) to 1.8 % (\f$ Z_{eff} = 4 \f$) of its peak
  with the Sauter coefficients and by 2.4 % to 12 % with the Redl coefficients, and by up to 11 % (Sauter) and
  40 % (Redl) in collisional cases.
- `imp_pressure`: the ion pressure and its gradient count \f$ n_{imp} = (n_e - n_i)^2/(Z_{eff} n_e - n_i) \f$ at
  \f$ T_i \f$, the density of the one impurity species that, with main ions of charge 1, gives quasineutrality and
  \f$ Z_{eff} \f$. The collisionalities and Coulomb logarithms keep the main-ion density. `eval_bootstrap` also takes
  an impurity density (`nimp`), e.g. the sum over several species.

`bootstrap.redl_bootstrap` is the same formula as an array function. With `formula_form='jboot1'`,
`use_legacy_L34=False`, `use_sign_q=False`, `ion_collisionality_model='Zavg_Zeff'`, \f$ |q| \f$ and the logarithms of
`calculate_ln_lambda(..., 'NRL', 'Zavg')` (or `'sauter'`, whose floor of 10 is rarely reached) it equals
`eval_bootstrap` node by node to round-off, given the same geometry and gradients (`test_TokaMaker_bootstrap_eval.py`).
Its own defaults differ (`'jB'`, \f$ Z_{eff}^4 \f$, \f$ \ln\Lambda = 17 \f$). Its `coefficient_model` (`'redl'`, `'sauter'`)
is `bs_model` (1, 2), and `bootstrap.sauter_coefficients` returns the Sauter coefficients alone; `use_legacy_L34`
is a variant of the Redl fits and is rejected with `'sauter'`. `ion_charge_model` = 2 is
`ion_collisionality_model='unity'` with `calculate_ln_lambda(..., ion_lnLambda_model='unity')`.

## Where these are used
- TokaMaker `jphi_bs_update` / `bootstrap_eval` (Fortran), `eval_bootstrap` and `solve_with_bootstrap(use_python_solve=True)`:
  the bootstrap enters `jphi_total` by (A7) with \f$ p'G \f$ (`boot_profs['jphi_pd_raw']`); `boot_profs['jdotB_bs_raw']`
  holds the \f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle \f$ of `bs_model`.
- TokaMaker `jphi_update` / `jphi_bs_update` \f$ I_p \f$ normalisation, the \f$ \alpha \f$ root of
  `solve_with_bootstrap(use_python_solve=True)`, and `util.get_dIdA_from_jphi`: (A9c).

## References
- J. P. Freidberg, *Ideal MHD* (Cambridge University Press, 2014): Grad-Shafranov equilibrium; (A1), (A2).
- J. Wesson, *Tokamaks*, 4th ed. (Oxford University Press, 2011): equilibrium and flux-surface averages.
- S. P. Hirshman and D. J. Sigmar, Nucl. Fusion **21**, 1079 (1981): neoclassical currents as
  \f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle \f$; Pfirsch-Schlueter current with zero \f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle \f$.
- O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas **6**, 2834 (1999); A. Redl et al.,
  Phys. Plasmas **28**, 022502 (2021): bootstrap current as \f$ \langle j_\parallel B \rangle \f$; collisionalities and
  Coulomb logarithms (S99 eqs. 18b-18e); coefficient fits (`bs_model` 2: S99 eqs. 14-17; 1: Redl eqs. 10-21).
- O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas **9**, 5140 (2002): erratum to S99 (+ sign of the
  \f$ 0.315\,\nu_{i*}^2 f_t^6 \f$ term of \f$ \alpha \f$ in eq. 17b; main-ion charge in eqs. 18c and 18e; the
  \f$ (1-R_{pe})/R_{pe} \f$ factor of the ion-temperature term of the bootstrap formula in the conclusion).
- OMFIT `omfit_classes.utils_fusion.sauter_bootstrap` (`version='jboot1'`): public implementation used as the
  reference values of `test_TokaMaker_bootstrap_eval.py`.
- J. D. Huba, *NRL Plasma Formulary* (Naval Research Laboratory): electron Coulomb logarithm (`ln_lambda_model` 1).
- O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. **184**, 293 (2013): COCOS (IMAS signs and \f$ 2\pi \f$).
- IMAS.jl `src/physics/currents.jl`, `JparB_2_JtoR`:
  \f$ \langle \mathbf{J}\cdot\mathbf{B} \rangle = \frac{1}{f}\frac{\langle B^2 \rangle}{\langle 1/R^2 \rangle}\left(\langle J_t/R \rangle + \frac{dp}{d\psi}\left(1 - \frac{f^2\langle 1/R^2 \rangle}{\langle B^2 \rangle}\right)\right) \f$, i.e. (A6).
