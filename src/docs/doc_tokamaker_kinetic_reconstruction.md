Kinetic equilibrium reconstruction in TokaMaker     {#doc_tokamaker_kinetic_reconstruction}
================

[TOC]

The \ref OpenFUSIONToolkit.TokaMaker.reconstruction.kinetic_reconstruction "kinetic_reconstruction" class fits one
time slice of a tokamak discharge: it finds the free-boundary equilibrium whose current profile follows from a
current model driven by the measured kinetic profiles, and whose magnetic signals best match the measurements. It
sits beside the \ref OpenFUSIONToolkit.TokaMaker.reconstruction.reconstruction "reconstruction" class (whose
constraint types are described in \ref doc_tokamaker_recon_constraints) and is a thin layer over the helper package
`OpenFUSIONToolkit.TokaMaker.kinetic`, which holds the measurement containers, the synthetic diagnostics, the
current model, the anchored forward solve and the fit. The class adds no computation of its own: a fit through it
gives exactly the result of calling `kinetic.fit.reconstruct` with the same inputs.

An example on a synthetic DIII-D-like case is \ref doc_tMaker_DIIID_ex5.

## What is solved for, and from which data
The unknowns are a few physical parameters of the current model (table below), not free coefficients of
\f$ F F' \f$ and \f$ P' \f$. For each trial value the current model and the free-boundary solve are iterated to
consistency ("consistent solve"), and the parameters are found as the maximum a posteriori (MAP) point of

\f[ \chi^2 = \sum_i \left( \frac{w_i (m_i - v_i)}{\sigma_i} \right)^2 + \sum_j \left( \frac{\theta_j - \mu_j}{s_j} \right)^2 \f]

with the modeled signal \f$ m_i \f$, the measured value \f$ v_i \f$, its uncertainty \f$ \sigma_i \f$ and weight
\f$ w_i \f$, and a Gaussian prior \f$ (\mu_j, s_j) \f$ per free parameter. The minimum is found by Levenberg-Marquardt
with forward-difference Jacobian columns, each a full consistent solve. The fit stops only through stated tests on a
fresh Jacobian (the undamped Gauss-Newton step below 0.1 posterior standard deviations, or the predicted decrease
below a floor set by the measured solver noise, or small relative decreases); the uncertainties are the Laplace
covariance at the final point. Every prior, bound, step rule and stopping test is a field of `kinetic.fit.FitSettings`
and is written to the fit record.

The rows of \f$ \chi^2 \f$ (groups of `kinetic.interfaces.MeasurementSet`, named after the IMAS data dictionary):

| Group | Rows used in the fit |
|---|---|
| `flux_loop` | every valid flux loop, in the measurement's own gauge (absolute, or relative to a reference loop) |
| `bpol_probe` | every valid poloidal-field probe, length-averaged when the probe has a length |
| `pf_current` | coils with the role `'fitted'` (their total current, including the vertical stability coil share); `'held'` coils are fixed at their measurement, `'free'` coils are unconstrained |
| `ip` | the plasma current |
| `mse` | MSE chords, only when the group is declared E_r-corrected (`meta['er_corrected']`) |

Other groups (Thomson scattering, diamagnetic flux) are scored and recorded but never weighted in the fit.

The free-boundary solve of each pass ("anchored forward solve", `kinetic.forward.ForwardModel`) takes the model's
\f$ \langle j_\phi \rangle \f$ and pressure as TokaMaker `jphi-linterp` and `linterp` profiles and fits the coil
currents in the same solve to the flux loops, probes and coil measurements, plus isoflux rows on a supplied
boundary outline (the anchor, from a prior magnetic equilibrium). A solve counts as converged only when the solver
converged and a warm re-solve reproduces the magnetic axis within 0.5 mm and \f$ q_0 \f$, \f$ l_i(3) \f$ and the
plasma current within \f$ 10^{-3} \f$.

## The current model in brief
On the geometry of the current equilibrium,

\f[ \langle \mathbf{J}\cdot\mathbf{B} \rangle = \sigma_{neo} \langle \mathbf{E}\cdot\mathbf{B} \rangle
    + s_{bs} \langle \mathbf{J}\cdot\mathbf{B} \rangle_{BS} + s_{cd} \langle \mathbf{J}\cdot\mathbf{B} \rangle_{CD}
    + \sum_k a_k g_k(\rho) \f]

converted to the plain surface average \f$ \langle j_\phi \rangle \f$ by the Grad-Shafranov identity (see
\ref doc_tokamaker_current_conventions), and \f$ p = p_{thermal}(n_e, T_e, n_i, T_i, n_{imp}) + s_{pf}\, p_{fast} \f$.

- Ohmic part: the steady state of the flux-diffusion equation with the plasma current as boundary condition and
  the neoclassical conductivity (Redl 2021) of the kinetic profiles.
- Core: by default the time-averaged state with a resistive q floor at 1 (where the steady state's \f$ q \f$ on
  axis falls below the floor, the resistivity in a central region is flattened until \f$ q \ge 1 \f$). What the
  floor did is recorded.
- Bootstrap current: Redl 2021, evaluated by TokaMaker's own bootstrap evaluation, multiplier \f$ s_{bs} \f$.
- Driven current and fast-ion pressure: supplied as arrays (`kinetic.current_model.SourceProfiles`); this package
  never computes them.
- Corrections: Gaussians \f$ g_k \f$ in \f$ \rho_{tor,norm} \f$ (centers 0.2, 0.5, 0.8 by default) with amplitudes
  \f$ a_k \f$ in fractions of \f$ I_p \f$, a model-discrepancy term with a prior width of 0.02.

| Parameter | Meaning | Prior (mean, sigma) |
|---|---|---|
| `a_corr` | the correction amplitudes (free by default) | (0, 0.02) |
| `ip_rel` | plasma-current target relative to the measured \f$ I_p \f$ (joins automatically, `ip_target='fitted'`) | none; its datum is the \f$ I_p \f$ row |
| `s_bs` | bootstrap multiplier | (1, 0.5) |
| `s_cd` | driven-current multiplier | (1, 0.25) |
| `s_pfast` | fast-ion pressure multiplier | (1, 0.2) |

**Stated assumptions.**
- Steady state: the ohmic current is the relaxed (steady-state) response; the slice carries no current history.
  A record (`steady_state_check`, on by default) shows the unfitted steady state next to the fit on the MSE chords
  by radius; it steers nothing.
- The kinetic profiles, \f$ Z_{eff} \f$ and the supplied sources are inputs, not fitted (only their multipliers).
- Fixed axisymmetric equilibrium, isotropic pressure (the fast-ion pressure enters as \f$ (p_\parallel + 2 p_\perp)/3 \f$).
- The plasma current must be positive in the discharge's frame (a reversed-current discharge is refused; the
  toroidal field may have either sign).

**Opt-in items, not validated.** The sawtooth core treatments `core_relaxation='periodic_orbit'` and
`'crash_times'`, the `'sharp'` q-floor form and the other non-default floor options, the edge pressure-gradient taper
(a workaround for solver builds without the cut-cell boundary quadrature), the experimental under-relaxation of the
consistency loop (`pass_relaxation`), lagged Jacobian columns, the noise retry and stall diagnostics of the fit, and
any driven current from a built-in source model. Each is off by default and recorded when on.

**Provisional defaults.** The anchor weight factor 3 (a lower factor fits the magnetics better and converges less
often) and the q-floor level 1 are proposals still under validation.

## Internal inductance
Two definitions are used and always named:

- \f$ l_i(1) = \dfrac{\langle B_p^2 \rangle_V}{B_{p,a}^2} \f$ with \f$ B_{p,a} = \mu_0 I_p / L \f$ and \f$ L \f$ the
  length of the boundary contour: TokaMaker's `get_stats()` default (`'std'`) normalization, which traces the
  \f$ \hat{\psi} = 0.99 \f$ surface. It is the quantity of the consistency loop's state test.
- \f$ l_i(3) = \dfrac{2 \int B_p^2\, dV}{\mu_0^2 I_p^2 R_{axis}} \f$: TokaMaker's `'iter'` normalization, which uses
  no traced length. It is the quantity of the verified-convergence test, because the traced length of a diverted
  boundary can jump by a few tenths of a percent between states that do not move.

## Conventions
- **Flux sign.** Flux-loop values in a `MeasurementSet` follow IMAS (COCOS 11): \f$ \psi_{11} = -2\pi\psi \f$ with
  TokaMaker's \f$ \psi \f$ per radian (see \ref doc_tokamaker_recon_constraints). The forward solve's flux-loop
  targets are converted to TokaMaker's \f$ 2\pi\psi \f$; field components, currents and the diamagnetic flux need no
  conversion.
- **MSE.** Coefficients in the IMAS order, frame and \f$ E_r \f$ status stated by the caller
  (`meta['field_orientation']`, `meta['er_corrected']`); the fit's own solve accepts only `field_orientation`
  (1, 1).
- **Radial coordinate.** Model profiles live on \f$ \rho_{tor,norm} \f$ of the current equilibrium; kinetic inputs
  are placed by the coordinate they carry (`kinetic.kinetics.KineticInputs` on \f$ \rho_{tor,norm} \f$,
  `kinetic.current_model.KineticProfiles` on \f$ \hat{\psi} \f$). When the kinetic inputs carry the outboard-midplane
  major radius of the equilibrium that produced them, the record `radial_coordinate` states how far the fitted flux
  surfaces sit from those surfaces in real space (the largest \f$ |\Delta R| \f$ and the value at the edge); it
  steers nothing.
- **Pressure frame.** TokaMaker solves with zero pressure at the boundary; the separatrix pressure is carried
  separately, and the g-file written by
  \ref OpenFUSIONToolkit.TokaMaker.reconstruction.kinetic_slice_result.save_geqdsk "save_geqdsk()" holds the full
  pressure (TokaMaker's `lcfs_pressure`).

## Use
```python
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker.reconstruction import kinetic_reconstruction, kinetic_slice_data
from OpenFUSIONToolkit.TokaMaker.kinetic.fit import FitSettings

myOFT = OFT_env(nthreads=2)
recon = kinetic_reconstruction(myOFT, 'mesh.h5', machine={'vsc': {'F9A': 1.0, 'F9B': -1.0}, 'r0': 1.6955},
                               fit_settings=FitSettings())      # settings objects of the helper package
data = kinetic_slice_data(measurements, kinetics, sources, boundary, F0, start=psi_nodes,
                          initial_profiles=(x, jphi, p))   # first solve of a new solver
result = recon.run(data, measurements.time, mode='slice')   # mode is required
print(result.status, result.free, result.theta, result.sigma)
result.save_geqdsk('g_fit')        # TokaMaker's writer, full pressure
result.record                      # every setting, default and diagnostic in effect
```

- `measurements`: `kinetic.interfaces.MeasurementSet` (built by hand, by `kinetic.adapters.measurements_from_imas`
  from IMAS JSON, or by a machine module).
- `kinetics`: `kinetic.kinetics.assemble_kinetic_inputs(...)` (on \f$ \rho_{tor,norm} \f$, with the checked
  \f$ Z_{eff} \f$ source rule and the required ion-temperature coverage check `ti_coverage`: a slice whose
  \f$ T_i \f$ is not measured inside the boundary is refused unless an assumption is named) or
  `kinetic.current_model.KineticProfiles` (on \f$ \hat{\psi} \f$).
- `sources`: `kinetic.current_model.SourceProfiles` (`SourceProfiles(psi_n)` for none).
- `boundary`: the anchor outline (R, Z) [m]; `F0` = \f$ R B_\phi \f$ [T m] signed; `start`: \f$ \psi \f$ on the mesh
  nodes (for example `kinetic.forward.psi_on_nodes` of a prior g-file). A new solver has no solved equilibrium to
  evaluate the current model on, so the first run from \f$ \psi \f$ also needs `initial_profiles` = \f$ (\hat{\psi},
  \langle j_\phi \rangle, p) \f$ for one starting solve (for example the prior equilibrium's current density and the
  kinetic pressure); it is a starting state, not part of the answer.
- Settings: `forward_settings` (`kinetic.forward.ForwardSettings`), `fit_settings` (`kinetic.fit.FitSettings`),
  `model_settings` (`kinetic.current_model.CurrentModelSettings`) and `speed` (`kinetic.speed.SpeedSettings`) are
  the helper package's own objects with their own defaults; nothing is renamed.

Set the environment variable `GFORTRAN_UNBUFFERED_PRECONNECTED=y` before Python starts (or at the very top of a
script, before numpy is imported: numpy's linear algebra library can load the Fortran runtime before
OpenFUSIONToolkit does): the forward model reads the solver's per-iteration output (residual history, vertical
stability coil current), and without it that output may arrive buffered. The record says whether the variable was
set when OpenFUSIONToolkit was imported (`history_capture`). With buffered output a solve can fail to find its own
iteration lines and stop with an error.

## What is device-specific: the machine module
Everything device-specific is an argument: the mesh, the vertical stability coil pairing and the reference radius
(`machine`), sensor geometry and uncertainties (inside the `MeasurementSet`), coil roles, the anchor outline, the
kinetic profiles and the sources. A machine module (a later pull request, `OpenFUSIONToolkit.machines`, starting
with DIII-D) supplies them from a device's own files: the device description (coil names, vertical stability pair,
flux-loop reference), EFIT k-file and stored-array readers with the device's uncertainty rules, the transport-code
source arrays, and a data fetch that only reuses a connection the user opened. Its outputs are the containers above;
the class does not import it.

## Known limits
- **Single slice only.** `run(..., mode='slices')` and `mode='series'` (independent slices; a time-dependent series
  with current diffusion between slices) raise `NotImplementedError`; they come with later pull requests.
- **The anchor's strength depends on the outline's point count.** Every fourth point of the supplied boundary
  becomes an isoflux row with the same weight, so an outline with more points anchors the boundary more strongly.
  The displacement of the fitted boundary from the anchor is recorded (`boundary`).
- **Ramp-up phases are not validated.** The model assumes a relaxed (steady-state) ohmic current.
- **Wall time is minutes per slice, not seconds**: each Jacobian column is a full consistent solve of several
  free-boundary solves. A fit of one or two parameters takes about 100 to 130 solves: about 4 minutes on the
  registered DIII-D-like test mesh (one thread) and about 1 minute on a mesh with a 5 cm plasma resolution (two
  threads); more free parameters cost proportionally more.
- **The forked workers are not included.** Parallel Jacobian columns in worker processes
  (`SpeedSettings(column_workers > 1)`) are refused; the in-process speed options (on by default) change no value.
- The solver offers no getter for the vertical stability coil current and the residual history; both are parsed
  from its printed output (see above).
