#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""TokaMaker tests: kinetic helper package, where a steady-form current correction returns its net current, and
the record of what the corrections do to the core (no solver; an analytic separatrix-like geometry, defined below,
with its edge non-inductive current, the package's steady diffusion solver).

* Default (``corr_return="conductivity"``): the correction shapes are the plain Gaussians, identical array for array
  to the earlier code (a frozen copy below), so every default result is unchanged.
* Option ``corr_return="local"``: each shape is ``G(c, w) - lam G(c, r w)`` with ``lam`` from the same current
  integrals as the normalization (``0 < lam <= 1``), so it carries zero net current: at fixed Ip the steady loop
  voltage changes by at most 1e-3 of the default shape's change (the solver-quadrature residual below).
* Steady response (Ip fixed) of a correction at rho 0.5 or 0.8, amplitude 1 % of Ip, on the steady state with the
  fixture's non-inductive current.  Default: the ohmic return sits on the axis, so the axis <J.B> moves by at least
  half of the amplitude times the axis value (the "-sum a" rule; the factor 1/2 only keeps the comparison away from
  two negligible numbers).  Local: the change of <J.B> at the innermost cell (rho_c) is bounded by the form itself:
  ``|shape(rho_c)| <= exp(-(c - rho_c)^2 / (2 (r w)^2))`` of the correction's peak (the larger of the two terms,
  since ``0 < lam <= 1`` and the wider Gaussian dominates the narrower one), i.e. 4.7 % at c = 0.5 and 0.04 % at
  c = 0.8, plus the steady return of the shape's residual net current in the SOLVER's quadrature (100 midpoint cells
  against the 201-point trapezoid that sets ``lam``): at most ``h^2/(24 w^2) = 4.2e-4`` of each Gaussian term's
  current for h = 0.01, w = 0.1, so below 1e-3 of the default's axis change.  Both bounds were fixed from the
  form before the first run.
* A correction at rho 0.2 is a core basis function in either form: with the local return its axis change is at
  least half of its analytic shape value there (coarse check that it acts; the factor only allows for the
  interpolation between grids).
* The record (:func:`current_model.correction_core_effect`) against steady states built directly:
  the axis <J.B> change and the model q change equal the direct ones to round-off (1e-9 relative), the amplitudes
  above their prior sigma and an axis change above the 5 % reporting level are flagged, and zero amplitudes record a
  zero effect with no flag.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic import coordinates as CO
from OpenFUSIONToolkit.TokaMaker.kinetic import current_model as CM
from OpenFUSIONToolkit.TokaMaker.kinetic.diffusion import CurrentState
from OpenFUSIONToolkit.TokaMaker.kinetic.geometry import MU0, FluxSurfaceGeometry, OuterLayerGeometry

# numpy's trapezoid-rule integrator: numpy.trapezoid (numpy 2) or numpy.trapz (numpy 1), the same function
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------- the analytic separatrix-like geometry
R0, B0, F0, DPSI, IP, C = 1.7, 2.0, 3.4, 0.18, 1.2e6, 0.995


def F(x):
    return F0 * (1.0 - 0.02 * x)


def g1(x):
    return (1.0 + 0.08 * x) / R0 ** 2


def q_ref(x, q0=1.1):
    return q0 + 2.2 * x ** 2 - 0.5 * np.log(np.maximum(1.0 - x, 1e-300))


def _lag(q0=1.1, sigma_scale=1.0):
    """A lagged-input dict like ``current_model.evaluate_geometry`` returns, on analytic surfaces."""
    xs = np.linspace(np.sqrt(1e-3), np.sqrt(C), 200) ** 2
    xs[-1] = C
    q = q_ref(xs, q0)
    vps = 4 * np.pi ** 2 * q / (F(xs) * g1(xs))
    iref = IP * (1.6 * xs - 0.6 * xs ** 2)
    bp2 = 2 * np.pi * MU0 * iref / vps
    geom = FluxSurfaceGeometry.from_psi_coordinates(xs, 0.0, DPSI, vps, bp2, g1(xs), F(xs), B0=B0, R0=R0, nrho=201,
                                                    psi_n_edge=C)
    xl = 1.0 - (1.0 - C) * np.geomspace(1.0, 1e-2, 40)
    xl[0] = C
    ql = q_ref(xl, q0)
    vl = 4 * np.pi ** 2 * ql / (F(xl) * g1(xl))
    layer = OuterLayerGeometry.from_psi_coordinates(xl, 0.0, DPSI, vl, g1(xl), F(xl),
                                                    avg_grad_psi2_r2=2 * np.pi * MU0 * IP * (1.6 * xl - 0.6 * xl ** 2) / vl,
                                                    fpol_lcfs=F(1.0))
    ge = np.unique(np.concatenate((np.linspace(1e-3, C, 300), xl)))
    sig = sigma_scale * 1.0e8 * (0.02 + 0.98 * np.clip(1.0 - ge ** 2, 0.0, None) ** 1.5)
    to_rho, to_psin = CM._psin_to_rho(np.asarray(geom.meta["psi_n_samples"]), np.asarray(geom.meta["rho_samples"]))
    rmap = CO.rho_map_from_geometry(geom, layer)
    phys = dict(psi_n=ge, rho=rmap.rho_of_psin(ge), q=q_ref(ge, q0), F=F(ge), inv_r2=g1(ge), b2=F(ge) ** 2 * g1(ge))
    return dict(geom=geom, layer=layer, psi_n_edge=C, to_rho=to_rho, to_psin=to_psin, rho_map=rmap, phys=phys,
                sigma=sig, jbs=B0 * 4.0e5 * np.exp(-((ge - 0.975) / 0.02) ** 2))


GEO = SimpleNamespace(IP=IP, R0=R0, DPSI=DPSI, _lag=_lag)
IP0 = GEO.IP
AMP = 0.01


def _frozen_default_shapes(lag, cfg, ip):
    """The plain-Gaussian shapes exactly as the code computed them before the local-return option."""
    geom, layer = lag["geom"], lag["layer"]
    rmap = lag["rho_map"]
    rho_ge = lag["phys"]["rho"]
    gc = geom.evaluate(geom.rho, ["dvol_drho", "fpol"])
    rho_std_g = geom.rho * rmap.rho_edge
    rho_l = rmap.rho_of_psin(layer.psi_n)
    out = []
    for rc0 in cfg.corr_centers_rho:
        w = float(cfg.corr_width_rho)
        dom = _trapezoid(gc["dvol_drho"] * np.exp(-0.5 * ((rho_std_g - rc0) / w) ** 2)
                         / (2 * np.pi * gc["fpol"] ** 2), geom.rho)
        lay = layer.integral(layer.source_density(np.exp(-0.5 * ((rho_l - rc0) / w) ** 2)))
        norm = abs(float(ip)) / (layer.fpol_lcfs * (dom + lay))
        out.append(np.exp(-0.5 * ((rho_ge - rc0) / w) ** 2) * norm)
    return out


def _net_current(lag, jdotb_ge):
    """``F_lcfs int <J.B>/(2 pi F^2) dV`` (domain trapezoid on the geometry's rho + the layer), the integral that
    normalizes the shapes."""
    geom, layer, rmap = lag["geom"], lag["layer"], lag["rho_map"]
    gc = geom.evaluate(geom.rho, ["dvol_drho", "fpol"])
    jg, jl = CM._domain_arrays(lag, jdotb_ge)
    dom = _trapezoid(gc["dvol_drho"] * jg / (2 * np.pi * gc["fpol"] ** 2), geom.rho)
    return float(layer.fpol_lcfs * (dom + layer.integral(layer.source_density(jl))))


def _gauss_current(lag, center, width):
    geom, layer, rmap = lag["geom"], lag["layer"], lag["rho_map"]
    gc = geom.evaluate(geom.rho, ["dvol_drho", "fpol"])
    rg, rl = geom.rho * rmap.rho_edge, rmap.rho_of_psin(layer.psi_n)
    dom = _trapezoid(gc["dvol_drho"] * np.exp(-0.5 * ((rg - center) / width) ** 2) / (2 * np.pi * gc["fpol"] ** 2),
                     geom.rho)
    return dom + layer.integral(layer.source_density(np.exp(-0.5 * ((rl - center) / width) ** 2)))


def _axis(lag, jni, rf):
    sol = CM.build_solver(lag, jni, IP0)
    prof = sol.profiles(sol.steady_state(rf))
    return float(prof["jdotb"][0]), float(prof["rho_cells"][0]) * lag["rho_map"].rho_edge


@pytest.mark.parametrize("explicit", [False, True])
def test_default_shapes_are_the_earlier_gaussians(explicit):
    lag = GEO._lag()
    cfg = CM.CurrentModelSettings(r0=GEO.R0, **(dict(corr_return="conductivity") if explicit else {}))
    for form in ("steady", "in_window"):
        for a, b in zip(CM.correction_shapes(lag, cfg, IP0, form=form), _frozen_default_shapes(lag, cfg, IP0)):
            assert np.array_equal(a, b)


def test_local_in_window_shapes_stay_gaussian_and_settings_are_checked():
    lag = GEO._lag()
    cfg = CM.CurrentModelSettings(r0=GEO.R0, corr_return="local")
    for a, b in zip(CM.correction_shapes(lag, cfg, IP0, form="in_window"), _frozen_default_shapes(lag, cfg, IP0)):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        CM.correction_shapes(lag, CM.CurrentModelSettings(r0=GEO.R0, corr_return="axis"), IP0)
    with pytest.raises(ValueError):
        CM.correction_shapes(lag, CM.CurrentModelSettings(r0=GEO.R0, corr_return="local",
                                                          corr_return_width_ratio=1.0), IP0)
    assert CM.CurrentModelSettings(r0=GEO.R0).as_dict()["corr_return"] == "conductivity"


def test_local_shapes_are_the_stated_zero_net_form():
    lag = GEO._lag()
    cfg = CM.CurrentModelSettings(r0=GEO.R0, corr_return="local")
    plain = _frozen_default_shapes(lag, cfg, IP0)
    loc = CM.correction_shapes(lag, cfg, IP0)
    rf = CurrentState.uniform_faces(cfg.n_cells)
    w, wr = cfg.corr_width_rho, cfg.corr_return_width_ratio * cfg.corr_width_rho
    v0 = CM.build_solver(lag, lag["jbs"], IP0).profiles(CM.build_solver(lag, lag["jbs"], IP0).steady_state(rf))
    for c, g, h in zip(cfg.corr_centers_rho, plain, loc):
        lam = _gauss_current(lag, c, w) / _gauss_current(lag, c, wr)
        assert 0.0 < lam <= 1.0
        # the shape is the stated formula with the default shape's normalization (the amplitude keeps its meaning:
        # the fraction of Ip carried by the Gaussian term)
        rho = lag["phys"]["rho"]
        norm = g / np.exp(-0.5 * ((rho - c) / w) ** 2)
        np.testing.assert_allclose(h, norm * CM.local_return_shape(rho, c, w, wr, lam), rtol=1e-12, atol=0.0)
        # zero net current as the steady solver sees it: the loop voltage at fixed Ip is unchanged, to the solver's
        # quadrature (bound 1e-3 of the default shape's change, module docstring)
        dv = {}
        for name, shp in (("default", g), ("local", h)):
            sol = CM.build_solver(lag, lag["jbs"] + AMP * shp, IP0)
            dv[name] = sol.profiles(sol.steady_state(rf))["v_loop_boundary"] - v0["v_loop_boundary"]
        assert abs(dv["default"]) > 0.0 and abs(dv["local"]) <= 1e-3 * abs(dv["default"]), (c, dv)


@pytest.mark.parametrize("k", [1, 2])        # centers rho 0.5 and 0.8
def test_local_return_leaves_the_axis_for_mid_and_outer_corrections(k):
    lag = GEO._lag()
    rf = CurrentState.uniform_faces(CM.CurrentModelSettings(r0=GEO.R0).n_cells)
    base, rho_c = _axis(lag, lag["jbs"], rf)
    cfg_d = CM.CurrentModelSettings(r0=GEO.R0)
    cfg_l = CM.CurrentModelSettings(r0=GEO.R0, corr_return="local")
    g = CM.correction_shapes(lag, cfg_d, IP0)[k]
    h = CM.correction_shapes(lag, cfg_l, IP0)[k]
    c, w = cfg_d.corr_centers_rho[k], cfg_d.corr_width_rho
    wr = cfg_l.corr_return_width_ratio * w
    d_default = _axis(lag, lag["jbs"] + AMP * g, rf)[0] - base
    d_local = _axis(lag, lag["jbs"] + AMP * h, rf)[0] - base
    assert d_default < 0.0 and abs(d_default) >= 0.5 * AMP * abs(base), (d_default, base)
    peak = float(np.max(g) / np.max(np.exp(-0.5 * ((lag["phys"]["rho"] - c) / w) ** 2)))   # the shape's normalization
    bound = AMP * peak * np.exp(-0.5 * ((c - rho_c) / wr) ** 2) + 1e-3 * abs(d_default)
    assert abs(d_local) <= bound, (c, d_local, bound, d_default)


def test_local_return_core_correction_still_acts_on_the_core():
    lag = GEO._lag()
    rf = CurrentState.uniform_faces(CM.CurrentModelSettings(r0=GEO.R0).n_cells)
    base, rho_c = _axis(lag, lag["jbs"], rf)
    cfg = CM.CurrentModelSettings(r0=GEO.R0, corr_return="local")
    h = CM.correction_shapes(lag, cfg, IP0)[0]
    c, w = cfg.corr_centers_rho[0], cfg.corr_width_rho
    wr = cfg.corr_return_width_ratio * w
    g = _frozen_default_shapes(lag, cfg, IP0)[0]
    norm = float(np.max(g) / np.max(np.exp(-0.5 * ((lag["phys"]["rho"] - c) / w) ** 2)))
    lam = _gauss_current(lag, c, w) / _gauss_current(lag, c, wr)
    expect = AMP * norm * float(CM.local_return_shape(rho_c, c, w, wr, lam))
    d_local = _axis(lag, lag["jbs"] + AMP * h, rf)[0] - base
    assert np.sign(d_local) == np.sign(expect) and abs(d_local) >= 0.5 * abs(expect), (d_local, expect)


# ============================================================== the record
def _fixture_inputs():
    lag = GEO._lag()
    ph = lag["phys"]
    ph["inv_r"] = np.sqrt(ph["inv_r2"])
    ph["r_avg"] = np.full_like(ph["psi_n"], GEO.R0)
    lag["dpsi"] = GEO.DPSI
    kin = SimpleNamespace(thermal_pressure_at=lambda psi_n=None, rho=None: 2.0e4 * (1.0 - np.asarray(psi_n, float)),
                          coordinate="psi_n")
    src = SimpleNamespace(jdotb_at=lambda ge, **kw: np.zeros_like(np.asarray(ge, float)),
                          p_fast_at=lambda x, rho=None: np.zeros_like(np.asarray(x, float)))
    return lag, kin, src


@pytest.mark.parametrize("ret", CM.CORRECTION_RETURNS)
def test_core_effect_record_against_direct_steady_states(ret):
    lag, kin, src = _fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0, corr_return=ret)
    amps = (0.03, 0.02, 0.01)
    asm = CM.assemble(lag, kin, src, dict(a_corr=amps), cfg, IP0)
    rec = CM.steady_correction_effect(lag, asm, kin, src, cfg, [0.02, 0.02, 0.02])
    assert rec["computed"] and rec["basis"]["corr_return"] == ret
    # direct: the two steady states on the same geometry
    rf = CurrentState.uniform_faces(cfg.n_cells)
    shapes = CM.correction_shapes(lag, cfg, IP0)
    p1 = CM.build_solver(lag, lag["jbs"] + sum(a * g for a, g in zip(amps, shapes)), IP0)
    p0 = CM.build_solver(lag, lag["jbs"], IP0)
    pr1, pr0 = p1.profiles(p1.steady_state(rf)), p0.profiles(p0.steady_state(rf))
    d_axis = (pr1["jdotb"][0] - pr0["jdotb"][0]) / pr0["jdotb"][0]
    assert rec["d_axis_jdotb_rel"] == pytest.approx(d_axis, rel=1e-9)
    assert rec["d_q_axis_model"] == pytest.approx(pr1["q"][0] - pr0["q"][0], rel=1e-9)
    assert [c["ratio_to_prior_sigma"] for c in rec["corrections"]] == pytest.approx([1.5, 1.0, 0.5], rel=1e-12)
    fl = " | ".join(rec["flags"])
    assert "rho 0.2" in fl and "rho 0.5" not in fl and "rho 0.8" not in fl      # 1.0 sigma is not above
    assert ("axis <J.B>" in fl) == bool(abs(d_axis) > CM.CORRECTION_AXIS_REPORT_LEVEL) and rec["flagged"]
    # the current inside rho 0.3 changes by the corrections' net effect there, as a fraction of Ip
    assert rec["d_current_inside_core_over_ip"] == pytest.approx(
        (rec["current_inside_core_A"]["corrected"] - rec["current_inside_core_A"]["uncorrected"]) / IP0, rel=1e-12)


def test_core_effect_zero_amplitudes_record_zero_and_no_flag():
    lag, kin, src = _fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    asm = CM.assemble(lag, kin, src, {}, cfg, IP0)
    rec = CM.steady_correction_effect(lag, asm, kin, src, cfg, [0.02, 0.02, 0.02])
    assert rec["computed"] and rec["d_axis_jdotb_rel"] == 0.0 and rec["d_q_axis_model"] == 0.0
    assert rec["d_current_inside_core_over_ip"] == 0.0 and rec["flags"] == [] and rec["flagged"] is False


def test_axis_flag_from_mid_and_outer_corrections_only_with_the_conductivity_return():
    lag, kin, src = _fixture_inputs()
    amps = (0.0, 0.04, 0.04)
    rec = {}
    for ret in CM.CORRECTION_RETURNS:
        cfg = CM.CurrentModelSettings(r0=GEO.R0, corr_return=ret)
        asm = CM.assemble(lag, kin, src, dict(a_corr=amps), cfg, IP0)
        rec[ret] = CM.steady_correction_effect(lag, asm, kin, src, cfg, [0.02] * 3)
    # 0.08 of Ip returned like the conductivity (no Gaussian of these two reaches the axis): past the 5 % level
    c = rec["conductivity"]
    assert c["d_axis_jdotb_rel"] < -CM.CORRECTION_AXIS_REPORT_LEVEL and any("axis <J.B>" in f for f in c["flags"])
    # local return: no axis flag; both amplitudes (2 prior sigmas) are flagged in both forms
    lo = rec["local"]
    assert not any("axis <J.B>" in f for f in lo["flags"]), lo["flags"]
    for r in rec.values():
        assert sum("prior sigmas" in f for f in r["flags"]) == 2


def test_bootstrap_multiplier_scales_the_bootstrap_source():
    """New in this package (not in the original tests; a deliberate bug that ignored ``s_bs`` passed them all): with
    the core treatment off, the state and <J_phi> assembled with ``s_bs = s`` are, bit for bit, those assembled from
    the fixture with its bootstrap current multiplied by ``s`` beforehand, and differ from the ``s_bs = 1`` ones."""
    lag, kin, src = _fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="none")
    base = CM.assemble(lag, kin, src, {}, cfg, GEO.IP)
    for s in (0.0, 0.5, 2.0):
        a = CM.assemble(lag, kin, src, dict(s_bs=s), cfg, GEO.IP)
        lag_s, _, _ = _fixture_inputs()
        lag_s["jbs"] = s * lag_s["jbs"]
        ref = CM.assemble(lag_s, kin, src, {}, cfg, GEO.IP)
        np.testing.assert_array_equal(a["state"].psi, ref["state"].psi)
        np.testing.assert_array_equal(a["jphi"], ref["jphi"])
        assert np.max(np.abs(a["state"].psi - base["state"].psi)) > 0.0
