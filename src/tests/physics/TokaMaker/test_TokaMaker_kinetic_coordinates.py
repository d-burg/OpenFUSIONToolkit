#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
# TokaMaker tests: kinetic helper package, radial coordinates (rho_tor,norm map, solver knots, q-consistency
# check, axis q), pure Python on synthetic profiles; bounds follow from the formulas or are stated beside each
# assertion
import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic import coordinates as CO
from OpenFUSIONToolkit.TokaMaker.kinetic import current_model as CM
from OpenFUSIONToolkit.TokaMaker.kinetic.geometry import MU0, FluxSurfaceGeometry, OuterLayerGeometry

C = 0.995

# ---------------------------------------------------------------- the analytic separatrix-like geometry
R0, B0, F0, DPSI, IP = 1.7, 2.0, 3.4, 0.18, 1.2e6


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


def _analytic_rho_map():
    '''A psi_N <-> rho_tor,norm table from an analytic toroidal-flux fraction s(psi_N) = (x + x^2/2)/1.5'''
    x = np.linspace(0.0, 1.0, 401)
    s = (x + 0.5*x**2)/1.5
    return CO.RhoMap(x, np.sqrt(s), C, (C + 0.5*C**2)/1.5, 1.0)


# ============================================================== rho map and knots
def test_analytic_rho_map_round_trip():
    '''The two maps invert each other to round-off; the domain-edge radius is sqrt(Phi(c)/Phi_LCFS)'''
    m = _analytic_rho_map()
    r = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(m.rho_of_psin(m.psin_of_rho(r)), r, atol=1e-10)
    assert m.rho_edge == pytest.approx(np.sqrt((C + 0.5*C**2)/1.5), rel=1e-15)
    with pytest.raises(ValueError):
        m.rho_of_psin([1.2])
    with pytest.raises(ValueError):
        CO.RhoMap(m.psi_n[::-1], m.rho, C, m.phi_edge, m.phi_lcfs)


@pytest.mark.parametrize("rule", CO.KNOT_RULES)
def test_knots_on_an_analytic_map(rule):
    '''Knots run from exactly 0 to exactly 1, include the domain edge, carry n_layer_knots in the layer, and are
    uniform in rho inside the domain for the rho rule'''
    m = _analytic_rho_map()
    s = CO.KnotSettings(rule=rule)
    x, rec = CO.solver_knots(m, s)
    assert x[0] == 0.0 and x[-1] == 1.0 and np.all(np.diff(x) > 0)
    assert np.any(np.abs(x - C) < 1e-15)
    assert np.count_nonzero((x > C) & (x < 1.0)) == s.n_layer_knots
    assert rec["n_layer"] == s.n_layer_knots + 1
    if rule == "rho_tor_uniform":
        rk = m.rho_of_psin(x[x < C])
        np.testing.assert_allclose(np.diff(rk), 1.0/(s.n_knots - 1), rtol=1e-6)
        assert x[1] < 1e-3


# ---------------------------------------------------------------- on the current model's geometry objects
def test_rho_map_identities_and_round_trip():
    lag = _lag()
    m = lag["rho_map"]
    geom = lag["geom"]
    # inside the domain: rho = rho_dom sqrt(Phi(c)/Phi_LCFS) exactly (table construction)
    xs, rs = np.asarray(geom.meta["psi_n_samples"]), np.asarray(geom.meta["rho_samples"])
    np.testing.assert_allclose(m.rho_of_psin(xs), rs * m.rho_edge, rtol=1e-12)
    assert 0.0 < m.rho_edge < 1.0 and m.phi_lcfs > m.phi_edge
    # Phi_LCFS = Phi(c) + int F <1/R^2>/(2 pi) dV over the layer (independent quadrature of the same tabulation)
    L = lag["layer"]
    dens = L.fpol * L.avg_inv_r2 / (2 * np.pi)
    assert m.phi_lcfs == pytest.approx(geom.phi_b + L.integral(dens), rel=1e-12)
    r = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(m.rho_of_psin(m.psin_of_rho(r)), r, atol=1e-10)
    with pytest.raises(ValueError):
        m.rho_of_psin([1.2])


@pytest.mark.parametrize("rule", CO.KNOT_RULES)
def test_knots_monotone_cover_axis_edge_and_layer(rule):
    m = _lag()["rho_map"]
    s = CO.KnotSettings(rule=rule)
    x, rec = CO.solver_knots(m, s)
    assert x[0] == 0.0 and x[-1] == 1.0 and np.all(np.diff(x) > 0)
    assert np.any(np.abs(x - C) < 1e-15)                         # the domain edge is a knot
    assert np.count_nonzero((x > C) & (x < 1.0)) == s.n_layer_knots
    assert rec["n_layer"] == s.n_layer_knots + 1
    if rule == "rho_tor_uniform":
        rk = m.rho_of_psin(x[x < C])
        # uniform in rho: interior spacing constant to round-off of the map inversion
        np.testing.assert_allclose(np.diff(rk), 1.0 / (s.n_knots - 1), rtol=1e-6)
        assert x[1] < 1e-3                                        # packed at the axis (psi_N ~ rho^2)


def test_knots_change_with_the_equilibrium_only_for_the_rho_rule():
    a, b = _lag(q0=1.1)["rho_map"], _lag(q0=0.8)["rho_map"]
    xa, _ = CO.solver_knots(a, CO.KnotSettings())
    xb, _ = CO.solver_knots(b, CO.KnotSettings())
    assert np.max(np.abs(xa - xb)) > 1e-3
    for rule in ("sqrt_psin_uniform", "psin_uniform"):
        np.testing.assert_array_equal(CO.solver_knots(a, CO.KnotSettings(rule=rule))[0],
                                      CO.solver_knots(b, CO.KnotSettings(rule=rule))[0])


# ============================================================== q check
def test_q_check_passes_on_identical_profiles_and_fails_on_a_core_mismatch():
    r = np.linspace(0.0, 1.0, 101)
    q = 1.0 + 2.5 * r ** 2
    ok = CO.q_consistency(r, q, r, q)
    assert ok["passed"] and ok["max_rel"] < 1e-14
    # the kind of mismatch the q check must catch: an equilibrium with q0 ~ 0.965 against a model
    # whose q has a reversed core with q(0) ~ 1.7 and a minimum ~ 0.97 near rho 0.11
    qm = 0.97 + 0.73 * np.exp(-(r / 0.06) ** 2) + 2.5 * np.clip(r - 0.11, 0, None) ** 2
    qe = 0.965 + 2.5 * np.clip(r - 0.11, 0, None) ** 2
    bad = CO.q_consistency(r, qe, r, qm)
    assert not bad["passed"] and bad["max_rel"] > 0.01
    # 1 % everywhere is exactly at the criterion's edge; 1.2 % fails
    assert CO.q_consistency(r, 1.012 * q, r, q)["passed"] is False
    assert CO.q_consistency(r, 1.009 * q, r, q)["passed"] is True
    # tables that do not cover the checked range are a failure, never a pass
    assert CO.q_consistency(r[r < 0.5], q[r < 0.5], r, q)["passed"] is False


def test_axis_q_is_the_regular_extrapolation_and_ignores_near_axis_artifacts():
    """The axis value is the regular-singular extrapolation q0 + q2 rho^2 + q4 rho^4 fitted on the axis
    window: exact for such a profile, and unaffected by scatter on the innermost surfaces (outside the
    window), where the raw innermost value is off by percent."""
    r = np.linspace(0.0244, 0.95, 186)
    q = 0.87 + 0.9 * r ** 2 - 0.3 * r ** 4
    a = CO.axis_q(r, q)
    assert a["q0"] == pytest.approx(0.87, rel=1e-12) and a["rms_rel"] < 1e-13
    qa = q.copy()
    inner = r < 0.045
    qa[inner] *= 1.0 + 0.02 * np.sin(97.0 * r[inner])          # trace scatter near the axis (outside the window)
    assert abs(qa[0] / 0.87 - 1) > 0.01
    assert CO.axis_q(r, qa)["q0"] == pytest.approx(0.87, rel=1e-12)
    with pytest.raises(ValueError):
        CO.axis_q(r[r < 0.06], q[r < 0.06])                    # too few points in the window


def test_q_check_tests_the_extrapolated_axis_at_the_same_criterion():
    """A model q that agrees with the equilibrium's to < 1 % on the interior range but bends away inside
    it (axis values differing by > 1 %) fails through the axis test when it is on; by default
    the axis values are recorded, not tested (the interior criterion is unchanged)."""
    re = np.linspace(0.0244, 0.95, 186)
    qe = 1.0 + 0.8 * re ** 2
    rm = np.linspace(0.0, 1.0, 101)
    qm = 1.0 + 0.8 * rm ** 2 - 0.025 * np.exp(-(rm / 0.05) ** 2)
    on = CO.QCheckSettings(test_axis=True)
    out = CO.q_consistency(re, qe, rm, qm, on)
    assert out["interior_passed"] is True and out["max_rel"] < 0.01
    assert out["axis"]["rel"] > 0.01 and out["axis"]["passed"] is False and out["passed"] is False
    off = CO.q_consistency(re, qe, rm, qm, CO.QCheckSettings(test_axis=False))
    assert off["passed"] is True and off["axis"]["tested"] is False
    same = CO.q_consistency(re, qe, rm, 1.0 + 0.8 * rm ** 2, on)
    assert same["passed"] and same["axis"]["rel"] < 1e-12
    dflt = CO.q_consistency(re, qe, rm, qm)                   # default: axis values recorded, not tested
    assert dflt["passed"] is True and dflt["axis"]["tested"] is False and dflt["axis"]["rel"] > 0.01
