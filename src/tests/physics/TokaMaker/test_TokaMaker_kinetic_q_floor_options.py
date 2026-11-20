#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""TokaMaker tests: kinetic helper package, the NON-DEFAULT options of the resistive q floor (``sawtooth``,
"Non-default options"), each off by default:

* ``total_current="flat_at_level"``: inside the floor's region the total current is that of q flat at the level (any
  level, not only 1); Ip is the solver's Ip condition; the region and the branch are the default rule's; nothing in
  branch (b);
* ``transition="blend_margin"``: the number-free blend of the level across the plain axis-q margin; identical to the
  default outside the band, continuous at the boundary inside it;
* ``total_current="flat_at_level_own_region"``: the same flat-at-level total current inside the option's own region
  (the default rule's criterion on the flat-at-level states): q at or above the level outside it, the smallest such
  edge;
* ``q_floor_branch_freeze="first_assembly"`` (diagnostic): outside a fit the plain branch; the frozen branch is
  carried by the freeze, not read from a record.

Synthetic geometry as in ``test_TokaMaker_kinetic_q_floor.py`` (no solver).  Ported from the reconstruction's own
tests with every bound unchanged.  Not ported here: the tests that run the reconstruction's fit or time-dependent
series (the default path never entering the new code across a fit and a series, the options in a time-dependent
step, the frozen branch over a whole fit, the series' option and Jacobian records): they need those modules.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic import coordinates as CO
from OpenFUSIONToolkit.TokaMaker.kinetic import current_model as CM
from OpenFUSIONToolkit.TokaMaker.kinetic import sawtooth as SAW
from OpenFUSIONToolkit.TokaMaker.kinetic.diffusion import CurrentState
from OpenFUSIONToolkit.TokaMaker.kinetic.geometry import MU0, FluxSurfaceGeometry, OuterLayerGeometry


# ---------------------------------------------------------------- the analytic separatrix-like geometry
# (verbatim from the original coordinates/kinetics/series test file; GEO stands for that file)
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


# (verbatim from the original correction-return test file; LR stands for that file)
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


LR = SimpleNamespace(_fixture_inputs=_fixture_inputs)


# (verbatim from the original series-driver test file; SD stands for that file)
def _lag_ni(sigma_scale=1.0, q0=1.1):
    """The analytic geometry WITH its non-inductive (edge bootstrap-like) current kept."""
    return GEO._lag(q0=q0, sigma_scale=sigma_scale)


SD = SimpleNamespace(_lag_ni=_lag_ni)


# (verbatim from the original continuous q-floor test file; QC stands for that file)
IP_MET = 1.86 * GEO.IP            # the floor 1 acts and is met on the fixture (q0 of the plain state ~0.8)
IP_LOW_Q = 2.2 * GEO.IP           # q < 1 out to rho ~0.56 (as in test_q_floor.py)
Q_LAND = 1e-9                     # q_min above the floor where met: the radius is bisected to 1e-14 in rho


def _solver(ip, jni=None):
    lag = GEO._lag()
    return CM.build_solver(lag, lag["jbs"] if jni is None else jni, ip)


def _fq(sol, st):
    return SAW.face_q(st, sol.geometry_at(st.time).phi_b)


def _reversed_shear():
    """Off-axis co-current source: q0 above the floor, an off-axis minimum below it (as in test_q_floor.py)."""
    lag = GEO._lag()
    rho = lag["phys"]["rho"]
    jni = lag["jbs"] + 8.0 * np.max(np.abs(lag["jbs"])) * np.exp(-0.5 * ((rho - 0.5) / 0.08) ** 2)
    return CM.build_solver(lag, jni, 2.3 * GEO.IP)


QC = SimpleNamespace(IP_MET=IP_MET, IP_LOW_Q=IP_LOW_Q, Q_LAND=Q_LAND, _solver=_solver, _fq=_fq,
                     _reversed_shear=_reversed_shear)


FLAT_ATOL = 1e-11      # q inside the region against the level: the closed form to round-off (measured ~4e-13)
IP_RTOL = 1e-12        # Ip against the solver's Ip condition (the same bound as the default rule's tests)
RF =CurrentState.uniform_faces(100)


def test_a_zero_source_change_leaves_the_solver_bitwise_unchanged():
    """The diffusion's source hook adds exactly what it is given: a modifier returning zeros gives the same
    coefficients and the same steady state as no modifier, to the bit."""
    import copy
    sol = QC._solver(QC.IP_MET)
    s2 = copy.copy(sol)
    s2.source_modifier = lambda t, rf, c, ip_mode, state: np.zeros(rf.size - 1)
    c0, c1 = sol.coefficients(0.0, RF), s2.coefficients(0.0, RF)
    np.testing.assert_array_equal(c0.s_c, c1.s_c)
    np.testing.assert_array_equal(c0.jni_c, c1.jni_c)
    np.testing.assert_array_equal(sol.steady_state(RF).psi, s2.steady_state(RF).psi)


def test_option_values_and_combinations_are_checked():
    sol = QC._solver(QC.IP_MET)
    for kw, msg in ((dict(total_current="flat"), "total_current"), (dict(transition="smooth"), "transition"),
                    (dict(branch="both"), "branch"),
                    (dict(total_current="flat_at_level", rule="grid"), "continuous"),
                    (dict(total_current="flat_at_level", transition="blend_margin"), "not defined"),
                    (dict(branch="axis", transition="blend_margin"), "not defined")):
        with pytest.raises(ValueError, match=msg):
            SAW.resistive_floor_steady(sol, RF, 1.0, **kw)
    for kw in (dict(q_floor_total_current="x"), dict(q_floor_transition="x"), dict(q_floor_branch_freeze="x"),
               dict(q_floor_total_current="flat_at_level", q_floor_form="sharp"),
               dict(q_floor_branch_freeze="first_assembly", q_floor_transition="blend_margin")):
        with pytest.raises(ValueError):
            CM.CurrentModelSettings(**kw)
    import copy
    s2 = copy.copy(sol)
    s2.source_modifier = lambda *a: 0.0
    with pytest.raises(ValueError, match="source modifier"):
        SAW.ResistiveFloor(s2, 1.0, rho_inv=0.3, total_current="flat_at_level").with_solver()


# ====================================================================== total-current flattening at the level
@pytest.mark.parametrize("ip, level", [(QC.IP_MET, 1.0), (QC.IP_MET, 0.95), (QC.IP_LOW_Q, 1.0), (QC.IP_LOW_Q, 0.9)])
def test_total_current_option_holds_q_flat_at_the_level_inside_the_floors_region(ip, level):
    """Inside the region the default rule decides (same branch, same radius) q equals the level on every face; Ip is
    the solver's Ip condition; the record states the option, the resistivity-only state's q and the non-inductive
    current the option adds."""
    sol = QC._solver(ip)
    st0 = sol.steady_state(RF)
    _, rd, _ = SAW.resistive_floor_steady(sol, RF, level)
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, level, total_current="flat_at_level")
    assert rec["floor_branch"] == rd["floor_branch"] == "axis" and rec["acted"]
    assert rec["rho_floor"] == rd["rho_floor"]
    q = QC._fq(s2, st)
    inside = RF <= rec["rho_floor"]
    assert np.max(np.abs(q[inside] - level)) <= FLAT_ATOL
    assert rec["q_flat_max_abs_dev"] <= FLAT_ATOL and abs(rec["q_axis_after"] - level) <= FLAT_ATOL
    assert s2.profiles(st)["ip"] == pytest.approx(sol.profiles(st0)["ip"], rel=IP_RTOL)
    assert st.psi_boundary == st0.psi_boundary
    assert rec["total_current"] == "flat_at_level" and np.isfinite(rec["total_current_ni_added_over_ip"])
    assert rec["q_min_resistive_only"] == pytest.approx(rd["q_min_after"], rel=1e-12)
    assert rec["q_floor_met"] == bool(np.all(q[~inside] >= level))
    assert st.meta["q_floor"]["total_current"] == "flat_at_level"
    # the source change is that of the closed form (the hook's ds), zero outside the region
    c_tc, c_r = s2.coefficients(0.0, RF), SAW.ResistiveFloor(sol, level, rho_inv=rec["rho_floor"],
                                                               rho_faces=RF).with_solver().coefficients(0.0, RF)
    ds = c_tc.s_c - c_r.s_c
    rc = 0.5 * (RF[1:] + RF[:-1])
    assert np.all(ds[RF[:-1] >= rec["rho_floor"]] == 0.0) and np.any(ds[rc < rec["rho_floor"]] != 0.0)


def test_total_current_option_does_nothing_in_branch_b_and_c():
    """Branch (b) (below the level off axis only) and (c) (nowhere): no region, so the option changes nothing."""
    sol = QC._reversed_shear()
    st_d, rd, _ = SAW.resistive_floor_steady(sol, RF, 1.0)
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, 1.0, total_current="flat_at_level")
    assert rd["floor_branch"] == rec["floor_branch"] == "off_axis" and not rec["acted"] and s2 is sol
    np.testing.assert_array_equal(st.psi, st_d.psi)
    sol_c = QC._solver(QC.IP_MET)
    _, rc_, s3 = SAW.resistive_floor_steady(sol_c, RF, 0.5, total_current="flat_at_level")
    assert rc_["floor_branch"] == "none" and not rc_["acted"] and s3 is sol_c


def test_total_current_option_converges_with_the_grid_and_q0_is_the_level_on_every_grid():
    sol = QC._solver(QC.IP_MET)
    out = {n: SAW.resistive_floor_steady(sol, CurrentState.uniform_faces(n), 1.0, total_current="flat_at_level")
           for n in (100, 200, 400)}
    r = {n: out[n][1]["rho_floor"] for n in out}
    assert abs(r[100] - r[200]) >= abs(r[200] - r[400])
    for n, (st, rec, s2) in out.items():
        assert abs(rec["q_axis_after"] - 1.0) <= FLAT_ATOL, n


def test_total_current_option_makes_q0_continuous_at_the_branch_boundary():
    """Where the plain axis q crosses the level (Ip scanned), the option puts q0 at the level on the (a) side, and
    the plain q0 on the (b)/(c) side is at the level too: q0 changes by the plain state's own slope only."""
    lag = SD._lag_ni()

    def q0_plain(ip_):
        s_ = CM.build_solver(lag, lag["jbs"], ip_)
        return float(QC._fq(s_, s_.steady_state(RF))[0])
    lo, hi = GEO.IP, 3.0 * GEO.IP
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if q0_plain(mid) > 1.0 else (lo, mid)
    q0s = {}
    for e in (-1e-6, 1e-6):
        s_ = CM.build_solver(lag, lag["jbs"], 0.5 * (lo + hi) * (1.0 + e))
        st, rec, s2 = SAW.resistive_floor_steady(s_, RF, 1.0, total_current="flat_at_level")
        q0s[e] = (float(QC._fq(s2, st)[0]), rec["floor_branch"])
    assert q0s[1e-6][1] == "axis" and q0s[-1e-6][1] != "axis"
    assert abs(q0s[1e-6][0] - q0s[-1e-6][0]) <= 1e-4


# ====================================================================== blended transition (number-free)
def test_blend_is_the_default_rule_outside_its_band():
    """Branch (a) and branch (c): the blend is the default rule to the bit; branch (b) beyond the margin (lambda 0):
    no action, as the default."""
    for sol in (QC._solver(QC.IP_MET), QC._solver(QC.IP_LOW_Q)):
        st_d, rd, _ = SAW.resistive_floor_steady(sol, RF, 1.0)
        st, rec, _ = SAW.resistive_floor_steady(sol, RF, 1.0, transition="blend_margin")
        np.testing.assert_array_equal(st.psi, st_d.psi)
        assert rec["rho_floor"] == rd["rho_floor"] and rec["blend_lambda"] == 1.0 and rec["level_used"] == 1.0
    sol = QC._reversed_shear()
    q = QC._fq(sol, sol.steady_state(RF))
    level = float(np.min(q)) + 0.4 * (float(q[0]) - float(np.min(q)))     # q0 - level > level - q_min: lambda 0
    st_d, rd, _ = SAW.resistive_floor_steady(sol, RF, level)
    st, rec, _ = SAW.resistive_floor_steady(sol, RF, level, transition="blend_margin")
    assert rd["floor_branch"] == "off_axis" and rec["blend_lambda"] == 0.0 and not rec["acted"]
    np.testing.assert_array_equal(st.psi, st_d.psi)


def test_blend_lands_q_min_on_the_blended_level_inside_the_band_and_is_continuous_at_the_boundary():
    """Inside the band the region is searched for the blended level (q_min on it where met); at the boundary
    (plain q0 = level) the blended state meets the default branch (a) state."""
    sol = QC._reversed_shear()
    q = QC._fq(sol, sol.steady_state(RF))
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, 1.0, transition="blend_margin")
    lam, lev = rec["blend_lambda"], rec["level_used"]
    qmn = float(np.min(q))
    assert 0.0 < lam < 1.0 and lev == pytest.approx(qmn + lam * (1.0 - qmn), rel=1e-14)
    assert rec["acted"] and rec["floor_branch"] == "off_axis" and rec["q_floor_met"] is False
    assert 0.0 <= rec["q_min_after"] - lev <= QC.Q_LAND
    # at the boundary (levels just above and just below the plain axis q): the blended state's q0 changes by an amount
    # that shrinks with the distance from the boundary (continuous); the default rule keeps its jump
    q0 = float(q[0])

    def gap(eps, **kw):
        a = SAW.resistive_floor_steady(sol, RF, q0 * (1.0 + eps), **kw)
        b = SAW.resistive_floor_steady(sol, RF, q0 * (1.0 - eps), **kw)
        assert a[1]["floor_branch"] == "axis" and b[1]["floor_branch"] == "off_axis"
        return abs(float(QC._fq(a[2], a[0])[0]) - float(QC._fq(b[2], b[0])[0]))
    g1, g2 = gap(1e-5, transition="blend_margin"), gap(1e-6, transition="blend_margin")
    p1, p2 = gap(1e-5), gap(1e-6)
    assert g2 <= 0.2 * g1 and g2 < 1e-2 * p2 and p2 >= 0.5 * p1


def test_frozen_branch_outside_a_fit_is_the_plain_branch_and_forced_branches_are_recorded():
    lag, kin, src = LR._fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0, q_floor_branch_freeze="first_assembly")
    a = CM.assemble(lag, kin, src, {}, cfg, QC.IP_MET)
    assert a["core"]["branch_freeze"].startswith("no open freeze")
    a_d = CM.assemble(lag, kin, src, {}, CM.CurrentModelSettings(r0=GEO.R0), QC.IP_MET)
    np.testing.assert_array_equal(a["state"].psi, a_d["state"].psi)
    sol = QC._solver(QC.IP_MET)           # plain branch (a)
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, 1.0, branch="off_axis")
    assert rec["floor_branch"] == "off_axis" and rec["floor_branch_plain"] == "axis" and not rec["acted"]
    sol_b = QC._reversed_shear()          # plain branch (b): forced (a) flattens from the axis
    st, rec, s2 = SAW.resistive_floor_steady(sol_b, RF, 1.0, branch="axis")
    assert rec["floor_branch"] == "axis" and rec["floor_branch_plain"] == "off_axis" and rec["acted"]
    _, rec, _ = SAW.resistive_floor_steady(sol, RF, 0.5, branch="axis")
    assert rec["floor_branch"] == "axis" and not rec["acted"] and "nothing to flatten" in rec["branch_frozen_note"]


# ====================================================================== pieces of the options tested directly
def test_total_current_cut_cell_is_the_overlap_weighted_blend_continuous_in_the_edge():
    """The cut cell takes the overlap-weighted source change, so the flat-at-level state is continuous in the edge:
    across a face the state changes in proportion to the distance from it (not by a whole cell's change)."""
    sol = QC._solver(QC.IP_MET)
    f_k = float(RF[30])

    def gap(eps):
        a, b = (SAW.ResistiveFloor(sol, 1.0, rho_inv=f_k * (1.0 + s * eps), rho_faces=RF,
                                   total_current="flat_at_level").with_solver() for s in (1.0, -1.0))
        sa, sb = a.steady_state(RF), b.steady_state(RF)
        return float(np.max(np.abs(QC._fq(a, sa) - QC._fq(b, sb))))
    g1, g2 = gap(1e-6), gap(1e-8)
    assert 0.0 < g2 <= 0.2 * g1
    # the change on a cut cell lies strictly between none and the whole cell's change
    c = SAW.ResistiveFloor(sol, 1.0, rho_inv=RF[30], rho_faces=RF).with_solver().coefficients(0.0, RF)
    r_mid = 0.5 * (RF[30] + RF[31])
    ds_mid = SAW.level_flattening_source_change(RF, c, True, r_mid, 1.0)
    ds_full = SAW.level_flattening_source_change(RF, c, True, RF[31], 1.0)
    assert 0.0 < ds_mid[30] / ds_full[30] < 1.0


@pytest.mark.parametrize("tc", ["flat_at_level", "flat_at_level_own_region"])
def test_the_source_hook_changes_the_non_inductive_current_with_the_sources(tc):
    """Where the option changes the cell sources, <J_NI.B> on the cells is changed with them: the coefficients keep
    the relation s_c = V' <J_NI.B> h / (2 pi F^2) of the plain solver, so the outputs stay consistent."""
    sol = QC._solver(QC.IP_LOW_Q)
    _, rec, s2 = SAW.resistive_floor_steady(sol, RF, 1.0, total_current=tc)
    h = np.diff(RF)
    for s_ in (sol, s2):
        c = s_.coefficients(0.0, RF)
        vc, fc = c.geo_c["dvol_drho"], c.geo_c["fpol"]
        np.testing.assert_allclose(c.s_c, vc * c.jni_c / (2.0 * np.pi * fc ** 2) * h, rtol=1e-12, atol=0.0)
    c0, c1 = sol.coefficients(0.0, RF), s2.coefficients(0.0, RF)
    assert np.any(c1.jni_c != c0.jni_c)


def test_blend_lambda_is_the_plain_axis_q_margin_fraction():
    """``lambda = 1 - (q0 - L) / (L - q_min)`` from the plain state: a level set so that ``q0 - L`` is half (a
    quarter) of ``L - q_min`` gives lambda 1/2 (3/4), and the blended level is ``q_min + lambda (L - q_min)``."""
    sol = QC._reversed_shear()
    q = SAW.face_q(sol.steady_state(RF), sol.geometry_at(0.0).phi_b)
    q0, qmn = float(q[0]), float(np.min(q[np.isfinite(q)]))
    for frac, lam in ((0.5, 0.5), (0.25, 0.75)):
        level = (q0 + frac * qmn) / (1.0 + frac)          # (q0 - level) / (level - q_min) = frac
        _, rec, _ = SAW.resistive_floor_steady(sol, RF, level, transition="blend_margin")
        assert rec["blend_lambda"] == pytest.approx(lam, rel=1e-12)
        assert rec["level_used"] == pytest.approx(qmn + lam * (level - qmn), rel=1e-12)


# ====================================================================== records do not steer; record keys
def test_the_frozen_branch_is_carried_by_the_freeze_not_read_from_the_record(monkeypatch):
    """The branch the freeze holds is the one the floor used (filled through ``branch_holder``), whatever the record
    says: a record with a garbage branch does not change it."""
    lag, kin, src = LR._fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0, q_floor_branch_freeze="first_assembly")
    real = SAW.resistive_floor_steady

    def garbage_record(*a, **k):
        st, rec, s_ = real(*a, **k)
        return st, dict(rec, floor_branch="garbage"), s_
    sol_m = CM.build_solver(lag, lag["jbs"], QC.IP_MET)
    hold, tok = CM.open_branch_freeze()
    try:
        with monkeypatch.context() as mp:
            mp.setattr(CM._saw, "resistive_floor_steady", garbage_record)
            CM._floor_with_options(sol_m, RF, cfg)
    finally:
        CM.close_branch_freeze(tok)
    assert hold["branch"] == "axis"
    used = {}
    real(QC._reversed_shear(), RF, 1.0, branch="axis", branch_holder=used)
    assert used == dict(branch="axis")
    real(QC._reversed_shear(), RF, 1.0, branch_holder=used)
    assert used == dict(branch="off_axis")


@pytest.mark.parametrize("tc", ["flat_at_level", "flat_at_level_own_region"])
def test_total_current_options_record_met_outside_the_region_under_its_own_key(tc):
    sol = QC._solver(QC.IP_LOW_Q)
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, 0.9, total_current=tc)
    q = QC._fq(s2, st)
    assert rec["q_floor_met_outside_region"] is bool(np.all(q[RF > rec["rho_floor"]] >= 0.9))
    assert rec["q_floor_met_outside_region_definition"] == SAW.Q_FLOOR_MET_OUTSIDE_REGION_DEFINITION
    assert rec["total_current_region"] == SAW.TOTAL_CURRENT_REGION_DEFINITIONS[tc]
    _, rd, _ = SAW.resistive_floor_steady(sol, RF, 0.9)
    assert "q_floor_met_outside_region" not in rd and "total_current_region" not in rd


# ====================================================================== total current with its own region
_OWN_CASES = [(QC.IP_MET, 1.0), (QC.IP_LOW_Q, 1.0), (QC.IP_LOW_Q, 0.9)]


def _own_criterion(sol, rf, level, r):
    """min q - level on the faces outside ``r`` of the flat-at-level state with the edge at ``r`` (the option's
    criterion), from the public pieces."""
    s_ = SAW.ResistiveFloor(sol, level, rho_inv=r, rho_faces=rf, total_current="flat_at_level").with_solver()
    q = QC._fq(s_, s_.steady_state(rf))
    return float(np.min(q[rf > r])) - level


@pytest.mark.parametrize("ip, level", _OWN_CASES)
@pytest.mark.parametrize("n", [100, 200, 400])
def test_own_region_option_meets_the_level_outside_its_region_and_its_region_is_the_smallest(ip, level, n):
    """Its own region: q is the level inside (to round-off), at or above it on every face outside; Ip is the solver's
    Ip condition; the region lies inside the plain outermost crossing, and the edge one face further in fails the
    criterion (the scan's bracket)."""
    sol = QC._solver(ip)
    rf = CurrentState.uniform_faces(n)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, level, total_current="flat_at_level_own_region")
    q = QC._fq(s2, st)
    r = rec["rho_floor"]
    assert rec["floor_branch"] == "axis" and rec["acted"]
    assert np.max(np.abs(q[rf <= r] - level)) <= FLAT_ATOL and rec["q_flat_max_abs_dev"] <= FLAT_ATOL
    assert np.all(q[rf > r] >= level) and rec["q_floor_met_outside_region"] is True
    assert rec["q_min_outside_region"] >= level
    assert s2.profiles(st)["ip"] == pytest.approx(sol.profiles(st0)["ip"], rel=IP_RTOL)
    plain_out = SAW.floor_region(rf, QC._fq(sol, st0), level)["rho_outer"]
    assert 0.0 < r <= plain_out
    k = int(np.searchsorted(rf, r))          # the first face at or beyond the edge; the face before it fails
    assert rf[k - 1] < r and (k - 1 == 0 or _own_criterion(sol, rf, level, float(rf[k - 1])) < 0.0)
    assert rec["total_current"] == "flat_at_level_own_region" and np.isfinite(rec["total_current_ni_added_over_ip"])


@pytest.mark.parametrize("ip, level", _OWN_CASES)
@pytest.mark.parametrize("n", [100, 200, 400])
def test_flat_at_level_state_with_its_edge_at_the_plain_outermost_crossing_meets_the_level_outside(ip, level, n):
    """The argument that the own-region search always finds an edge: with the edge at the plain state's outermost
    upward crossing the enclosed current of q = level equals the plain one there, so the state outside is the plain
    state (exact in the continuum; with the cut cell blended, approximately), which is at or above the level."""
    sol = QC._solver(ip)
    rf = CurrentState.uniform_faces(n)
    r = SAW.floor_region(rf, QC._fq(sol, sol.steady_state(rf)), level)["rho_outer"]
    assert _own_criterion(sol, rf, level, r) >= 0.0
