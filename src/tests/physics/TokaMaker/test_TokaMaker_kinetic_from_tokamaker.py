#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""TokaMaker tests: kinetic helper package, the current model's geometry read from a solved equilibrium
(``geometry.from_tokamaker``) on the ITER reference equilibrium of the other TokaMaker tests
(``tokamaker_testing.solve_ITER_reference`` inputs, order 2, solver defaults), read with the domain-edge rule
(default ``psi_n_edge="auto"``).  Checked:

(a) the edge choice: 0.995 with a stated reason (no 2-D grid; route agreement not checked unless given); never
    beyond 0.999; the previous behavior is available as ``psi_n_edge=None``;
(b) internal consistency: TokaMaker q vs the geometric q at the domain samples (same trace, round-off) and the Phi
    identity of the resampled geometry;
(c) independent routes: domain + layer volume and toroidal flux against TokaMaker's own volume and toroidal-flux
    integrals (``get_globals``), and the layer's Ampere current at its outermost sample against Ip;
(d) a lumped-layer steady state on the 0.995 and 0.99 geometries: total Ip conserved, layer current small and
    positive, interior q independent of the cutoff.

Ported from the reconstruction's live-solver test, whose equilibrium (a synthetic case on a mesh and g-file that are
not part of this repository) is replaced here by the ITER case; every bound is the original one.  The original test
states bounds only, no reference values.
"""
import numpy as np
import pytest
from tokamaker_testing import mp_run, create_ITER_mesh, setup_ITER_gs, set_ITER_constraints, solve_ITER_eq
from OpenFUSIONToolkit import OFT_env

# numpy's trapezoid-rule integrator: numpy.trapezoid (numpy 2) or numpy.trapz (numpy 1), the same function
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

ITER_R0 = 6.2      # major radius of the ITER reference case [m] (F0 = 5.3 T * 6.2 m in setup_ITER_gs)


def _sigma(x):
    """Synthetic conductivity on psi_N (pedestal-like edge drop)."""
    return 1.0e8 * (0.02 + 0.98 * np.clip(1.0 - np.asarray(x) ** 2, 0.0, None) ** 1.5)


def _jni(x):
    """Synthetic <J_NI.B> on psi_N with an edge bootstrap-like peak."""
    x = np.asarray(x)
    return 2.0 * (1.5e5 * np.exp(-(x / 0.35) ** 2) + 4.0e5 * np.exp(-((x - 0.975) / 0.02) ** 2))


def run_ITER_from_tokamaker(fe_order, mp_q):
    from scipy.interpolate import PchipInterpolator
    from OpenFUSIONToolkit.TokaMaker.kinetic.diffusion import CurrentDiffusion, CurrentState, IpBoundary, OuterLayer
    from OpenFUSIONToolkit.TokaMaker.kinetic.geometry import choose_psi_n_edge, from_tokamaker
    rec = None
    try:
        create_ITER_mesh()
        gs = setup_ITER_gs(OFT_env(nthreads=-1), fe_order)
        set_ITER_constraints(gs)
        solve_ITER_eq(gs)
        rec = {}
        rec["settings"] = dict(nl_tol=gs.settings.nl_tol, urf=gs.settings.urf, maxits=gs.settings.maxits)
        ip_tm, _, vol_tm, _, _, tflux_tm, _ = gs.get_globals()
        R0 = ITER_R0
        geos = {}
        for key, kw in (("auto", {}), ("0.99", dict(psi_n_edge=0.99))):
            geo = from_tokamaker(gs, R0, **kw)
            lay = geo.meta["outer_layer_geometry"]
            phi_layer = lay.integral(lay.fpol * lay.avg_inv_r2 / (2.0 * np.pi))
            vol_dom = float(_trapezoid(geo.dvol_drho, geo.rho))
            rec[key] = dict(psi_n_edge=geo.meta["psi_n_edge"], edge_choice=geo.meta["edge_choice"],
                            q_geo_vs_tm=geo.meta["q_geometric_vs_tokamaker_max_rel"],
                            phi_identity=geo.consistency()["max_rel_error"],
                            volume_rel=(vol_dom + lay.total_volume) / abs(vol_tm) - 1.0,
                            tflux_rel=(geo.phi_b + phi_layer) / abs(tflux_tm) - 1.0,
                            ampere_last_rel=float(lay.ampere_current[-1] / abs(ip_tm) - 1.0),
                            layer_last_psi_n=float(lay.psi_n[-1]), tail_fraction=lay.meta["tail_volume_fraction"],
                            n_layer=int(lay.psi_n.size))
            geos[key] = geo
        rec["too_far_raises"] = False
        try:
            from_tokamaker(gs, R0, psi_n_edge=0.9995)
        except ValueError:
            rec["too_far_raises"] = True
        leg = from_tokamaker(gs, R0, psi_n_edge=None)
        rec["legacy"] = dict(psi_n_edge=leg.meta["psi_n_edge"], has_layer="outer_layer_geometry" in leg.meta,
                             edge_choice=leg.meta["edge_choice"])
        rec["rule_default"] = choose_psi_n_edge().psi_n_edge
        # lumped steady state on both cutoffs
        xs = np.linspace(0.02, 0.98, 300)
        qs = {}
        for key, geo in geos.items():
            lay = geo.meta["outer_layer_geometry"]
            rs = np.concatenate(([0.0], geo.meta["rho_samples"]))
            xsamp = np.concatenate(([0.0], geo.meta["psi_n_samples"]))
            layer = OuterLayer.from_profiles(lay, (lay.psi_n, _sigma(lay.psi_n)), (lay.psi_n, _jni(lay.psi_n)))
            sol = CurrentDiffusion(geo, (rs, _sigma(xsamp)), IpBoundary(abs(ip_tm)), j_ni=(rs, _jni(xsamp)),
                                   outer_layer=layer)
            ss = sol.steady_state(CurrentState.uniform_faces(400))
            p = sol.profiles(ss)
            r2x = PchipInterpolator(rs, xsamp)
            qs[key] = PchipInterpolator(r2x(p["rho_faces"]), p["q"])(xs)
            rec[key].update(ip_rel=p["ip"] / abs(ip_tm) - 1.0, layer_fraction=p["outer_layer_current"] / abs(ip_tm),
                            q95=p["q95"], li3=p["li3"], surfaces=p["surfaces"])
        rec["interior_q_cutoff_diff"] = float(np.max(np.abs(qs["0.99"] / qs["auto"] - 1.0)))
    finally:
        mp_q.put(rec)


@pytest.fixture(scope="module")
def result():
    out = mp_run(run_ITER_from_tokamaker, (2,), timeout=300)
    assert out is not None
    return out


@pytest.mark.slow
def test_default_solver_settings(result):
    assert result["settings"] == dict(nl_tol=1e-6, urf=0.2, maxits=40)


@pytest.mark.slow
def test_edge_default_and_requested_cutoffs(result):
    assert result["auto"]["psi_n_edge"] == result["rule_default"] == 0.995
    assert "default" in result["auto"]["edge_choice"] and "route_disagreement" in result["auto"]["edge_choice"]
    assert result["0.99"]["psi_n_edge"] == 0.99 and "requested" in result["0.99"]["edge_choice"]
    assert result["too_far_raises"]
    assert result["legacy"]["psi_n_edge"] == 1.0 and not result["legacy"]["has_layer"]
    assert "previous behavior" in result["legacy"]["edge_choice"]


@pytest.mark.slow
def test_internal_consistency(result):
    for key in ("auto", "0.99"):
        r = result[key]
        # same TokaMaker trace for q and the averages: agreement to round-off
        assert r["q_geo_vs_tm"] < 1e-10, r
        # Phi identity of the resampled geometry: the construction accuracy bounded in the geometry tests
        # (1e-4 at 200 samples)
        assert r["phi_identity"] < 1e-4, r


@pytest.mark.slow
def test_independent_routes_agree(result):
    """Domain + layer volume and toroidal flux vs TokaMaker's own integrals: two geometry routes, bound 1e-3
    (the domain-edge rule's route-agreement tolerance)."""
    for key in ("auto", "0.99"):
        r = result[key]
        assert abs(r["volume_rel"]) < 1e-3, r
        assert abs(r["tflux_rel"]) < 1e-3, r


@pytest.mark.slow
def test_layer_ampere_current_matches_ip(result):
    """The layer's Ampere current at its outermost sample (psi_N = 1 - 1e-2 (1 - c)) vs Ip; bound 1e-3 (the
    domain-edge rule's route-agreement tolerance)."""
    for key in ("auto", "0.99"):
        r = result[key]
        assert abs(r["ampere_last_rel"]) < 1e-3, r


@pytest.mark.slow
def test_lumped_steady_state_on_tokamaker_geometry(result):
    for key in ("auto", "0.99"):
        r = result[key]
        assert abs(r["ip_rel"]) < 1e-14
        assert 0.0 < r["layer_fraction"] < 0.01
        assert "last closed surface" in r["surfaces"]["ip"]
        assert np.isfinite(r["q95"]) and np.isfinite(r["li3"])
    # interior q independent of the cutoff: <= 1e-3
    assert result["interior_q_cutoff_diff"] <= 1e-3
