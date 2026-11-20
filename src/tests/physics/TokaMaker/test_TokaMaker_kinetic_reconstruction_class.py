#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""TokaMaker tests: the user class of the kinetic reconstruction (``reconstruction.kinetic_reconstruction``).

Interface checks (no solver): the mode is required and only ``'slice'`` is implemented (the later modes raise
``NotImplementedError``), the device description is checked, the settings objects are the helper package's own with
their defaults, and the inputs of a slice are selected by time without any tolerance.

One small end-to-end fit (NOT marked slow), in one child interpreter at 2 threads:

* mesh: a coarse DIII-D-like mesh built in code (``kinetic_fit_testing.coarse_mesh``, plasma resolution 5 cm);
* truth: the in-code synthetic L-mode slice of ``kinetic_fit_testing`` (recipe profiles and boundary, Ip 1.3 MA),
  solved to consistency with the DEFAULT current model (q-floor core, bootstrap multiplier 1) in generation mode;
* data: the synthetic measurement set of ``kinetic_fit_testing`` on the truth (40 flux loops, 60 finite-length
  probes, coils, Ip, 16 MSE chords) with seeded Gaussian noise; anchor = the truth's own boundary; start = the truth's
  flux on the mesh nodes with the free parameter moved off its true value;
* fit: through the class, one free parameter (the bootstrap multiplier ``s_bs``; the plasma-current target joins as
  the fit's default ``ip_target='fitted'`` makes it), every other setting the helper package's default.

Checked: the fit's status is a convergence, each fitted parameter returns to its true value within 3 of its own Laplace
sigmas (the noisy-closure bound of ``test_TokaMaker_kinetic_fit_solver``), the g-file of the fitted equilibrium is
written by TokaMaker's writer and reads back, and the class gives EXACTLY the result of calling
``kinetic.fit.reconstruct`` directly with the same inputs (every numeric leaf of the fit result and record, and the
solver's final flux, equal; timings excluded).

Not covered here: the fine-mesh fit and its sensitivity tests (``test_TokaMaker_kinetic_fit_solver``, slow), the
correction amplitudes and other free parameters, MSE without E_r correction, the opt-in core treatments, a second run
on the same object, and wall time on the registered mesh.
"""
import json
import math
import os
import re
import sys
import time

import numpy as np
import pytest

from kinetic_fit_testing import (EC, IP, JPHI, LCFS, MESH, R0, B0, _measurements, _sensors, coarse_mesh,
                                 lmode_boundary, lmode_kinetics, prof, run_child)

_SEED = 20261120
_VSC = {"F9A": 1.0, "F9B": -1.0}
_FREE = ("s_bs",)
_TRUTH = {"s_bs": 1.0}
_P0 = {"s_bs": 1.2}
_THREADS = 2
_SKIP = re.compile(r"(^|_)wall|_s$|perf")


# ------------------------------------------------------------------ comparison of two results, leaf by leaf
def _leaves(obj, path, out):
    """Flatten ``obj`` to ``{path: value}``: numbers, strings, booleans, None; dataclass fields and dict items by
    name; arrays and sequences by index; any other object by its type name (no memory address)."""
    import dataclasses
    key = path.rsplit("/", 1)[-1]
    if _SKIP.search(key):
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _leaves(getattr(obj, f.name), f"{path}/{f.name}", out)
    elif isinstance(obj, dict):
        for k in obj:
            _leaves(obj[k], f"{path}/{k}", out)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in "biuf":
            for i, v in enumerate(obj.ravel().tolist()):
                out[f"{path}[{i}]"] = v
        else:
            for i, v in enumerate(obj.ravel().tolist()):
                _leaves(v, f"{path}[{i}]", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _leaves(v, f"{path}[{i}]", out)
    elif isinstance(obj, (np.floating, np.integer, np.bool_)):
        out[path] = obj.item()
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        out[path] = obj
    else:
        out[path] = f"<{type(obj).__name__}>"


def compare(a, b):
    """Leaf-by-leaf comparison: counts of equal and different numeric leaves, other differences, and examples."""
    la, lb = {}, {}
    _leaves(a, "", la)
    _leaves(b, "", lb)
    res = dict(num_equal=0, num_diff=0, other_diff=0, only_one=len(set(la) ^ set(lb)), examples=[])
    for k in sorted(set(la) & set(lb)):
        x, y = la[k], lb[k]
        num = isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool)
        same = x == y or (num and isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))
        if num:
            res["num_equal" if same else "num_diff"] += 1
        elif not same:
            res["other_diff"] += 1
        if not same and len(res["examples"]) < 20:
            res["examples"].append(f"{k}: {x!r} vs {y!r}")
    res["examples"] += [f"only in one: {k}" for k in sorted(set(la) ^ set(lb))[:10]]
    return res


# ------------------------------------------------------------------ the end-to-end child
def child(out):
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker.kinetic import current_model as CM
    from OpenFUSIONToolkit.TokaMaker.kinetic import fit as FIT
    from OpenFUSIONToolkit.TokaMaker.kinetic import forward as FW
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.reconstruction import kinetic_reconstruction, kinetic_slice_data
    from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
    rec = {}
    t_all = time.perf_counter()
    env = OFT_env(nthreads=_THREADS)
    mesh_file = coarse_mesh(os.path.join(os.getcwd(), "kinetic_coarse_mesh.h5"))
    rec["n_cells"] = int(load_gs_mesh(mesh_file)[1].shape[0])
    pn, ne, te, ni, ti, zeff = lmode_kinetics()
    kin = CM.KineticProfiles(pn, ne, te, ni, ti, zeff, source="in-code synthetic L-mode (recipe profiles)")
    src = CM.SourceProfiles(pn, source="none (no driven current, no fast ions)")
    # truth: the default current model in generation mode on the coarse mesh
    t0 = time.perf_counter()
    fm_t = FW.ForwardModel(load_gs_mesh(mesh_file), R0 * B0, None, lmode_boundary(),
                           settings=FW.ForwardSettings(vsc=dict(_VSC)), oft_env=env, case_id="truth", ip=IP,
                           generation_coil_reg=1e-2)
    rz = np.array([(R0 + dr, z) for dr, z in LCFS])
    a = 0.5 * (rz[:, 0].max() - rz[:, 0].min())
    fm_t.gs.init_psi(0.5 * (rz[:, 0].max() + rz[:, 0].min()), 0.0, a,
                     0.5 * (rz[:, 1].max() - rz[:, 1].min()) / a, 0.3)
    xj = np.linspace(0.0, 1.0, 129)
    pk = EC * (np.interp(xj, pn, ne) * np.interp(xj, pn, te) + np.interp(xj, pn, ni) * np.interp(xj, pn, ti))
    init = (xj, 1e6 * prof(xj, JPHI), pk)
    tr = CM.consistent_solve(fm_t, kin, src, dict(_TRUTH), CM.CurrentModelSettings(r0=R0), label="truth",
                             initial_profiles=init)
    rec["truth"] = dict(status=tr.status, passes=tr.passes, n_solves=fm_t.n_solves, wall=time.perf_counter() - t0,
                        stats=tr.solve.stats if tr.converged else None)
    if not tr.converged:
        json.dump(rec, open(out, "w"), indent=1, default=str)
        return
    gs_t = fm_t.gs
    geo = _sensors(gs_t)
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted")) for n in gs_t.coil_sets}
    ms = _measurements(gs_t, geo, roles, IP, np.random.default_rng(_SEED))
    bnd = np.asarray(gs_t.trace_surf(0.999), float)
    psi0 = np.array(gs_t.get_psi(False))
    rec["ip_measured"] = float(ms.groups["ip"].measured[0])
    # the fit through the class
    data = kinetic_slice_data(ms, kin, src, bnd, R0 * B0, start=psi0, coil_roles=roles, initial_profiles=init,
                              params0=dict(_P0))
    recon = kinetic_reconstruction(env, mesh_file, machine={"name": "DIII-D-like test case", "vsc": dict(_VSC), "r0": R0},
                                   fit_settings=FIT.FitSettings(free=_FREE))
    t0 = time.perf_counter()
    res = recon.run(data, ms.time, mode="slice", label="slice")
    rec["class"] = dict(status=res.status, free=list(res.fit.free), theta=res.theta.tolist(),
                        sigma=res.sigma.tolist(), chi2=res.chi2, n_solves=res.record["n_solves"],
                        wall=time.perf_counter() - t0)
    try:
        grec = res.save_geqdsk("kinetic_fit.geqdsk")
        g = read_eqdsk("kinetic_fit.geqdsk")
        rec["geqdsk"] = dict(record=grec, ip=float(g["ip"]), nr=g["nr"], nz=g["nz"],
                             ip_fit=float(recon.gs.get_stats()["Ip"]))
    except Exception as exc:  # noqa: BLE001
        rec["geqdsk"] = dict(error=f"{type(exc).__name__}: {exc}")
    # the same fit by a direct call of the helper package
    t0 = time.perf_counter()
    fm_d = FW.ForwardModel(load_gs_mesh(mesh_file), R0 * B0, ms, bnd, coil_roles=roles,
                           settings=FW.ForwardSettings(vsc=dict(_VSC)), oft_env=env, case_id="slice")
    fr = FIT.reconstruct(fm_d, kin, src, CM.CurrentModelSettings(r0=R0), settings=FIT.FitSettings(free=_FREE),
                         params0=dict(_P0), start=psi0, initial_profiles=init, label="slice")
    rec["direct"] = dict(status=fr.status, wall=time.perf_counter() - t0)
    rec["identity"] = compare(dict(fit=res.fit, psi=np.array(recon.gs.get_psi(False))),
                              dict(fit=fr, psi=np.array(fm_d.gs.get_psi(False))))
    rec["wall_total"] = time.perf_counter() - t_all
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    try:
        return run_child(__file__, tmp_path_factory.mktemp("kinetic_class"), threads=_THREADS)
    except RuntimeError as exc:
        pytest.fail(str(exc))


def test_end_to_end_truth_consistent(result):
    assert result["truth"]["status"] == "consistent", result["truth"]


def test_end_to_end_fit_returns_to_truth(result):
    c = result["class"]
    assert c["status"].startswith("converged"), c["status"]
    truth = dict(_TRUTH, ip_rel=IP / result["ip_measured"] - 1.0)
    for name, th, sg in zip(c["free"], c["theta"], c["sigma"]):
        assert np.isfinite(sg) and sg > 0, (name, sg)
        assert abs(th - truth[name]) <= 3.0 * sg, (name, th, truth[name], sg)


def test_end_to_end_geqdsk_written(result):
    g = result["geqdsk"]
    assert "error" not in g, g
    assert g["record"]["pressure_frame"] == "full"
    assert (g["nr"], g["nz"]) == (65, 65)
    assert abs(g["ip"]) == pytest.approx(g["ip_fit"], rel=1e-6)


def test_class_equals_direct_call(result):
    idn = result["identity"]
    assert result["direct"]["status"] == result["class"]["status"]
    assert idn["num_diff"] == 0 and idn["other_diff"] == 0 and idn["only_one"] == 0, idn
    assert idn["num_equal"] > 1000, idn


# ------------------------------------------------------------------ interface checks (no solver)
def _recon(**kw):
    from OpenFUSIONToolkit.TokaMaker.reconstruction import kinetic_reconstruction
    args = dict(machine={"vsc": dict(_VSC), "r0": R0})
    args.update(kw)
    return kinetic_reconstruction(None, MESH, **args)


def test_settings_are_the_helper_package_defaults():
    from OpenFUSIONToolkit.TokaMaker.kinetic import current_model as CM
    from OpenFUSIONToolkit.TokaMaker.kinetic import fit as FIT
    from OpenFUSIONToolkit.TokaMaker.kinetic import forward as FW
    r = _recon()
    assert r.forward_settings == FW.ForwardSettings(vsc=dict(_VSC))
    assert r.fit_settings == FIT.FitSettings()
    assert r.model_settings == CM.CurrentModelSettings(r0=R0)
    assert r.speed is None and r.forward_model is None and r.gs is None
    given = FIT.FitSettings(free=("s_bs",))
    assert _recon(fit_settings=given).fit_settings is given


def test_machine_description_checked():
    from OpenFUSIONToolkit.TokaMaker.kinetic import forward as FW
    with pytest.raises(ValueError, match="vsc"):
        _recon(machine={"r0": R0})
    with pytest.raises(ValueError, match="unknown machine"):
        _recon(machine={"vsc": dict(_VSC), "r0": R0, "shot": 1})
    with pytest.raises(ValueError, match="reference major radius"):
        _recon(machine={"vsc": dict(_VSC)})
    with pytest.raises(ValueError, match="disagrees"):
        _recon(forward_settings=FW.ForwardSettings(vsc={"F9A": 1.0}))
    same = FW.ForwardSettings(vsc=dict(_VSC))
    assert _recon(forward_settings=same).forward_settings is same


def test_mode_required_and_later_modes_refused():
    r = _recon()
    with pytest.raises(TypeError):
        r.run({}, 1.0)
    for mode in ("slices", "series"):
        with pytest.raises(NotImplementedError, match="later pull request"):
            r.run({}, 1.0, mode=mode)
    with pytest.raises(ValueError, match="mode must be one of"):
        r.run({}, 1.0, mode="auto")


def test_slice_selected_by_time_without_tolerance():
    from OpenFUSIONToolkit.TokaMaker.kinetic.interfaces import MeasurementSet
    from OpenFUSIONToolkit.TokaMaker.reconstruction import kinetic_reconstruction, kinetic_slice_data
    d = kinetic_slice_data(MeasurementSet("test", 1.0, 0.01, {}), None, None, None, -3.4, start=None)
    assert kinetic_reconstruction._select(d, 1.0) is d
    assert kinetic_reconstruction._select({1.0: d}, 1.0) is d
    with pytest.raises(ValueError, match="differs"):
        kinetic_reconstruction._select(d, 1.0 + 1e-12)
    with pytest.raises(ValueError, match="no inputs"):
        kinetic_reconstruction._select({1.0: d}, 2.0)


def test_psi_start_of_a_new_solver_needs_initial_profiles():
    from OpenFUSIONToolkit.TokaMaker.kinetic.interfaces import MeasurementSet
    from OpenFUSIONToolkit.TokaMaker.reconstruction import kinetic_slice_data
    d = kinetic_slice_data(MeasurementSet("test", 1.0, 0.01, {}), None, None, None, -3.4, start=np.zeros(3))
    r = _recon()
    with pytest.raises(ValueError, match="initial_profiles"):
        r.run(d, 1.0, mode="slice")
    assert r.forward_model is None


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    child(sys.argv[1])
