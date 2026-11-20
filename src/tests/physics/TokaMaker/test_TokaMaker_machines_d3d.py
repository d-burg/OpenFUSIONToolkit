#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
# TokaMaker tests: machines.d3d (device layer of kinetic reconstruction).  No network and no experimental data:
# synthetic machine and k-files written by the tests, fetch commands composed and run only with a fake runner.
import ast
import inspect
import os
import re

import numpy as np
import pytest

from OpenFUSIONToolkit.machines import d3d

TWO_PI = 2.0 * np.pi
LOOPS = ["PSF1A", "LOOPB", "LOOPC"]
PROBES = ["PRB1", "PRB2"]


def _fmt(v):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, str):
        return f"'{v}'"
    return repr(float(v))


def _block(name, d, per_line=1):
    items = [f"{k} = " + ", ".join(_fmt(x) for x in v) for k, v in d.items()]
    lines = ["  " + "  ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return f"&{name}\n" + "\n".join(lines) + "\n/\n"


def _mhdin(per_line):
    d = {"LPNAME": LOOPS, "RSI": [1.0, 1.5, 2.0], "ZSI": [0.5, 1.0, -0.5], "MPNAM2": PROBES,
         "XMP2": [1.1, 2.2], "YMP2": [0.1, -0.2], "AMP2": [90.0, 30.0], "SMP2": [0.1, -0.05],
         "RF": [1.0 + 0.05 * i for i in range(18)], "ZF": [0.1 * i - 0.9 for i in range(18)],
         "WF": [0.05] * 18, "HF": [0.2] * 18, "TURNFC": [58.0] * 18, "FCID": list(range(1, 19)),
         "ECID": [1, 1, 2, 3, 4, 5, 6], "RE": [0.8] * 7, "ZE": [0.0] * 7, "WE": [0.01] * 7, "HE": [0.1] * 7,
         "NSMP2": [25]}
    return _block("MACHINEIN", d, per_line) + "comment: synthetic description\n"


@pytest.fixture
def device(tmp_path):
    p = tmp_path / "mhdin.dat"
    p.write_text(_mhdin(per_line=3))
    return d3d.device_from_mhdin(str(p))


def _in1(**over):
    d = dict(ISHOT=[12345.0], ITIME=[2500.0], IAVEM=[10.0], SERROR=[0.03], VBIT=[10.0], PLASMA=[1.0e6], FWTCUR=[1.0],
             BITIP=[100.0], COILS=[0.0, 0.5, -0.25], FWTSI=[1.0, 1.0, 1.0], PSIBIT=[1e-3, 2e-3, 1e-3], SIREF=[0.2],
             FITSIREF=[False], KERSIL=[3.0], NSLREF=[1.0], IECURR=[1.0], EXPMP2=[0.1, -0.2], FWTMP2=[1.0, 0.0],
             BITMPI=[1e-4, 1e-3], BRSP=[1.0e3] * 18 + [0.0, 0.0], FWTFC=[1.0] * 17 + [0.0], BITFC=[10.0] * 18,
             ECURRT=[2.0e3] * 6, FWTEC=[0.0] * 6, BITEC=[10.0] * 6, DFLUX=[1.5], FWTDLC=[1.0], SIGDLC=[0.1])
    d.update(over)
    return {k: v for k, v in d.items() if v is not None}


def _ins(**over):
    d = dict(TGAMMA=[0.1, 0.05, -0.05], SGAMMA=[0.01, 0.02, 0.01], FWTGAM=[1.0, 1.0, 0.0], RRRGAM=[1.8, 1.9, 2.0],
             ZZZGAM=[0.0, 0.01, 0.02], AA1GAM=[0.7, 0.71, 0.72], AA2GAM=[0.6, 0.61, 0.62], AA3GAM=[0.1, 0.11, 0.12],
             AA4GAM=[0.02, 0.03, 0.04], AA5GAM=[0.5, 0.5, 0.5], AA6GAM=[0.0, 0.0, 0.0], AA7GAM=[0.3, 0.3, 0.3])
    d.update(over)
    return {k: v for k, v in d.items() if v is not None}


def _kfile(tmp_path, in1=None, ins=None, extra=""):
    p = tmp_path / "kfile_test"
    p.write_text(_block("IN1", in1 or _in1(), per_line=4) + _block("INS", ins or _ins(), per_line=2) + extra)
    return str(p)


# ------------------------------------------------------------------------------------------------ naming rule
def test_module_source_carries_no_private_names():
    src = inspect.getsource(d3d)
    assert not re.search(r"(^|[^0-9.])(1[4-9][0-9]{4}|20[0-9]{4})([^0-9]|$)", src)   # no discharge-like numbers
    assert not re.search(r"[A-Za-z0-9-]+\.(com|edu|gov|org|net)\b", src)              # no host names
    assert not re.search(r"\w@\w", src)                                              # no user@host
    assert not re.search(r"/(Users|home|mnt|fusion|cscratch)/", src)                 # no private paths
    assert not re.search(r"(EFIT|ZIPFIT)0[0-9]", src)                                # tree names are arguments
    assert src.isascii()


def test_node_spec_trees_are_arguments():
    with pytest.raises(ValueError, match="no node set"):
        d3d.node_spec()
    spec = d3d.node_spec(efit_tree="MYEFIT", profile_tree="MYPROF", transport_tree="MYTC", beams=True)
    trees = {t for t, _ in spec}
    assert trees == {"MYEFIT", "MYPROF", "MYTC", "NB"}
    assert ("MYEFIT", "\\MYEFIT::TOP.MEASUREMENTS.SILOPT") in spec
    assert ("MYPROF", "\\TOP.PROFILES.EDENSFIT") in spec
    assert ("MYTC", "\\MYTC::TOP.HCD:J_CD") in spec
    assert ("NB", "\\NB::TOP.NB30L.PINJ_30L") in spec and ("NB", "\\NB::TOP.NB33R.GAS") in spec
    assert len(spec) == len(d3d.EFIT_MEASUREMENT_NODES) + 5 + len(d3d.TRANSPORT_CODE_POINTS) + 4 * 8
    with pytest.raises(ValueError, match="tab or newline"):
        d3d.node_spec(extra=[("A\t", "x")])


# ------------------------------------------------------------------------------------------------ parameter sets
def test_parameter_sets_are_the_engine_values():
    th = d3d.zeff_rule_thresholds()
    from OpenFUSIONToolkit.TokaMaker.kinetic import zeff_sources as Z
    assert isinstance(th, Z.ZeffRuleThresholds) and Z.ZeffRuleThresholds.coerce(th) is th
    pinned = dict(coverage_rho_inner_max=0.30, coverage_rho_outer_min=0.80, coverage_min_points=8,
                  coverage_window_s=0.025, placeholder_rel_spread_min=1e-3, placeholder_rel_sigma_max=1.0,
                  ni_fraction_min=0.0, ni_fraction_warn=0.1, points_abs_median_z_max=2.0, points_ratio_min=2.0 / 3.0,
                  points_ratio_max=1.5, points_rho_max=0.9, points_core_rho_max=0.5, line_rel_sigma_max=0.25,
                  copy_rel_diff_min=1e-3, flat_ptp_min=0.02, core_cv_advisory=0.05, tension_z_max=2.0,
                  psi_n_test_max=0.9, psi_n_plasma_max=1.0, psi_n_core_median_max=0.5)
    assert th.as_dict() == pinned and list(th.as_dict()) == list(Z.THRESHOLD_KEYS)
    assert d3d.ZEFF_IMPURITY_Z == 6.0 and d3d.TC_MAX_GAP_S == 0.040
    assert d3d.EFIT_SIGMA_CONSTANTS == {"ERRSIL": 0.03, "SICONT": 1.0e-9, "NSLREF": 1}
    assert d3d.source_prior_widths() == {"fast_ion_pressure_scale": 0.20, "driven_current_scale": 0.25}
    assert set(d3d.PARAMETER_SETS) >= {"zeff_rule_thresholds", "efit_sigma_constants", "source_prior_widths"}


# ------------------------------------------------------------------------------------------------ namelists, device
def test_parse_namelists_blocks_repeats_indices_nulls():
    text = ("&IN1\n A = 1.0, 2*3.0 B=.TRUE. C='x/y' ! comment\n D(3) = 7.0 E = 1.5D+01\n F = 1.0, , 3.0\n"
            " FWT = 1.0 2.0\n/\n&INWANT\n FWT = 9.0\n/\n")
    nl = d3d.parse_namelists(text)
    assert nl["IN1"]["A"] == [1.0, 3.0, 3.0] and nl["IN1"]["B"] == [True] and nl["IN1"]["C"] == ["x/y"]
    assert nl["IN1"]["D"] == [None, None, 7.0] and nl["IN1"]["E"] == [15.0] and nl["IN1"]["F"] == [1.0, None, 3.0]
    assert nl["IN1"]["FWT"] == [1.0, 2.0] and nl["INWANT"]["FWT"] == [9.0]
    with pytest.raises(ValueError, match="multi-dimensional"):
        d3d.parse_namelists("&X\n A(1,2) = 1.0\n/\n")
    with pytest.raises(ValueError, match="not a number"):
        d3d.parse_namelists("&X\n A = 1.0 x1\n/\n")
    with pytest.raises(ValueError, match="no integer count"):
        d3d.parse_namelists("&X\n A = ****\n/\n")


def test_device_from_mhdin_matches_solver_reader(tmp_path, device):
    assert device.labels("flux_loop") == LOOPS and device.labels("bpol_probe") == PROBES
    assert device.flux_loops["r"] == [1.0, 1.5, 2.0] and device.probes["length"] == [0.1, -0.05]
    assert device.labels("pf_current") == list(d3d.F_COILS) + list(d3d.E_COILS)
    assert device.probe_n_avg == 25
    # the solver's reader on the same content written one key per line (the layout it reads)
    from OpenFUSIONToolkit.TokaMaker.util import read_mhdin
    p = tmp_path / "mhdin_one.dat"
    p.write_text(_mhdin(per_line=1))
    mdict, _ = read_mhdin(str(p), e_coil_names=list(d3d.E_COILS), f_coil_names=list(d3d.F_COILS))
    assert list(mdict["LPNAME"]) == LOOPS and list(mdict["MPNAM2"]) == PROBES
    assert [mdict["AMP2"][k] for k in PROBES] == device.probes["angle_deg"]
    for i, name in enumerate(d3d.F_COILS):
        r, z, w, h, t = mdict["FCOIL"][name]
        assert (r, z, w, h, t) == tuple(device.f_coils[k][i] for k in ("r", "z", "w", "h", "turns"))
    assert sorted(mdict["ECOIL"]) == sorted(device.e_coils["labels"])
    assert d3d.device_from_mhdin(str(p)).flux_loops == device.flux_loops


# ------------------------------------------------------------------------------------------------ k-file tables
def test_kfile_default_rules(tmp_path, device):
    st = d3d.kfile_tables(_kfile(tmp_path), device, er_corrected=True, efit_constants=d3d.EFIT_SIGMA_CONSTANTS)
    assert st.time == 2.5 and st.time_window == pytest.approx(0.02)
    assert st.meta["sigma_model_id"] == "efit_kersil:siref"
    fl = st.tables["flux_loop"]
    coils, r, ip, siref = np.array([0.0, 0.5, -0.25]), np.array([1.0, 1.5, 2.0]), 1.0e6, 0.2
    bit = 10.0 * np.array([1e-3, 2e-3, 1e-3])
    sig = np.maximum.reduce([0.03 * np.abs(coils), 1e-9 * r * ip, bit])
    sig[0] = max(0.03 * siref, 1e-9 * r[0] * ip, bit[0])
    assert np.array_equal(fl.measured, -TWO_PI * coils) and np.allclose(fl.sigma, TWO_PI * sig, rtol=1e-15)
    assert fl.meta["coils_form"] == "undetermined" and fl.meta["absolute_offset"] == -TWO_PI * siref
    pr = st.tables["bpol_probe"]
    assert np.allclose(pr.sigma, np.maximum(0.03 * np.abs([0.1, -0.2]), 10.0 * np.array([1e-4, 1e-3])))
    assert list(pr.valid) == [True, False]
    assert st.tables["ip"].sigma[0] == max(0.03 * 1e6, 10.0 * 100.0)
    pf = st.tables["pf_current"]
    assert pf.labels == list(d3d.F_COILS) + list(d3d.E_COILS) and pf.valid.all()
    assert pf.weight[17] == 1.0 and pf.meta["not_fitted_by_efit"][0] == "F9B"
    assert pf.meta["coil_roles"]["F1A"] == "fitted" and pf.meta["coil_roles"]["ECOILA"] == "held"
    dm = st.tables["diamagnetic_flux"]
    assert dm.measured[0] == pytest.approx(1.5e-3) and dm.sigma[0] == pytest.approx(1e-4)
    ms = st.tables["mse"]
    assert list(ms.valid) == [True, True, False]
    assert np.allclose(ms.geometry["coefficients"][0], [0.7, 0, 0, 0.5, 0.02, 0.1, 0.6, 0.3, 0.0])


def test_kfile_gauge_and_storage(tmp_path, device):
    kf = _kfile(tmp_path)
    rel = d3d.kfile_tables(kf, device, groups=["flux_loop"], flux_reference="PSF1A",
                           efit_constants=d3d.EFIT_SIGMA_CONSTANTS).tables["flux_loop"]
    assert rel.meta["coils_form_rule"] == "reference_entry_zero"
    kf0 = _kfile(tmp_path, in1=_in1(SIREF=[0.0]))
    ab = d3d.kfile_tables(kf0, device, groups=["flux_loop"], efit_constants=d3d.EFIT_SIGMA_CONSTANTS)
    assert ab.tables["flux_loop"].meta["coils_form"] == "absolute"
    with pytest.raises(ValueError, match="not in relative form"):
        d3d.kfile_tables(kf0, device, groups=["flux_loop"], flux_reference="PSF1A",
                         efit_constants=d3d.EFIT_SIGMA_CONSTANTS)
    fs = d3d.kfile_tables(_kfile(tmp_path, in1=_in1(FITSIREF=[True])), device, groups=["flux_loop"],
                          efit_constants=d3d.EFIT_SIGMA_CONSTANTS)
    assert fs.tables["flux_loop"].meta["coils_form_rule"] == "fitsiref"
    # several undecided slices of one discharge: the reference entry stays constant while SIREF changes
    sl = [d3d.parse_namelists(_block("IN1", _in1(COILS=[0.01, 0.5, -0.25], SIREF=[s]))) for s in (0.2, 0.3, 0.5)]
    ser = d3d.flux_storage_series(sl, 0)
    assert ser["coils_form"] == "relative" and ser["n_slices"] == 3
    one = d3d.kfile_tables(sl[0], device, groups=["flux_loop"], flux_storage=ser, flux_reference="PSF1A",
                           efit_constants=d3d.EFIT_SIGMA_CONSTANTS)
    assert one.tables["flux_loop"].meta["coils_form_rule"] == "series"


def test_kfile_stored_sigma_and_vbit(tmp_path, device):
    kf = _kfile(tmp_path)
    stored = {"flux_loop": [1e-3, 0.0, 2e-3], "bpol_probe": [1e-3, 1e-3], "ip": [5e3], "pf_current": [10.0] * 24,
              "diamagnetic_flux": [2e-5], "mse": [0.02, 0.02, 0.02]}
    st = d3d.kfile_tables(kf, device, er_corrected=False, stored_sigma=stored)
    fl = st.tables["flux_loop"]
    assert st.meta["sigma_model"] == "efit_stored" and np.isnan(fl.sigma[1]) and not fl.valid[1]
    assert fl.sigma[0] == pytest.approx(TWO_PI * 1e-3) and st.tables["mse"].sigma[0] == 0.02
    vb = d3d.kfile_tables(kf, device, sigma_model="efit_vbit", groups=["flux_loop", "ip"], flux_reference="PSF1A")
    assert vb.tables["flux_loop"].sigma[0] == pytest.approx(TWO_PI * 10.0 * 1e-3)    # reference loop: VBIT bit
    assert vb.tables["ip"].sigma[0] == pytest.approx(0.03e6)                          # SERROR |PLASMA|


def test_kfile_refusals(tmp_path, device):
    kw = dict(efit_constants=d3d.EFIT_SIGMA_CONSTANTS, er_corrected=True)
    with pytest.raises(ValueError, match="SERROR"):
        d3d.kfile_tables(_kfile(tmp_path, in1=_in1(SERROR=None)), device, groups=["bpol_probe"], **kw)
    with pytest.raises(ValueError, match="VBIT"):
        d3d.kfile_tables(_kfile(tmp_path, in1=_in1(VBIT=None)), device, groups=["bpol_probe"], **kw)
    assert d3d.kfile_tables(_kfile(tmp_path, in1=_in1(VBIT=None)), device, groups=["bpol_probe"], vbit=10.0,
                            **kw).meta["VBIT_source"] == "caller"
    with pytest.raises(ValueError, match="calibration-error"):
        d3d.kfile_tables(_kfile(tmp_path, in1=_in1(IMAGSIGMA=[1.0])), device, **kw)
    with pytest.raises(ValueError, match="KERSIL = 1"):
        d3d.kfile_tables(_kfile(tmp_path, in1=_in1(KERSIL=[1.0])), device, groups=["flux_loop"], **kw)
    with pytest.raises(ValueError, match="ERRSIL"):
        d3d.kfile_tables(_kfile(tmp_path), device, groups=["flux_loop"])
    with pytest.raises(ValueError, match="er_corrected"):
        d3d.kfile_tables(_kfile(tmp_path), device, groups=["mse"])
    with pytest.raises(ValueError, match="only some chords"):
        d3d.kfile_tables(_kfile(tmp_path, ins=_ins(TGAMMAUNCOR=[0.1, 0.06, -0.05])), device, groups=["mse"], **kw)
    with pytest.raises(ValueError, match="counted twice"):
        d3d.kfile_tables(_kfile(tmp_path, in1=_in1(KEECUR=[1.0])), device, groups=["mse"], **kw)
    with pytest.raises(ValueError, match="beyond"):
        d3d.kfile_tables(_kfile(tmp_path, in1=_in1(COILS=[0.0, 0.5, -0.25, 0.3], FWTSI=[1.0] * 4)), device,
                         groups=["flux_loop"], **kw)
    with pytest.raises(ValueError, match="requested but absent"):
        d3d.kfile_tables(_kfile(tmp_path, in1=_in1(DFLUX=None)), device, groups=["diamagnetic_flux"], **kw)


# ------------------------------------------------------------------------------------------------ EFIT arrays
def test_efit_array_tables(device):
    A = {"MTIME": [2500.0], "SILOPT": [0.1, 0.2, 0.3], "SIGSIL": [1e-3, 0.0, 1e-3], "FWTSI": [1, 1, 0],
         "CSILOP": [0.1, 0.2, 0.3], "SAISIL": [0, 0, 0], "EXPMPI": [0.1, 0.2], "SIGMPI": [1e-3, 1e-3],
         "FWTMP2": [1, 1], "CMPR2": [0.1, 0.2], "SAIMPI": [0, 0], "FCCURT": [1e3] * 18, "SIGFCC": [10.0] * 18,
         "FWTFC": [1.0] * 17 + [0.0], "CCBRSP": [1e3] * 18, "CHIFCC": [0.0] * 18, "ECCURT": [1.0] * 6,
         "SIGECC": [1.0] * 6, "FWTEC": [0.0] * 6, "CECURR": [1.0] * 6, "CHIECC": [0.0] * 6,
         "plasma": [1e6], "SIGPASMA": [3e3], "FWTPASMA": [1.0], "CPASMA": [1e6], "CHIPASMA": [0.0],
         "TANGAM": [0.1, 0.0], "SIGGAM": [0.01, 0.01], "FWTGAM": [1, 1], "CMGAM": [0.1, 0.0], "CHIGAM": [0, 0],
         "RRGAM": [1.8, 1.9], "ZZGAM": [0, 0], "A1GAM": [0.7, 0.7], "A2GAM": [0.6, 0.6], "A3GAM": [0.1, 0.1],
         "A4GAM": [0.0, 0.0]}
    st = d3d.efit_array_tables(A, device, er_corrected=True)
    assert st.time == 2.5 and set(st.tables) == {"flux_loop", "bpol_probe", "pf_current", "ip", "mse"}
    fl = st.tables["flux_loop"]
    assert list(fl.valid) == [True, False, False] and np.allclose(fl.measured, -TWO_PI * np.array([0.1, 0.2, 0.3]))
    assert st.tables["pf_current"].valid.all() and st.tables["pf_current"].weight[17] == 1.0
    assert list(st.tables["mse"].valid) == [True, False]
    with pytest.raises(ValueError, match="MTIME"):
        d3d.efit_array_tables({"SILOPT": [0.1]}, device)


# ------------------------------------------------------------------------------------------------ constraints
def test_to_constraints(tmp_path, device):
    from OpenFUSIONToolkit.TokaMaker import reconstruction as R
    st = d3d.kfile_tables(_kfile(tmp_path), device, er_corrected=True, flux_reference="PSF1A",
                          efit_constants=d3d.EFIT_SIGMA_CONSTANTS)
    idx = {c: i for i, c in enumerate(d3d.F_COILS + ("ECOILA", "ECOILB"))}
    with pytest.raises(ValueError, match="no index"):
        d3d.to_constraints(st, idx)
    out = d3d.to_constraints(st, idx, missing_coils="skip", probe_n_avg=5, mse_E_R=[1e3, 2e3, 0.0])
    c = out["constraints"]
    fl = st.tables["flux_loop"]
    assert all(isinstance(x, R.fluxLoop_con) for x in c["flux_loop"]) and len(c["flux_loop"]) == 3
    assert c["flux_loop"][1].val == pytest.approx(TWO_PI * (0.5 + 0.2))          # 2 pi (COILS + SIREF)
    assert c["flux_loop"][1].err == pytest.approx(fl.sigma[1])
    p0 = c["bpol_probe"][0]
    assert np.allclose(p0.norm, [0.0, 0.0, 1.0], atol=1e-15) and p0.length == 0.1 and len(c["bpol_probe"]) == 1
    assert out["skipped"]["pf_current"] == ["E567UP", "E567DN", "E89DN", "E89UP"]
    assert [x.ind for x in c["pf_current"]][:3] == [0, 1, 2]
    m = c["mse"]
    assert len(m) == 2 and m[1].E_R == 2e3 and m[0].coefs[6] == 0.6
    keep = d3d.to_constraints(st, idx, missing_coils="skip", unused="keep")
    assert len(keep["constraints"]["bpol_probe"]) == 2 and keep["constraints"]["bpol_probe"][1].use is False
    und = d3d.kfile_tables(_kfile(tmp_path), device, groups=["flux_loop"], efit_constants=d3d.EFIT_SIGMA_CONSTANTS)
    with pytest.raises(ValueError, match="storage form"):
        d3d.to_constraints(und)


# ------------------------------------------------------------------------------------------------ fetch and cache
FETCH = dict(host="userhost", control_path="/tmp/cm-x", remote_dir="scratch/pull", server="dataserver",
             setup="module load something")


def test_compose_fetch_only_reuses_the_user_connection(tmp_path):
    spec = d3d.node_spec(efit_tree="MYEFIT")
    plan = d3d.compose_fetch(12345, spec, local_dir=str(tmp_path), **FETCH)
    names = [s[0] for s in plan.steps]
    assert names == ["check_master", "mkdir", "send_spec", "send_script", "pull", "fetch_result"]
    assert plan.steps[0][1][:3] == ["ssh", "-O", "check"]
    for name, argv, _ in plan.steps[1:]:
        assert "ControlMaster=no" in argv and "BatchMode=yes" in argv and "ControlPath=/tmp/cm-x" in argv, name
    assert plan.files[f"d3d_12345_spec.txt"].count("\n") == len(spec)
    ast.parse(plan.files["d3d_12345_pull.py"])                     # the remote script compiles (never run here)
    assert "dataserver" in plan.steps[4][1][-1] and "module load something" in plan.steps[4][1][-1]
    for k in FETCH:
        bad = dict(FETCH, **{k: ""})
        if k != "setup":
            with pytest.raises(ValueError, match=k):
                d3d.compose_fetch(1, spec, local_dir=str(tmp_path), **bad)


def test_run_fetch_stops_at_first_failure(tmp_path):
    plan = d3d.compose_fetch(7, d3d.node_spec(efit_tree="T"), local_dir=str(tmp_path), **FETCH)
    calls = []

    def master_down(argv, timeout):
        calls.append(argv)
        return 255

    with pytest.raises(d3d.FetchError, match="master connection is not open"):
        d3d.run_fetch(plan, str(tmp_path), master_down)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="runner"):
        d3d.run_fetch(plan, str(tmp_path), None)

    def ok(argv, timeout):
        calls.append(argv)
        if argv[0] == "scp" and argv[-1].endswith(".npz"):
            np.savez(argv[-1], nodes=np.array(["T\tx", "T\ty"]), errors='{"1": "TreeNNF: no node"}',
                     n0_data=np.arange(3.0), n0_dim0=np.arange(3.0), n0_units=np.array("V"))
        return 0

    calls.clear()
    path = d3d.run_fetch(plan, str(tmp_path), ok)
    assert len(calls) == 6 and os.path.isfile(path)
    pulled = d3d.load_pull(path)
    assert pulled[("T", "x")]["units"] == "V" and "error" in pulled[("T", "y")]


def test_fetch_cache(tmp_path):
    spec = d3d.node_spec(efit_tree="T")
    src = tmp_path / "pulled.npz"
    np.savez(src, nodes=np.array(["T\tx"]), errors="{}", n0_data=np.ones(2))
    cache = d3d.FetchCache(str(tmp_path / "cache"))
    assert not cache.has(5, spec)
    with pytest.raises(FileNotFoundError):
        cache.load(5, spec)
    p = cache.store(5, spec, str(src))
    assert cache.has(5, spec) and not cache.has(6, spec) and not cache.has(5, spec[:-1])
    assert np.array_equal(cache.load(5, spec)[("T", "x")]["data"], np.ones(2))
    with open(p, "ab") as fh:
        fh.write(b"0")
    with pytest.raises(ValueError, match="manifest"):
        cache.load(5, spec)


# ------------------------------------------------------------------------------------------------ transport code
def _tc_pulled(times_ms, tree="TC"):
    rho = np.linspace(0.0, 1.0, 11)
    nt = len(times_ms)
    prof = np.outer(1.0 - rho ** 2, 1.0 + 0.1 * np.arange(nt))          # (rho, time): time on the last axis
    rec = lambda d: {"data": d, "dim0": np.asarray(times_ms, float)}      # noqa: E731
    return {(tree, f"\\{tree}::TOP.EQU:PRESS_BEAM"): rec(1e4 * prof), (tree, f"\\{tree}::TOP.HCD:J_CD"): rec(1e5 * prof),
            (tree, f"\\{tree}::TOP.NBI:J_NBCD"): rec(5e4 * prof), (tree, f"\\{tree}::TOP.NBI:ENBEAM"): rec(1e18 * prof),
            (tree, f"\\{tree}::TOP.EQU:RMAJ0"): rec(1.7 * np.ones(nt))}, rho


def test_transport_code_route():
    cov = d3d.transport_code_coverage([2.48, 2.50, 2.52, 2.60], 2.51, 0.040)
    assert cov["covered"] and cov["bracket"] == [1, 2] and cov["weight"] == pytest.approx(0.5)
    assert d3d.transport_code_coverage([2.48, 2.50, 2.52, 2.60], 2.50, 0.040)["rule"] == "stored time"
    assert not d3d.transport_code_coverage([2.48, 2.50, 2.52, 2.60], 2.55, 0.040)["covered"]
    assert not d3d.transport_code_coverage([2.48, 2.50], 2.70, 0.040)["covered"]
    pulled, rho = _tc_pulled([2480.0, 2500.0, 2520.0])
    v, _ = d3d.transport_code_values(pulled, "TC", 2.51, d3d.TC_MAX_GAP_S)
    assert np.allclose(v["EQU:PRESS_BEAM"], 1e4 * (1 - rho ** 2) * 1.15) and v["EQU:RMAJ0"] == pytest.approx(1.7)
    sp, cov = d3d.transport_code_sources(pulled, "TC", 2.51, rho, d3d.TC_MAX_GAP_S)
    assert sp.coordinate == "rho_tor_norm" and sp.r0_cd == pytest.approx(1.7)
    assert np.allclose(sp.jtor_r0_cd, 1e5 * (1 - rho ** 2) * 1.15)
    sp2, _ = d3d.transport_code_sources(pulled, "TC", 2.51, rho, d3d.TC_MAX_GAP_S, ec_supplied_separately=True)
    assert np.allclose(sp2.jtor_r0_cd, 5e4 * (1 - rho ** 2) * 1.15)
    with pytest.raises(ValueError, match="does not cover"):
        d3d.transport_code_values(pulled, "TC", 3.0, d3d.TC_MAX_GAP_S)
    bad = dict(pulled)
    bad[("TC", "\\TC::TOP.EQU:PRESS_BEAM")] = {"data": -np.ones((11, 3)), "dim0": np.array([2480.0, 2500.0, 2520.0])}
    with pytest.raises(ValueError, match="refused"):
        d3d.transport_code_sources(bad, "TC", 2.51, rho, d3d.TC_MAX_GAP_S)
