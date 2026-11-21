#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""Shared fixture of the live-solver tests of the kinetic reconstruction (``TokaMaker.kinetic``): a synthetic
sawtoothing L-mode slice built IN CODE on the DIII-D-like mesh of the TokaMaker tests (``DIIID_mesh.h5``), so that
no equilibrium or profile file is needed.

The reconstruction's own tests read a synthetic g-file and a profile-fit file on a coarser copy of this
mesh.  Here the same case is rebuilt from its generating recipe: the kinetic profiles are the recipe's analytic
L-mode profiles (``Hmode_profiles`` families, parameters below; carbon impurity, main-ion density from Z_eff), the
plasma boundary is the recipe's 16 boundary points (upper single null in the recipe; here isoflux points only, no
X-point constraint), interpolated to 128 points, with Ip = 1.3 MA and B0 = -2 T at R0 = 1.7 m.  The starting state
is a uniform-current flux initialization followed by one solve with the recipe's <j_phi> shape and the kinetic
pressure.  The equilibria therefore differ from the original fixture's (mesh, boundary and starting state differ):
reference values measured on this fixture are NEW, while bounds and tolerances are the original tests' bounds.

Live work runs in ONE child interpreter per test module (``OFT_env`` is a per-process singleton); the child
environment sets ``GFORTRAN_UNBUFFERED_PRECONNECTED=y`` before the solver library is loaded, which the forward
model needs to read the solver's per-iteration output (residual history, vertical-stability coil current).
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(HERE, "DIIID_mesh.h5")

R0, B0 = 1.7, -2.0             # m, T (signed as the recipe's g-file)
IP = 1.3e6                     # A
Z_IMP = 6.0                    # carbon
EC = 1.602176634e-19
# boundary points of the recipe (R - R0 [m], Z [m])
LCFS = ((-0.62, 0.087), (-0.6, 0.32), (-0.56, 0.6), (-0.43, 1.0), (-0.01, 0.92), (0.24, 0.69), (0.4, 0.47),
        (0.49, 0.29), (0.56, 0.07), (0.56, -0.16), (0.48, -0.44), (0.3, -0.72), (-0.01, -0.92), (-0.38, -0.85),
        (-0.56, -0.49), (-0.6, -0.21))
# analytic profile parameters of the recipe: Hmode_profiles(edge, ped, core, expin, expout, widthp, xphalf)
NE = (1.5, 2.1, 4.4, 1.0, 0.58, 0.1, 1.0)          # 1e19 m^-3
TE = (0.088, 0.45, 3.0, 1.2, 1.5, 0.15, 0.98)      # keV
TI = (0.17, 0.55, 4.0, 1.0, 1.7, 0.16, 1.0)        # keV
ZEFF = (1.4, 1.5, 1.6, 3.6, 3.2, 0.1, 1.0)
JPHI = (0.0, 0.11, 2.4, 1.1, 1.7, 0.011, 0.98)     # <j_phi> shape [MA/m^2] (amplitude set by Ip)
SOL_DECAY = 0.05
# the fit tests' truth (a 50 ms period at f = 0.37 of the opt-in periodic orbit) and noise seed
_SEED = 20261110
_F_TRUE, _RATE_TRUE = 0.37, 20.0
_XS = np.linspace(0.0, 1.0, 4001)


def prof(x, p, sol="decay"):
    """Recipe profile ``p`` at psi_N ``x``; past psi_N = 1 an exponential decay (or the separatrix value)."""
    from OpenFUSIONToolkit.TokaMaker.bootstrap import Hmode_profiles
    edge, ped, core, expin, expout, widthp, xphalf = p
    v_all = Hmode_profiles(edge=edge, ped=ped, core=core, rgrid=_XS.size, expin=expin, expout=expout,
                           widthp=widthp, xphalf=xphalf)
    x = np.asarray(x, dtype=float)
    v = np.interp(np.clip(x, 0.0, 1.0), _XS, v_all)
    if sol == "hold":
        return v
    return np.where(x > 1.0, v_all[-1] * np.exp(-(x - 1.0) / SOL_DECAY), v)


def lmode_kinetics():
    """``(psi_N, ne, te, ni, ti, zeff)`` [m^-3, eV] on the recipe's profile-fit grid (0.01 steps to 0.9, then 60
    nodes to 1.2)."""
    pn = np.r_[0.01 * np.arange(90), np.linspace(0.9, 1.2, 60)]
    ne = 1e19 * prof(pn, NE)
    zeff = prof(pn, ZEFF, sol="hold")
    ni = ne * (Z_IMP - zeff) / (Z_IMP - 1.0)
    return pn, ne, 1e3 * prof(pn, TE), ni, 1e3 * prof(pn, TI), zeff


def lmode_boundary(n=128):
    """The recipe's 16 boundary points, interpolated (periodic cubic spline in the poloidal angle about their
    centroid) to ``n`` points [m]."""
    from scipy.interpolate import CubicSpline
    b = np.array([(R0 + dr, z) for dr, z in LCFS], dtype=float)
    c = b.mean(axis=0)
    th = np.unwrap(np.arctan2(b[:, 1] - c[1], b[:, 0] - c[0]))
    order = np.argsort(th)
    th, b = th[order], b[order]
    th = np.r_[th, th[0] + 2 * np.pi]
    b = np.vstack([b, b[:1]])
    sp = CubicSpline(th, b, bc_type="periodic")
    return sp(np.linspace(th[0], th[0] + 2 * np.pi, n, endpoint=False))


def edge_remedy_for_this_build():
    """``ForwardSettings`` keywords for a solve whose pressure gradient is finite at the boundary, chosen for the
    solver build that is loaded: none on a build with the cut-cell setting (the forward model turns it on by
    default, no taper), else the explicit edge taper ``on`` (the workaround for a build without it).  The one place
    where a test selects between the two.  Returns ``(label, keywords)``."""
    from OpenFUSIONToolkit.TokaMaker.kinetic import edge as E
    from OpenFUSIONToolkit.TokaMaker.kinetic import forward as FW
    if FW.solver_build()["provides_lcfs_cut_cells"]:
        return "cut_cells", {}
    return "taper_on", dict(edge_taper=E.EdgeTaperSetting("on"))


def _setup(**forward_kw):
    """The in-code synthetic case in generation mode; ``forward_kw`` are extra ``ForwardSettings`` keywords (none
    by default).  Returns ``(fm, kin, src, cfg, init, ip)`` as the original fixture's ``_setup``."""
    from OpenFUSIONToolkit.TokaMaker.kinetic import current_model as CM
    from OpenFUSIONToolkit.TokaMaker.kinetic import forward as FW
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    pn, ne, te, ni, ti, zeff = lmode_kinetics()
    kin = CM.KineticProfiles(pn, ne, te, ni, ti, zeff, source="in-code synthetic L-mode (recipe profiles)")
    src = CM.SourceProfiles(pn, source="none (no driven current, no fast ions)")
    st = FW.ForwardSettings(vsc={"F9A": 1.0, "F9B": -1.0}, **forward_kw)
    bnd = lmode_boundary()
    fm = FW.ForwardModel(load_gs_mesh(MESH), R0 * B0, None, bnd, settings=st, case_id="truth", ip=IP,
                         generation_coil_reg=1e-2)
    rz = np.array([(R0 + dr, z) for dr, z in LCFS])
    a = 0.5 * (rz[:, 0].max() - rz[:, 0].min())
    fm.gs.init_psi(0.5 * (rz[:, 0].max() + rz[:, 0].min()), 0.0, a, 0.5 * (rz[:, 1].max() - rz[:, 1].min()) / a, 0.3)
    xj = np.linspace(0.0, 1.0, 129)
    nk = np.interp(xj, pn, ne)
    pk = EC * (nk * np.interp(xj, pn, te) + np.interp(xj, pn, ni) * np.interp(xj, pn, ti))
    init = (xj, 1e6 * prof(xj, JPHI), pk)
    # the truth is a sawtoothing slice made with the opt-in periodic orbit (the fits free its strength)
    cfg = CM.CurrentModelSettings(r0=R0, core_relaxation="periodic_orbit")
    return fm, kin, src, cfg, init, IP


def _build_equilibrium():
    """A free-boundary equilibrium of the in-code case solved directly with TokaMaker (OFT default solver
    settings): isoflux on every 4th fixture boundary point, coils regularized toward zero, linearly interpolated
    FF' and p' of fixed analytic shapes, Ip and the kinetic axis pressure as targets.  Replaces the scoring tests'
    g-file equilibrium.  Returns ``(gs, g)`` with ``g["ip"]`` [A] and ``g["pres"][0]`` [Pa] as the g-file fields
    those tests read."""
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    pn, ne, te, ni, ti, zeff = lmode_kinetics()
    pax = EC * (ne[0] * te[0] + ni[0] * ti[0])
    g = {"ip": IP, "pres": np.array([pax])}
    oft = OFT_env(nthreads=1)
    gs = TokaMaker(oft)
    pts, lc, reg, coil_dict, cond_dict = load_gs_mesh(MESH)
    gs.setup_model(pts, lc, reg, cond_dict=cond_dict, coil_dict=coil_dict, order=2, F0=R0 * B0)
    gs.set_coil_vsc({"F9A": 1.0, "F9B": -1.0})
    gs.set_targets(Ip=IP, pax=pax)
    gs.set_isoflux_constraints(lmode_boundary()[::4].copy())
    regs = [gs.coil_reg_term({n: 1.0}, target=0.0, weight=1.e-2) for n in gs.coil_sets]
    regs.append(gs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=1.e-2))
    gs.set_coil_reg(reg_terms=regs)
    x = np.linspace(0.0, 1.0, 50)
    gs.set_profiles(ffp_prof={"type": "linterp", "x": x, "y": (1.0 - x ** 1.5) ** 1.5},
                    pp_prof={"type": "linterp", "x": x, "y": (1.0 - x ** 2) ** 1.2})
    rz = np.array([(R0 + dr, z) for dr, z in LCFS])
    a = 0.5 * (rz[:, 0].max() - rz[:, 0].min())
    gs.init_psi(0.5 * (rz[:, 0].max() + rz[:, 0].min()), 0.0, a, 0.5 * (rz[:, 1].max() - rz[:, 1].min()) / a, 0.3)
    gs.solve()        # OFT defaults: nl_tol 1e-6, urf 0.2, maxits 40 (raises if not converged)
    return gs, g


#: Resolutions [m] of the coarse in-code mesh (plasma, coils, vessel, vacuum); the registered DIII-D-like mesh of the
#: TokaMaker tests uses 0.02, 0.03, 0.04, 0.10
COARSE_MESH_DX = (0.05, 0.06, 0.08, 0.20)


def coarse_mesh(filename, dx=COARSE_MESH_DX):
    """Build a coarse DIII-D-like mesh IN CODE from the device geometry of the TokaMaker DIII-D example
    (``DIIID_geom.json``: limiter, vessel segments, F- and E-coils), with the region layout of that example's mesh
    script and the resolutions ``dx``; save it to ``filename`` and return ``filename``."""
    from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain, save_gs_mesh
    plasma_dx, coil_dx, vv_dx, vac_dx = dx
    with open(os.path.join(HERE, "DIIID_geom.json")) as fh:
        geom = json.load(fh)
    dom = gs_Domain()
    dom.define_region("air", vac_dx, "boundary")
    dom.define_region("plasma", plasma_dx, "plasma")
    dom.define_region("vacuum", vv_dx, "vacuum", allow_xpoints=True)
    for i, seg in enumerate(geom["vv"]):
        dom.define_region(f"vv{i}", vv_dx, "conductor", eta=seg["eta"])
    for key, coil in geom["coils"].items():
        if key.startswith("ECOIL"):
            for i, sub in enumerate(coil):
                dom.define_region(f"{key}_{i}", coil_dx, "coil", coil_set=key, nTurns=sub["nturns"])
        else:
            dom.define_region(key, coil_dx, "coil", nTurns=coil["nturns"])
    dom.add_polygon(geom["limiter"], "plasma", parent_name="vacuum")
    dom.add_enclosed([1.75, 1.25], "vacuum")
    for i, seg in enumerate(geom["vv"]):
        dom.add_polygon(seg["pts"], f"vv{i}", parent_name="air")
    for key, coil in geom["coils"].items():
        if key.startswith("ECOIL"):
            for i, sub in enumerate(coil):
                dom.add_polygon(sub["pts"], f"{key}_{i}", parent_name="air")
        else:
            dom.add_polygon(coil["pts"], key, parent_name="air")
    pts, lc, reg = dom.build_mesh()
    save_gs_mesh(pts, lc, reg, dom.get_coils(), dom.get_conductors(), filename)
    return filename


def _sensors(gs):
    lim = np.asarray(gs.lim_contour, float)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(lim, axis=0).T))]

    def resample(n, inset):
        s = np.linspace(0.0, seg[-1], n, endpoint=False)
        p = np.column_stack((np.interp(s, seg, lim[:, 0]), np.interp(s, seg, lim[:, 1])))
        t = np.column_stack((np.interp(s + 1e-3, seg, lim[:, 0]), np.interp(s + 1e-3, seg, lim[:, 1]))) - p
        t /= np.hypot(t[:, 0], t[:, 1])[:, None]
        d = np.array([1.75, 0.0]) - p
        d /= np.hypot(d[:, 0], d[:, 1])[:, None]
        return p + inset * d, t

    fl, _ = resample(40, 0.03)
    pr, tan = resample(60, 0.03)
    n = 16
    coef = np.column_stack((np.linspace(0.70, 0.95, n), np.zeros(n), np.zeros(n), np.full(n, 1.5e-6),
                            np.full(n, 0.05), np.linspace(0.2, 0.4, n), -np.linspace(0.6, 0.8, n),
                            np.full(n, 1.0e-7), np.zeros(n)))
    return {"flux_loop": {"r": fl[:, 0], "z": fl[:, 1]},
            "bpol_probe": {"r": pr[:, 0], "z": pr[:, 1], "direction_r": tan[:, 0], "direction_z": tan[:, 1],
                           "length": np.where(np.arange(60) % 2 == 0, 0.04, -0.03)},
            "pf_current": {"coil_names": list(gs.coil_sets)},
            "mse": {"r": np.linspace(1.78, 2.20, n), "z": np.full(n, 0.04), "coefficients": coef}}


def _measurements(gs, geo, roles, ip, rng):
    from OpenFUSIONToolkit.TokaMaker.kinetic import diagnostics as D
    from OpenFUSIONToolkit.TokaMaker.kinetic.interfaces import ConstraintGroup, MeasurementSet
    model = {"flux_loop": D.flux_loop(gs, geo["flux_loop"]).value,
             "bpol_probe": D.bpol_probe(gs, geo["bpol_probe"]).value,
             "pf_current": D.pf_current(gs, geo["pf_current"]).value, "ip": D.ip(gs).value,
             "mse": D.mse(gs, geo["mse"], quantity="tan_polarisation_angle").value}
    sig = {"flux_loop": np.maximum(0.01 * np.abs(model["flux_loop"]), 2e-3),
           "bpol_probe": np.maximum(0.02 * np.abs(model["bpol_probe"]), 2e-3),
           "pf_current": np.maximum(0.01 * np.abs(model["pf_current"]), 200.0),
           "ip": np.array([5e-3 * ip]), "mse": np.full(16, 3.5e-3)}
    meta = {"flux_loop": dict(coils_form="absolute", absolute_offset=0.0),
            "mse": dict(er_corrected=True, quantity="tan_polarisation_angle")}
    groups = {}
    for name, v in model.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        m = np.where(ok, v, 0.0) + (rng.standard_normal(v.size) * sig[name] if rng is not None else 0.0)
        if name == "pf_current":          # held coils are held at the measurement: give them the true value
            held = np.array([roles[n] == "held" for n in geo["pf_current"]["coil_names"]])
            m = np.where(held, v, m)
        groups[name] = ConstraintGroup(name=name, labels=[f"{name}_{i}" for i in range(v.size)], measured=m,
                                       sigma=sig[name], valid=ok, geometry=geo.get(name, {}),
                                       meta=dict(sigma_model="synthetic_test", **meta.get(name, {})))
    return MeasurementSet("synthetic", 1.0, 0.01, groups)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def child_env(threads=1):
    """Environment of a live child: the OFT Python package importable, one BLAS thread, ``threads`` OpenMP
    threads, and unbuffered Fortran output (set before the solver library loads in the child)."""
    import OpenFUSIONToolkit
    oft = os.path.dirname(os.path.dirname(os.path.abspath(OpenFUSIONToolkit.__file__)))
    path = os.pathsep.join(p for p in (HERE, oft, os.environ.get("PYTHONPATH", "")) if p)
    return dict(os.environ, PYTHONPATH=path, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(threads),
                GFORTRAN_UNBUFFERED_PRECONNECTED="y")


def run_child(script, out_dir, args=(), timeout=3000, threads=1):
    """Run ``script out.json *args`` in a child interpreter from ``out_dir``; return the JSON it wrote."""
    out = os.path.join(str(out_dir), "result.json")
    proc = subprocess.run([sys.executable, os.path.abspath(script), out, *map(str, args)], cwd=str(out_dir),
                          env=child_env(threads), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)
