#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
r'''! Device layer of kinetic equilibrium reconstruction for DIII-D

What is specific to the device, kept out of the solver and the kinetic helper packages:

* **device description** -- coil names, the vertical stability coil pair and the flux-loop
  reference, and :func:`device_from_mhdin`, which reads the sensor tables (flux loops, probes,
  F-coils) of the user's EFIT ``mhdin.dat`` with the solver's own reader
  (:func:`OpenFUSIONToolkit.TokaMaker.util.read_mhdin`);
* **named parameter sets** -- values that the kinetic and source modules take as required
  arguments because they were calibrated on this device (:data:`PARAMETER_SETS`);
* **file adapters** -- an EFIT k-file (:func:`kfile_tables`) or an EFIT run's stored
  measurement arrays (:func:`efit_array_tables`) to per-channel tables of value, uncertainty,
  weight and validity, and the tables to the solver's reconstruction constraints
  (:func:`to_constraints`); a stored transport-code run to the driven-current and fast-ion
  pressure profiles of the current model (:func:`transport_code_sources`);
* **fetch and cache** -- :func:`compose_fetch` writes down the commands that pull a node list
  through an ssh connection the USER has already opened (OpenSSH ``ControlMaster``), and
  :func:`run_fetch` runs them only with a runner the caller passes in.  Nothing here opens a
  new connection or authenticates: every ssh/scp command carries ``ControlMaster=no`` and
  ``BatchMode=yes``, the first command only checks the user's master, and a failed step stops
  the fetch without a retry.  :class:`FetchCache` keeps the pulled files by discharge and node list.

Names.  Host names, user names, paths, discharge numbers and analysis-tree names are always
arguments: none is written here.  The node and signal names used are those of the public
data-mapping package OMAS (``gafusion/omas``, ``omas/machine_mappings``: ``_common.py``,
``_efit.json``, ``d3d.py``); the coil names also appear in this project's own DIII-D example
(``src/examples/TokaMaker/AdvancedWorkflows/Pulse_Design/d3d_pulse_ex.py``) and the vertical
stability pair in its tests.  The MSE geometry nodes and the stored transport-code point names
are named as such.  Beam geometry, apertures and optics are never here: they are arguments of
:mod:`OpenFUSIONToolkit.TokaMaker.sources`.

Sign conventions.  The k-file adapter keeps the conventions of the kinetic reconstruction it was
written for: flux-loop values are ``flux_scale * COILS`` with ``flux_scale = -2 pi`` by default,
the IMAS ``magnetics.flux_loop[:].flux`` sign (k-files store ``COILS`` [Wb/rad] in the sign of
TokaMaker's ``psi``).  :func:`to_constraints` converts to the solver's flux-loop model
``2 pi psi`` [Wb].
'''
import hashlib
import json
import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "F_COILS", "E_COILS", "VSC_PAIRING", "FLUX_REFERENCE_LOOP", "Device", "device_from_mhdin",
    "PARAMETER_SETS", "EFIT_SIGMA_CONSTANTS", "ZEFF_IMPURITY_Z", "TC_MAX_GAP_S", "zeff_rule_thresholds",
    "source_prior_widths", "parse_namelists", "read_kfile", "ChannelTable", "SliceTables", "flux_gauge",
    "flux_storage_series", "kersil_loop_sigma", "kfile_tables", "efit_array_tables", "efit_to_imas_coefficients",
    "to_constraints", "EFIT_MEASUREMENT_NODES", "MSE_GEOMETRY_NODES", "PROFILE_FIT_NODES", "BEAM_NAMES",
    "BEAM_NODE_FIELDS", "TRANSPORT_CODE_POINTS", "node_spec", "FetchPlan", "FetchError", "compose_fetch",
    "run_fetch", "load_pull", "FetchCache", "transport_code_coverage", "transport_code_values",
    "transport_code_sources",
]

# =====================================================================================================
# Device description
# =====================================================================================================
#: F-coil names in the order of the EFIT coil tables (``BRSP``); OMAS ``d3d.py``, ``pf_active_hardware``.
F_COILS = ("F1A", "F2A", "F3A", "F4A", "F5A", "F6A", "F7A", "F8A", "F9A",
           "F1B", "F2B", "F3B", "F4B", "F5B", "F6B", "F7B", "F8B", "F9B")
#: E-coil names in the order of ``ECURRT``; OMAS ``d3d.py``, ``pf_active_hardware``.
E_COILS = ("ECOILA", "ECOILB", "E567UP", "E567DN", "E89DN", "E89UP")
#: Vertical stability coil pair (``TokaMaker.set_coil_vsc``), as in this project's DIII-D tests.
VSC_PAIRING = {"F9A": 1.0, "F9B": -1.0}
#: Flux loop the k-file loop fluxes are stored relative to when stored relative (OMAS ``d3d.py``,
#: the loop-voltage mapping names it as the standard loop).
FLUX_REFERENCE_LOOP = "PSF1A"


@dataclass
class Device:
    '''Sensor and coil tables of one EFIT machine description, in the order of the EFIT green tables.

    ``flux_loops``: ``labels``, ``r``, ``z`` [m]; ``probes``: ``labels``, ``r``, ``z`` [m], ``angle_deg``
    (EFIT ``AMP2``: the probe normal is ``(cos, sin)`` of it in the (R, Z) plane), ``length`` [m] (``SMP2``);
    ``f_coils``: ``labels``, ``r``, ``z``, ``w``, ``h`` [m], ``turns``; ``e_coils``: ``labels``.
    ``probe_n_avg``: EFIT's probe averaging count ``NSMP2`` when the file states it (for
    :func:`to_constraints`), else None.  ``source``: file name and SHA-256 of the description it was read from.
    '''
    flux_loops: Dict[str, list]
    probes: Dict[str, list]
    f_coils: Dict[str, list]
    e_coils: Dict[str, list]
    probe_n_avg: Optional[int] = None
    source: Dict[str, str] = field(default_factory=dict)

    def labels(self, group):
        '''Channel labels of a group (``flux_loop``, ``bpol_probe``, ``pf_current``)'''
        if group == "flux_loop":
            return list(self.flux_loops["labels"])
        if group == "bpol_probe":
            return list(self.probes["labels"])
        if group == "pf_current":
            return list(self.f_coils["labels"]) + list(self.e_coils["labels"])
        raise ValueError(f"unknown group {group!r}")


def _floats(raw, key, n):
    if key not in raw:
        raise ValueError(f"machine description has no {key}")
    a = np.array([np.nan if x is None else float(x) for x in raw[key]], dtype=float)
    if a.size != n or not np.all(np.isfinite(a)):
        raise ValueError(f"machine description: {key} has {a.size} entries (or nulls) for {n} channels")
    return a.tolist()


def device_from_mhdin(path, f_coil_names=F_COILS, e_coil_names=E_COILS):
    '''Read the sensor and coil tables of an EFIT ``mhdin.dat`` (the user's file).

    The same tables as the solver's ``TokaMaker.util.read_mhdin`` (loop names ``LPNAME``, probe names ``MPNAM2`` and
    angles ``AMP2``, F-coils ``RF ZF WF HF TURNFC``, E-coil groups by ``ECID``) plus loop positions ``RSI``/``ZSI``
    and probe positions ``XMP2``/``YMP2`` and lengths ``SMP2``, read with :func:`parse_namelists`: the solver's
    namelist reader takes one key per line, and machine files with several ``KEY=value`` on one line (as this
    device's are) shift its arrays.  ``e_coil_names`` are indexed by the file's ``ECID`` (1-based).
    '''
    with open(path) as fh:
        text = fh.read()
    raw = {}
    for block in parse_namelists(text).values():
        for k, v in block.items():
            raw.setdefault(k, v)

    def names(key):
        if key not in raw:
            raise ValueError(f"machine description has no {key}")
        return [str(s).strip() for s in raw[key]]

    loops, probes = names("LPNAME"), names("MPNAM2")
    nl, nprb, nf = len(loops), len(probes), len(raw.get("RF", []))
    if nf != len(f_coil_names):
        raise ValueError(f"machine description has {nf} F-coils, {len(f_coil_names)} names given")
    ecid = sorted({int(x) for x in raw.get("ECID", []) if x is not None})
    if ecid and (ecid[0] < 1 or ecid[-1] > len(e_coil_names)):
        raise ValueError(f"ECID values {ecid} do not index the {len(e_coil_names)} E-coil names given")
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return Device(
        flux_loops={"labels": loops, "r": _floats(raw, "RSI", nl), "z": _floats(raw, "ZSI", nl)},
        probes={"labels": probes, "r": _floats(raw, "XMP2", nprb), "z": _floats(raw, "YMP2", nprb),
                "angle_deg": _floats(raw, "AMP2", nprb), "length": _floats(raw, "SMP2", nprb)},
        f_coils={"labels": list(f_coil_names), "r": _floats(raw, "RF", nf), "z": _floats(raw, "ZF", nf),
                 "w": _floats(raw, "WF", nf), "h": _floats(raw, "HF", nf), "turns": _floats(raw, "TURNFC", nf)},
        e_coils={"labels": [e_coil_names[i - 1] for i in ecid]},
        probe_n_avg=int(raw["NSMP2"][0]) if raw.get("NSMP2") and raw["NSMP2"][0] is not None else None,
        source={"file": os.path.basename(str(path)), "sha256": sha})


# =====================================================================================================
# Named parameter sets (values calibrated on this device; required arguments elsewhere)
# =====================================================================================================
#: Constants of EFIT's flux-loop uncertainty rule (``KERSIL`` 2/3) for k-files that do not carry them:
#: fitted to EFIT's stored uncertainties on a set of EFIT re-runs of this device (an extrapolation to other runs).
EFIT_SIGMA_CONSTANTS = {"ERRSIL": 0.03, "SICONT": 1.0e-9, "NSLREF": 1}
#: Impurity charge of the Z_eff rule's impurity test (carbon, the dominant impurity of this device's walls).
ZEFF_IMPURITY_Z = 6.0
#: Largest spacing [s] of the two stored transport-code times around a slice for the transport-code route
#: (the stored runs' time grids are 20-40 ms in the fit windows).
TC_MAX_GAP_S = 0.040
_ZEFF_THRESHOLDS = dict(
    coverage_rho_inner_max=0.30, coverage_rho_outer_min=0.80, coverage_min_points=8, coverage_window_s=0.025,
    placeholder_rel_spread_min=1e-3, placeholder_rel_sigma_max=1.0, ni_fraction_warn=0.1,
    points_abs_median_z_max=2.0, points_ratio_min=2.0 / 3.0, points_ratio_max=3.0 / 2.0, points_rho_max=0.9,
    points_core_rho_max=0.5, line_rel_sigma_max=0.25, copy_rel_diff_min=1e-3, flat_ptp_min=0.02,
    core_cv_advisory=0.05, tension_z_max=2.0, psi_n_test_max=0.9, psi_n_core_median_max=0.5)
#: Every named set with its basis (recorded with results that use it).
PARAMETER_SETS = {
    "zeff_rule_thresholds": {"value": dict(_ZEFF_THRESHOLDS), "basis": "calibrated on this device's profile "
                             "files (Z_eff source rule); passed as thresholds= of the kinetic Z_eff rule"},
    "zeff_impurity_z": {"value": ZEFF_IMPURITY_Z, "basis": "carbon"},
    "efit_sigma_constants": {"value": dict(EFIT_SIGMA_CONSTANTS), "basis": "fitted to EFIT's stored flux-loop "
                             "uncertainties on a set of EFIT re-runs; for k-files without them"},
    "transport_code_max_gap_s": {"value": TC_MAX_GAP_S, "basis": "the stored transport-code runs' time grids"},
    "source_prior_widths": {"value": "OpenFUSIONToolkit.TokaMaker.sources.priors (decided defaults)",
                            "basis": "development checks on this device's discharges (sources.priors)"},
}


def zeff_rule_thresholds():
    '''The Z_eff rule thresholds of this device as a ``kinetic.zeff_sources.ZeffRuleThresholds``'''
    from OpenFUSIONToolkit.TokaMaker.kinetic.zeff_sources import ZeffRuleThresholds
    return ZeffRuleThresholds(**_ZEFF_THRESHOLDS)


def source_prior_widths():
    '''The decided source prior widths (1 sigma, fraction) used on this device: those of ``sources.priors``'''
    from OpenFUSIONToolkit.TokaMaker.sources import priors
    return {"fast_ion_pressure_scale": priors.FAST_ION_PRESSURE_SCALE_SIGMA,
            "driven_current_scale": priors.DRIVEN_CURRENT_SCALE_SIGMA}


# =====================================================================================================
# Fortran namelists (k-files)
# =====================================================================================================
# The solver's read_fortran_namelist merges all blocks into one dictionary and splits a line at its first "=",
# so k-files that repeat a key in two blocks (FWTSI in IN1 and INWANT) or put several keys on one line give
# wrong arrays.  This reader keeps blocks apart and assigns positionally.
_BLOCK = re.compile(r"[&$]\s*([A-Za-z_][A-Za-z0-9_]*)(.*?)(?:^\s*/\s*$|[&$]END\b|\Z)", re.S | re.M | re.I)
_KEY = re.compile(r"(?<![\w.'\"])([A-Za-z_][A-Za-z0-9_]*)\s*(\(\s*([^)]*?)\s*\))?\s*=")
_STR = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_TOKEN = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|,|[^\s,]+")
_COMMENT = re.compile(r"!.*$", re.M)


def _value(tok):
    t = tok.strip()
    if not t:
        return None
    if t[0] in "'\"":
        return t[1:-1].replace(t[0] * 2, t[0])
    low = t.lower()
    if low in (".true.", "t", ".t."):
        return True
    if low in (".false.", "f", ".f."):
        return False
    try:
        return float(low.replace("d", "e"))
    except ValueError:
        raise ValueError(f"namelist value {t!r} is not a number, logical or string") from None


def _values(text):
    out, prev_comma = [], True
    for tok in _TOKEN.findall(text):
        if tok == ",":
            if prev_comma:
                out.append(None)
            prev_comma = True
            continue
        prev_comma = False
        if "*" in tok and tok[0] not in "'\"":
            k, v = tok.split("*", 1)
            try:
                count = int(k)
            except ValueError:
                raise ValueError(f"namelist repeat {tok!r} has no integer count") from None
            out.extend([_value(v) if v else None] * count)
        else:
            out.append(_value(tok))
    if out and out[-1] is None and text.rstrip().endswith(","):
        out.pop()
    return out


def parse_namelists(text):
    '''Fortran namelist text to ``{BLOCK: {KEY: [values]}}`` (names upper-cased).

    ``n*v`` repeats are expanded, ``KEY(i) =`` / ``KEY(i:j) =`` assign from 1-based position ``i``, a null value
    (``a, , b``) is kept as ``None`` at its position, a key repeated in a block keeps its last assignment and
    ``!`` starts a comment outside strings.  Multi-dimensional entries, text before a block's first key and tokens
    that are not numbers, logicals or strings raise ``ValueError``.
    '''
    text = _STR.sub(lambda s: s.group(0).replace("!", "\1"), text)
    text = _COMMENT.sub("", text).replace("\1", "!")
    out = {}
    for m in _BLOCK.finditer(text):
        name, body = m.group(1).upper(), m.group(2)
        masked = _STR.sub(lambda s: "\0" * len(s.group(0)), body)
        keys = list(_KEY.finditer(masked))
        head = body[:keys[0].start()] if keys else body
        if head.strip():
            raise ValueError(f"namelist block {name}: unparsed text before the first key: {head.strip()[:40]!r}")
        block = out.setdefault(name, {})
        for i, k in enumerate(keys):
            end = keys[i + 1].start() if i + 1 < len(keys) else len(body)
            vals = _values(body[k.end():end])
            key, sub = k.group(1).upper(), k.group(3)
            if sub is not None:
                if "," in sub:
                    raise ValueError(f"namelist block {name}: multi-dimensional entry {key}({sub}) is not supported")
                rng = re.fullmatch(r"(\d+)\s*(?::\s*(\d+))?", sub)
                if rng is None:
                    raise ValueError(f"namelist block {name}: cannot read the subscript of {key}({sub})")
                start = int(rng.group(1)) - 1
                if rng.group(2) is not None:
                    stop = int(rng.group(2))
                    if stop <= start:
                        raise ValueError(f"namelist block {name}: empty range {key}({sub})")
                    if len(vals) > stop - start:
                        raise ValueError(f"namelist block {name}: {len(vals)} values for the range {key}({sub})")
                cur = list(block.get(key, []))
                cur.extend([None] * max(0, start + len(vals) - len(cur)))
                for j, v in enumerate(vals):
                    if v is not None:
                        cur[start + j] = v
                vals = cur
            else:
                prev = block.get(key)
                if prev is not None and any(v is None for v in vals):
                    vals = [prev[j] if (v is None and j < len(prev)) else v for j, v in enumerate(vals)] \
                        + list(prev[len(vals):])
            block[key] = vals
    return out


def read_kfile(path):
    '''Parse an EFIT k-file into ``{BLOCK: {KEY: [values]}}`` (:func:`parse_namelists`)'''
    with open(path) as fh:
        return parse_namelists(fh.read())


def _num(block, key, n=None, default=None):
    '''Array ``key`` as floats (null -> NaN).  With ``n``, trailing entries beyond ``n`` are dropped only when all
    are zero or null (EFIT pads to the namelist dimension); a short array is padded with NaN.'''
    v = block.get(key)
    if v is None:
        if default is None:
            return None
        v = [default] * (n or 1)
    a = np.array([np.nan if x is None else float(x) for x in v], dtype=float)
    if n is not None:
        if a.size < n:
            a = np.concatenate([a, np.full(n - a.size, np.nan)])
        elif a.size > n:
            extra = a[n:]
            if np.any(np.isfinite(extra) & (extra != 0.0)):
                raise ValueError(f"k-file array {key} has {a.size} entries with nonzero values beyond the {n} "
                                 "channels of the machine description: the description does not match the file")
            a = a[:n]
    return a


def _scalar(block, key, default=None):
    a = _num(block, key)
    return default if a is None or a.size == 0 or not np.isfinite(a[0]) else float(a[0])


def _flag_bool(block, key):
    v = block.get(key)
    if not v or v[0] is None:
        return None
    return bool(v[0]) if isinstance(v[0], bool) else bool(float(v[0]))


def _flag(nl, key, blocks=("INS", "EFITIN", "IN1")):
    for b in blocks:
        v = nl.get(b, {}).get(key)
        if v and v[0] is not None:
            return float(v[0])
    return None


# =====================================================================================================
# k-file -> channel tables
# =====================================================================================================
@dataclass
class ChannelTable:
    '''One measurement group: per-channel value, 1-sigma uncertainty, EFIT fit weight and validity.

    ``geometry`` holds what the solver's constraint needs (positions, MSE coefficients); ``meta`` the rule
    used and its inputs.  The fit weight multiplies the normalized residual (EFIT's ``FWT*``).
    '''
    group: str
    labels: List[str]
    measured: np.ndarray
    sigma: np.ndarray
    weight: np.ndarray
    valid: np.ndarray
    geometry: Dict[str, object] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class SliceTables:
    '''The groups of one time slice: ``time`` [s], full averaging window ``time_window`` [s]'''
    time: float
    time_window: float
    tables: Dict[str, ChannelTable]
    meta: Dict[str, object] = field(default_factory=dict)


ALL_GROUPS = ("flux_loop", "bpol_probe", "ip", "diamagnetic_flux", "pf_current", "mse")
#: Uncertainty models of :func:`kfile_tables`; ``"default"`` selects one.
KFILE_SIGMA_MODELS = ("default", "efit_stored", "efit_kersil", "efit_vbit")
#: Readings of the reference flux in the ``efit_kersil`` flux-loop rule.
KERSIL_REFERENCE_READINGS = ("source", "siref")
#: EFIT zeroes the weight of a channel whose uncertainty does not exceed this (file units).
EFIT_SIGMA_FLOOR = 1.0e-10
#: Names of the EFIT constants of the KERSIL flux-loop rule.
KERSIL_CONSTANTS = ("ERRSIL", "SICONT", "ERSIL8")


def flux_gauge(in1, coils, ref_index, flux_scale, series=None):
    '''Storage form of the k-file loop fluxes (``"relative"``, ``"absolute"`` or ``"undetermined"``) and ``SIREF``.

    Decided by the first rule that applies: ``FITSIREF = T`` (relative to a reference offset EFIT fits); ``SIREF = 0``
    or absent (absolute); the reference loop's entry exactly 0 with ``SIREF != 0`` (relative); then ``series`` (a
    :func:`flux_storage_series` record, or ``"relative"`` / ``"absolute"`` stated by the caller).  The absolute flux in
    the group's units is ``measured + absolute_offset`` with ``absolute_offset = flux_scale * SIREF``.
    '''
    siref = _scalar(in1, "SIREF")
    s = 0.0 if siref is None or not np.isfinite(siref) else float(siref)
    fitsiref = _flag_bool(in1, "FITSIREF")
    r = None if ref_index is None else float(coils[ref_index])
    form, rule = "undetermined", None
    if fitsiref:
        form, rule, ev = "relative", "fitsiref", "FITSIREF = T: loops relative to a reference offset that EFIT fits"
    elif s == 0.0 and (r is None or r != 0.0):
        form, rule, ev = "absolute", "siref_zero", "SIREF = 0 (or absent): absolute storage"
    elif r is not None and r == 0.0 and s != 0.0:
        form, rule, ev = "relative", "reference_entry_zero", "reference entry exactly 0 and SIREF != 0: relative"
    else:
        ev = "SIREF != 0, FITSIREF not true, reference entry nonzero or not checked: one slice cannot decide"
        if isinstance(series, str):
            if series not in ("relative", "absolute"):
                raise ValueError("flux_storage must be 'relative', 'absolute' or a flux_storage_series record")
            form, rule = series, "caller"
            ev += f"; stated by the caller: {series}"
        elif series is not None:
            if series.get("coils_form") in ("relative", "absolute"):
                form, rule = series["coils_form"], "series"
            ev += f"; series evidence: {series.get('evidence')}"
    return {"siref": s, "siref_source": "k-file &IN1 SIREF" if siref is not None else "absent (taken as 0)",
            "coils_form": form, "coils_form_rule": rule, "coils_form_evidence": ev, "FITSIREF": fitsiref,
            "absolute_offset": float(flux_scale) * s}


def flux_storage_series(kfiles, ref_index):
    '''Storage form from several slices of ONE discharge (no threshold): over the slices one file cannot decide,
    relative when the reference loop's entry ``c`` stays closer to its mean than to ``SIREF`` (``sum (c - mean c)^2 <
    sum (c - SIREF)^2``), absolute in the opposite case; undetermined with fewer than two such slices, a constant
    ``SIREF`` or a tie.  Pass the record to :func:`kfile_tables` as ``flux_storage``.'''
    c, s = [], []
    for k in kfiles:
        nl = read_kfile(k) if isinstance(k, str) else k
        in1 = nl.get("IN1", {})
        coils = _num(in1, "COILS")
        if coils is None or ref_index >= coils.size:
            raise ValueError("flux_storage_series: a slice has no COILS entry for the reference loop")
        g = flux_gauge(in1, coils, ref_index, 1.0)
        if g["coils_form_rule"] is None:
            c.append(float(coils[ref_index]))
            s.append(g["siref"])
    c, s = np.asarray(c), np.asarray(s)
    out = {"n_slices": int(c.size), "ss_constant": None, "ss_follows_siref": None, "coils_form": "undetermined"}
    if c.size < 2:
        out["evidence"] = f"{c.size} undecided slice(s): at least two are needed"
        return out
    if np.ptp(s) == 0.0:
        out["evidence"] = "SIREF identical on every undecided slice: the readings cannot be told apart"
        return out
    a, b = float(np.sum((c - c.mean()) ** 2)), float(np.sum((c - s) ** 2))
    out.update(ss_constant=a, ss_follows_siref=b)
    if a < b:
        out["coils_form"] = "relative"
    elif b < a:
        out["coils_form"] = "absolute"
    out["evidence"] = f"{c.size} slices: spread about mean {a:.3g}, about SIREF {b:.3g} -> {out['coils_form']}"
    return out


def kersil_loop_sigma(in1, coils, psibit, r_loop, plasma, vbit, consts, reading, nslref=None):
    '''EFIT's flux-loop uncertainty under ``KERSIL`` 2 or 3, in file units (Wb/rad), and a record::

        sigma_m   = max(ERRSIL |psi_m - psi_ref|, SICONT R_m |Ip|, VBIT PSIBIT_m)
        sigma_ref = max(ERRSIL |psi_ref|, SICONT R_ref |Ip|, VBIT PSIBIT_ref)    (KERSIL = 3)
        sigma_ref = ERSIL8                                                        (KERSIL = 2)

    any other ``KERSIL`` is refused.  The reference loop is ``|NSLREF|`` (file, else ``nslref``).  ``reading``:
    ``"siref"`` (reference flux ``SIREF``; other loops ``COILS_m`` when ``SIREF != 0``, ``COILS_m - COILS_ref`` when
    ``SIREF = 0``; inferred from EFIT output, the default) or ``"source"`` (``psi = COILS`` plus ``SIREF`` on every
    loop when ``FITSIREF`` is false and ``IECURR > 0`` or ``NSLREF < 0``; ``psi_ref = psi[reference]``).
    '''
    kersil = _scalar(in1, "KERSIL")
    if kersil is None:
        raise ValueError("efit_kersil needs KERSIL in the k-file")
    nsl = _scalar(in1, "NSLREF")
    nsl_src = "k-file &IN1"
    if nsl is None:
        if nslref is None:
            raise ValueError("NSLREF is absent from the k-file: state the reference-loop number with nslref=...")
        nsl, nsl_src = float(nslref), "caller"
    iref, n = int(abs(nsl)) - 1, coils.size
    if not 0 <= iref < n:
        raise ValueError(f"reference loop |NSLREF| = {abs(nsl):g} outside the {n} loops")
    if kersil not in (2.0, 3.0):
        raise ValueError(f"KERSIL = {kersil:g}: the flux-loop rule of this switch has not been checked against EFIT "
                         "output and is refused; choose another uncertainty model explicitly")
    for k in ("ERRSIL", "SICONT") + (("ERSIL8",) if kersil == 2.0 else ()):
        if k not in consts:
            raise ValueError(f"efit_kersil with KERSIL = {kersil:g} needs {k}: not in the k-file nor efit_constants")
    fitsiref, iecurr, siref = _flag_bool(in1, "FITSIREF"), _scalar(in1, "IECURR"), _scalar(in1, "SIREF")
    ip, bit = abs(float(plasma)), vbit * np.abs(psibit)
    r_loop = np.asarray(r_loop, dtype=float)[:n]
    if reading == "source":
        if fitsiref is None or iecurr is None:
            raise ValueError("the 'source' reading needs FITSIREF and IECURR in the k-file")
        add = (not fitsiref) and (iecurr > 0 or nsl < 0)
        psi = coils + ((siref or 0.0) if add else 0.0)
        rel, ref_mag = np.abs(psi - psi[iref]), abs(psi[iref])
    else:
        s = siref or 0.0
        rel, ref_mag = np.abs(coils if s != 0.0 else coils - coils[iref]), abs(s)
    sig = np.maximum.reduce([consts["ERRSIL"] * rel, consts["SICONT"] * r_loop * ip, bit])
    if kersil == 3.0:
        sig[iref] = max(consts["ERRSIL"] * ref_mag, consts["SICONT"] * r_loop[iref] * ip, bit[iref])
    else:
        sig[iref] = consts["ERSIL8"]
    rec = {"KERSIL": kersil, "NSLREF": nsl, "NSLREF_source": nsl_src, "reading": reading, "FITSIREF": fitsiref,
           "IECURR": iecurr, "SIREF": siref, "reference_index": iref,
           "constants": {k: v for k, v in consts.items() if k != "_source"}, "constants_source": consts.get("_source")}
    return sig, rec


def _efit_constants(in1, given):
    out, src = {}, {}
    for k in KERSIL_CONSTANTS:
        v = _scalar(in1, k)
        if v is not None:
            out[k], src[k] = v, "k-file &IN1"
        elif given and k in given:
            out[k], src[k] = float(given[k]), "caller (efit_constants)"
    out["_source"] = src
    return out


def efit_to_imas_coefficients(a1, a2, a3, a4, a5=None, a6=None, a7=None, a8=None):
    '''EFIT MSE coefficients ``tan gamma = (A1 B_Z + A8 B_R + A5 E_R) / (A2 B_phi + A3 B_R + A4 B_Z + A6 E_Z + A7 E_R)``
    to the 9 IMAS coefficients ``[A1, A8, 0, A5, A4, A3, A2, A7, A6]`` (the order of ``MSE_con``); absent ones 0.'''
    a1 = np.atleast_1d(np.asarray(a1, dtype=float))
    z = np.zeros_like(a1)

    def _c(a):
        return z if a is None else np.broadcast_to(np.asarray(a, float), a1.shape)

    return np.stack([a1, _c(a8), z, _c(a5), _c(a4), _c(a3), _c(a2), _c(a7), _c(a6)], axis=-1)


def _stored(stored, group, n):
    if group not in stored:
        raise ValueError(f"sigma_model 'efit_stored': stored_sigma has no entry for group '{group}'")
    a = np.asarray(stored[group], dtype=float).ravel()
    if n is not None and a.size < n:
        raise ValueError(f"stored_sigma['{group}'] has {a.size} entries for {n} channels")
    return a if n is None else a[:n]


def kfile_tables(kfile, device, groups=None, *, er_corrected=None, flux_scale=-2.0 * np.pi, flux_reference=None,
                 sigma_model="default", score_unfitted_coils=True, serror=None, time=None, time_window=None,
                 allow_missing_bits=False, efit_constants=None, kersil_reference=None, nslref=None,
                 stored_sigma=None, vbit=None, flux_storage=None, coil_current_per_turn=(),
                 allow_calibration_sigma_option=False):
    '''Channel tables of one EFIT k-file (namelists ``IN1`` and ``INS``).

    @param kfile Path, or the dict of :func:`read_kfile`
    @param device :class:`Device` whose channels are in the file's order
    @param groups Groups to build (default: those present; a requested group absent from the file raises)
    @param er_corrected Whether ``TGAMMA`` is E_r-corrected (stated; required when the MSE group is built)
    @param flux_scale Multiplies ``COILS``, ``PSIBIT`` and ``SIREF`` (default ``-2 pi``, IMAS flux-loop sign)
    @param flux_reference Label of the reference loop for relative storage (refused on absolute data)
    @param sigma_model ``"default"`` (EFIT's stored sigma when ``stored_sigma`` is given, else ``"efit_kersil"`` with
    the ``"siref"`` reading), ``"efit_stored"``, ``"efit_kersil"`` (``kersil_reference`` required) or ``"efit_vbit"``
    (``max(SERROR |COILS|, VBIT PSIBIT)`` on the stored loop fluxes, ``SERROR |PLASMA|`` for Ip)
    @param score_unfitted_coils Coils with fit weight 0 stay valid with weight 1 (default) or are invalid
    @param serror, time, vbit Used only when the file lacks ``SERROR``, ``ITIME`` [s], ``VBIT`` (else refused)
    @param time_window Full averaging window [s] (default ``2 IAVEM``)
    @param allow_missing_bits Take an absent digitizer-bit array as 0 (recorded) instead of refusing
    @param efit_constants ``ERRSIL``, ``SICONT``, ``ERSIL8`` (and ``NSLREF``) for files without them, e.g.
    :data:`EFIT_SIGMA_CONSTANTS`
    @param kersil_reference ``"siref"`` or ``"source"`` (:func:`kersil_loop_sigma`)
    @param nslref Reference-loop number (1-based) when the file has no ``NSLREF`` (else ``efit_constants``)
    @param stored_sigma EFIT's own sigma per group in EFIT units (``efit_stored``)
    @param flux_storage Storage-form evidence for slices one file cannot decide (:func:`flux_gauge`)
    @param coil_current_per_turn Arrays (``"BRSP"``, ``"ECURRT"``, or True for both) divided by the coil turns
    @param allow_calibration_sigma_option Accept a file with ``IMAGSIGMA > 0`` under a rule-based model
    @result :class:`SliceTables`
    '''
    if sigma_model not in KFILE_SIGMA_MODELS:
        raise ValueError(f"sigma_model must be one of {KFILE_SIGMA_MODELS}, got {sigma_model!r}")
    requested_model = sigma_model
    if sigma_model == "default":
        if stored_sigma is not None:
            sigma_model, selection = "efit_stored", "default: EFIT's stored sigma supplied"
        else:
            if kersil_reference not in (None, "siref"):
                raise ValueError("the default model uses the 'siref' reading; name sigma_model='efit_kersil'")
            sigma_model, kersil_reference = "efit_kersil", "siref"
            selection = "default: no stored sigma; EFIT's KERSIL flux-loop rule (reading 'siref') and EFIT's rule"
    else:
        selection = "named by the caller"
    if sigma_model == "efit_kersil" and kersil_reference not in KERSIL_REFERENCE_READINGS:
        raise ValueError(f"sigma_model 'efit_kersil' needs kersil_reference in {KERSIL_REFERENCE_READINGS}")
    if sigma_model == "efit_stored" and not isinstance(stored_sigma, dict):
        raise ValueError("sigma_model 'efit_stored' needs stored_sigma={group: EFIT sigma array}")
    nl = read_kfile(kfile) if isinstance(kfile, str) else kfile
    in1 = nl.get("IN1", {})
    imag = _flag(nl, "IMAGSIGMA")
    if imag is not None and imag > 0 and sigma_model != "efit_stored" and not allow_calibration_sigma_option:
        raise ValueError(f"the k-file turns on EFIT's calibration-error uncertainty model (IMAGSIGMA = {imag:g}): no "
                         "rule-based model reproduces it; use EFIT's stored sigma or allow_calibration_sigma_option")
    serror_file = _scalar(in1, "SERROR")
    if serror_file is not None:
        if serror is not None:
            raise ValueError("serror given by the caller but the k-file sets SERROR: remove one")
        serror, serror_src = serror_file, "k-file &IN1"
    else:
        serror_src = "caller" if serror is not None else None
    vbit_file = _scalar(in1, "VBIT")
    if vbit_file is not None:
        if vbit is not None:
            raise ValueError("vbit given by the caller but the k-file sets VBIT: remove one")
        vbit, vbit_src = vbit_file, "k-file &IN1"
    else:
        vbit_src = "caller" if vbit is not None else None
    itime, iavem = _scalar(in1, "ITIME"), _scalar(in1, "IAVEM")
    if itime is not None:
        if time is not None:
            raise ValueError("time given by the caller but the k-file sets ITIME: remove one")
        t_mid = itime * 1e-3
    elif time is not None:
        t_mid = float(time)
    else:
        raise ValueError("ITIME is absent from the k-file's &IN1: state the slice time with time=... [s]")
    if time_window is None:
        if iavem is None:
            raise ValueError("IAVEM is absent from the k-file's &IN1: state time_window=... [s] (full width)")
        time_window = 2.0 * iavem * 1e-3
    model_id = f"efit_kersil:{kersil_reference}" if sigma_model == "efit_kersil" else sigma_model
    out = SliceTables(time=t_mid, time_window=float(time_window), tables={},
                      meta={"source": "efit_kfile", "SERROR": serror, "SERROR_source": serror_src, "VBIT": vbit,
                            "VBIT_source": vbit_src, "sigma_model": sigma_model, "sigma_model_id": model_id,
                            "sigma_model_requested": requested_model, "sigma_model_selection": selection,
                            "IMAGSIGMA": imag, "device_source": dict(device.source)})
    if isinstance(kfile, str):
        out.meta["file"] = os.path.basename(kfile)
    requested = groups is not None
    groups = list(groups) if requested else list(ALL_GROUPS)

    def need(name, present):
        if not present and requested:
            raise ValueError(f"group '{name}' requested but absent from the k-file")
        return present

    def arrays(name, vkey, fkey, bkey, n, bit_needed=True):
        if serror is None:
            raise ValueError(f"group '{name}': SERROR is absent from the k-file's &IN1; state it with serror=...")
        if vbit is None and sigma_model in ("efit_vbit", "efit_kersil"):
            raise ValueError(f"group '{name}': VBIT is absent from the k-file's &IN1; state it with vbit=...")
        value = _num(in1, vkey, n)
        if np.count_nonzero(np.isfinite(value)) < n:
            raise ValueError(f"group '{name}': k-file has {np.count_nonzero(np.isfinite(value))} values for {n} "
                             "channels in the machine description")
        if fkey not in in1:
            raise ValueError(f"group '{name}': {vkey} present but {fkey} (fit weights) absent from &IN1")
        fwt = _num(in1, fkey, n)
        if bkey in in1:
            bit, bsrc = _num(in1, bkey, n), f"k-file &IN1 {bkey}"
        elif not bit_needed:
            bit, bsrc = np.zeros(n), f"{bkey} absent (not used by this rule)"
        elif allow_missing_bits:
            bit, bsrc = np.zeros(n), f"{bkey} absent: floor taken as 0 (allow_missing_bits=True)"
        else:
            raise ValueError(f"group '{name}': digitizer-bit array {bkey} absent from &IN1; pass "
                             "allow_missing_bits=True to accept a zero floor explicitly")
        if np.any(~np.isfinite(fwt)):
            raise ValueError(f"group '{name}': {fkey} has null or non-finite entries")
        return value, fwt, bit, bsrc

    def table(name, value, fwt, bit, labels, geometry, scale=1.0, extra=None, rule="bit", floor_only=None,
              unfitted_valid=False, sigma_override=None, override_text=None):
        n = len(labels)
        value, fwt, bit = value[:n] * scale, fwt[:n], bit[:n] * np.abs(scale)
        if sigma_override is not None:
            sigma = np.asarray(sigma_override, dtype=float)[:n] * np.abs(scale)
            sigma = np.where(sigma > EFIT_SIGMA_FLOOR * np.abs(scale), sigma, np.nan)
            text = override_text
        elif rule == "relative" and sigma_model == "efit_vbit":
            sigma, text = serror * np.abs(value), "SERROR |x|"
        else:
            sigma, text = np.maximum(serror * np.abs(value), vbit * bit), "max(SERROR |x|, VBIT bit)"
            if floor_only is not None and np.any(floor_only):
                sigma = np.where(floor_only, vbit * bit, sigma)
                text += "; reference loop: VBIT bit"
        good = np.isfinite(value) & np.isfinite(sigma) & (sigma > 0)
        unfitted = fwt <= 0
        if unfitted_valid:
            valid, weight = good, np.where(unfitted, 1.0, fwt)
        else:
            valid, weight = good & ~unfitted, fwt
        meta = {"sigma_model_id": model_id, "sigma_rule": text}
        if unfitted_valid:
            meta["efit_fit_weight"] = fwt.copy()
            meta["not_fitted_by_efit"] = [labels[i] for i in np.nonzero(unfitted)[0]]
        meta.update(extra or {})
        return ChannelTable(group=name, labels=list(labels), measured=value, sigma=sigma, weight=weight,
                            valid=valid, geometry=geometry, meta=meta)

    if "flux_loop" in groups and need("flux_loop", "COILS" in in1):
        labels = device.labels("flux_loop")
        n = len(labels)
        if flux_reference is not None and flux_reference not in labels:
            raise ValueError(f"flux_reference {flux_reference!r} is not a flux-loop label")
        ref = np.zeros(n, dtype=bool)
        if flux_reference is not None:
            ref[labels.index(flux_reference)] = True
        coils, fwt_l, bit_l, bsrc = arrays("flux_loop", "COILS", "FWTSI", "PSIBIT", n)
        gauge = flux_gauge(in1, coils, labels.index(flux_reference) if flux_reference else None, flux_scale,
                           series=flux_storage)
        if flux_reference is not None and gauge["coils_form"] != "relative":
            raise ValueError(f"flux_reference {flux_reference!r} given, but this slice's loop fluxes are not in "
                             f"relative form ({gauge['coils_form']}: {gauge['coils_form_evidence']})")
        extra = dict(gauge, flux_scale=flux_scale, flux_reference=flux_reference, bit_source=bsrc)
        override, otext = None, None
        if sigma_model == "efit_kersil":
            plasma = _scalar(in1, "PLASMA")
            if plasma is None:
                raise ValueError("efit_kersil flux-loop sigma needs PLASMA (Ip) in &IN1")
            nsl = nslref if nslref is not None else (efit_constants or {}).get("NSLREF")
            override, krec = kersil_loop_sigma(in1, coils, bit_l, device.flux_loops["r"], plasma, vbit,
                                               _efit_constants(in1, efit_constants), kersil_reference, nsl)
            otext, ref = f"EFIT KERSIL = {krec['KERSIL']:g} rule, reading '{kersil_reference}'", np.zeros(n, bool)
            extra["kersil"] = krec
        elif sigma_model == "efit_stored":
            override, otext = _stored(stored_sigma, "flux_loop", n), "EFIT's stored sigma x |flux_scale|"
            ref = np.zeros(n, dtype=bool)
        out.tables["flux_loop"] = table(
            "flux_loop", coils, fwt_l, bit_l, labels,
            {"r": list(device.flux_loops["r"]), "z": list(device.flux_loops["z"])}, scale=flux_scale,
            floor_only=ref, sigma_override=override, override_text=otext, extra=extra)
    if "bpol_probe" in groups and need("bpol_probe", "EXPMP2" in in1):
        labels = device.labels("bpol_probe")
        n = len(labels)
        v, w, b, bsrc = arrays("bpol_probe", "EXPMP2", "FWTMP2", "BITMPI", n)
        geo = {k: list(device.probes[k]) for k in ("r", "z", "angle_deg", "length")}
        out.tables["bpol_probe"] = table(
            "bpol_probe", v, w, b, labels, geo, extra={"bit_source": bsrc},
            sigma_override=_stored(stored_sigma, "bpol_probe", n) if sigma_model == "efit_stored" else None,
            override_text="EFIT's stored sigma")
    if "ip" in groups and need("ip", "PLASMA" in in1):
        v, w, b, bsrc = arrays("ip", "PLASMA", "FWTCUR", "BITIP", 1, bit_needed=sigma_model != "efit_vbit")
        out.tables["ip"] = table(
            "ip", v, w, b, ["ip"], {}, rule="relative", extra={"bit_source": bsrc},
            sigma_override=_stored(stored_sigma, "ip", 1) if sigma_model == "efit_stored" else None,
            override_text="EFIT's stored sigma")
    if "pf_current" in groups and need("pf_current", "BRSP" in in1 or "ECURRT" in in1):
        out.tables["pf_current"] = _coil_table(in1, device, table, arrays, coil_current_per_turn,
                                               score_unfitted_coils,
                                               stored_sigma if sigma_model == "efit_stored" else None)
    if "diamagnetic_flux" in groups and need("diamagnetic_flux", "DFLUX" in in1):
        v = _num(in1, "DFLUX", 1) * 1e-3
        if "FWTDLC" not in in1:
            raise ValueError("group 'diamagnetic_flux': DFLUX present but FWTDLC absent from &IN1")
        w = _num(in1, "FWTDLC", 1)
        if sigma_model == "efit_stored":
            s, text = _stored(stored_sigma, "diamagnetic_flux", 1), "EFIT's stored sigma [Wb]"
        else:
            if "SIGDLC" not in in1:
                raise ValueError("group 'diamagnetic_flux': DFLUX present but SIGDLC absent from &IN1")
            s, text = _num(in1, "SIGDLC", 1) * 1e-3, "SIGDLC x 1e-3"
        out.tables["diamagnetic_flux"] = ChannelTable(
            group="diamagnetic_flux", labels=["diamagnetic_flux"], measured=v, sigma=s, weight=w,
            valid=(w > 0) & np.isfinite(s) & (s > 0), meta={"sigma_rule": text, "units": "Wb (file mV s x 1e-3)"})
    if "mse" in groups and need("mse", "TGAMMA" in nl.get("INS", {})):
        out.tables["mse"] = _mse_table(nl, er_corrected,
                                       _stored(stored_sigma, "mse", None) if sigma_model == "efit_stored" else None)
    return out


def _coil_table(in1, device, table, arrays, per_turn, score_unfitted, stored):
    parts = [p for p in (("BRSP", "FWTFC", "BITFC"), ("ECURRT", "FWTEC", "BITEC")) if p[0] in in1]
    names = {"BRSP": list(device.f_coils["labels"]), "ECURRT": list(device.e_coils["labels"])}
    if per_turn is True:
        per_turn = tuple(k for k, _, _ in parts)
    per_turn = tuple(per_turn or ())
    unknown = [k for k in per_turn if k not in ("BRSP", "ECURRT")]
    if unknown:
        raise ValueError(f"coil_current_per_turn names unknown arrays {unknown}")
    if "ECURRT" in per_turn:
        raise ValueError("coil_current_per_turn: the machine description carries no E-coil turns")
    labels, val, fwt, bit, scale, src = [], [], [], [], [], []
    for k, fk, bk in parts:
        n = len(names[k])
        v, w, b, s = arrays("pf_current", k, fk, bk, n)
        labels += names[k]
        val.append(v)
        fwt.append(w)
        bit.append(b)
        src.append(s)
        scale.append(1.0 / np.asarray(device.f_coils["turns"], float) if k in per_turn else np.ones(n))
    val, fwt, bit, scale = (np.concatenate(x) for x in (val, fwt, bit, scale))
    iecurr = _scalar(in1, "IECURR")
    roles = {}
    for k, _, _ in parts:
        for lab in names[k]:
            w = float(fwt[labels.index(lab)])
            if k == "BRSP" or iecurr == 2.0:
                roles[lab] = "fitted" if w > 0 else "free"
            else:
                roles[lab] = "held" if iecurr == 1.0 else "undetermined"
    extra = {"coil_roles": roles, "IECURR": iecurr, "bit_source": "; ".join(src),
             "current_convention": {k: ("k-file value / turns" if k in per_turn else "as written") for k, _, _ in parts},
             "coil_arrays": {k: names[k] for k, _, _ in parts}}
    return table("pf_current", val, fwt, bit, labels, {}, scale=scale, extra=extra,
                 unfitted_valid=bool(score_unfitted),
                 sigma_override=None if stored is None else _stored(stored, "pf_current", len(labels)),
                 override_text="EFIT's stored sigma")


def _mse_table(nl, er_corrected, stored=None):
    ins = nl.get("INS", {})
    sig_key = "SGAMMA" if ins.get("SGAMMA") else "SIGGAM"
    core = ["TGAMMA", sig_key, "FWTGAM", "RRRGAM", "ZZZGAM", "AA1GAM", "AA2GAM", "AA3GAM", "AA4GAM"]
    missing = [k for k in core if not ins.get(k)]
    if missing:
        raise ValueError(f"k-file MSE block lacks {missing}")
    if er_corrected is None or not isinstance(er_corrected, (bool, np.bool_)):
        raise ValueError("MSE in the k-file: state er_corrected=True or False (whether TGAMMA is E_r-corrected); "
                         "leave 'mse' out of groups to build the slice without MSE")
    n = min(len(ins[k]) for k in core)
    tg = _num(ins, "TGAMMA", n)
    sg = _num(ins, sig_key, n) if stored is None else _stored({"mse": stored}, "mse", n)
    fw = _num(ins, "FWTGAM", n, 0.0)
    a = {i: _num(ins, f"AA{i}GAM", n, 0.0) for i in range(1, 8)}
    valid = (fw > 0) & np.isfinite(sg) & (sg > 0) & np.isfinite(tg) & (np.abs(tg) > 1e-8)
    unc = _num(ins, "TGAMMAUNCOR", n) if ins.get("TGAMMAUNCOR") and len(ins["TGAMMAUNCOR"]) >= n else None
    keecur = _flag(nl, "KEECUR", ("IN1", "EFITIN", "INS"))
    n_act = int(valid.sum())
    n_diff = None if unc is None else int(np.count_nonzero(tg[valid] != unc[valid]))
    if n_diff is not None and 0 < n_diff < n_act:
        raise ValueError(f"MSE in the k-file: TGAMMA differs from TGAMMAUNCOR on {n_diff} of {n_act} active chords "
                         "(an E_r correction on only some chords): refused")
    if keecur is not None and keecur > 0 and (er_corrected or (n_diff or 0) > 0):
        raise ValueError(f"MSE in the k-file: KEECUR = {keecur:g} (E_r fitted inside EFIT on the raw TGAMMA) with an "
                         "already E_r-corrected TGAMMA: the E_r effect would be counted twice; refused")
    meta = {"sigma_rule": sig_key if stored is None else "EFIT's stored sigma", "quantity": "tan",
            "er_corrected": bool(er_corrected), "KEECUR": keecur, "MSE_USECER": _flag(nl, "MSE_USECER"),
            "n_active": n_act, "n_tgamma_differs_from_uncorrected": n_diff}
    if unc is not None:
        meta["tan_gamma_uncorrected"] = unc
    geo = {"r": _num(ins, "RRRGAM", n), "z": _num(ins, "ZZZGAM", n),
           "coefficients": efit_to_imas_coefficients(a[1], a[2], a[3], a[4], a[5], a[6], a[7])}
    return ChannelTable(group="mse", labels=[f"mse_{i:03d}" for i in range(n)], measured=tg, sigma=sg, weight=fw,
                        valid=valid, geometry=geo, meta=meta)


# =====================================================================================================
# EFIT stored measurement arrays (one time) -> channel tables
# =====================================================================================================
#: Measurement-node names of an EFIT run per group: (measured, stored sigma, weight, fitted, chi^2).
#: OMAS ``_efit.json`` (``scalar_constraint_data`` / ``vector_constraint_data`` / ``concat_constraint_data``).
EFIT_GROUP_NODES = {
    "flux_loop": ("SILOPT", "SIGSIL", "FWTSI", "CSILOP", "SAISIL"),
    "bpol_probe": ("EXPMPI", "SIGMPI", "FWTMP2", "CMPR2", "SAIMPI"),
    "f_coil": ("FCCURT", "SIGFCC", "FWTFC", "CCBRSP", "CHIFCC"),
    "e_coil": ("ECCURT", "SIGECC", "FWTEC", "CECURR", "CHIECC"),
    "ip": ("PLASMA", "SIGPASMA", "FWTPASMA", "CPASMA", "CHIPASMA"),
    "diamagnetic_flux": ("DIAMAG", "SIGDIA", "FWTDIA", "CDFLUX", "CHIDFLUX"),
    "mse": ("TANGAM", "SIGGAM", "FWTGAM", "CMGAM", "CHIGAM"),
}
#: MSE geometry nodes of an EFIT run (MSE point names).
MSE_GEOMETRY_NODES = ("RRGAM", "ZZGAM", "A1GAM", "A2GAM", "A3GAM", "A4GAM", "A5GAM", "A6GAM", "A7GAM")
#: Every measurement node pulled for :func:`efit_array_tables` (``MTIME``: OMAS ``_common.py``).
EFIT_MEASUREMENT_NODES = ("MTIME",) + tuple(n for v in EFIT_GROUP_NODES.values() for n in v) + MSE_GEOMETRY_NODES


def efit_array_tables(arrays, device, *, groups=None, flux_scale=-2.0 * np.pi, er_corrected=None,
                      score_unfitted_coils=True):
    '''Channel tables from the stored measurement arrays of one EFIT time (EFIT's own sigma).

    @param arrays ``{node name: array at this time}`` (names of :data:`EFIT_MEASUREMENT_NODES`, case-blind;
    ``MTIME`` [ms] gives the time)
    @param device :class:`Device` in the arrays' channel order
    @param flux_scale Multiplies ``SILOPT`` and ``SIGSIL`` (stored absolute, in TokaMaker's ``psi`` sign)
    @param er_corrected Whether ``TANGAM`` is E_r-corrected (stated; required for the MSE group)
    A channel is valid when its weight is positive and its sigma finite and above EFIT's floor; coils with weight 0
    stay valid with weight 1 when ``score_unfitted_coils``.  The stored fit and chi^2 are kept in ``meta``.
    '''
    A = {str(k).upper(): np.asarray(v, dtype=float) for k, v in arrays.items()}
    if "MTIME" not in A:
        raise ValueError("the EFIT arrays have no MTIME: the time of the slice is not stated")
    groups = list(groups or [g for g in ("flux_loop", "bpol_probe", "pf_current", "ip", "diamagnetic_flux", "mse")
                             if EFIT_GROUP_NODES["f_coil" if g == "pf_current" else g][0] in A])

    def get(key, n):
        if key not in A:
            raise ValueError(f"EFIT arrays: {key} absent")
        a = np.atleast_1d(A[key]).ravel()
        if a.size < n:
            raise ValueError(f"EFIT arrays: {key} has {a.size} entries for {n} channels")
        return a[:n]

    def build(name, keys_list, labels, geometry, scale=1.0, unfitted_valid=False):
        n = len(labels)
        val, sig, wt, fit, chi = (np.concatenate([get(k[i], m) for k, m in keys_list]) for i in range(5))
        val, sig, fit = val * scale, sig * abs(scale), fit * scale
        good = np.isfinite(val) & np.isfinite(sig) & (sig > EFIT_SIGMA_FLOOR * abs(scale))
        sig = np.where(good, sig, np.nan)
        if unfitted_valid:
            valid, weight = good, np.where(wt <= 0, 1.0, wt)
        else:
            valid, weight = good & (wt > 0), wt
        assert val.size == n
        return ChannelTable(group=name, labels=list(labels), measured=val, sigma=sig, weight=weight, valid=valid,
                            geometry=geometry, meta={"sigma_rule": "EFIT's stored sigma", "efit_fit_weight": wt,
                                                     "efit_fitted": fit, "efit_chi2": chi})

    out = SliceTables(time=float(np.atleast_1d(A["MTIME"])[0]) * 1e-3, time_window=float("nan"), tables={},
                      meta={"source": "efit_arrays", "device_source": dict(device.source)})
    for g in groups:
        if g == "flux_loop":
            labs = device.labels(g)
            out.tables[g] = build(g, [(EFIT_GROUP_NODES[g], len(labs))], labs,
                                  {"r": list(device.flux_loops["r"]), "z": list(device.flux_loops["z"])},
                                  scale=flux_scale)
            out.tables[g].meta.update(coils_form="absolute", absolute_offset=0.0, flux_scale=flux_scale)
        elif g == "bpol_probe":
            labs = device.labels(g)
            out.tables[g] = build(g, [(EFIT_GROUP_NODES[g], len(labs))], labs,
                                  {k: list(device.probes[k]) for k in ("r", "z", "angle_deg", "length")})
        elif g == "pf_current":
            nf, ne = len(device.f_coils["labels"]), len(device.e_coils["labels"])
            out.tables[g] = build(g, [(EFIT_GROUP_NODES["f_coil"], nf), (EFIT_GROUP_NODES["e_coil"], ne)],
                                  device.labels(g), {}, unfitted_valid=bool(score_unfitted_coils))
        elif g in ("ip", "diamagnetic_flux"):
            out.tables[g] = build(g, [(EFIT_GROUP_NODES[g], 1)], [g], {})
        elif g == "mse":
            if er_corrected is None or not isinstance(er_corrected, (bool, np.bool_)):
                raise ValueError("MSE: state er_corrected=True or False (whether TANGAM is E_r-corrected)")
            n = np.atleast_1d(A["TANGAM"]).size
            co = {i: (np.atleast_1d(A[f"A{i}GAM"]).ravel()[:n] if f"A{i}GAM" in A else None) for i in range(1, 8)}
            if any(co[i] is None for i in range(1, 5)):
                raise ValueError("MSE: A1GAM..A4GAM are required")
            t = build(g, [(EFIT_GROUP_NODES[g], n)], [f"mse_{i:03d}" for i in range(n)],
                      {"r": get("RRGAM", n), "z": get("ZZGAM", n),
                       "coefficients": efit_to_imas_coefficients(*[co[i] for i in range(1, 8)])})
            t.valid &= np.abs(t.measured) > 1e-8
            t.meta.update(quantity="tan", er_corrected=bool(er_corrected))
            out.tables[g] = t
        else:
            raise ValueError(f"unknown group {g!r}")
    return out


# =====================================================================================================
# Channel tables -> the solver's reconstruction constraints
# =====================================================================================================
def to_constraints(slice_tables, coil_index=None, *, probe_n_avg=None, mse_E_R=None, unused="skip",
                   missing_coils="refuse"):
    '''Constraint objects of ``TokaMaker.reconstruction`` from :class:`SliceTables`.

    The fit weight is folded into the error (``err = sigma / weight``, as EFIT folds ``FWT`` into ``1/sigma``).
    Flux loops become ``fluxLoop_con`` on the solver's ``2 pi psi`` [Wb]: the ABSOLUTE flux
    ``(2 pi / flux_scale) (measured + absolute_offset)``; a slice whose storage form is undetermined is refused.
    Probes are ``Mirnov_con`` with normal ``(cos AMP2, 0, sin AMP2)``; finite-length probes when ``probe_n_avg`` is
    given (length ``SMP2``), point probes otherwise.  Coil currents are ``coil_current_con`` on the index
    ``coil_index[name]`` (e.g. ``{k: v['id'] for k, v in mygs.coil_sets.items()}``); a coil the mesh does not have
    raises unless ``missing_coils="skip"`` (listed in the result).  MSE chords are ``MSE_con`` (``quantity='tan'``)
    with ``E_R`` per chord from ``mse_E_R`` (default 0, recorded).

    @param unused ``"skip"`` (default) drops invalid channels; ``"keep"`` adds those with a finite value and
    uncertainty as ``use=False`` constraints (evaluated, zero weight)
    @result ``{"constraints": {group: [objects]}, "labels": {group: [labels]}, "skipped": {group: [labels]}}``
    '''
    from OpenFUSIONToolkit.TokaMaker.reconstruction import (Ip_con, MSE_con, Mirnov_con, coil_current_con,
                                                           dFlux_con, fluxLoop_con)
    if unused not in ("skip", "keep"):
        raise ValueError("unused must be 'skip' or 'keep'")
    if missing_coils not in ("refuse", "skip"):
        raise ValueError("missing_coils must be 'refuse' or 'skip'")
    cons, labs, skipped = {}, {}, {}
    for g, t in slice_tables.tables.items():
        cons[g], labs[g], skipped[g] = [], [], []
        if g == "flux_loop":
            if t.meta.get("coils_form") not in ("relative", "absolute"):
                raise ValueError(f"flux loops: storage form {t.meta.get('coils_form')!r}: the absolute flux is "
                                 "not defined (state flux_storage)")
            f = 2.0 * np.pi / float(t.meta["flux_scale"])
            val = f * (t.measured + float(t.meta["absolute_offset"]))
        for i, lab in enumerate(t.labels):
            ok = bool(t.valid[i]) and t.weight[i] > 0
            if not ok and not (unused == "keep" and np.isfinite(t.measured[i]) and np.isfinite(t.sigma[i])
                               and t.sigma[i] > 0):
                skipped[g].append(lab)
                continue
            err = float(t.sigma[i] / t.weight[i]) if ok else float(t.sigma[i])
            if g == "flux_loop":
                c = fluxLoop_con(loc=[t.geometry["r"][i], t.geometry["z"][i]], val=float(val[i]), err=abs(f) * err,
                                 use=ok)
            elif g == "bpol_probe":
                a = np.deg2rad(t.geometry["angle_deg"][i])
                kw = {} if probe_n_avg is None else {"length": float(t.geometry["length"][i]),
                                                     "n_avg": int(probe_n_avg)}
                c = Mirnov_con(loc=[t.geometry["r"][i], t.geometry["z"][i]], norm=[np.cos(a), 0.0, np.sin(a)],
                               val=float(t.measured[i]), err=err, use=ok, **kw)
            elif g == "ip":
                c = Ip_con(val=float(t.measured[i]), err=err, use=ok)
            elif g == "diamagnetic_flux":
                c = dFlux_con(val=float(t.measured[i]), err=err, use=ok)
            elif g == "pf_current":
                if coil_index is None or lab not in coil_index:
                    if missing_coils == "skip":
                        skipped[g].append(lab)
                        continue
                    raise ValueError(f"coil {lab!r} has no index in coil_index (pass missing_coils='skip' to drop it)")
                c = coil_current_con(ind=int(coil_index[lab]), val=float(t.measured[i]), err=err, use=ok)
            elif g == "mse":
                er = 0.0 if mse_E_R is None else float(np.asarray(mse_E_R, float)[i])
                c = MSE_con(loc=[float(t.geometry["r"][i]), float(t.geometry["z"][i])],
                            coefs=t.geometry["coefficients"][i], val=float(t.measured[i]), err=err, quantity="tan",
                            E_R=er, use=ok)
            else:
                raise ValueError(f"no solver constraint for group {g!r}")
            cons[g].append(c)
            labs[g].append(lab)
    return {"constraints": cons, "labels": labs, "skipped": skipped,
            "record": {"mse_E_R": "caller" if mse_E_R is not None else "0 (not supplied)",
                       "probe_model": "finite length" if probe_n_avg is not None else "point",
                       "probe_n_avg": probe_n_avg, "error": "sigma / EFIT fit weight"}}


# =====================================================================================================
# Fetch (through a connection the user opened) and cache
# =====================================================================================================
#: Profile-fit nodes of a profile-fit tree (tree name an argument); OMAS ``d3d.py`` core-profiles mapping.
PROFILE_FIT_NODES = ("EDENSFIT", "ETEMPFIT", "ITEMPFIT", "ZDENSFIT", "TROTFIT")
#: Neutral-beam names and per-beam node fields; OMAS ``d3d.py``, ``nbi_active_hardware`` (tree ``NB``).
BEAM_NAMES = ("30L", "30R", "15L", "15R", "21L", "21R", "33L", "33R")
BEAM_NODE_FIELDS = ("PINJ", "TINJ", "VBEAM", "GAS")
#: Stored transport-code point names used by :func:`transport_code_sources` (tree name an argument).
TRANSPORT_CODE_POINTS = ("EQU:PRESS_BEAM", "NBI:ENBEAM", "HCD:J_CD", "NBI:J_NBCD", "EQU:RMAJ0", "EQU:RHO")


def node_spec(*, efit_tree=None, profile_tree=None, transport_tree=None, beams=False, extra=()):
    '''Node list ``[(tree, TDI expression)]`` of the named node sets.  Every tree name is the caller's.

    ``efit_tree``: :data:`EFIT_MEASUREMENT_NODES` as ``\\TREE::TOP.MEASUREMENTS.NAME`` (the OMAS form);
    ``profile_tree``: :data:`PROFILE_FIT_NODES` as ``\\TOP.PROFILES.NAME``; ``transport_tree``:
    :data:`TRANSPORT_CODE_POINTS` as ``\\TREE::TOP.POINT``; ``beams``: the ``NB`` tree nodes of every beam;
    ``extra``: further ``(tree, expression)`` pairs (``tree = ""`` for expressions that need no tree).
    '''
    spec = []
    if efit_tree:
        spec += [(efit_tree, f"\\{efit_tree}::TOP.MEASUREMENTS.{n}") for n in EFIT_MEASUREMENT_NODES]
    if profile_tree:
        spec += [(profile_tree, f"\\TOP.PROFILES.{n}") for n in PROFILE_FIT_NODES]
    if transport_tree:
        spec += [(transport_tree, f"\\{transport_tree}::TOP.{p}") for p in TRANSPORT_CODE_POINTS]
    if beams:
        for b in BEAM_NAMES:
            spec += [("NB", f"\\NB::TOP.NB{b}.{f}_{b}") for f in ("PINJ", "TINJ")]
            spec += [("NB", f"\\NB::TOP.NB{b}.{f}") for f in ("VBEAM", "GAS")]
    for tree, expr in extra:
        spec.append((str(tree), str(expr)))
    if not spec:
        raise ValueError("node_spec: no node set selected")
    for tree, expr in spec:
        if "\t" in tree + expr or "\n" in tree + expr:
            raise ValueError(f"node_spec: tab or newline in {tree!r} / {expr!r}")
    return spec


# Server-side pull: a thin MDSplus client, run on the user's login host with the user's environment.
# argv: server spec_file out_file discharge.  Read only; writes out_file only.
_REMOTE_PULL = '''import json, sys
import numpy as np
import MDSplus
server, spec_file, out_file, discharge = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
conn = MDSplus.Connection(server)
spec = [l.rstrip("\\n").split("\\t") for l in open(spec_file) if l.strip()]
d, errs = {}, {}
for i, (tree, expr) in enumerate(spec):
    try:
        if tree:
            conn.openTree(tree, discharge)
        v = np.asarray(conn.get(expr).data())
        try:
            d["n%d_data" % i] = v.astype(float)
        except (TypeError, ValueError):
            d["n%d_data" % i] = np.array(str(v.tolist())[:4000])
        for ax in (0, 1):
            try:
                d["n%d_dim%d" % (i, ax)] = np.asarray(conn.get("dim_of(%s,%d)" % (expr, ax)).data(), float)
            except Exception:
                pass
        try:
            d["n%d_units" % i] = np.array(str(conn.get("units_of(%s)" % expr).data()))
        except Exception:
            pass
    except Exception as e:
        errs[str(i)] = "%s: %s" % (type(e).__name__, str(e)[:160])
np.savez_compressed(out_file, nodes=np.array(["%s\\t%s" % (t, x) for t, x in spec]), errors=json.dumps(errs), **d)
print("PULL_DONE", len(spec) - len(errs), "of", len(spec))
'''


class FetchError(RuntimeError):
    '''A fetch step failed; the fetch stopped there (no retry)'''


@dataclass
class FetchPlan:
    '''The commands of one pull, composed by :func:`compose_fetch` and never run by it.

    ``steps``: ``[(name, argv, timeout_s)]`` in order; ``files``: ``{local name: text}`` written before the steps
    run; ``result``: local name of the pulled file.
    '''
    discharge: int
    spec: list
    steps: list
    files: Dict[str, str]
    result: str


def compose_fetch(discharge, spec, *, host, control_path, remote_dir, server, local_dir, setup="",
                  python="python3", timeout_s=3600):
    '''Commands that pull ``spec`` for one discharge through the user's open ssh master.

    @param discharge Discharge number (the caller's)
    @param spec Node list (:func:`node_spec`)
    @param host The user's ssh destination for the login host (an alias of his ssh configuration)
    @param control_path The ``ControlPath`` of the master connection the user opened
    @param remote_dir Writable directory on the login host (created; the spec, script and result go there)
    @param server Data server as seen from the login host
    @param local_dir Local directory for the spec, script and pulled file
    @param setup Shell commands that prepare the user's Python with MDSplus on the login host
    The first step only checks the master (``ssh -O check``); every other ssh/scp step carries
    ``ControlMaster=no`` and ``BatchMode=yes`` (no new connection, no authentication prompt).
    '''
    discharge = int(discharge)
    for name, val in (("host", host), ("control_path", control_path), ("remote_dir", remote_dir),
                      ("server", server), ("local_dir", local_dir)):
        if not val or not isinstance(val, str):
            raise ValueError(f"compose_fetch: {name} is required (no default)")
    opts = ["-o", f"ControlPath={control_path}", "-o", "ControlMaster=no", "-o", "BatchMode=yes"]
    tag = f"d3d_{discharge}"
    spec_name, script_name, result = f"{tag}_spec.txt", f"{tag}_pull.py", f"{tag}_pull.npz"
    rq = shlex.quote
    rspec, rscript, rout = (f"{remote_dir}/{n}" for n in (spec_name, script_name, result))
    run = (f"{setup.strip()}\n" if setup.strip() else "") + " ".join(
        [rq(python), rq(rscript), rq(server), rq(rspec), rq(rout), str(discharge)])
    steps = [
        ("check_master", ["ssh", "-O", "check", "-o", f"ControlPath={control_path}", host], 15),
        ("mkdir", ["ssh"] + opts + [host, f"mkdir -p {rq(remote_dir)}"], 60),
        ("send_spec", ["scp", "-q"] + opts + [os.path.join(local_dir, spec_name), f"{host}:{rspec}"], 60),
        ("send_script", ["scp", "-q"] + opts + [os.path.join(local_dir, script_name), f"{host}:{rscript}"], 60),
        ("pull", ["ssh"] + opts + [host, f"bash -l -c {rq(run)}"], int(timeout_s)),
        ("fetch_result", ["scp", "-q"] + opts + [f"{host}:{rout}", os.path.join(local_dir, result)], 900),
    ]
    files = {spec_name: "".join(f"{t}\t{e}\n" for t, e in spec), script_name: _REMOTE_PULL}
    return FetchPlan(discharge=discharge, spec=list(spec), steps=steps, files=files, result=result)


def run_fetch(plan, local_dir, runner):
    '''Run a :class:`FetchPlan` with ``runner(argv, timeout=...)`` (e.g. ``subprocess.run``; required, no default).

    Writes the plan's files to ``local_dir``, runs the steps in order and stops at the first failure with
    :class:`FetchError` (no retry; a failed master check means the user has to open the connection).  Returns the
    path of the pulled file.
    '''
    if runner is None:
        raise ValueError("run_fetch needs a runner (e.g. subprocess.run): nothing runs by default")
    os.makedirs(local_dir, exist_ok=True)
    for name, text in plan.files.items():
        with open(os.path.join(local_dir, name), "w") as fh:
            fh.write(text)
    for name, argv, timeout in plan.steps:
        res = runner(argv, timeout=timeout)
        rc = getattr(res, "returncode", res)
        if rc != 0:
            hint = (" -- the ssh master connection is not open: open it yourself, then call again"
                    if name == "check_master" else "")
            raise FetchError(f"fetch step '{name}' failed (return code {rc}){hint}; stopped, nothing retried")
    path = os.path.join(local_dir, plan.result)
    if not os.path.isfile(path):
        raise FetchError(f"fetch finished but {path} is missing")
    return path


def load_pull(path):
    '''A pulled file to ``{(tree, expression): {"data", "dim0", "dim1", "units"} or {"error": text}}``'''
    with np.load(path, allow_pickle=False) as z:
        nodes = [str(n) for n in z["nodes"]]
        errs = json.loads(str(z["errors"]))
        out = {}
        for i, n in enumerate(nodes):
            tree, expr = n.split("\t", 1)
            if str(i) in errs:
                out[(tree, expr)] = {"error": errs[str(i)]}
                continue
            rec = {}
            for k in ("data", "dim0", "dim1", "units"):
                key = f"n{i}_{k}"
                if key in z.files:
                    rec[k] = z[key] if k != "units" else str(z[key])
            out[(tree, expr)] = rec
    return out


class FetchCache:
    '''Pulled files by discharge and node list under ``root``: ``d3d_<discharge>_<key>.npz`` with a JSON manifest
    (the node list and the file's SHA-256).  A changed node list is a different entry; a file whose SHA-256 does
    not match its manifest is refused.'''

    def __init__(self, root):
        self.root = str(root)

    @staticmethod
    def key(discharge, spec):
        text = json.dumps({"discharge": int(discharge), "spec": [list(map(str, s)) for s in spec]}, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def path(self, discharge, spec):
        return os.path.join(self.root, f"d3d_{int(discharge)}_{self.key(discharge, spec)}.npz")

    def has(self, discharge, spec):
        p = self.path(discharge, spec)
        return os.path.isfile(p) and os.path.isfile(p + ".json")

    def store(self, discharge, spec, pulled):
        os.makedirs(self.root, exist_ok=True)
        p = self.path(discharge, spec)
        shutil.copyfile(pulled, p)
        with open(p, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        with open(p + ".json", "w") as fh:
            json.dump({"discharge": int(discharge), "spec": [list(s) for s in spec], "sha256": sha}, fh, indent=1)
        return p

    def load(self, discharge, spec):
        p = self.path(discharge, spec)
        if not self.has(discharge, spec):
            raise FileNotFoundError(f"no cache entry for discharge {int(discharge)} with this node list ({p})")
        with open(p + ".json") as fh:
            man = json.load(fh)
        with open(p, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        if sha != man["sha256"] or man["spec"] != [list(s) for s in spec]:
            raise ValueError(f"cache entry {p} does not match its manifest: refused")
        return load_pull(p)


# =====================================================================================================
# Stored transport-code run -> current-model sources
# =====================================================================================================
def transport_code_coverage(times, t, max_gap_s):
    '''Whether a stored transport-code time grid ``times`` [s] covers the slice time ``t`` [s]: a stored time, or the
    two stored times around ``t`` at most ``max_gap_s`` apart (linear in time between them).  No extrapolation.'''
    times = np.asarray(times, float)
    if t < times[0] - 1e-9 or t > times[-1] + 1e-9:
        return {"covered": False, "reason": f"t outside {times[0]:.4f}-{times[-1]:.4f} s"}
    j = np.flatnonzero(np.abs(times - t) < 1e-9)
    if j.size:
        return {"covered": True, "bracket": [int(j[0])] * 2, "weight": 0.0, "gap_s": 0.0, "rule": "stored time"}
    k = int(np.clip(np.searchsorted(times, t) - 1, 0, times.size - 2))
    gap = float(times[k + 1] - times[k])
    if gap > max_gap_s + 1e-12:
        return {"covered": False, "reason": f"stored times {gap:.4f} s apart > {max_gap_s:g} s"}
    return {"covered": True, "bracket": [k, k + 1], "weight": float((t - times[k]) / gap), "gap_s": gap,
            "rule": f"linear between stored times at most {max_gap_s:g} s apart"}


def transport_code_values(pulled, tree, t, max_gap_s, points=("EQU:PRESS_BEAM", "NBI:ENBEAM", "HCD:J_CD",
                                                                  "EQU:RMAJ0")):
    '''Values of stored transport-code points at ``t`` [s] from a :func:`load_pull` dict, all from the bracket of
    the first point.  Time is the node's ``dim0`` [ms] and the LAST axis of its data (the stored layout); every
    point must share the first point's time grid.  Refused when the slice is not covered.'''
    def node(p):
        rec = pulled.get((tree, f"\\{tree}::TOP.{p}"))
        if rec is None or "error" in rec or "data" not in rec or "dim0" not in rec:
            raise ValueError(f"transport-code point {p} not in the pulled data")
        d, tp = np.asarray(rec["data"], float), np.asarray(rec["dim0"], float) * 1e-3
        if d.shape[-1] != tp.size:
            raise ValueError(f"transport-code point {p}: last data axis {d.shape} does not match its time axis")
        return d, tp
    d0, t0 = node(points[0])
    cov = transport_code_coverage(t0, t, max_gap_s)
    if not cov["covered"]:
        raise ValueError(f"transport-code run does not cover t = {t} s: {cov['reason']}")
    out = {}
    for p in points:
        d, tp = node(p)
        if tp.shape != t0.shape or not np.array_equal(tp, t0):
            raise ValueError(f"transport-code point {p} is on another time grid than {points[0]}")
        k0, k1 = cov["bracket"]
        w = cov["weight"]
        out[p] = (1 - w) * d[..., k0] + w * d[..., k1] if d.ndim == 2 else (1 - w) * d[k0] + w * d[k1]
    return out, cov


def transport_code_sources(pulled, tree, t, rho, max_gap_s, ec_supplied_separately=False):
    '''``kinetic.current_model.SourceProfiles`` from a stored transport-code run at ``t`` [s] on its ``rho``
    (rho_tor,norm grid of the stored profiles, the caller's statement): fast-ion pressure ``EQU:PRESS_BEAM`` and the
    driven current ``HCD:J_CD`` (total) with ``EQU:RMAJ0`` in the ``<J_phi R0/R>`` form -- or ``NBI:J_NBCD`` (beam
    only) when an electron-cyclotron current is supplied to the fit separately.  A negative pressure is refused.'''
    from OpenFUSIONToolkit.TokaMaker.kinetic.current_model import SourceProfiles
    cd = "NBI:J_NBCD" if ec_supplied_separately else "HCD:J_CD"
    v, cov = transport_code_values(pulled, tree, t, max_gap_s, points=("EQU:PRESS_BEAM", cd, "EQU:RMAJ0"))
    pb = np.asarray(v["EQU:PRESS_BEAM"], float)
    if np.any(pb < 0):
        raise ValueError("EQU:PRESS_BEAM < 0: refused (no clipping)")
    sp = SourceProfiles(np.asarray(rho, float), p_fast=pb, jtor_r0_cd=np.asarray(v[cd], float),
                        r0_cd=float(v["EQU:RMAJ0"]), coordinate="rho_tor_norm",
                        convention={"jtor_r0_cd": f"stored transport-code {cd} <J_phi R0/R>",
                                    "p_fast": "stored transport-code EQU:PRESS_BEAM (isotropic equivalent)"},
                        source=f"stored transport-code run at {t} s ({cov['rule']})")
    return sp, cov
