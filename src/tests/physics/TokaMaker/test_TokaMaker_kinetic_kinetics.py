#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""TokaMaker tests: kinetic helper package, kinetic inputs (kinetic/kinetics.py).

Quasineutrality with fast ions and the Z_eff identity, the required Z_eff source and fast-ion density,
refusals without clipping, the psi_N -> rho_tor,norm placement and the producer's midplane table; and
``assemble_kinetic_inputs(..., zeff_source="checked")``: the checked-source Z_eff rule attached to
the kinetic inputs (the rule itself is tested in ``test_TokaMaker_kinetic_zeff_sources.py``), at the
rule's calibration thresholds (test data; the rule has no default thresholds).

The checked route on synthetic profiles: a qualified impurity density is
taken; an unverified one yields to a qualified line-integrated value; an unverified impurity
density with no alternative is used, labeled and warned; neither usable is refused; an explicit
override is honored and recorded, refused for a candidate that fails unless explicitly accepted,
and refused outright for an impurity density that fails I-3; the fast-ion density reaches I-3; the
full record is stored and JSON-serializable; the named route is unchanged.
"""
import json
import warnings

import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic import kinetics as KI
from OpenFUSIONToolkit.TokaMaker.kinetic import zeff_sources as Z
from OpenFUSIONToolkit.TokaMaker.kinetic.kinetics import CHECKED, assemble_kinetic_inputs

#: The Z_eff rule's thresholds as calibrated on the profile files of one carbon-wall device (test data).
TH = Z.ZeffRuleThresholds(
    coverage_rho_inner_max=0.30, coverage_rho_outer_min=0.80, coverage_min_points=8,
    coverage_window_s=0.025, placeholder_rel_spread_min=1e-3, placeholder_rel_sigma_max=1.0,
    ni_fraction_min=0.0, ni_fraction_warn=0.1, points_abs_median_z_max=2.0,
    points_ratio_min=2.0 / 3.0, points_ratio_max=3.0 / 2.0, points_rho_max=0.9,
    points_core_rho_max=0.5, line_rel_sigma_max=0.25, copy_rel_diff_min=1e-3, flat_ptp_min=0.02,
    core_cv_advisory=0.05, tension_z_max=2.0, psi_n_test_max=0.9, psi_n_plasma_max=1.0,
    psi_n_core_median_max=0.5)

PSI = np.round(np.linspace(0.0, 1.2, 121), 10)
NE = 5e19 * (1 - 0.6 * (PSI / 1.2) ** 2) + 1e18
TE = 3e3 * (1 - 0.9 * np.minimum(PSI, 1.0) ** 2) + 50.0
TI = 0.9 * TE
RHO_MAP = (PSI[PSI <= 1.0], np.sqrt(PSI[PSI <= 1.0]))
RAW_RHO = np.linspace(0.05, 0.85, 17)


def _nimp(fc0=0.01, slope=0.5):
    return fc0 * (1.0 + slope * PSI) * NE


def _zline(level=1.9, ptp=0.5):
    return level + ptp * (PSI - 0.45)


def _points_at(rho, scale=1.0, n_imp=None, t=None, kind=Z.OTHER, label="points"):
    n_imp = _nimp() if n_imp is None else n_imp
    psi = np.interp(rho, RHO_MAP[1], RHO_MAP[0])
    v = scale * np.interp(psi, PSI, n_imp)
    return Z.MeasurementPoints(value=v, sigma=0.1 * v, rho=np.asarray(rho, float),
                               time=None if t is None else np.full(np.size(v), t), kind=kind, label=label)


def _imp_spec(points=True, raw=True, n_imp=None, sigma=True):
    n_imp = _nimp() if n_imp is None else n_imp
    d = dict(kind="impurity_density", value=n_imp, provenance="synthetic CER density")
    if sigma:
        d["sigma"] = 0.1 * n_imp
    if points:
        d["fit_points"] = _points_at(np.linspace(0.05, 0.95, 10), n_imp=n_imp, label="fit points")
    if raw:
        d["raw_points"] = [_points_at(RAW_RHO, n_imp=n_imp, t=1.0, kind=Z.AUTOMATIC, label="auto")]
    return d


def _line_spec(level=1.9, ptp=0.5, rel_sigma=0.08, sigma=True):
    zl = _zline(level, ptp)
    d = dict(kind="zeff", value=zl, provenance="synthetic line-integrated Z_eff")
    if sigma:
        d["sigma"] = rel_sigma * zl
    return d


def _build(cands, rule=None, n_fast="none", source=CHECKED, **kw):
    return assemble_kinetic_inputs(PSI, NE, TE, TI, zeff_candidates=cands, zeff_source=source, z_imp=6.0,
                                   coordinate="psi_n", psi_n_to_rho=RHO_MAP, n_fast=n_fast,
                                   zeff_rule=(dict(dict(slice_time=1.0, thresholds=TH), **(rule or {}))
                                              if source == CHECKED else None),
                                   **kw)


def _quiet(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Z.ZeffSourceWarning)
        return _build(**kw)


def test_a_qualified_impurity_density_is_taken():
    k = _quiet(cands={"cer": _imp_spec(), "vb": _line_spec(level=1.0 + 30 * 0.0125)})
    assert k.record["zeff_source"] == "cer"
    r = k.record["zeff_rule"]
    assert r["decision"]["source"] == Z.IMPURITY and r["decision"]["source_state"] == Z.QUALIFIED
    assert r["candidates"][Z.LINE]["state"] == Z.QUALIFIED
    assert r["candidate_names"] == {Z.IMPURITY: "cer", Z.LINE: "vb"}
    np.testing.assert_allclose(k.zeff, 1 + 30 * _nimp()[PSI <= 1.0] / NE[PSI <= 1.0], rtol=1e-12)
    assert "checked-source rule" in k.provenance["zeff"] and "qualified" in k.provenance["zeff"]
    assert k.record["candidates"]["cer"]["provenance"] == "synthetic CER density"
    json.dumps(r)                       # archivable


def test_two_qualified_candidates_in_tension_warn_and_take_the_impurity_density():
    with pytest.warns(Z.ZeffSourceWarning, match="disagree"):
        k = _build(cands={"cer": _imp_spec(), "vb": _line_spec(level=4.0)})
    assert k.record["zeff_source"] == "cer"
    assert k.record["zeff_rule"]["tension"]["exceeds"] is True


def test_an_unverified_impurity_density_yields_to_a_qualified_line_value():
    k = _quiet(cands={"cer": _imp_spec(points=False, raw=False), "vb": _line_spec()})
    assert k.record["zeff_source"] == "vb"
    r = k.record["zeff_rule"]
    assert r["candidates"][Z.IMPURITY]["state"] == Z.UNVERIFIED
    assert r["decision"]["rank"] == 2
    np.testing.assert_allclose(k.zeff, _zline()[PSI <= 1.0])


def test_an_unverified_impurity_density_alone_is_used_labeled_and_warned():
    with pytest.warns(Z.ZeffSourceWarning, match="UNVERIFIED"):
        k = _build(cands={"cer": _imp_spec(points=False, raw=False)})
    assert k.record["zeff_source"] == "cer"
    assert k.record["zeff_rule"]["decision"]["source_state"] == Z.UNVERIFIED
    assert any("I-1" in s for s in k.record["zeff_rule"]["not_evaluated"])
    assert "unverified" in k.provenance["zeff"]


def test_neither_usable_is_refused_with_the_record():
    flat = _line_spec(ptp=0.0)                                  # L-4 fails
    thin = _imp_spec(points=False, raw=False)
    thin["fit_points"] = _points_at(np.linspace(0.5, 0.95, 10))  # I-1 fails: innermost at 0.5
    with pytest.raises(Z.ZeffSourceRefused) as e:
        _quiet(cands={"cer": thin, "vb": flat})
    assert e.value.record["decision"]["source"] is None
    assert "I-1" in e.value.record["candidates"][Z.IMPURITY]["failed"]
    assert "L-4" in e.value.record["candidates"][Z.LINE]["failed"]


def test_an_override_onto_a_failing_candidate_is_refused_unless_accepted():
    flat = _line_spec(ptp=0.0)
    with pytest.raises(Z.ZeffSourceRefused, match="accept_failed"):
        _quiet(cands={"cer": _imp_spec(), "vb": flat}, rule=dict(override="vb"))
    with pytest.warns(Z.ZeffSourceWarning, match="override"):
        k = _build(cands={"cer": _imp_spec(), "vb": flat}, rule=dict(override="vb", accept_failed=True))
    assert k.record["zeff_source"] == "vb"
    r = k.record["zeff_rule"]
    assert r["override"]["source"] == Z.LINE and r["override"]["named_failed_tests"] == ["L-4"]
    assert r["accept_failed"] is True
    assert "explicit override" in k.provenance["zeff"]


def test_an_override_onto_a_qualified_candidate_is_honored_and_recorded():
    with pytest.warns(Z.ZeffSourceWarning, match="override"):
        k = _build(cands={"cer": _imp_spec(), "vb": _line_spec()}, rule=dict(override="vb"))
    assert k.record["zeff_source"] == "vb"
    assert k.record["zeff_rule"]["override"]["rule_source"] == Z.IMPURITY


def test_an_override_cannot_make_an_unphysical_impurity_density_usable():
    big = _nimp(fc0=0.2, slope=0.1)                  # Z n_imp > n_e: n_i < 0
    with pytest.raises(Z.ZeffSourceRefused, match="I-3"):
        _quiet(cands={"cer": _imp_spec(n_imp=big), "vb": _line_spec()},
               rule=dict(override="cer", accept_failed=True))


def test_the_fast_ion_density_reaches_i3():
    nimp = _nimp(fc0=0.08, slope=0.2)                # n_i/n_e >= 0.42 without fast ions
    k = _quiet(cands={"cer": _imp_spec(n_imp=nimp)})
    assert k.record["zeff_rule"]["candidates"][Z.IMPURITY]["tests"]["I-3"]["verdict"] == Z.PASS
    assert k.record["zeff_rule"]["candidates"][Z.IMPURITY]["tests"]["I-3"]["fast_ions"] == Z.FAST_IONS_DECLARED_NONE
    with pytest.raises(ValueError, match="n_fast is required"):
        _quiet(cands={"cer": _imp_spec(n_imp=nimp)}, n_fast=None)
    nf = 0.5 * NE                                    # with fast ions: n_i/n_e < 0 on axis
    with pytest.raises(Z.ZeffSourceRefused):
        _quiet(cands={"cer": _imp_spec(n_imp=nimp)}, n_fast=nf)
    k2 = _quiet(cands={"cer": _imp_spec(n_imp=nimp)}, n_fast=0.1 * NE)
    t = k2.record["zeff_rule"]["candidates"][Z.IMPURITY]["tests"]["I-3"]
    assert t["verdict"] == Z.PASS and t["fast_ions"] == Z.FAST_IONS_SUPPLIED


def test_a_declared_absence_of_fast_ions_is_recorded():
    k = _quiet(cands={"cer": _imp_spec()}, rule=dict(fast_ions="none"))
    assert k.record["zeff_rule"]["candidates"][Z.IMPURITY]["tests"]["I-3"]["fast_ions"] == Z.FAST_IONS_DECLARED_NONE
    with pytest.raises(ValueError, match="declares no fast ions"):
        _quiet(cands={"cer": _imp_spec()}, rule=dict(fast_ions="none"), n_fast=0.01 * NE)


def test_a_rho_table_needs_the_psi_n_of_its_points():
    rho = np.linspace(0.0, 1.0, 101)
    psi = rho ** 2
    ne = np.interp(psi, PSI, NE)
    te, ti = np.interp(psi, PSI, TE), np.interp(psi, PSI, TI)
    nimp = np.interp(psi, PSI, _nimp())
    cands = {"cer": dict(kind="impurity_density", value=nimp, sigma=0.1 * nimp,
                         fit_points=Z.MeasurementPoints(value=np.interp(np.linspace(0.05, 0.95, 10), rho, nimp),
                                                        sigma=None, rho=np.linspace(0.05, 0.95, 10)))}
    with pytest.raises(ValueError, match="psi_n"):
        assemble_kinetic_inputs(rho, ne, te, ti, zeff_candidates=cands, zeff_source=CHECKED, z_imp=6.0, n_fast="none",
                                zeff_rule=dict(thresholds=TH))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Z.ZeffSourceWarning)
        k = assemble_kinetic_inputs(rho, ne, te, ti, zeff_candidates=cands, zeff_source=CHECKED, z_imp=6.0,
                                    zeff_rule=dict(psi_n=psi, thresholds=TH), n_fast="none")
    assert k.record["zeff_source"] == "cer"
    assert k.record["zeff_rule"]["candidates"][Z.IMPURITY]["tests"]["I-1"]["verdict"] == Z.PASS


def test_the_named_route_is_unchanged_and_refuses_rule_options():
    old = assemble_kinetic_inputs(PSI, NE, TE, TI, zeff_candidates={"vb": ("zeff", _zline())}, zeff_source="vb",
                                  z_imp=6.0, coordinate="psi_n", psi_n_to_rho=RHO_MAP, n_fast="none")
    new = assemble_kinetic_inputs(PSI, NE, TE, TI, zeff_candidates={"vb": dict(kind="zeff", value=_zline())},
                                  zeff_source="vb", z_imp=6.0, coordinate="psi_n", psi_n_to_rho=RHO_MAP, n_fast="none")
    np.testing.assert_array_equal(old.zeff, new.zeff)
    np.testing.assert_array_equal(old.ni, new.ni)
    assert "zeff_rule" not in old.record
    assert "not tested" in old.provenance["zeff"]
    with pytest.raises(ValueError, match="only with zeff_source"):
        assemble_kinetic_inputs(PSI, NE, TE, TI, zeff_candidates={"vb": ("zeff", _zline())}, zeff_source="vb",
                                z_imp=6.0, coordinate="psi_n", zeff_rule=dict(override="vb"), n_fast="none")


def test_no_default_and_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="no default"):
        assemble_kinetic_inputs(PSI, NE, TE, TI, zeff_candidates={"vb": ("zeff", _zline())}, zeff_source=None,
                                z_imp=6.0, coordinate="psi_n")
    with pytest.raises(ValueError, match="reserved"):
        assemble_kinetic_inputs(PSI, NE, TE, TI, zeff_candidates={CHECKED: ("zeff", _zline())},
                                zeff_source=CHECKED, z_imp=6.0, coordinate="psi_n")
    with pytest.raises(ValueError, match="at most one candidate of each kind"):
        _quiet(cands={"a": _line_spec(), "b": _line_spec()})
    with pytest.raises(ValueError, match="unknown zeff_rule"):
        _quiet(cands={"cer": _imp_spec()}, rule=dict(bogus=1))
    with pytest.raises(ValueError, match="not one of the candidates"):
        _quiet(cands={"cer": _imp_spec()}, rule=dict(override="vb"))
    with pytest.raises(ValueError, match="may carry only"):
        _quiet(cands={"cer": dict(_imp_spec(), extra=1)})


def test_the_checked_route_needs_the_rule_thresholds():
    """The rule's thresholds have no default: the checked route without them is refused; a dict of the
    fields is accepted and gives the same record."""
    with pytest.raises(ValueError, match="thresholds"):
        _quiet(cands={"cer": _imp_spec()}, rule=dict(thresholds=None))
    a = _quiet(cands={"cer": _imp_spec()})
    b = _quiet(cands={"cer": _imp_spec()}, rule=dict(thresholds=TH.as_dict()))
    assert json.dumps(a.record, sort_keys=True) == json.dumps(b.record, sort_keys=True)
    assert a.record["zeff_rule"]["thresholds"] == TH.as_dict()


# ============================================================== quasineutrality, refusals, coordinates
def _kin_arrays():
    x = np.linspace(0.0, 1.0, 51)
    ne = 5e19 * (1.1 - 0.6 * x ** 2)
    te = 3e3 * (1.05 - x ** 2) + 50
    ti = 2.5e3 * (1.05 - x ** 2) + 60
    nfast = 4e18 * np.exp(-(x / 0.4) ** 2)
    return x, ne, te, ti, nfast


def test_quasineutrality_with_fast_ions_and_zeff_identity():
    x, ne, te, ti, nf = _kin_arrays()
    zvb = 1.8 + 0.4 * x
    k = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"vb": ("zeff", zvb)}, zeff_source="vb", z_imp=6.0,
                                   n_fast=nf, n_fast_source="provider")
    np.testing.assert_allclose(k.ni + 6.0 * k.nimp + k.n_fast, ne, rtol=1e-14)                       # (K1)
    np.testing.assert_allclose((k.ni + 36.0 * k.nimp + k.n_fast) / ne, zvb, rtol=1e-13)               # (K2)
    # the fast ions are removed from the thermal ions, not from the impurity
    k0 = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"vb": ("zeff", zvb)}, zeff_source="vb", z_imp=6.0,
                                    n_fast="none")
    np.testing.assert_allclose(k0.ni - k.ni, nf, rtol=1e-12)
    np.testing.assert_allclose(k0.nimp, k.nimp, rtol=1e-14)
    assert k0.provenance["n_fast"] == "declared by the caller: no fast ions"
    # a missing fast-ion density is refused, never taken as zero silently
    with pytest.raises(ValueError, match="n_fast is required"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"vb": ("zeff", zvb)}, zeff_source="vb", z_imp=6.0)
    with pytest.raises(ValueError, match="declared: no fast ions"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"vb": ("zeff", zvb)}, zeff_source="vb", z_imp=6.0,
                                   n_fast="zero")
    p = k.thermal_pressure_at(rho=x)
    np.testing.assert_allclose(p, KI.EC * (ne * te + (k.ni + k.nimp) * ti), rtol=1e-12)
    assert k.coordinate == "rho_tor_norm" and "quasineutrality" in k.provenance["ni"]


def test_zeff_source_is_required_and_disagreement_recorded():
    x, ne, te, ti, nf = _kin_arrays()
    zvb = np.full_like(x, 2.4)
    nc = ne * 0.6 / 30.0                                   # CER-like carbon density -> Z_eff = 1.6
    cands = {"vb": ("zeff", zvb), "cer": ("impurity_density", nc)}
    with pytest.raises(ValueError, match="no\\s+default"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates=cands, zeff_source=None, z_imp=6.0)
    k = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates=cands, zeff_source="cer", z_imp=6.0, n_fast="none")
    np.testing.assert_allclose(k.zeff, 1.6, rtol=1e-13)
    d = k.record["pairwise_disagreement"]["cer vs vb"]
    assert d["median_rel"] == pytest.approx(1.6 / 2.4 - 1.0, rel=1e-12)


def test_kinetic_refusals_without_clipping():
    x, ne, te, ti, nf = _kin_arrays()
    with pytest.raises(ValueError, match="< 1"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"z": ("zeff", np.full_like(x, 0.9))},
                                   zeff_source="z", z_imp=6.0, n_fast="none")
    with pytest.raises(ValueError, match="<= 0"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"z": ("zeff", np.full_like(x, 1.5))},
                                   zeff_source="z", z_imp=6.0, n_fast=0.99 * ne)
    with pytest.raises(ValueError, match="cover"):
        KI.assemble_kinetic_inputs(x[:40], ne[:40], te[:40], ti[:40],
                                   zeff_candidates={"z": ("zeff", np.full(40, 1.5))}, zeff_source="z", z_imp=6.0,
                                   n_fast="none")


def test_psi_n_inputs_mapped_through_the_producers_table_or_kept():
    x, ne, te, ti, nf = _kin_arrays()                      # read as psi_N of a foreign equilibrium
    tab = (np.linspace(0, 1, 101), np.sqrt(np.linspace(0, 1, 101)) ** 1.0)       # rho = sqrt(psi_N)
    z = {"z": ("zeff", np.full_like(x, 1.5))}
    k = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates=z, zeff_source="z", z_imp=6.0, coordinate="psi_n",
                                   psi_n_to_rho=tab, mapping_source="the profile fit's equilibrium", n_fast="none")
    assert k.coordinate == "rho_tor_norm" and "own table" in k.record["mapping"]
    np.testing.assert_allclose(k.profiles_at(rho=np.sqrt(0.36))["ne"], np.interp(0.36, x, ne), rtol=2e-3)
    k2 = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates=z, zeff_source="z", z_imp=6.0, coordinate="psi_n",
                                    n_fast="none")
    assert k2.coordinate == "psi_n" and "ours" in k2.record["mapping"]


# ============================================================== the producer's midplane table (a record only)
def _inputs():
    x = np.linspace(0.0, 1.0, 51)
    ne = 4e19 * (1.0 - 0.8 * x ** 2)
    te = 2e3 * (1.0 - 0.9 * x ** 2)
    return x, ne, te, 0.8 * te, dict(zeff_candidates={"vb": ("zeff", np.full_like(x, 1.8))}, zeff_source="vb",
                                     z_imp=6.0, n_fast="none")


def test_midplane_table_is_kept_and_stated_and_changes_no_profile():
    x, ne, te, ti, kw = _inputs()
    mp = np.linspace(0.0, 1.02, 30)
    mr = 1.7 + 0.6 * np.sqrt(mp)
    a = KI.assemble_kinetic_inputs(x, ne, te, ti, **kw)
    b = KI.assemble_kinetic_inputs(x, ne, te, ti, midplane_radius=(mp, mr, "producer"), **kw)
    assert a.midplane_radius is None and a.record["midplane_radius"] == "not supplied"
    assert b.midplane_radius["source"] == "producer"
    assert np.array_equal(b.midplane_radius["psi_n"], mp) and np.array_equal(b.midplane_radius["r_m"], mr)
    assert b.record["midplane_radius"].startswith("supplied: producer, 30 points")
    for k in ("x", "ne", "te", "ti", "zeff", "ni", "nimp", "n_fast"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert np.array_equal(a.thermal_pressure_at(rho=x), b.thermal_pressure_at(rho=x))
    c = KI.assemble_kinetic_inputs(x, ne, te, ti, midplane_radius=(mp, mr), mapping_source="eq", **kw)
    assert c.midplane_radius["source"] == "eq"


@pytest.mark.parametrize("bad", ["psi_flat", "r_down", "nan", "length"])
def test_midplane_table_refused_unless_strictly_increasing_and_finite(bad):
    x, ne, te, ti, kw = _inputs()
    mp = np.linspace(0.0, 1.0, 10)
    mr = 1.7 + 0.5 * mp
    if bad == "psi_flat":
        mp[3] = mp[2]
    elif bad == "r_down":
        mr[-1] = mr[-3]
    elif bad == "nan":
        mr[4] = np.nan
    else:
        mr = mr[:-1]
    with pytest.raises(ValueError, match="midplane_radius"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, midplane_radius=(mp, mr), **kw)
