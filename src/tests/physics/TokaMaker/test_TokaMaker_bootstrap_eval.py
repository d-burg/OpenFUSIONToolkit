#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
# TokaMaker tests: eval_bootstrap (the internal solver's bootstrap evaluation on a given equilibrium) against the
# Redl formula of bootstrap.py, its options, and the evaluation inside solve_bootstrap, on ITER
import os
import pytest
import numpy as np
from tokamaker_testing import mp_run, setup_ITER_bootstrap_internal
from OpenFUSIONToolkit._interface import oftpy_dump_cov

## Elementary charge of bootstrap.py and the Fortran evaluation (OpenFUSIONToolkit.util.eC is rounded)
EC = 1.602176634e-19
## eval_bootstrap profiles compared with the Python formula
EVAL_KEYS = ('jdotB_bs_raw', 'jphi_bs_raw', 'jphi_pd_raw', 'f_t', 'nu_e_star', 'nu_i_star')


def _carbon_case(ne, Zeff):
    '''Main-ion and carbon (Z = 6) densities with quasineutrality and the given Z_eff'''
    return ne*(6.0 - Zeff)/5.0, ne*(Zeff - 1.0)/30.0


def _python_bootstrap(mygs, x, Te, Ti, ne, ni, Zeff, nimp=None, ln_lambda='NRL', solver_gradients=True,
                      coefficient_model='redl'):
    '''bootstrap.redl_bootstrap with the internal solver's settings at the interior nodes x[1:-1], with the
    geometry of the current equilibrium sampled as solve_with_bootstrap(use_python_solve=True) samples it.
    Gradients are taken as the solver takes them (on psi in its node order, 0 at the LCFS, without the LCFS
    node) or as solve_with_bootstrap does (on all of x, divided by the flux range).'''
    from OpenFUSIONToolkit.TokaMaker.bootstrap import redl_bootstrap, calculate_ln_lambda
    xi = x[1:-1]
    _, f, _, _, pp = mygs.get_profiles(psi=xi)
    _, fc, _, b_avgs, eps = mygs.sauter_fc(psi=xi, return_eps=True)
    _, q, ravgs, _, _, _ = mygs.get_q(psi=xi)
    fsa = mygs.get_fsa(psi=xi)
    R_geo = 0.5*(fsa['R_min'] + fsa['R_max'])
    psi_a, psi_0 = mygs.psi_bounds
    nit = ni if nimp is None else ni + nimp
    if solver_gradients:
        psi = psi_a + (1.0 - x[::-1])[1:]*(psi_0 - psi_a)

        def grad(y):
            return -np.gradient(y[::-1][1:], psi, edge_order=2)[::-1][1:]
    else:
        def grad(y):
            return (np.gradient(y, x, edge_order=2)/(psi_0 - psi_a))[1:-1]
    dTe, dTi, dne, dnit = grad(Te), grad(Ti), grad(ne), grad(nit)
    Te, Ti, ne, ni, nit, Zeff = Te[1:-1], Ti[1:-1], ne[1:-1], ni[1:-1], nit[1:-1], Zeff[1:-1]
    ln_le, ln_lii = calculate_ln_lambda(Te, Ti, ne, ni, Zeff, electron_lnLambda_model=ln_lambda,
                                        ion_lnLambda_model='Zavg')
    jdotB, coeffs = redl_bootstrap(psi_N=xi, Te=Te, Ti=Ti, ne=ne, ni=ni, pe=EC*ne*Te, pi=EC*nit*Ti, Zeff=Zeff,
                                   R=R_geo, q=np.abs(q), eps=eps, fT=1.0 - fc, I_psi=f,
                                   dT_e_dpsi=dTe, dT_i_dpsi=dTi, dn_e_dpsi=dne, dn_i_dpsi=dnit,
                                   dp_dpsi=(ne*dTe + Te*dne + nit*dTi + Ti*dnit)*EC,
                                   ln_lambda_e=ln_le, ln_lambda_ii=ln_lii,
                                   use_legacy_L34=False, use_sign_q=False, ion_collisionality_model='Zavg_Zeff',
                                   formula_form='jboot1', coefficient_model=coefficient_model)
    inv_R, B2 = ravgs['<1/R>'], b_avgs[1]
    return {'jdotB_bs_raw': jdotB, 'jphi_bs_raw': jdotB*f*inv_R/B2, 'jphi_pd_raw': pp*(ravgs['<R>'] - f**2*inv_R/B2),
            'f_t': 1.0 - fc, 'nu_e_star': coeffs['nu_e_star'], 'nu_i_star': coeffs['nu_i_star'],
            'ln_lambda_min': float(min(np.min(ln_le), np.min(ln_lii)))}


def _rel(a, b):
    '''Largest difference relative to the largest magnitude of b'''
    return float(np.max(np.abs(a - b))/np.max(np.abs(b)))


def _rejects(fn, **kwargs):
    '''True if fn(**kwargs) raises ValueError (any other outcome is False)'''
    try:
        fn(**kwargs)
    except ValueError:
        return True
    except Exception:
        return False
    return False


def run_eval_parity(mesh_resolution, fe_order, mp_q):
    setup = setup_ITER_bootstrap_internal(mesh_resolution, fe_order)
    if setup is None:
        mp_q.put(None)
        return
    mygs, x, ne, Te = setup['mygs'], setup['psi_sample'], setup['ne'], setup['Te']
    n = len(x)
    Zeff = np.full(n, setup['Zeff_val'])
    Te_k = Te/1e3
    Te_eV = Te_k*1e3  # as the solver converts keV
    ni_c, nimp_c = _carbon_case(ne, setup['Zeff_val'])
    results = {}
    try:
        # n_i = n_e with the NRL logarithms; n_i < n_e (Z = n_e/n_i in ln Lambda_ii) with the Sauter 1999 ones;
        # and with the carbon density in the ion pressure
        cases = {'nrl': (ne, None, 1, 'NRL'), 'sauter': (ni_c, None, 2, 'sauter'), 'imp': (ni_c, nimp_c, 1, 'NRL')}
        evals = {}
        for name, (ni, nimp, ln_model, ln_py) in cases.items():
            F = mygs.eval_bootstrap(x, Te_k, Te_k, ne, ni, setup['Zeff_val'], nimp=nimp, ln_lambda_model=ln_model)
            evals[name] = F
            for solver_gradients in (True, False):
                P = _python_bootstrap(mygs, x, Te_eV, Te_eV, ne, ni, Zeff, nimp=nimp, ln_lambda=ln_py,
                                      solver_gradients=solver_gradients)
                results[f'{name}/ln_lambda_min'] = P['ln_lambda_min']
                for key in EVAL_KEYS:
                    Fi = F[key][1:-1]
                    if solver_gradients:
                        results[f'{name}/{key}'] = _rel(Fi, P[key])
                    else:  # The solver's gradients leave out the LCFS node: compare all but the outermost node
                        results[f'{name}/solve_with_bootstrap/{key}'] = _rel(Fi[:-1], P[key][:-1])
                        results[f'{name}/outermost/{key}'] = _rel(Fi[-1:], P[key][-1:])
            results[f'{name}/lcfs_nan'] = bool(np.all(np.isnan([F[key][-1] for key in EVAL_KEYS[3:]])))
        # Impurity density of imp_pressure (one impurity with quasineutrality and Z_eff) = the carbon density
        F = mygs.eval_bootstrap(x, Te_k, Te_k, ne, ni_c, setup['Zeff_val'], imp_pressure=True)
        results['imp_derived'] = _rel(F['jdotB_bs_raw'], evals['imp']['jdotB_bs_raw'])
        # Size of each option's effect on <j.B>
        F = mygs.eval_bootstrap(x, Te_k, Te_k, ne, ni_c, setup['Zeff_val'])
        results['imp_effect'] = _rel(evals['imp']['jdotB_bs_raw'], F['jdotB_bs_raw'])
        results['sauter_effect'] = _rel(evals['sauter']['jdotB_bs_raw'], F['jdotB_bs_raw'])
        # Invalid inputs are rejected before the Fortran call
        for kwargs in ({'psi': x[::-1]}, {'psi': x, 'ln_lambda_model': 3}, {'psi': x[:-1]}):
            args = dict(psi=x, te=Te_k, ti=Te_k, ne=ne, ni=ne, Zeff=1.5)
            args.update(kwargs)
            try:
                mygs.eval_bootstrap(**args)
            except ValueError:
                continue
            raise AssertionError(f"eval_bootstrap accepted invalid input {list(kwargs)}")
        # Each further rejection, with one input changed (node 100 is inside the LCFS, the last node is on it)
        base = dict(psi=x, te=Te_k, ti=Te_k, ne=ne, ni=ne, Zeff=1.5)

        def at(y, i, v):
            y = np.array(y, dtype=float)
            y[i] = v
            return y
        x4 = np.array([0.0, 0.3, 0.6, 1.0])
        invalid = {'psi_duplicate': {'psi': at(x, 100, x[99])}, 'psi_nan': {'psi': at(x, 100, np.nan)},
                   'te_nan': {'te': at(Te_k, 100, np.nan)}, 'te_inf_lcfs': {'te': at(Te_k, -1, np.inf)},
                   'te_zero': {'te': at(Te_k, 100, 0.0)}, 'ti_negative': {'ti': at(Te_k, 100, -1.0E-3)},
                   'ne_zero': {'ne': at(ne, 100, 0.0)}, 'ni_zero': {'ni': at(ne, 100, 0.0)}, 'Zeff_below_1': {'Zeff': 0.5},
                   'nimp_negative': {'nimp': at(np.zeros(n), 100, -1.0E17)}, 'te_scalar': {'te': 5.0},
                   'nimp_scalar': {'nimp': 0.0}, 'nimp_imp_pressure_false': {'nimp': np.zeros(n), 'imp_pressure': False},
                   'ln_lambda_float': {'ln_lambda_model': 2.7}, 'imp_pressure_string': {'imp_pressure': 'False'},
                   'too_few_interior': {'psi': x4, 'te': np.full(4, 5.0), 'ti': np.full(4, 5.0), 'ne': np.full(4, 1.0E20),
                                        'ni': np.full(4, 1.0E20)}}
        for name, kwargs in invalid.items():
            results[f'rejects/{name}'] = _rejects(mygs.eval_bootstrap, **dict(base, **kwargs))
        results['rejects/set_boot_ops_ln_lambda_float'] = _rejects(mygs.set_boot_ops, ln_lambda_model=2.7)
        results['rejects/set_boot_ops_imp_pressure_string'] = _rejects(mygs.set_boot_ops, imp_pressure='False')
        # A point on the LCFS is not used: T_e = 0 there is accepted and changes nothing
        F = mygs.eval_bootstrap(**base)
        F_lcfs = mygs.eval_bootstrap(**dict(base, te=at(Te_k, -1, 0.0)))
        results['lcfs_te_zero_unused'] = all(F[key].tobytes() == F_lcfs[key].tobytes() for key in EVAL_KEYS)
        # psi_convention = 1 (0 at the LCFS) gives the mirror of psi_convention = 0, bitwise
        F = mygs.eval_bootstrap(x, Te_k, Te_k, ne, ni_c, setup['Zeff_val'], nimp=nimp_c, ln_lambda_model=2)
        mygs._tMaker_equil.psi_convention = 1
        try:
            F_mirror = mygs.eval_bootstrap(1.0 - x[::-1], Te_k[::-1], Te_k[::-1], ne[::-1], ni_c[::-1],
                                           setup['Zeff_val'], nimp=nimp_c[::-1], ln_lambda_model=2)
        finally:
            mygs._tMaker_equil.psi_convention = 0
        results['mirror'] = all(F[key].tobytes() == F_mirror[key][::-1].tobytes() for key in EVAL_KEYS)
        # Without an equilibrium (before init_psi) the evaluation reports it
        from OpenFUSIONToolkit.TokaMaker._core import TokaMaker_equilibrium
        fresh = TokaMaker_equilibrium(mygs)
        try:
            fresh.eval_bootstrap(**base)
            results['no_equilibrium'] = False
        except ValueError:
            results['no_equilibrium'] = False
        except Exception as e:
            results['no_equilibrium'] = 'No equilibrium' in str(e)
        del fresh
    except Exception:
        import traceback
        traceback.print_exc()
        mp_q.put(None)
        return
    mp_q.put([results])
    oftpy_dump_cov()


@pytest.mark.parametrize("order", (2,))
def test_ITER_bootstrap_eval_parity(order):
    '''eval_bootstrap on the ITER equilibrium with H-mode profiles equals bootstrap.redl_bootstrap with the internal
    solver's settings (jboot1, L34 = L31, Zavg_Zeff ion charge, |q|, NRL or Sauter 1999 Coulomb logarithms) and
    the same geometry and gradients, node by node: <j.B>, both parts of the toroidal current, f_t and both
    collisionalities; with a carbon impurity in the ion pressure; and with the impurity density imp_pressure derives.

    Differences are relative to the largest magnitude; the bounds are about ten times the largest difference
    measured. With the gradients taken as the solver takes them, all interior nodes agree to round-off (1E-15
    for the collisionalities; f_t and the pressure-driven part are identical). The two currents agree to 4E-16
    (GCC, x86) up to 3.7E-14 (Intel compilers) and 6.7E-14 (GCC, Apple silicon): Fortran and numpy use different
    math libraries, and some compilers fuse multiply and add, which rounds once instead of twice. Their bound,
    1E-12, covers that spread. With the
    gradients of solve_with_bootstrap (on the normalized flux, LCFS node included) the currents differ by 4E-14,
    from the differently rounded abscissae, and the outermost interior node differs by the gradient stencil
    (one-sided in the solver; 1.0E-3 of its value), which is checked to exceed round-off. The derived impurity
    density equals the carbon density to round-off, which the gradient raises to 1.5E-14 in <j.B>.
    Invalid input (non-finite, non-physical on the nodes used, too few nodes, scalars other than Z_eff,
    conflicting or unconverted options, duplicate points) raises ValueError, T_e = 0 on the unused LCFS point
    changes nothing, psi_convention = 1 gives the mirror of psi_convention = 0 bitwise, and without an
    equilibrium the evaluation reports it.

    References: O. Sauter et al., Phys. Plasmas 6, 2834 (1999); A. Redl et al., Phys. Plasmas 28, 022502 (2021).
    '''
    results = mp_run(run_eval_parity, (1.0, order), timeout=300)
    assert results is not None
    res = results[0]
    bounds = {'jdotB_bs_raw': 1.0E-12, 'jphi_bs_raw': 1.0E-12, 'jphi_pd_raw': 1.0E-14, 'f_t': 1.0E-14,
              'nu_e_star': 1.0E-14, 'nu_i_star': 1.0E-14}
    for name in ('nrl', 'sauter', 'imp'):
        assert res[f'{name}/lcfs_nan']
        assert res[f'{name}/ln_lambda_min'] > 10.0  # The floor of calculate_ln_lambda('sauter') is inactive
        for key, bound in bounds.items():
            assert res[f'{name}/{key}'] < bound, f'{name}/{key}'
            bound_swb = 4.0E-13 if key in ('jdotB_bs_raw', 'jphi_bs_raw') else bound
            assert res[f'{name}/solve_with_bootstrap/{key}'] < bound_swb, f'{name}/solve_with_bootstrap/{key}'
        for key in ('jdotB_bs_raw', 'jphi_bs_raw'):
            assert res[f'{name}/outermost/{key}'] > 1.0E-4, f'{name}/outermost/{key}'
    assert res['imp_derived'] < 1.5E-13
    # Each option changes <j.B> (measured: 8.0E-3 and 1.7E-3 of the peak)
    assert res['imp_effect'] > 4.0E-3
    assert res['sauter_effect'] > 9.0E-4
    # Each invalid input raises ValueError; a point on the LCFS is not checked beyond being finite
    rejects = [key for key in res if key.startswith('rejects/')]
    assert len(rejects) == 18
    for key in rejects:
        assert res[key], key
    assert res['lcfs_te_zero_unused']
    assert res['mirror']
    assert res['no_equilibrium']


def run_eval_insolve(mesh_resolution, fe_order, mp_q):
    setup = setup_ITER_bootstrap_internal(mesh_resolution, fe_order)
    if setup is None:
        mp_q.put(None)
        return
    mygs, x, ne, Te = setup['mygs'], setup['psi_sample'], setup['ne'], setup['Te']
    Zeff = setup['Zeff_val']
    ni, nimp = _carbon_case(ne, Zeff)
    kwargs = dict(ffp_prof={'type': 'jphi-split-bootstrap', 'x': x, 'y': setup['inductive_jphi']},
                  te_prof={'type': 'linterp', 'x': x, 'y': Te/1e3}, ne_prof={'type': 'linterp', 'x': x, 'y': ne},
                  ti_prof={'type': 'linterp', 'x': x, 'y': Te/1e3}, ni_prof={'type': 'linterp', 'x': x, 'y': ni},
                  Zeff=Zeff, Ip_target=setup['Ip_target'])
    results = {}
    try:
        bp = mygs.solve_bootstrap(**kwargs)
        # eval_bootstrap must leave the equilibrium and the bootstrap state as they are
        state = (mygs.get_psi(False), mygs.get_boot_profs(), dict(mygs._tMaker_equil._boot_ops), mygs.get_globals(),
                 mygs.get_profiles(npsi=50))
        F = mygs.eval_bootstrap(x, Te/1e3, Te/1e3, ne, ni, Zeff)
        F_opts = mygs.eval_bootstrap(x, Te/1e3, Te/1e3, ne, ni, Zeff, ln_lambda_model=2, imp_pressure=True)
        state_after = (mygs.get_psi(False), mygs.get_boot_profs(), dict(mygs._tMaker_equil._boot_ops),
                       mygs.get_globals(), mygs.get_profiles(npsi=50))
        results['state_unchanged'] = bool(
            np.array_equal(state[0], state_after[0])
            and all(np.array_equal(state[1][k], state_after[1][k]) for k in state[1])
            and state[2] == state_after[2]
            and all(np.array_equal(u, v) for u, v in zip(state[3], state_after[3]))
            and all(np.array_equal(u, v) for u, v in zip(state[4], state_after[4])))
        # The in-solve values come from the last evaluation, one or more iterations before the converged
        # equilibrium (freeze at djbs_tol): eval_bootstrap on the converged equilibrium agrees to that level
        results['insolve_default'] = _rel(F['jdotB_bs_raw'], bp['jdotB_bs_raw'])
        results['insolve_default_bs'] = _rel(F['jphi_bs_raw'], bp['jphi_bs_raw'])
        results['insolve_default_vs_opts'] = _rel(F_opts['jdotB_bs_raw'], bp['jdotB_bs_raw'])
        # Both options in the solve; eval_bootstrap takes them from the solve's options by default
        bp_default = bp
        bp = mygs.solve_bootstrap(**kwargs, ln_lambda_model=2, imp_pressure=True)
        results['insolve_opts_vs_default'] = _rel(bp['jdotB_bs_raw'], bp_default['jdotB_bs_raw'])
        F = mygs.eval_bootstrap(x, Te/1e3, Te/1e3, ne, ni, Zeff)
        results['insolve_opts'] = _rel(F['jdotB_bs_raw'], bp['jdotB_bs_raw'])
        results['insolve_opts_pd'] = _rel(F['jphi_pd_raw'], bp['jphi_pd_raw'])
        results['insolve_opts_bs'] = _rel(F['jphi_bs_raw'], bp['jphi_bs_raw'])
        # The options survive save and load
        mygs._tMaker_equil.save_TokaMaker('ITER_boot_eval_test.h5')
        mygs.boot_reset()
        mygs.replace_eq(source_file='ITER_boot_eval_test.h5')
        ops = mygs._tMaker_equil._boot_ops
        results['ops_loaded'] = bool(ops['ln_lambda_model'] == 2 and ops['imp_pressure'])
        # An invalid ln_lambda_model in the file loads the default (with a warning)
        import h5py
        paths = []
        with h5py.File('ITER_boot_eval_test.h5', 'r+') as fid:
            fid.visit(lambda name: paths.append(name) if name.endswith('BOOT_OPS/LN_LAMBDA_MODEL') else None)
            for path in paths:
                fid[path][...] = 7
        mygs.replace_eq(source_file='ITER_boot_eval_test.h5')
        os.remove('ITER_boot_eval_test.h5')
        ops = mygs._tMaker_equil._boot_ops
        results['ln_lambda_loaded_invalid'] = bool(len(paths) == 1 and ops['ln_lambda_model'] == 1
                                                   and ops['imp_pressure'])
    except Exception:
        import traceback
        traceback.print_exc()
        mp_q.put(None)
        return
    mp_q.put([results])
    oftpy_dump_cov()


@pytest.mark.parametrize("order", (2,))
def test_ITER_bootstrap_eval_insolve(order):
    '''eval_bootstrap is the evaluation solve_bootstrap makes: after a self-consistent ITER bootstrap solve with a
    carbon impurity, eval_bootstrap on the converged equilibrium reproduces the solver's last <j.B> to the level of
    its freeze tolerance, with the default options and with both options set in the solve, and differs from it by
    the options' effect otherwise. eval_bootstrap changes neither the equilibrium nor the bootstrap state, and the
    options are saved and loaded with the equilibrium; an invalid ln_lambda_model in the file loads the default.

    Differences are relative to the largest magnitude. The bound is about three times the largest difference
    measured (5.8E-5, 9.0E-5 and 1.1E-4 of the peak); each option alone changes <j.B> by more (8.0E-3 and
    1.7E-3 in test_ITER_bootstrap_eval_parity). The stored field-aligned part jphi_bs_raw agrees with
    eval_bootstrap's to the same level (measured 7.4E-5 and 1.15E-4 of the peak; bound 3.5E-4).
    '''
    results = mp_run(run_eval_insolve, (1.0, order), timeout=300)
    assert results is not None
    res = results[0]
    assert res['state_unchanged']
    assert res['insolve_default'] < 3.0E-4
    assert res['insolve_opts'] < 3.0E-4
    assert res['insolve_opts_pd'] < 3.0E-4
    assert res['insolve_default_bs'] < 3.5E-4  # measured 7.4E-5
    assert res['insolve_opts_bs'] < 3.5E-4  # measured 1.15E-4
    assert res['insolve_default_vs_opts'] > 3.0E-3  # measured 7.5E-3
    assert res['insolve_opts_vs_default'] > 3.0E-3  # the options reach the solve; measured 9.1E-3
    assert res['ops_loaded']
    assert res['ln_lambda_loaded_invalid']


## Smallest accepted change of <j.B> (relative to its peak) between bs_model 1 and 2 on the ITER case: a quarter of
## the smallest measured (7.8E-2); an option that does not reach the evaluation changes nothing
EFFECT_BOUND = 2.0E-2


## Sauter et al. (1999) coefficients with the 2002 erratum from the public OMFIT function
## omfit_classes.utils_fusion.sauter_bootstrap (version 'jboot1'), run once offline: L31, L32 and L34 recovered
## from its returned terms with dT/dpsi = T and dn/dpsi = 0 (term1 = L31 p/p_e, term2 = L32,
## term3 = L34 alpha (1-R_pe)/R_pe), alpha returned directly, at the collisionalities its nclass_conductivity
## returns for the same inputs; 13 significant digits. Rows: f_t, nu_e*, nu_i*, Z_eff, L31, L32, L34, alpha.
SAUTER_REFERENCE = (
    (0.3, 0.0011580234468042622, 0.010164169864218536, 1.0,
     4.1782939093464e-01, -2.0722863870829e-01, 4.1780020973256e-01, -8.2852618809245e-01),
    (0.7, 0.007714358034249874, 0.0666576134705423, 1.5,
     7.3743481607982e-01, -1.2154049464633e-01, 7.3699733844869e-01, -3.8354702412615e-01),
    (0.9, 0.0023160468936085244, 0.02032833972843707, 2.0,
     8.9467711646289e-01, -4.7773163852354e-02, 8.9451176776512e-01, -1.6211822976899e-01),
    (0.5, 0.15119538018609732, 1.2590332852046313, 1.5,
     4.6846241534750e-01, -4.2670878303651e-02, 4.6500222915857e-01, -3.0111405082665e-01),
    (0.5, 1.956082319240309, 15.540250849985801, 1.0,
     2.7268860465565e-01, 7.5956063278709e-02, 2.5297445426743e-01, 7.6864136088056e-01),
    (0.3, 5.8682469577209275, 46.620752549957395, 3.0,
     9.7792491806567e-02, 4.0409111273131e-02, 9.4447884949263e-02, 5.2410098109149e-01),
    (0.1, 17.158296730857277, 132.45881210353147, 2.0,
     1.6287672619070e-02, 6.3614274891763e-03, 1.5908757580498e-02, 2.6682814763846e-01),
    (0.7, 149.99695774597808, 1120.0591517731846, 4.0,
     4.9166483227364e-02, 1.4043300140095e-02, 3.6149613221117e-02, 2.0999148304661e+00),
    (0.5, 8.579148365428638, 66.22940605176575, 1.0,
     1.3673810540140e-01, 6.0980102427029e-02, 1.1664292854801e-01, 1.9283796707512e+00),
    (0.7, 2.9341234788604638, 23.31037627497869, 1.5,
     3.3561175566860e-01, 1.7397516232900e-01, 3.0388854734697e-01, 1.9058150783296e+00),
    (0.9, 5.8682469577209275, 46.620752549957395, 3.0,
     3.3444447576622e-01, 2.2969006394289e-01, 2.9840083731019e-01, 2.0881361719473e+00),
    (0.98, 7.824329276961236, 62.16100339994321, 4.0,
     3.2651182226506e-01, 2.2907791081579e-01, 2.9034129139124e-01, 2.0959332455707e+00),
    (0.02, 0.15119538018609732, 1.2590332852046313, 1.5,
     2.1558283861819e-02, -8.8084009162762e-03, 2.1550779281985e-02, -5.5818851084792e-01),
    (0.3, 1415.7181083774653, 9704.651951311202, 2.0,
     1.5418174145317e-03, 5.0637660390035e-04, 1.2999793160442e-03, 2.0998376805786e+00),
)


def test_sauter_coefficients():
    '''bootstrap.sauter_coefficients (Sauter et al., Phys. Plasmas 6, 2834 (1999), eqs. 14-17, with the Erratum,
    Phys. Plasmas 9, 5140 (2002)) against the public reference values of SAUTER_REFERENCE (banana, plateau and
    collisional regimes, Z_eff 1 to 4), the erratum sign, limits that can be checked by hand, and the selection
    in redl_bootstrap.

    The coefficients are of order 1e-3 to 2; the reference values are rounded to 13 significant digits (at most
    1E-13 absolute) and evaluated in a different order (a few 1E-16 measured), so the absolute bound 1E-12 is
    ten times the rounding. The pre-erratum sign changes alpha by 2(0.315 w)/(1 + 0.15 w), w = nu_i*^2 f_t^6,
    which exceeds 0.1 on the eight rows with w > 1; the hand limits are exact sums of a few O(1) terms (1E-14).
    '''
    from OpenFUSIONToolkit.TokaMaker.bootstrap import sauter_coefficients, redl_bootstrap
    ref = np.array(SAUTER_REFERENCE)
    ft, nue, nui, Z = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
    c = sauter_coefficients(ft, nue, nui, Z)
    for k, key in enumerate(('L31', 'L32', 'L34', 'alpha')):
        assert np.max(np.abs(c[key] - ref[:, 4 + k])) < 1.0E-12, key
    assert np.array_equal(c['L32'], c['F32_ee'] + c['F32_ei'])
    # The erratum sign: the 1999 sign of the 0.315 term disagrees with the reference where it matters
    w = nui**2*ft**6
    alpha_1999 = c['alpha'] - 2.0*0.315*w/(1.0 + 0.15*w)
    assert np.count_nonzero(w > 1.0) == 8
    assert np.min(np.abs(alpha_1999 - ref[:, 7])[w > 1.0]) > 0.1
    # Limits: no trapped particles; f_t = 1 without collisions (F31(1) = 1, F32_ee(1) = -F32_ei(1), alpha0 = 0);
    # the banana-regime alpha0 = -1.17; and nu* -> infinity, where alpha -> 0.315/0.15 = 2.1 (-2.1 before the erratum)
    Zs = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    zero, one = np.zeros(5), np.ones(5)
    c = sauter_coefficients(zero, 0.3*one, 0.3*one, Zs)
    assert all(np.all(c[key] == 0.0) for key in ('L31', 'L32', 'L34'))
    c = sauter_coefficients(one, zero, zero, Zs)
    assert np.max(np.abs(c['L31'] - 1.0)) < 1.0E-14 and np.max(np.abs(c['L34'] - 1.0)) < 1.0E-14
    assert np.max(np.abs(c['L32'])) < 1.0E-14 and np.max(np.abs(c['alpha'])) < 1.0E-14
    c = sauter_coefficients(zero, zero, zero, Zs)
    assert np.max(np.abs(c['alpha'] + 1.17)) < 1.0E-14
    c = sauter_coefficients(0.5*one, 1.0E12*one, 1.0E12*one, Zs)
    assert np.max(np.abs(c['alpha'] - 2.1)) < 1.0E-12
    assert max(np.max(np.abs(c[key])) for key in ('L31', 'L32', 'L34')) < 1.0E-9
    # redl_bootstrap: 'sauter' uses these coefficients in the shared jboot1 assembly; invalid selections are rejected
    n = 5
    args = dict(psi_N=np.linspace(0.1, 0.9, n), Te=np.full(n, 3.0E3), Ti=np.full(n, 2.5E3), ne=np.full(n, 5.0E19),
                ni=np.full(n, 4.0E19), pe=np.full(n, 2.4E4), pi=np.full(n, 1.6E4), Zeff=Zs, R=np.full(n, 6.2),
                q=np.full(n, 2.0), eps=np.linspace(0.05, 0.3, n), fT=np.linspace(0.2, 0.6, n), I_psi=np.full(n, 33.0),
                dT_e_dpsi=np.full(n, -2.0E3), dT_i_dpsi=np.full(n, -1.5E3), dn_e_dpsi=np.full(n, -1.0E19),
                dn_i_dpsi=np.full(n, -8.0E18), formula_form='jboot1')
    jb_s, cs = redl_bootstrap(**args, coefficient_model='sauter')
    jb_r, cr = redl_bootstrap(**args)
    c = sauter_coefficients(args['fT'], cs['nu_e_star'], cs['nu_i_star'], Zs)
    assert all(np.array_equal(cs[key], c[key]) for key in ('L31', 'L32', 'L34', 'alpha'))
    assert np.array_equal(cs['nu_e_star'], cr['nu_e_star']) and np.array_equal(cs['nu_i_star'], cr['nu_i_star'])
    pe, Rpe = args['pe'], cs['R_pe']
    dp = (args['ne']*args['dT_e_dpsi'] + args['Te']*args['dn_e_dpsi'] + args['ni']*args['dT_i_dpsi']
          + args['Ti']*args['dn_i_dpsi'])*EC
    jb = -args['I_psi']*pe*(c['L31']*dp/pe + c['L32']*args['dT_e_dpsi']/args['Te']
                            + c['L34']*c['alpha']*(1.0 - Rpe)/Rpe*args['dT_i_dpsi']/args['Ti'])
    assert np.array_equal(jb_s, jb)
    assert _rel(jb_s, jb_r) > 1.0E-2
    assert _rejects(redl_bootstrap, **args, coefficient_model='Sauter')
    assert _rejects(redl_bootstrap, **args, coefficient_model='sauter', use_legacy_L34=True)


## Sauter cases of test_ITER_bootstrap_eval_sauter beyond the ITER profiles: name -> (Z_eff, temperature scale,
## density scale)
SAUTER_SCAN = {'zeff1': (1.0, 1.0, 1.0), 'zeff4_t005': (4.0, 0.05, 1.0), 'zeff4_t001_n3': (4.0, 0.01, 3.0)}


def run_eval_sauter(mesh_resolution, fe_order, mp_q):
    setup = setup_ITER_bootstrap_internal(mesh_resolution, fe_order)
    if setup is None:
        mp_q.put(None)
        return
    mygs, x, ne, Te = setup['mygs'], setup['psi_sample'], setup['ne'], setup['Te']
    n = len(x)
    Zeff = np.full(n, setup['Zeff_val'])
    Te_k = Te/1e3
    ni_c, nimp_c = _carbon_case(ne, setup['Zeff_val'])
    results = {}
    try:
        # bs_model = 2 against redl_bootstrap(coefficient_model='sauter'), with both Coulomb logarithms and with the
        # carbon impurity; the parts that do not depend on the coefficients equal those of bs_model = 1 bitwise
        cases = {'nrl': (ne, None, 1, 'NRL'), 'imp': (ni_c, nimp_c, 2, 'sauter')}
        for name, (ni, nimp, ln_model, ln_py) in cases.items():
            F = mygs.eval_bootstrap(x, Te_k, Te_k, ne, ni, setup['Zeff_val'], nimp=nimp, ln_lambda_model=ln_model,
                                    bs_model=2)
            F1 = mygs.eval_bootstrap(x, Te_k, Te_k, ne, ni, setup['Zeff_val'], nimp=nimp, ln_lambda_model=ln_model)
            P = _python_bootstrap(mygs, x, Te_k*1e3, Te_k*1e3, ne, ni, Zeff, nimp=nimp, ln_lambda=ln_py,
                                  coefficient_model='sauter')
            for key in EVAL_KEYS:
                results[f'{name}/{key}'] = _rel(F[key][1:-1], P[key])
            results[f'{name}/shared'] = all(np.array_equal(F[key], F1[key], equal_nan=True)
                                            for key in ('jphi_pd_raw', 'f_t', 'nu_e_star', 'nu_i_star'))
            results[f'{name}/effect'] = _rel(F['jdotB_bs_raw'], F1['jdotB_bs_raw'])
        # The same over Z_eff and collisionality (above: Z_eff 1.5, nu_e* < 0.3): Z_eff 1 and 4 with carbon, and the
        # temperatures scaled down (and the density up) into the plateau and collisional regimes
        for name, (zv, t_scale, n_scale) in SAUTER_SCAN.items():
            ne_s, Te_s = ne*n_scale, Te_k*t_scale
            ni_s, nimp_s = _carbon_case(ne_s, zv)
            F = mygs.eval_bootstrap(x, Te_s, Te_s, ne_s, ni_s, zv, nimp=nimp_s, bs_model=2)
            P = _python_bootstrap(mygs, x, Te_s*1e3, Te_s*1e3, ne_s, ni_s, np.full(n, zv), nimp=nimp_s,
                                  coefficient_model='sauter')
            for key in EVAL_KEYS:
                results[f'{name}/{key}'] = _rel(F[key][1:-1], P[key])
            results[f'{name}/nu_e_max'] = float(np.max(P['nu_e_star']))
        # Invalid selections are rejected in Python; the Fortran entries reject them too
        base = dict(psi=x, te=Te_k, ti=Te_k, ne=ne, ni=ne, Zeff=1.5)
        for name, val in (('3', 3), ('0', 0), ('float', 2.0), ('bool', True), ('string', '2')):
            results[f'rejects/eval_{name}'] = _rejects(mygs.eval_bootstrap, **dict(base, bs_model=val))
            results[f'rejects/set_boot_ops_{name}'] = _rejects(mygs.set_boot_ops, bs_model=val)
        from OpenFUSIONToolkit.TokaMaker._interface import tokamaker_eval_bootstrap
        eq = mygs._tMaker_equil
        err = eq._oft_env.get_c_errorbuff()
        vals = [np.ascontiguousarray(v) for v in (1.0 - x[::-1], Te_k[::-1], Te_k[::-1], ne[::-1], ne[::-1],
                                                  np.full(n, 1.5), np.zeros(n))]
        tokamaker_eval_bootstrap(eq.c_ptr, n, *vals, 1, 0, 3, 1, np.zeros((6, n)), err)
        results['fortran_eval_rejects'] = b'Invalid bs_model' in err.value
        # psi_convention = 1 gives the mirror of psi_convention = 0 bitwise
        F = mygs.eval_bootstrap(x, Te_k, Te_k, ne, ni_c, setup['Zeff_val'], nimp=nimp_c, bs_model=2)
        mygs._tMaker_equil.psi_convention = 1
        try:
            F_mirror = mygs.eval_bootstrap(1.0 - x[::-1], Te_k[::-1], Te_k[::-1], ne[::-1], ni_c[::-1],
                                           setup['Zeff_val'], nimp=nimp_c[::-1], bs_model=2)
        finally:
            mygs._tMaker_equil.psi_convention = 0
        results['mirror'] = all(F[key].tobytes() == F_mirror[key][::-1].tobytes() for key in EVAL_KEYS)
    except Exception:
        import traceback
        traceback.print_exc()
        mp_q.put(None)
        return
    mp_q.put([results])
    oftpy_dump_cov()


@pytest.mark.parametrize("order", (2,))
def test_ITER_bootstrap_eval_sauter(order):
    '''eval_bootstrap with bs_model = 2 (Sauter 1999 coefficients with the 2002 erratum) on the ITER equilibrium
    with H-mode profiles equals bootstrap.redl_bootstrap(coefficient_model='sauter') with the internal solver's
    settings, node by node, with both Coulomb logarithms and with a carbon impurity, and at Z_eff 1 and 4 with the
    temperatures scaled down into the plateau and collisional regimes; the pressure-driven part, f_t and the
    collisionalities equal those of bs_model = 1 bitwise (shared code), and <j.B> differs.

    The bound 1E-12 of the peak is not a measured value: Fortran and numpy evaluate the same expressions in the same
    order, and may differ only by the rounding of single operations (integer powers by multiplication or by pow,
    fused multiply-add and division as compiled), a few units in the last place each over about a hundred
    operations (1E-13), and by the cancellation of the three jboot1 terms against the peak (at most ten).
    Invalid bs_model values raise ValueError in Python and an error from the Fortran entry, and psi_convention = 1
    gives the mirror of psi_convention = 0 bitwise.
    '''
    results = mp_run(run_eval_sauter, (1.0, order), timeout=300)
    assert results is not None
    res = results[0]
    for name in ('nrl', 'imp'):
        for key in EVAL_KEYS:
            assert res[f'{name}/{key}'] < 1.0E-12, f'{name}/{key}'
        assert res[f'{name}/shared']
        assert res[f'{name}/effect'] > EFFECT_BOUND, f'{name}/effect'
    for name in SAUTER_SCAN:
        for key in EVAL_KEYS:
            assert res[f'{name}/{key}'] < 1.0E-12, f'{name}/{key}'
    # The scan reaches the collisional regime
    assert res['zeff4_t001_n3/nu_e_max'] > 10.0
    rejects = [key for key in res if key.startswith('rejects/')]
    assert len(rejects) == 10
    for key in rejects:
        assert res[key], key
    assert res['fortran_eval_rejects']
    assert res['mirror']


def run_sauter_insolve(mesh_resolution, fe_order, mp_q):
    setup = setup_ITER_bootstrap_internal(mesh_resolution, fe_order)
    if setup is None:
        mp_q.put(None)
        return
    mygs, x, ne, Te = setup['mygs'], setup['psi_sample'], setup['ne'], setup['Te']
    Zeff = setup['Zeff_val']
    ni, _ = _carbon_case(ne, Zeff)
    kwargs = dict(ffp_prof={'type': 'jphi-split-bootstrap', 'x': x, 'y': setup['inductive_jphi']},
                  te_prof={'type': 'linterp', 'x': x, 'y': Te/1e3}, ne_prof={'type': 'linterp', 'x': x, 'y': ne},
                  ti_prof={'type': 'linterp', 'x': x, 'y': Te/1e3}, ni_prof={'type': 'linterp', 'x': x, 'y': ni},
                  Zeff=Zeff, Ip_target=setup['Ip_target'])
    results = {}
    try:
        import ctypes
        from OpenFUSIONToolkit.TokaMaker._interface import (tokamaker_boot_ops_struct, tokamaker_set_boot_ops,
                                                            tokamaker_get_boot_ops)
        bp_redl = mygs.solve_bootstrap(**kwargs)
        bp = mygs.solve_bootstrap(**kwargs, bs_model=2)
        results['insolve_model_effect'] = _rel(bp['jdotB_bs_raw'], bp_redl['jdotB_bs_raw'])
        # eval_bootstrap takes bs_model from the solve and reproduces the solver's last evaluation (freeze tolerance)
        F = mygs.eval_bootstrap(x, Te/1e3, Te/1e3, ne, ni, Zeff)
        results['insolve'] = _rel(F['jdotB_bs_raw'], bp['jdotB_bs_raw'])
        results['insolve_bs'] = _rel(F['jphi_bs_raw'], bp['jphi_bs_raw'])
        # The option is in the Fortran state; an invalid value from the C interface is refused and changes nothing
        eq = mygs._tMaker_equil
        bops, initialized, err = tokamaker_boot_ops_struct(), ctypes.c_bool(False), eq._oft_env.get_c_errorbuff()
        tokamaker_get_boot_ops(eq.c_ptr, ctypes.byref(bops), ctypes.byref(initialized), err)
        results['fortran_ops'] = bool(initialized.value and bops.bs_model == 2)
        bops.bs_model = 3
        tokamaker_set_boot_ops(eq.c_ptr, ctypes.byref(bops), err)
        results['fortran_set_rejects'] = b'Invalid bs_model' in err.value
        bops, err = tokamaker_boot_ops_struct(), eq._oft_env.get_c_errorbuff()
        tokamaker_get_boot_ops(eq.c_ptr, ctypes.byref(bops), ctypes.byref(initialized), err)
        results['fortran_ops_kept'] = bool(bops.bs_model == 2)
        # Saved and loaded with the equilibrium; a file without it (written before the option) loads the default
        import h5py
        eq.save_TokaMaker('ITER_boot_sauter_test.h5')
        mygs.boot_reset()
        results['reset'] = bool(mygs._tMaker_equil._boot_ops['bs_model'] == 1)
        mygs.replace_eq(source_file='ITER_boot_sauter_test.h5')
        results['loaded'] = bool(mygs._tMaker_equil._boot_ops['bs_model'] == 2)
        paths = []
        with h5py.File('ITER_boot_sauter_test.h5', 'r+') as fid:
            fid.visit(lambda name: paths.append(name) if name.endswith('BOOT_OPS/BS_MODEL') else None)
            for path in paths:
                fid[path][...] = 7
        # An invalid value in the file loads the default (with a warning)
        mygs.replace_eq(source_file='ITER_boot_sauter_test.h5')
        results['loaded_invalid'] = bool(mygs._tMaker_equil._boot_ops['bs_model'] == 1)
        with h5py.File('ITER_boot_sauter_test.h5', 'r+') as fid:
            for path in paths:
                del fid[path]
        results['n_removed'] = len(paths)
        mygs.replace_eq(source_file='ITER_boot_sauter_test.h5')
        os.remove('ITER_boot_sauter_test.h5')
        results['loaded_old'] = bool(mygs._tMaker_equil._boot_ops['bs_model'] == 1)
        # The deprecated Python solve passes bs_model to redl_bootstrap (recorded at its first evaluation, which
        # then stops the solve) and rejects invalid values and the combination with the OMFIT (Redl) evaluation
        import OpenFUSIONToolkit.TokaMaker.bootstrap as bootstrap_mod

        class _Stop(Exception):
            pass
        seen = []

        def _spy(**kw):
            seen.append(kw.get('coefficient_model', 'redl'))
            raise _Stop
        py_args = dict(mygs=mygs, ne=ne, Te=Te, ni=ni, Ti=Te, Zeff=np.full(len(x), Zeff), Ip_target=setup['Ip_target'],
                       inductive_jphi=setup['inductive_jphi'], x=x, use_python_solve=True, verbose=False)
        redl_orig = bootstrap_mod.redl_bootstrap
        bootstrap_mod.redl_bootstrap = _spy
        try:
            for model in (None, 1, 2):
                try:
                    bootstrap_mod.solve_with_bootstrap(**py_args, bs_model=model)
                except _Stop:
                    pass
        finally:
            bootstrap_mod.redl_bootstrap = redl_orig
        results['python_solve_models'] = seen == ['redl', 'redl', 'sauter']
        results['python_solve_rejects'] = (_rejects(bootstrap_mod.solve_with_bootstrap, **py_args, bs_model=3)
                                           and _rejects(bootstrap_mod.solve_with_bootstrap, **py_args, bs_model=2.0)
                                           and _rejects(bootstrap_mod.solve_with_bootstrap, **py_args, bs_model=2,
                                                        use_OMFIT_sauter=True))
    except Exception:
        import traceback
        traceback.print_exc()
        mp_q.put(None)
        return
    mp_q.put([results])
    oftpy_dump_cov()


@pytest.mark.parametrize("order", (2,))
def test_ITER_bootstrap_sauter_insolve(order):
    '''bs_model = 2 reaches the internal solve: the self-consistent ITER bootstrap solve with a carbon impurity
    differs from the one with bs_model = 1, eval_bootstrap (taking bs_model from the solve) reproduces the solver's
    last <j.B> to the level of its freeze tolerance (the bound of test_ITER_bootstrap_eval_insolve, 3E-4 of the
    peak, and 3.5E-4 for the stored field-aligned part), the option is in the Fortran state and an invalid value
    from the C interface is refused, boot_reset restores the default, and the option is saved and loaded with the
    equilibrium, a file with an invalid value or without it loading the default. The deprecated Python solve
    (use_python_solve=True) evaluates the coefficients of bs_model and rejects invalid values.
    '''
    results = mp_run(run_sauter_insolve, (1.0, order), timeout=300)
    assert results is not None
    res = results[0]
    assert res['insolve_model_effect'] > EFFECT_BOUND
    assert res['insolve'] < 3.0E-4
    assert res['insolve_bs'] < 3.5E-4
    for key in ('fortran_ops', 'fortran_set_rejects', 'fortran_ops_kept', 'reset', 'loaded', 'loaded_invalid',
                'loaded_old', 'python_solve_models', 'python_solve_rejects'):
        assert res[key], key
    assert res['n_removed'] == 1
