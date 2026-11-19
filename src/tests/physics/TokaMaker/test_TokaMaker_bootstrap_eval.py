#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
# TokaMaker tests: eval_bootstrap (the internal solver's bootstrap evaluation on a given equilibrium) against the
# Redl formula of bootstrap.py, its options, and the evaluation inside solve_bootstrap, on ITER
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


def _python_bootstrap(mygs, x, Te, Ti, ne, ni, Zeff, nimp=None, ln_lambda='NRL', solver_gradients=True):
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
                                   formula_form='jboot1')
    inv_R, B2 = ravgs['<1/R>'], b_avgs[1]
    return {'jdotB_bs_raw': jdotB, 'jphi_bs_raw': jdotB*f*inv_R/B2, 'jphi_pd_raw': pp*(ravgs['<R>'] - f**2*inv_R/B2),
            'f_t': 1.0 - fc, 'nu_e_star': coeffs['nu_e_star'], 'nu_i_star': coeffs['nu_i_star'],
            'ln_lambda_min': float(min(np.min(ln_le), np.min(ln_lii)))}


def _rel(a, b):
    '''Largest difference relative to the largest magnitude of b'''
    return float(np.max(np.abs(a - b))/np.max(np.abs(b)))


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
    measured. With the gradients taken as the solver takes them, all interior nodes agree to round-off (6E-16
    for the currents, 1E-15 for the collisionalities; f_t and the pressure-driven part are identical). With the
    gradients of solve_with_bootstrap (on the normalized flux, LCFS node included) the currents differ by 4E-14,
    from the differently rounded abscissae, and the outermost interior node differs by the gradient stencil
    (one-sided in the solver; 1.0E-3 of its value), which is checked to exceed round-off. The derived impurity
    density equals the carbon density to round-off, which the gradient raises to 1.5E-14 in <j.B>.

    References: O. Sauter et al., Phys. Plasmas 6, 2834 (1999); A. Redl et al., Phys. Plasmas 28, 022502 (2021).
    '''
    results = mp_run(run_eval_parity, (1.0, order), timeout=300)
    assert results is not None
    res = results[0]
    bounds = {'jdotB_bs_raw': 6.0E-15, 'jphi_bs_raw': 6.0E-15, 'jphi_pd_raw': 1.0E-14, 'f_t': 1.0E-14,
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
        results['insolve_default_vs_opts'] = _rel(F_opts['jdotB_bs_raw'], bp['jdotB_bs_raw'])
        # Both options in the solve; eval_bootstrap takes them from the solve's options by default
        bp_default = bp
        bp = mygs.solve_bootstrap(**kwargs, ln_lambda_model=2, imp_pressure=True)
        results['insolve_opts_vs_default'] = _rel(bp['jdotB_bs_raw'], bp_default['jdotB_bs_raw'])
        F = mygs.eval_bootstrap(x, Te/1e3, Te/1e3, ne, ni, Zeff)
        results['insolve_opts'] = _rel(F['jdotB_bs_raw'], bp['jdotB_bs_raw'])
        results['insolve_opts_pd'] = _rel(F['jphi_pd_raw'], bp['jphi_pd_raw'])
        # The options survive save and load
        mygs._tMaker_equil.save_TokaMaker('ITER_boot_eval_test.h5')
        mygs.boot_reset()
        mygs.replace_eq(source_file='ITER_boot_eval_test.h5')
        ops = mygs._tMaker_equil._boot_ops
        results['ops_loaded'] = bool(ops['ln_lambda_model'] == 2 and ops['imp_pressure'])
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
    options are saved and loaded with the equilibrium.

    Differences are relative to the largest magnitude. The bound is about three times the largest difference
    measured (5.8E-5, 9.0E-5 and 1.1E-4 of the peak); each option alone changes <j.B> by more (8.0E-3 and
    1.7E-3 in test_ITER_bootstrap_eval_parity).
    '''
    results = mp_run(run_eval_insolve, (1.0, order), timeout=300)
    assert results is not None
    res = results[0]
    assert res['state_unchanged']
    assert res['insolve_default'] < 3.0E-4
    assert res['insolve_opts'] < 3.0E-4
    assert res['insolve_opts_pd'] < 3.0E-4
    assert res['insolve_default_vs_opts'] > 3.0E-3  # measured 7.5E-3
    assert res['insolve_opts_vs_default'] > 3.0E-3  # the options reach the solve; measured 9.1E-3
    assert res['ops_loaded']
