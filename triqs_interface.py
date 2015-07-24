import pytriqs.operators as triqs_ops
import pytriqs.operators.util as triqs_ops_util
import pytriqs.applications.impurity_solvers.cthyb_mod as triqs_solver
import pytriqs.gf.local as triqs_gf

from pytriqs.archive import HDFArchive
import pytriqs.utility.mpi as mpi

import os
import sys
import time
from numpy import *

from share_fun import readParameters


def load_parms_from_file(filename):
    parms = readParameters(filename)
    for s in ('NSPINS', 'NFLAVORS', 'N_CUTOFF', 'N_MAX_FREQ', 'N_TAU',
              'MEASURE',
              'n_cycles', 'length_cycle', 'n_warmup_cycles', 'max_time'):
        if s in parms: parms[s] = int(parms[s])
    for s in ('BETA', 'U', 'J'):
        if s in parms: parms[s] = float(parms[s])
    return parms


def assign_weiss_field(G0, parms, nspins, spin_names, nflavors, flavor_names):
    hyb_mat_prefix = parms.get('HYB_MAT', '%s.hybmat'%parms['PREFIX'])
    hyb_mat = genfromtxt('%s.real'%hyb_mat_prefix)[:,1:]\
              + 1j*genfromtxt('%s.imag'%hyb_mat_prefix)[:,1:]
    hyb_tail = genfromtxt('%s.tail'%hyb_mat_prefix)
    mu_vec = genfromtxt(parms.get('MU_VECTOR', '%s.mu_eff'%parms['PREFIX'])) 
    for s in range(nspins):
        for f in range(nflavors):
            hyb_w = triqs_gf.GfImFreq(indices=[0], beta=parms['BETA'], 
                                      n_points=parms['N_MAX_FREQ'])
            hyb_w.data[:, 0, 0] = hyb_mat[:, nspins*f+s] 
            for n in range(len(hyb_tail)):
                hyb_w.tail[n+1][0, 0] = hyb_tail[n, nspins*f+s]
            block, i = mkind(spin_names[s], flavor_names[f])
            G0[block][i, i] << triqs_gf.inverse(triqs_gf.iOmega_n\
                                                +mu_vec[nspins*f+s]-hyb_w)


def get_interaction_hamiltonian(parms, spin_names, flavor_names, is_kanamori):
    U_int = parms['U']
    J_hund = parms['J']
    if is_kanamori:
        U, Uprime = triqs_ops_util.U_matrix_kanamori(parms['NFLAVORS'],
                                                     U_int, J_hund)
        ham = triqs_ops_util.h_int_kanamori(spin_names, flavor_names,
                                            U, Uprime, J_hund, off_diag=False)
    else: # Slater-type interaction
        l_number = (parms['NFLAVORS']-1)/2
        U = triqs_ops_util.U_matrix(l=lnumber, U_int=U_int, J_hund=J_hund,
                                    basis='cubic')
        ham = triqs_ops_util.h_int_slater(spin_names, flavor_names,
                                          U, off_diag=False)
    return ham


def get_quantum_numbers(parms, spin_names, flavor_names, is_kanamori):
    qn = []
    for s in spin_names:
        tmp = triqs_ops.Operator()
        for o in flavor_names:
            tmp += triqs_ops.n(*mkind(s, o))
        qn.append(tmp)
    if is_kanamori:
        for o in flavor_names:
            dn = triqs_ops.n(*mkind(spin_names[0], o))\
                 - triqs_ops.n(*mkind(spin_names[1],o))
            qn.append(dn*dn)
    return qn


def get_static_observables(parms, spin_names, flavor_names):
    ret = {
            'N' : triqs_ops.Operator(),
            'Sz' : triqs_ops.Operator(),
            }
    for sn, s in enumerate(spin_names):
        for o in flavor_names:
            sp = mkind(s, o)
            ret['N'] += triqs_ops.n(*sp)
            ret['Sz'] += (-1)**sn * triqs_ops.n(*sp)
    return ret


if __name__ == '__main__':
    mkind = triqs_ops_util.get_mkind(off_diag=False,
                                     map_operator_structure=None)
    parms = load_parms_from_file(sys.argv[1])
    if parms['INTERACTION'].upper() not in ('SLATER', 'KANAMORI'):
        raise ValueError('Key INTERACTION must be either "Slater" or "Kanamori"')
    is_kanamori = True if parms['INTERACTION'].upper() == 'KANAMORI'\
                  else False
    assert parms['NSPINS'] == 2
    nspins = parms['NSPINS']
    spin_names = ('up', 'dn')
    nflavors = parms['NFLAVORS']
    flavor_names = [str(i) for i in range(nflavors)]
    gf_struct = triqs_ops_util.set_operator_structure(spin_names, flavor_names,
                                                      off_diag=False)

    solver = triqs_solver.Solver(beta=parms['BETA'], gf_struct=gf_struct,
                                 n_tau=parms['N_TAU'], n_iw=parms['N_MAX_FREQ'])
    solver_parms = {}
    for s in parms:
        if s.lower() == s: solver_parms[s] = parms[s]
    assign_weiss_field(solver.G0_iw, parms, nspins, spin_names, 
                       nflavors, flavor_names)
    ham_int = get_interaction_hamiltonian(parms, spin_names, flavor_names,
                                          is_kanamori)
    if solver_parms['partition_method'] == 'quantum_numbers':
        solver_parms['quantum_numbers'] = get_quantum_numbers(parms, 
                                            spin_names, flavor_names, 
                                            is_kanamori)

    solver_parms.update({
        'h_int' : ham_int,
        'random_seed' : int(1e6*time.time()*(mpi.rank+1) % 1e6),
        'use_trace_estimator' : False,
        'measure_g_tau' : True,
        'measure_g_l' : False,
        'performance_analysis' : False,
        'perform_tail_fit' : False,
        'perform_post_proc' : True,
        'move_shift' : True,
        'move_double' : False,
        })
    if parms.get('MEASURE', 0) > 0:
        solver_parms['static_observables'] = get_static_observables(parms,
                                                    spin_names, flavor_names)

    # run the solver
    solver.solve(**solver_parms)

    # save data
    if mpi.is_master_node():
        h5file = HDFArchive(parms.get('HDF5_OUTPUT', 
                                      '%s.triqs.out.h5'%parms['PREFIX']), 'w')
        h5file['Gtau'] = solver.G_tau
        h5file['Giwn'] = solver.G_iw
        h5file['Siwn'] = solver.Sigma_iw
        h5file['Occupancy'] = solver.G_iw.density()
        h5file['G0iwn'] = solver.G0_iw
        h5file['average_sign'] = solver.average_sign
        if len(solver.static_observables) > 0:
            h5file['Observables'] = solver.static_observables

        r = solver.eigensystems
        eigvals = []
        for rr in r:
            eigvals = r_[eigvals, rr[0]]
        savetxt('%s.eigvals'%parms['PREFIX'], sort(eigvals))
