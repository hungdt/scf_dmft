from pytriqs.archive import HDFArchive
from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.applications.impurity_solvers.cthyb_matrix import Solver
import pytriqs.utility.mpi as mpi

import sys, os, time
from share_fun import val_def, readParameters;
from numpy import *;


#
# Interface for TRIQS solver
# Caution: works for diagonal Weiss field only
#


# input data
parms = readParameters(sys.argv[1]);
NCOR = int(parms['NCOR']); 
SPINS = 2; spins = ('up', 'dn');
BETA = float(parms['BETA']);
hyb_mat = genfromtxt(parms['HYB_MAT']+'.real')[:,1:] + 1j*genfromtxt(parms['HYB_MAT']+'.imag')[:,1:];
hyb_coefs = genfromtxt(parms['HYB_MAT']+'.tail');
MUvector = genfromtxt(parms['MU_VECTOR']); 

solver_parms = {
        'n_cycles'        : int(parms['SWEEPS_EACH_NODE']),
        'length_cycle'    : int(parms['N_MEAS']),
        'n_warmup_cycles' : int(parms['THERMALIZATION']),
        'random_seed'     : int(1e6*time.time()*(mpi.rank+1) % 1e6)
        };


# prepare Green function structure, local H and quantum numbers
# Slater-Kanamori style
Umat = genfromtxt(parms['U_MATRIX']); 
GFstruct = [ ('%s%d'%(s,f), [0]) for f in range(NCOR) for s in spins ];
H_Local = None;
for n in range(SPINS*NCOR):
    s = n % SPINS; f = n / SPINS;
    for n1 in range(n+1, SPINS*NCOR):
        s1 = n1 % SPINS; f1 = n1 / SPINS;
        tmp = Umat[n, n1] * N('%s%d'%(spins[s],f),0) * N('%s%d'%(spins[s1],f1), 0);
        if H_Local is None: H_Local = tmp;
        else: H_Local += tmp;


# exchange and pair hopping terms
if int(parms['SPINFLIP']) > 0:
    Jmat = Umat[::2,1::2] - Umat[::2,::2];
    for f1 in range(NCOR):
        for f2 in range(NCOR):
            if f1 == f2: continue;
            u1 = '%s%d'%(spins[0],f1);
            u2 = '%s%d'%(spins[0],f2);
            d1 = '%s%d'%(spins[1],f1);
            d2 = '%s%d'%(spins[1],f2);
            H_Local += Jmat[f1,f2]*Cdag(u1,0)*C(u2,0)*Cdag(d2,0)*C(d1,0) + Jmat[f1,f2]*Cdag(u1,0)*C(u2,0)*Cdag(d1,0)*C(d2,0);
    Ntot = sum( [ N('%s%d'%(s,f),0) for s in spins for f in range(NCOR) ]);
    Sz   = sum( [ N('%s%d'%(spins[0],f),0) - N('%s%d'%(spins[1],f),0) for f in range(NCOR) ]);
    Quantum_Numbers = { 'Ntot' : Ntot, 'Sztot' : Sz };
    for f in range(NCOR):
        Quantum_Numbers['Sz2_%d'%f] = N('%s%d'%(spins[0],f),0) + N('%s%d'%(spins[1],f),0) - 2*N('%s%d'%(spins[0],f),0)*N('%s%d'%(spins[1],f),0)
else:
    Quantum_Numbers = {};
    for sp in spins:
        for f in range(NCOR): Quantum_Numbers['N%s%d'%(sp,f)] = N('%s%d'%(sp,f),0);
solver_parms['quantum_numbers'] = Quantum_Numbers;
solver_parms['use_segment_picture'] = int(parms['SPINFLIP']) == 0;
solver_parms['H_local'] = H_Local;


# create a solver object
solver = Solver(beta = BETA, gf_struct = GFstruct, n_w = int(val_def(parms, 'N_MATSUBARA', len(hyb_mat))));


# Legendre or Time accumulation
accumulation = val_def(parms, 'ACCUMULATION', 'time');
if accumulation not in ['time', 'legendre']: exit('ACCUMULATION should be either "time" or "legendre"');
if accumulation == 'time': 
    solver_parms['time_accumulation'] = True;
    solver_parms['legendre_accumulation'] = False;
    solver_parms['fit_start'] = len(hyb_mat)-10;
    solver_parms['fit_stop']  = len(hyb_mat)-1;  # I don't want to use the fitTails()
elif accumulation == 'legendre':
    solver_parms['legendre_accumulation'] = True;
    solver_parms['n_legendre'] = int(val_def(parms, 'N_LEGENDRE', 50));


# prepare input G0 from hybridization \Gamma
if len(MUvector) == 1: MUvector = MUvector[0]*ones(SPINS*NCOR);
assert len(MUvector) == SPINS*NCOR, 'Length of MUvector must be equal to #FLAVORS';
Delta = BlockGf(name_block_generator = solver.G0, make_copies = True, name = "Delta");
for s, sp in enumerate(spins):
    for f in range(NCOR):
        name = '%s%d'%(sp,f);
        Delta[name].data[:, 0, 0] = hyb_mat[:, 2*f+s];
        for n in range(size(hyb_coefs, 0)): 
            Delta[name].tail[n+1][0, 0] = hyb_coefs[n, 2*f+s];
        MU = MUvector[2*f+s];
        solver.G0[name] <<= inverse(iOmega_n + MU - Delta[name]);


# operators for measurement
Measured_Operators = {};
for sp in spins:
    for f in range(NCOR): 
        Measured_Operators['N_%s%d'%(sp,f)] = N('%s%d'%(sp,f), 0);
for n1 in range(SPINS*NCOR):
    for n2 in range(n1+1, SPINS*NCOR):
        f1 = n1 / SPINS; sp1 = spins[n1 % SPINS];
        f2 = n2 / SPINS; sp2 = spins[n2 % SPINS];
        Measured_Operators['nn_%d_%d'%(n2,n1)] = N('%s%d'%(sp1, f1), 0) * N('%s%d'%(sp2, f2), 0);
solver_parms['measured_operators'] = Measured_Operators;

solver_parms['measured_time_correlators'] = {}
if int(val_def(parms, 'MEASURE', 0)) > 0:
    if 'Sztot' in Quantum_Numbers:
        solver_parms['measured_time_correlators'] = {
                'Sztot' : [ Quantum_Numbers['Sztot'], 300 ]
                }


# run solver
solver.solve(**solver_parms);


# save data
NfileMax = 100;
if mpi.is_master_node():
    R = HDFArchive(parms['HDF5_OUTPUT'], 'w');
    if accumulation == 'legendre':
        for s, gl in solver.G_legendre: solver.G[s] <<= LegendreToMatsubara(gl);
        R['G_Legendre'] = solver.G_legendre;
        Gl_out = None;
        for f in range(NCOR):
            for sp in spins:
                tmp = solver.G_legendre['%s%d'%(sp, f)]._data.array[0, 0, :];
                Gl_out = tmp if Gl_out is None else c_[Gl_out, tmp];
        for n in range(1,NfileMax):
            filename = 'Green_Legendre.%03d'%n;
            if not os.path.isfile(filename): break;
        savetxt(filename, c_[arange(len(Gl_out)), Gl_out]);

    solver.Sigma <<= solver.G0_inv - inverse(solver.G);
    R['G'] = solver.G;
    R['Sigma'] = solver.Sigma;
    R['Observables'] = solver.measured_operators_results;
    if len(solver_parms['measured_time_correlators']) > 0:
        R['TimeCorrelators'] = solver.measured_time_correlators_results


