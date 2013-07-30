from pytriqs.Base.GF_Local import *
from pytriqs.Base.Utility.myUtils import Sum
from pytriqs.Solvers.Operators import *
from pytriqs.Solvers.HybridizationExpansion import Solver

import sys, os
from share_fun import val_def, readParameters;
from numpy import *;


#
# Interface for TRIQS solver
# Caution: works for diagonal Weiss field only
#


# input data
parms = readParameters(sys.argv[1]);
NCOR = int(parms['NCOR']); SPINS = 2; spins = ('up', 'dn');
BETA = float(parms['BETA']);
hyb_mat = genfromtxt(parms['HYB_MAT']+'.real')[:,1:] + 1j*genfromtxt(parms['HYB_MAT']+'.imag')[:,1:];
hyb_coefs = genfromtxt(parms['HYB_MAT']+'.tail');
MUvector = genfromtxt(parms['MU_VECTOR']); 

# overwrite parameters
if os.path.isfile('settings'):
    parms_new = readParameters('settings');
    print 'Overwrite parameters'
    for k, v in parms_new.iteritems(): 
        print k, ' = ', v;
        parms[k] = v;


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
    Ntot = Sum( [ N('%s%d'%(s,f),0) for s in spins for f in range(NCOR) ]);
    Sz   = Sum( [ N('%s%d'%(spins[0],f),0) - N('%s%d'%(spins[1],f),0) for f in range(NCOR) ]);
    Quantum_Numbers = { 'Ntot' : Ntot, 'Sztot' : Sz };
    for f in range(NCOR):
        Quantum_Numbers['Sz2_%d'%f] = N('%s%d'%(spins[0],f),0) + N('%s%d'%(spins[1],f),0) - 2*N('%s%d'%(spins[0],f),0)*N('%s%d'%(spins[1],f),0)
else:
    Quantum_Numbers = {};
    for sp in spins:
        for f in range(NCOR): Quantum_Numbers['N%s%d'%(sp,f)] = N('%s%d'%(sp,f),0);

# create a solver object
solver = Solver(
    Beta = BETA, 
    GFstruct = GFstruct,
    H_Local = H_Local,
    Quantum_Numbers = Quantum_Numbers,
    N_Cycles = int(parms['SWEEPS_EACH_NODE']),
    Length_Cycle = int(parms['N_MEAS']),
    N_Warmup_Cycles = int(parms['THERMALIZATION']),
    N_Matsubara_Frequencies = int(val_def(parms, 'N_MATSUBARA', len(hyb_mat))),
    Use_Segment_Picture = int(parms['SPINFLIP']) == 0
    );


# Legendre or Time accumulation
accumulation = val_def(parms, 'ACCUMULATION', 'legendre');
if accumulation not in ['time', 'legendre']: exit('ACCUMULATION should be either "time" or "legendre"');
if accumulation == 'time': 
    solver.Time_Accumulation = True;
    solver.Legendre_Accumulation = False;
    solver.Fitting_Frequency_Start = len(hyb_mat)-10;
    solver.N_Frequencies_Accumulated = len(hyb_mat)-1;  # I don't want to use the fitTails()
elif accumulation == 'legendre':
    solver.Legendre_Accumulation = True;
    solver.N_Legendre_Coeffs = int(val_def(parms, 'N_LEGENDRE', 50));

# prepare input G0 from hybridization \Gamma
if len(MUvector) == 1: MUvector = MUvector[0]*ones(SPINS*NCOR);
assert len(MUvector) == SPINS*NCOR, 'Length of MUvector must be equal to #FLAVORS';
Delta = GF(Name_Block_Generator = solver.G0, Copy = True, Name = "Delta");
for s, sp in enumerate(spins):
    for f in range(NCOR):
        name = '%s%d'%(sp,f);
        Delta[name]._data[0,0,:] = hyb_mat[:, 2*f+s];
        for n in range(size(hyb_coefs, 0)): 
            Delta[name]._tail[n+1][0, 0] = hyb_coefs[n, 2*f+s];
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


solver.Measured_Operators = Measured_Operators;


# run solver
solver.Solve();


# save data
import pytriqs.Base.Utility.MPI as MPI
from pytriqs.Base.Archive import HDF_Archive
NfileMax = 100;
if MPI.IS_MASTER_NODE():
    R = HDF_Archive(parms['HDF5_OUTPUT'], 'w');
    if accumulation == 'legendre':
        for s, gl in solver.G_Legendre: solver.G[s] <<= LegendreToMatsubara(gl);
        R['G_Legendre'] = solver.G_Legendre;
        Gl_out = None;
        for f in range(NCOR):
            for sp in spins:
                tmp = solver.G_Legendre['%s%d'%(sp, f)]._data.array[0, 0, :];
                Gl_out = tmp if Gl_out is None else c_[Gl_out, tmp];
        for n in range(1,NfileMax):
            filename = 'Green_Legendre.%03d'%n;
            if not os.path.isfile(filename): break;
        savetxt(filename, c_[arange(len(Gl_out)), Gl_out]);

    solver.Sigma <<= solver.G0_inv - inverse(solver.G);
    R['G'] = solver.G;
    R['Sigma'] = solver.Sigma;
    R['Observables'] = solver.Measured_Operators_Results;


