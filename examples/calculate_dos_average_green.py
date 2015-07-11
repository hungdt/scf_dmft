# demo: generate the noninteracting DOS from the DFT+MLWF Hamiltonian
#       using average_green module

from numpy import *
import sys
# directory for the main code
sys.path.append('..')

from functions import rotate_all, irotate
from share_fun import grule
import init
import average_green

N_LAYERS = 4
FLAVORS  = 3
SPINS = 1

mu = 15.45
double_counting = 0.
magnetic_field = 0.

# the file for the Hamiltonian produced by DFT+MLWF
rham_file = "LaTiO3_tilted_t2g_only/lto_rham.py"

nbin = 300
emin = -1.5
emax = 1.5
broadening = 0.02
numk = 26

# parallelization: parallel
#   0: no parallelization
#   1: OpenMP with number of threads set by OMP_NUM_THREADS
#   2: MPI with number of processes from the input
parallel = 2
num_processes = 10
if len(sys.argv) > 1: num_processes = int(sys.argv[1])

HR, R = init.getHamiltonian(rham_file, 4)
NORB  = size(HR, 1)
rot_mat = init.getRotationMatrix(N_LAYERS, FLAVORS, 
                                 H0 = HR[nonzero(sum(R**2, 1)==0)[0][0]])
w = linspace(emin, emax, nbin)

# set zero self energy and convert it into the flattened format
SelfEnergy = zeros((SPINS, nbin, N_LAYERS*FLAVORS), dtype = 'c16')
SelfEnergy_rot = array([irotate(SelfEnergy[s], rot_mat)
                        for s in range(SPINS)])
SE = array([array([s.flatten() for s in SelfEnergy_rot[n]])
            for n in range(SPINS)])

# prepare the parameter dict
parms = {
        'H' : magnetic_field,
        'N_LAYERS' : N_LAYERS,
        'FLAVORS' : FLAVORS,
        'SPINS' : SPINS,
        'NORB' : NORB,
        'INTEGRATE_MOD' : 'int_donly_tilted_3bands',
        'np' : num_processes,
        }

extra = {
        'HR' : HR,
        'R' : R,
        'GaussianData' : grule(numk),
        }

# calculate the k-integral
Gavg = average_green.integrate(w+1j*broadening, double_counting, mu, SE,
                               parms, extra, parallel)
Gavg = array([rotate_all(Gavg[s], rot_mat, need_extra = True) for s in range(SPINS)])

dos = -1/pi*Gavg[0].imag 
savetxt('dos_dft_mlwf.mpi', c_[w, dos], fmt='%.6f')

