# demo: generate the noninteracting DOS from the Hamiltonian produced by DFT+MLWF

import sys
# directory for the main code
sys.path.append('..')

from int_donly_tilted_3bands import calc_Gavg
from numpy import *
from functions import *
from share_fun import grule
import init

N_LAYERS = 4
FLAVORS  = 3
SPINS = 1

mu = 15.45
double_counting = 0.

nbin = 300
emin = -1.5
emax = 1.5
broadening = 0.02
magnetic_field = 0.
nthreads = 8

rham_file = "lto_rham.py"  # the file for the Hamiltonian produced by DFT+MLWF

HR, R = init.getHamiltonian(rham_file, 4)
NORB  = size(HR, 1)
rot_mat = init.getRotationMatrix(N_LAYERS, FLAVORS, 
                                 H0 = HR[nonzero(sum(R**2, 1)==0)[0][0]])

numk = 26
bp, wf = grule(numk)       # the Gaussian points and weights

w = linspace(emin, emax, nbin)
SelfEnergy = zeros((SPINS, nbin, N_LAYERS*FLAVORS), dtype = 'c16')
# convert self energy to the C++ form
SelfEnergy_rot = array([irotate(SelfEnergy[s], rot_mat) for s in range(SPINS)])
SE = array([array([s.flatten() for s in SelfEnergy_rot[n]]) for n in range(SPINS)])
Gavg = array([calc_Gavg(w+1j*broadening, double_counting, mu, 
                        SE[n].copy(), HR, R, magnetic_field*(-1)**n, 
                        bp, wf, nthreads).reshape(nbin, NORB, NORB) 
                        for n in range(SPINS)])

# swap the Gavg to the format of the main code
swap_vec = zeros((2, N_LAYERS*FLAVORS), dtype = int)
for L in range(N_LAYERS):
    for f in range(FLAVORS): swap_vec[:,f*N_LAYERS+L] = array([f*N_LAYERS+L, L*FLAVORS+f])
for s in range(SPINS):
    for n in range(len(Gavg[s])):
        Gavg[s, n,:,swap_vec[0]] = Gavg[s, n, :, swap_vec[1]]
        Gavg[s, n,swap_vec[0],:] = Gavg[s, n, swap_vec[1], :]
Gavg = array([rotate_all(Gavg[s], rot_mat, need_extra = True) for s in range(SPINS)])

dos = -1/pi*Gavg[0].imag 
savetxt('dos_dft_mlwf.out', c_[w, dos], fmt='%.6f')

