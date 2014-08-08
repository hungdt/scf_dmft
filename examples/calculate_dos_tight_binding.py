# demo: generate the noninteracting DOS from the tight binding parameters

import sys
# directory for the main code
sys.path.append('..')

from int_2dhopping import *
from numpy import *
from share_fun import grule
from functions import rotate_all

nflavors = 3

nbin = 500
emin = -6.
emax = 6.
broadening = 0.05

mu = 0.
double_counting = 0.        # unused in models containing only correlated orbitals

magnetic_field  = 0.0
numk = 100
bp, wf = grule(numk)        # the Gaussian points and weights

w = linspace(emin, emax, nbin) + 1j*broadening
self_energy = zeros((nbin, 1, nflavors, nflavors), dtype = complex)
self_energy = array([s.flatten() for s in self_energy], dtype = complex)
tight_binding_parameters = array([1.,0.3])

Gavg = calc_Gavg(w, double_counting, mu, self_energy, tight_binding_parameters, 
                 magnetic_field, bp, wf, 0).reshape(nbin, nflavors, nflavors)
G = rotate_all(Gavg, [matrix(eye(nflavors))])
dos = -1/pi*G.imag
savetxt('dos_tight_binding.out', c_[w.real, dos], fmt='%.6f')
