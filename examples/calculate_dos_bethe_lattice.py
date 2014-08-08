# demo: generate the noninteracting DOS for the Bethe lattice

import sys
# directory for the main code
sys.path.append('..')

from int_bethe_lattice import *
from numpy import *
from share_fun import grule
from functions import rotate_all

nflavors = 3

emin = -3
emax = 3
nbin = 500
broadening = 0.005

double_counting = 0.
mu = 0.
magnetic_field  = 0.0
bp, wf = grule(1000)

w = linspace(emin, emax, nbin) + 1j*broadening
self_energy = zeros((nbin, nflavors*nflavors), dtype = 'c16')
quarter_bandwidth = array([1.,0])

Gavg = calc_Gavg(w, double_counting, mu, self_energy, quarter_bandwidth, 
                 magnetic_field, bp, wf, 0).reshape(nbin, nflavors, nflavors)
G = rotate_all(Gavg, [matrix(eye(nflavors))])
dos = -1/pi*G.imag
savetxt('dos_bethe_lattice.out', c_[w.real, dos])
