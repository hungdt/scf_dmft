import sys, os, argparse
from share_fun import load_parms, val_def, grule, readParameters;
from numpy import *;
from init import getHamiltonian, getRotationMatrix;
from average_green import integrate, averageGreen;
from functions import *;

# This is different from calculate_spectra.py 
#   no need hdf5 file
#   calculate directly from rham.py
#   can provide self energy data separately


def getGavgFromSelfEnergy(parms, se_filename = None):
    N_LAYERS = int(parms['N_LAYERS']); 
    FLAVORS  = int(parms['FLAVORS']);
    SPINS = int(parms['SPINS']);
    parms['BETA'] = 10;

    # prepare data
    NMaxFreq = int(parms['N_MAX_FREQ']);
    if se_filename is not None:
        print 'load self energy from file: ', se_filename;
        tmp = genfromtxt(se_filename);
        if NMaxFreq > len(tmp): NMaxFreq = len(tmp);
        w = tmp[:,0] + 1j*float(parms['BROADENING']);
        tmp = tmp[:, 1:];
        tmp = tmp[:NMaxFreq, 0::2] + 1j*tmp[:NMaxFreq, 1::2];
        se = zeros((SPINS, NMaxFreq, N_LAYERS*FLAVORS), dtype = complex);
        for s in range(SPINS):
            for f in range(N_LAYERS*FLAVORS):
                se[s, :, f] = tmp[:, SPINS*f+s];
    else: 
        se = zeros((SPINS, NMaxFreq, N_LAYERS*FLAVORS), dtype = complex); 
        w = linspace(float(parms['EMIN']), float(parms['EMAX']), NMaxFreq) + 1j*float(parms['BROADENING']); 

    # tight binding Hamiltonian
    HR, R = getHamiltonian(parms['RHAM'], 4);
    parms['NORB'] = len(HR[0])
    extra = { 'HR' : HR, 'R': R };
    if int(val_def(parms, 'FORCE_DIAGONAL', 0)) > 0:
        print 'FORCE_DIAGONAL is used';
        ind = nonzero(sum(R**2, 1)==0)[0][0];
        H0 = HR[ind];
    else: H0 = None;
    rot_mat = getRotationMatrix(N_LAYERS, FLAVORS, val_def(parms, 'ROT_MAT', None), H0);
 
    # prepare for k-integrate
    bp, wf = grule(int(parms['NUMK']));
    extra.update({
            'GaussianData' : [bp, wf],
            'rot_mat'      : rot_mat
            });
    delta = float(parms['DELTA']);
    mu    = float(parms['MU']);

    # running
    Gavg = averageGreen(delta, mu, w, se, parms, -1, -1, 0, extra)[1];
    # swap the Gavg to the format of my code
#    swap_vec = zeros((2, N_LAYERS*FLAVORS), dtype = int);
#    for L in range(N_LAYERS):
#        for f in range(FLAVORS): swap_vec[:,f*N_LAYERS+L] = array([f*N_LAYERS+L, L*FLAVORS+f]);
#    for s in range(SPINS): Gavg[s, :, swap_vec[1]] = Gavg[s, :, swap_vec[0]];

    spec = -1/pi * Gavg.imag;
    if SPINS == 1: savetxt('spec.dat', c_[w.real, spec[0]]);
    elif SPINS > 1: 
        savetxt('spec_up.dat', c_[w.real, spec[0]]);
        savetxt('spec_dn.dat', c_[w.real, spec[1]]);


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DMFT process.');
    parser.add_argument('-p', dest='parmsfile', type = str, required = True, help='Parameter file');
    parser.add_argument('-se', dest='sefilename', type = str, default = None, help='Matsubara frequency self energy');
    args = parser.parse_args(sys.argv[1:]);

    set_printoptions(linewidth=150, suppress=True, precision=4);
    parms = readParameters(args.parmsfile);

    getGavgFromSelfEnergy(parms, args.sefilename);

