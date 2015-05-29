import sys, os, argparse
from share_fun import load_parms, val_def, grule, readParameters;
from numpy import *;
from init import getHamiltonian, getRotationMatrix;
from average_green import integrate;
from functions import *;

# This is different from calculate_spectra.py 
#   no need hdf5 file
#   calculate directly from rham.py
#   can provide self energy data separately


def getGavgFromSelfEnergy(parms, se_filename = None):
    N_LAYERS = int(parms['N_LAYERS'])
    FLAVORS  = int(parms['FLAVORS'])
    SPINS = int(parms['SPINS'])
    beta = float(parms['BETA'])

    # prepare data
    NMaxFreq = int(round((beta*float(parms['MAX_FREQ'])/pi - 1)/2.))
    iwn = 1j * (2*arange(NMaxFreq)+1)*pi/beta
    if se_filename is not None:
        print 'load self energy from file: ', se_filename;
        tmp = genfromtxt(se_filename);
        if NMaxFreq > len(tmp): NMaxFreq = len(tmp);
        tmp = tmp[:, 1:]
        tmp = tmp[:NMaxFreq, 0::2] + 1j*tmp[:NMaxFreq, 1::2]
        se = zeros((SPINS, NMaxFreq, N_LAYERS*FLAVORS), dtype = complex)
        for s in range(SPINS):
            for f in range(N_LAYERS*FLAVORS):
                se[s, :, f] = tmp[:NMaxFreq, SPINS*f+s]
    else: 
        se = zeros((SPINS, NMaxFreq, N_LAYERS*FLAVORS), dtype = complex)
        

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
    SelfEnergy_rot = array([irotate(se[s], rot_mat) for s in range(SPINS)])
    SE = array([array([s.flatten() for s in SelfEnergy_rot[n]]) for n in range(SPINS)])
    Gavg = integrate(iwn, delta, mu, SE, parms, extra, parallel = False)
    g = array([rotate_green(Gavg, rot_mat, layer=L) for L in range(N_LAYERS)])

    return iwn, g

def rotate_green(g, rot_mat, layer=0):
    nlayers = len(rot_mat)
    nspins = size(g, 0)
    nfreq = size(g, 1)
    nflavors = size(g, 2) / nlayers

    out = zeros((nspins, nfreq, nflavors, nflavors), dtype = g.dtype)
    rot = matrix(rot_mat[layer])
    rot_inv = matrix(linalg.inv(rot))
    for s in range(nspins):
        for n in range(nfreq):
            mat = matrix(g[s, n, layer::nlayers, layer::nlayers])
            out[s, n] = rot * mat * rot_inv
    return out

def get_density_matrix(beta, g):
    nspins = size(g, 0)
    nfreq = size(g, 1)
    nflavors = size(g, 2)

    density = zeros((nspins, nflavors, nflavors))
    for s in range(nspins):
        density[s] = 2./beta*real(sum(g[s], 0)) + 0.5*eye(nflavors)
    return density


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DMFT process.');
    parser.add_argument('-p', dest='parmsfile', type = str, required = True, help='Parameter file');
    parser.add_argument('-se', dest='sefilename', type = str, default = None, help='Matsubara frequency self energy');
    args = parser.parse_args(sys.argv[1:]);

    set_printoptions(linewidth=150, suppress=True, precision=4);
    parms = readParameters(args.parmsfile);

    beta = float(parms['BETA'])
    iwn, g = getGavgFromSelfEnergy(parms, args.sefilename);
    density = []
    for ll in range(len(g)):
        density.append(get_density_matrix(beta, g[ll]).flatten())
    savetxt('density.dat', density)


