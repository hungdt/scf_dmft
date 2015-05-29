import sys, os, h5py, argparse, system_dependence as system;
from share_fun import load_parms, val_def, grule;
from numpy import *;
import maxent;
from scipy.interpolate import *;
from init import getHamiltonian, getRotationMatrix;
from average_green import averageGreen;



def getSpectraFromSelfEnergy(h5, se_filename, rham, rotmat, numk = None, setail_filename = None, it = 0):
    
    # prepare data
    w, se_refreq = ProcessSelfEnergy(se_filename, emin = -5, emax = 5, NFreq = 500);

    it = h5['iter'][0] - it;
    parms = load_parms(h5, it);
    print 'work on iteration ', it;
    if rham is not None: print 'new path for rham file is: ', rham; parms['RHAM'] = rham;
    if rotmat is not None: print 'new path for rot_mat file is ', rotmat; parms['ROT_MAT'] = rotmat;
    BETA = float(parms['BETA']);
    N_LAYERS = int(parms['N_LAYERS']);
    FLAVORS  = int(parms['FLAVORS']);
    SPINS = int(parms['SPINS']);
    NORB  = int(parms['NORB']);
    dmft_id = system.getDMFTCorrIndex(parms, all = False);
    dmft_id_len = len(dmft_id);

    # get the se tails
    tmp = h5['SolverData/selfenergy_asymp_coeffs'][:];
    se_tail = tmp[tmp[:,0] == it, 1:].reshape(SPINS, 2, -1)[:, :, ::N_LAYERS];
    if setail_filename is not None:
        print 'use the tail from external source: ', setail_filename;
        tmp = genfromtxt(setail_filename);
        se_tail[:, :, dmft_id] = array([tmp[:, s::SPINS] for s in range(SPINS)]); 
    print se_tail;

    # restore SelfEnergy
    se = zeros((SPINS, len(se_refreq), N_LAYERS*FLAVORS), dtype = complex);
    for s in range(SPINS):
        for f in range(N_LAYERS*FLAVORS):
            if f/N_LAYERS not in dmft_id: se[s,:,f] = se_tail[s, 0, f/N_LAYERS];
            else: 
                f1 = nonzero(f/N_LAYERS == dmft_id)[0][0];
                se[s, :, f] = se_refreq[:, SPINS*f1+s]*se_tail[s, 1, f/N_LAYERS] + se_tail[s, 0, f/N_LAYERS]; 

    # tight binding Hamiltonian
    if 'RHAM' in parms: 
        HR, R = getHamiltonian(parms['RHAM'], 4);
        if parms['DTYPE'] == '3bands': FLAVORS = 3;
        extra = { 'HR' : HR, 'R': R };

    # rotation matrix
    if int(val_def(parms, 'FORCE_DIAGONAL', 0)) > 0:
        print 'FORCE_DIAGONAL is used';
        ind = nonzero(sum(R**2, 1)==0)[0][0];
        H0 = HR[ind];
    else: H0 = None;
    rot_mat = getRotationMatrix(N_LAYERS, FLAVORS, val_def(parms, 'ROT_MAT', None), H0);


    # prepare for k-integrate
    parms['NUMK'] = 16 if numk is None else numk;
    bp, wf = grule(int(parms['NUMK']));
    broadening = 0.01;
    extra.update({
            'GaussianData' : [bp, wf],
            'rot_mat'      : rot_mat
            });
    delta = float(parms['DELTA']);
    mu    = float(parms['MU']);

    # running
    print 'generating interacting DOS with parameters'
    for k, v in parms.iteritems(): print '%s = %s'%(k, v);

    Gr = averageGreen(delta, mu, w+1j*broadening, se, parms, float(parms['ND']), float(parms['DENSITY']), 0, extra)[1];
    if SPINS == 1: savetxt(parms['ID']+'.idos', c_[w, -1/pi*Gr[0].imag], fmt = '%g');
    elif SPINS == 2:
        savetxt(parms['ID']+'_up.idos', c_[w, -1/pi*Gr[0].imag], fmt = '%g'); 
        savetxt(parms['ID']+'_dn.idos', c_[w, -1/pi*Gr[1].imag], fmt = '%g'); 
    
    # calculate original G(iwn), only consider one "LAYERS"
    Giwn_orig = h5['ImpurityGreen/%d'%it][:,:,::N_LAYERS];
    NMatsubara = size(Giwn_orig, 1);
    wn = (2*arange(NMatsubara) + 1)*pi/BETA;
    Giwn = zeros((NMatsubara, 2*FLAVORS*SPINS), dtype = float); # 2 for real and imag
    for f in range(FLAVORS):
        for s in range(SPINS):
            Giwn[:, 2*(SPINS*f+s)] = Giwn_orig[s, :, f].real;
            Giwn[:, 2*(SPINS*f+s)+1] = Giwn_orig[s, :, f].imag;
    savetxt(parms['ID']+'.gmat', c_[wn, Giwn]);

    # calculate G(iwn) for reference, only consider one "LAYERS"
    NMatsubara = 200;
    wn = (2*arange(NMatsubara) + 1)*pi/BETA;
    Giwn = zeros((NMatsubara, 2*FLAVORS*SPINS), dtype = float); # 2 for real and imag
    for f in range(FLAVORS):
        for s in range(SPINS):
            A = -1/pi * Gr[s, :, f*N_LAYERS].imag;
            for n in range(NMatsubara):
                tck_re = splrep(w, real(A / (1j*wn[n] - w)));
                tck_im = splrep(w, imag(A / (1j*wn[n] - w)));
                Giwn[n, 2*(SPINS*f+s)] = splint(w[0], w[-1], tck_re);
                Giwn[n, 2*(SPINS*f+s)+1] = splint(w[0], w[-1], tck_im);
    savetxt(parms['ID']+'.gmat.ref', c_[wn, Giwn]);


def ProcessSelfEnergy(se_filename, emin = None, emax = None, NFreq = None, delta = None):
    # remove irrelevant energy ranges
    se_refreq = genfromtxt(se_filename);
    min_id = 0;
    max_id = len(se_refreq);
    tmp = sum(se_refreq[:, 2::2], 1);
    min_spec_val = 3e-3;
    print 'min = %g, max = %g'%(abs(max(tmp)), abs(min(tmp)));
#    if min_spec_val > abs(min(tmp)) or min_spec_val < abs(max(tmp)): exit('min_spec_val=%g is out of range'%min_spec_val)
    stop1 = False; stop2 = False;
    L = len(se_refreq);
    for i in xrange(L):
        if abs(tmp[i]) > min_spec_val: stop1 = True; min_id = max(0,i-1);
        if abs(tmp[L-1-i]) > min_spec_val: stop2 = True; max_id = L-i;
        if stop1 and stop2: break;
    id_emin = inf; id_emax = -inf;
    w = se_refreq[:,0];
    for i in xrange(L):
        if emin is not None:
            if w[i] < emin: id_emin = i;
        if emax is not None:
            if w[L-1-i] > emax: id_emax = L-1-i;
    if id_emin is not None: min_id = id_emin;
    if id_emax is not None: max_id = id_emax;

    se_refreq0 = se_refreq[min_id:max_id, :];
    w0 = se_refreq0[:,0];
    w = linspace(w0[0], w0[-1], NFreq);
    se_refreq = None;
    for f in range((size(se_refreq0, 1)-1)/2):
        tck_re = splrep(w0, se_refreq0[:, 1+2*f])
        tck_im = splrep(w0, se_refreq0[:, 1+2*f+1]);
        if se_refreq is None: se_refreq = splev(w, tck_re) + 1j*splev(w, tck_im);
        else: se_refreq = c_[se_refreq, splev(w, tck_re) + 1j*splev(w, tck_im)];

    print 'get self energy from file %s'%se_filename
    print 'total number of frequencies considered is %d from wmin=%.4f to wmax=%.4f'%(NFreq, w[0], w[-1])
    return w, se_refreq;


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DMFT process.');
    parser.add_argument('-f', dest='h5filename', type = str, default = None, help='Input HDF5 file');
    parser.add_argument('-g', dest='h5groupname', type = str, default = None, help='HDF5 group name');
    parser.add_argument('-rham', dest='rham', type = str, default = None, help='Tight binding Hamiltonian');
    parser.add_argument('-rotmat', dest='rotmat', type = str, default = None, help='Rotation matrix');
    parser.add_argument('-nk', dest='numk', type = int, default = None, help='Number of kpoints');
    parser.add_argument('-se', dest='sefilename', type = str, default = None, help='Real frequency self energy (normalized to 1)');
    parser.add_argument('-se_tail', dest='setail_filename', type = str, default = None, help='Self eneryg tail (up to 1/iwn)');
    parser.add_argument('-iter', dest='it', type = int, default = 0, help='Iteration number for continuation');
    args = parser.parse_args(sys.argv[1:]);

    if not os.path.isfile(args.sefilename): 
        exit('Spectra file %s not found'%args.sefilename);
    try: h5file = h5py.File(args.h5filename, 'r');
    except: exit('Unable to load hdf5 file %s'%args.h5filename);

    if args.h5groupname is not None:
        if args.h5groupname not in h5file: 
            h5file.close();
            exit('No group %s'%args.h5groupname);
        else: h5 = h5file[args.h5groupname];
    else: h5 = h5file[h5file.keys()[0]];

    set_printoptions(linewidth=150, suppress=True, precision=4);
    getSpectraFromSelfEnergy(h5, args.sefilename, args.rham, args.rotmat, args.numk, args.setail_filename, args.it);

