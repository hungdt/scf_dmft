#!/usr/bin/env python

import sys, os, h5py;
import hartree, rotation_matrix, system_dependence as system;

from numpy import *;
from average_green import averageGreen;
from functions import *;
from share_fun import *;


def getHamiltonian(rham_file, dist = inf):
    # get DFT+MLWF Hamiltonian
    if not os.path.isfile(rham_file): sys.exit("Hamiltonian file %s not found..."%rham_file);
    execfile(rham_file)
    exec("rham = Hopping.copy(); del Hopping;");
    for k in rham.keys():
        if linalg.norm(array(k)) > dist: rham.pop(k)
    nr = len(rham);
    NORB = len(rham[(0,0,0)]);
    R  = zeros((nr, 3), dtype = 'i4');
    HR = zeros((nr, NORB, NORB), dtype = complex);
    for i,ii in enumerate(sorted(rham.keys())):
        R[i, :] = array(ii);
        HR[i, :, :] = array(rham[ii]);
    return HR, R;


def getRotationMatrix(N_LAYERS, FLAVORS, rot_mat_file = None, H0 = None):
    rot_mat = [];
    if rot_mat_file is not None: rot_mat = rotation_matrix.generate_transform_mat(rot_mat_file)[2]; # for l=2
    elif H0 is not None:
        for c in range(N_LAYERS):
            h = H0[c*FLAVORS:(c+1)*FLAVORS, c*FLAVORS:(c+1)*FLAVORS];
            d, v = linalg.eig(h);
            rot_mat.append(mat(v).H);
    else:
        for c in range(N_LAYERS): rot_mat.append(mat(eye(FLAVORS)));
    return rot_mat;


def initialize(h5file, parms):
    if parms['ID'] in h5file.keys():
        # group exists
        h5 = h5file[parms['ID']];
        it = h5["iter"][0];
        if it < 0: del h5file[parms['ID']];
        else:
            parms1 = parms.copy();
            # not allow to change these fixed parameters
            for s in ('DELTA', 'U', 'J', 'UJRAT', 'BETA'):
                if s in parms1: del parms1[s];
            parms = load_parms(h5, it+1);
            for k, v in parms1.iteritems():
                try: 
                    if parms[k] != parms1[k]: parms[k] = parms1[k];
                except: parms[k] = parms1[k];
        
    # NOTE: basic information for the structure
    N_LAYERS = int(parms['N_LAYERS']);
    SPINS    = int(val_def(parms, 'SPINS', 2));
    if int(val_def(parms, 'PARAMAGNET', 0)) > 0: SPINS = 1;
    DENSITY  = N_LAYERS*float(parms['DENSITY']);
    BETA     = float(parms["BETA"]);
    Nd       = N_LAYERS*float(val_def(parms, 'ND', -1));
    if 'DTYPE' not in parms: parms['DTYPE'] = '';
    if 'RHAM' in parms: 
        FLAVORS = 5;   # 5 d-bands
        HR, R = getHamiltonian(parms['RHAM'], 4); # max distance is 4
        NORB = len(HR[0]);
        if parms['DTYPE'] == '3bands': FLAVORS = 3;
    else: 
        FLAVORS = int(parms['FLAVORS']);
        NORB = int(parms['NORB']);

    if int(val_def(parms, 'AFM', 0)) > 0: 
        print 'This is AFM self consistent loop!';
        if SPINS == 1: exit('SPINS must be 2 for AFM calculation');

    if int(val_def(parms, 'FORCE_DIAGONAL', 0)) > 0:
        print 'FORCE_DIAGONAL is used';
        ind = nonzero(sum(R**2, 1)==0)[0][0];
        H0 = HR[ind];
    else: H0 = None;
    rot_mat = getRotationMatrix(N_LAYERS, FLAVORS, val_def(parms, 'ROT_MAT', None), H0);


    if 'MAX_FREQ' in parms.keys(): parms['N_MAX_FREQ'] = int(round((BETA*float(parms['MAX_FREQ'])/pi - 1)/2.));
    if 'N_CUTOFF' not in parms.keys(): 
        cutoff_factor = float(val_def(parms, 'CUTOFF_FACTOR', 7));
        parms['N_CUTOFF'] = int(round((BETA/pi*float(val_def(parms,'CUTOFF_FREQ', cutoff_factor*float(parms['U']))) - 1)/2.));
    parms['CUTOFF_FREQ'] = (2*int(parms['N_CUTOFF'])+1)*pi/BETA;
    parms['MAX_FREQ'] = (2*int(parms['N_MAX_FREQ'])+1)*pi/BETA;
    parms['FLAVORS']  = FLAVORS;
    
    # mixing
    mixer = Mixing(float(val_def(parms, "MIXING", 1)), int(val_def(parms, 'MIXING_FIXED', 1)));
    mixer_SE = Mixing(float(val_def(parms, "MIXING", 1)), int(val_def(parms, 'MIXING_FIXED', 1)));
    
    wn = (2*arange(int(parms['N_MAX_FREQ']))+1)*pi/BETA;
    extra = {
            'correction'   : zeros(SPINS),
            'GaussianData' : grule(int(val_def(parms, 'NUMK', 30))),
            'rot_mat'      : rot_mat
            };
    if 'RHAM' in parms: extra.update({'HR' : HR, 'R' : R });
    else: extra.update({ 'tight_binding_parms' : array([float(s) for s in parms['TB_PARMS'].split()]) });

    corr_id = system.getCorrIndex(parms);
    NCOR = len(corr_id);
    
    
    if not parms['ID'] in h5file.keys():
        it = 0;
    
        # create main group ID and its subgroups
        h5 = h5file.create_group(parms['ID']);
        h5.create_dataset("iter", (1,), int, data = -1);
    
        crt_tuple = ("ImpurityGreen", "avgGreen", "SelfEnergy", "WeissField", "parms", "StaticCoulomb",
                "SolverData", "SolverData/Gtau", "SolverData/Hybmat", "SolverData/Hybtau", "SolverData/Observables");
        for obj in crt_tuple: h5.create_group(obj);
    
        parms['SPINS']    = SPINS;
        parms['NORB']     = NORB;
        parms['ND']       = Nd/float(N_LAYERS);
        parms['NCOR']     = NCOR;
        parms['N_TAU']    = val_def(parms, 'N_TAU', 400);
        if 'UJRAT' in parms.keys(): parms['J'] = float(parms['U']) / float(parms['UJRAT']);
    
       
        # generate initial conditions
        if 'USE_DATAFILE' in parms.keys():
            print 'Get initial data from file ' + parms['USE_DATAFILE'];
            is_hdf5 = True
            if os.path.abspath(parms['USE_DATAFILE']) != os.path.abspath(parms['DATA_FILE']):
                try: 
                    g5file = h5py.File(parms['USE_DATAFILE'], 'r');
                    g5 = g5file[val_def(parms, 'USE_DATAFILE_ID', g5file.keys()[0])];
                except: is_hdf5 = False
            else:
                g5file = None;
                g5     = h5file[val_def(parms, 'USE_DATAFILE_ID', h5file.keys()[0])];
            if is_hdf5:
                g5it = g5['iter'][0];
                parms['MU'] = val_def(parms, 'MU0', str(g5['parms/%d/MU'%(g5it+1)][...]));
                parms['DELTA'] = val_def(parms, 'DELTA', str(g5['parms/%d/DELTA'%(g5it+1)][...]));
                SelfEnergy, se_coef = get_self_energy_hdf5(g5, parms, wn)
                if not g5file is None: g5file.close();
                del g5file, g5;
            else:
                parms['MU'] = val_def(parms, 'MU0', 0);
                parms['DELTA'] = val_def(parms, 'DELTA', 0);
                SelfEnergy, se_coef = get_self_energy_text(parms['USE_DATAFILE'], parms, wn)
        else:
            SelfEnergy = zeros((SPINS, int(parms["N_MAX_FREQ"]), NCOR), dtype = complex);
            if int(val_def(parms, 'HARTREE_INIT', 0)) == 1: 
                delta, mu, nf, VCoulomb = hartree.HartreeRun(parms, extra);
                nf = nf[:, corr_id];
                parms['MU'] = mu; parms['DELTA'] = delta;
            else:
                mu    = float(val_def(parms, 'MU0', 0)); # don't know which default MU is good
                delta = float(val_def(parms, 'DELTA', 0));
                parms['MU'] = mu; parms['DELTA'] = delta;
                Gavg, Gavg0, delta, mu, VCoulomb = averageGreen(delta, mu, 1j*wn, SelfEnergy, parms, Nd, DENSITY, False, extra);
                nf = getDensityFromGmat(Gavg, BETA, extra);
            se_coef = zeros((SPINS, 2, NCOR), dtype = float);
            for L in range(N_LAYERS): se_coef[:, :, L::N_LAYERS] = get_asymp_selfenergy(parms, nf[:, L::N_LAYERS]);
            for s in range(SPINS):
                for f in range(NCOR): SelfEnergy[s, :, f] = se_coef[s, 0, f];
            if int(val_def(parms, 'NO_TUNEUP', 0)) == 0: parms['MU'] = mu; parms['DELTA'] = delta;

        log_data(h5['SolverData'], 'selfenergy_asymp_coeffs', it, se_coef.flatten(), data_type = float);
        save_data(h5, it, ['SelfEnergy'], [SelfEnergy]);
        save_parms(h5, it,   parms);
        save_parms(h5, it+1, parms);

        # get average dispersion up to nth order
        NthOrder = 3;
        dispersion_avg = system.getAvgDispersion(parms, NthOrder, extra);
        h5.create_dataset('SolverData/AvgDispersion', dispersion_avg.shape, dtype = float, data = dispersion_avg);

        h5["iter"][...] = it; # this is the mark that iteration 'it' is done

    return {
            'parms'   : parms,
            'h5'      : h5,
            'mixer'   : mixer,
            'mixer_SE': mixer_SE,
            'extra'   : extra,
            'N_LAYERS': N_LAYERS,
            'NCOR'    : NCOR,
            'Nd'      : Nd,
            'DENSITY' : DENSITY,
            'wn'      : wn,
            'corr_id' : corr_id
            }


if __name__ == '__main__':
    # read input arguments
    parms, np, parms_file = getOptions(sys.argv[1:]);
    
    # create or open database (HDF5)
    h5file = h5py.File(parms['DATA_FILE'], 'a');

    vars_dict = initialize(h5file, parms);
    
