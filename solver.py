import os, h5py;
import user_config, solver_types, cppext;
import system_dependence as system;

from numpy import *;
from functions import getDensity, get_asymp_hybmat, smooth_selfenergy, get_asymp_selfenergy, assign;
from share_fun import val_def, log_data, load_parms, save_data;


def run_solver(AvgDispersion, nf, w, it, parms, aWeiss, np = 1, VCoulomb = None):
    ID        = parms["ID"];
    N_LAYERS  = int(parms["N_LAYERS"]); FLAVORS = int(parms["FLAVORS"]); SPINS = int(parms['SPINS']);
    DATA_FILE = parms["DATA_FILE"];
    TMPH5FILE = "." + DATA_FILE + ".id" + str(ID) + ".i" + str(it) + ".solver_out.h5";
    if VCoulomb is None: VCoulomb = zeros(N_LAYERS);
    solver    = solver_types.init_solver(parms, np);
    corr_id   = system.getCorrIndex(parms);
    NCOR      = int(parms['NCOR']);
    NDMFT     = 2*len(system.getDMFTCorrIndex(parms));    # 2 for SPINS

    # check save point and initialize for new iteration
    try: 
        tmph5 = h5py.File(TMPH5FILE, 'r+');
        hyb_tau = tmph5['Hybtau'][:];
        hyb_mat = tmph5['Hybmat'][:];
        hyb_coefs = tmph5['hyb_asym_coeffs'][:].reshape(SPINS, -1, NCOR);
    except: 
        try: tmph5.close();
        except: pass;
        tmph5 = h5py.File(TMPH5FILE, 'w');
        tmph5.create_dataset('L', (2,), dtype = int, data = array([it, 0]));

        # asymptotic coefficients, upto 3rd order for hyb
        hyb_coefs = zeros((SPINS, 3, NCOR), dtype = float);
        # electric chemical potential
        eMU = float(parms['MU']) - VCoulomb;
        for L in range(N_LAYERS): 
            hyb_coefs[:, :, L::N_LAYERS] = get_asymp_hybmat(parms, nf[:, L::N_LAYERS], eMU[L], AvgDispersion[:, :, corr_id[L:NCOR:N_LAYERS]]);

        # get practical hybmat, and hybtau
        Eav = AvgDispersion[:, 0, corr_id];
        hyb_mat = zeros((SPINS, int(parms['N_MAX_FREQ']), NCOR), dtype = complex);
        hyb_tau = zeros((SPINS, int(parms['N_TAU'])+1, NCOR), dtype = float); 
        for s in range(SPINS):
            for f in range(NCOR):
                hyb_mat[s, :, f] = w + eMU[f%N_LAYERS] - Eav[s, f] - aWeiss[s, :, f];
                tmp = cppext.IFT_mat2tau(hyb_mat[s, :,f].copy(), int(parms['N_TAU'])+1, float(parms['BETA']), float(hyb_coefs[s, 0, f]), float(hyb_coefs[s, 1, f]));

                # set value >= 0 to be smaller than 0, the mean of left and right neighbors
                ind = nonzero(tmp >= 0)[0];
                for i in ind:
                    lefti = righti = i;
                    while tmp[lefti] >= 0 and lefti > 0: lefti -= 1;
                    while tmp[righti] >= 0 and righti < len(tmp)-1: righti += 1;
                    leftval = tmp[lefti] if tmp[lefti] < 0 else 0;
                    rightval = tmp[righti] if tmp[righti] < 0 else 0;
                    tmp[i] = (leftval + rightval)/2.;
                hyb_tau[s, :, f] = tmp;
        tmph5.create_dataset("Hybmat", hyb_mat.shape, dtype = complex, data = hyb_mat);
        tmph5.create_dataset("Hybtau", hyb_tau.shape, dtype = float  , data = hyb_tau);
       
        # initialize output dataset
        Gtau_shape = (int(parms['N_TAU'])+1, NDMFT); 
        tmph5.create_dataset("Gtau", Gtau_shape, dtype = float, data = zeros(Gtau_shape, dtype = float));
        tmph5.create_group("Observables");
        tmph5.create_dataset('hyb_asym_coeffs', hyb_coefs.flatten().shape, dtype = float, data = hyb_coefs.flatten());

    # run
    hyb_data = [hyb_tau, hyb_mat, hyb_coefs];
    MEASURE_freq = True if 'Gw' in tmph5 else False;
    startL = tmph5['L'][1];
    sym_layers = getSymmetricLayers(tmph5, parms);
    for L in range(startL, N_LAYERS):
        print "Processing task ", ID, ": iteration ", it, ", layer ", L;
        tmph5['L'][1] = L;
        TMPFILE = "." + DATA_FILE + ".id" + str(ID) + ".i" + str(it) + ".L" + str(L);

        if float(parms['U']) == 0: break;

        if (sym_layers is None) or (L not in sym_layers[:, 1]):
            solver.prepare(TMPFILE, solver_input_data(parms, L, hyb_data, AvgDispersion, VCoulomb, nf));
            tmph5.close();
            ret_val = solver.run();
            tmph5 = h5py.File(TMPH5FILE, 'r+');
            if ret_val > 0:
                print "Not finish running impurity solver or problem occurs while running the solver.";
                os.system('rm ' + TMPFILE + '.*');
                tmph5.close();
                return None;
            solver_out = solver.collect();
            if solver_out is None: tmph5.close(); return None;
            Gtau = solver_out[0]; obs = solver_out[1];
            if len(solver_out) > 2: 
                MEASURE_freq = True; Giwn = solver_out[2]; Siwn = solver_out[3];
            os.system('rm ' + TMPFILE + '.*');

        elif L in sym_layers[:, 1]: # symmetric layer, no need to calculate
            sym_index = nonzero(sym_layers[:, 1] == L)[0];
            sym_L = sym_layers[sym_index, 0][0];
            print "L=%d is the symmetric layer of layer L=%d"%(L,sym_L);
            Gtau = tmph5['Gtau'][:, sym_L::N_LAYERS];
            obs = None;
            if tmph5['Observables'].keys() != []:
                obs = dict();
                for k, v in tmph5["Observables/L"+str(sym_L)].iteritems(): obs[k] = v;
            if MEASURE_freq:
                Giwn = tmph5['Gw'][:, sym_L::N_LAYERS];
                Siwn = tmph5['Sw'][:, sym_L::N_LAYERS];

        tmph5['Gtau'][:, L::N_LAYERS] = Gtau;
        if MEASURE_freq:
            if 'Gw' not in tmph5:
                matsubara_shape = (len(Giwn), NDMFT);
                tmph5.create_dataset("Gw", matsubara_shape, dtype = complex, data = zeros(matsubara_shape, dtype = complex));
                tmph5.create_dataset("Sw", matsubara_shape, dtype = complex, data = zeros(matsubara_shape, dtype = complex));
            tmph5['Gw'][:, L::N_LAYERS] = Giwn;
            tmph5['Sw'][:, L::N_LAYERS] = Siwn;

        if obs is not None:
            new_group_str = "Observables/L"+str(L);
            tmph5.create_group(new_group_str);
            for k, v in obs.iteritems(): tmph5.create_dataset(new_group_str+"/"+k, v.shape, dtype = v.dtype, data = v);

        print "Finish iteration ", it, ", layer ", L, "\n";
    print "DONE: iteration %d\n"%it;
    tmph5['L'][1] = N_LAYERS;
    tmph5.close();
    return TMPH5FILE;


def solver_input_data(parms, L, hyb_data_all, AvgDispersion, VCoulomb, nf):
    # prepare hybtau file for CTQMC
    N_LAYERS =int(parms['N_LAYERS']); FLAVORS = int(parms['FLAVORS']); SPINS   = int(parms['SPINS']);
    corr_id = system.getCorrIndex(parms);
    dmft_id = system.getDMFTCorrIndex(parms);
    dmft_FLAVORS = 2*len(dmft_id)/N_LAYERS; # 2 for SPINS

    hyb_data = [];
    for n, d in enumerate(hyb_data_all):
        data = hyb_data_all[n][:, :, dmft_id[L::N_LAYERS]];
        data_out = zeros((size(data,1), dmft_FLAVORS), dtype = data.dtype);
        data_out[:, ::2] = data[0];
        data_out[:,1::2] = data[0] if SPINS == 1 else data[1];
        hyb_data.append(data_out);

    Eav = AvgDispersion[:, 0, dmft_id[L::N_LAYERS]];
    inertHFSelfEnergy = get_inert_band_HF(parms, nf[:, L::N_LAYERS]);

    MU = array([float(parms['MU']) - VCoulomb[L] - Eav[s] - inertHFSelfEnergy[s] for s in range(SPINS)]);
    MU_out = zeros(dmft_FLAVORS);
    MU_out[::2] = MU[0]; MU_out[1::2] = MU[0] if SPINS == 1 else MU[1];
    parms_copy = parms.copy();
    parms_copy['FLAVORS'] = dmft_FLAVORS; 

    print 'Inert HF Self Energy: ', inertHFSelfEnergy;
    return {'hybtau' : hyb_data[0], 'hybmat' : hyb_data[1], 'hybtail' : hyb_data[2],
            'MU' : MU_out, 'parms' : parms_copy};


def solver_post_process(parms, aWeiss, h5, tmph5filename):
    N_LAYERS = int(parms["N_LAYERS"]); FLAVORS = int(parms["FLAVORS"]); NCOR = int(parms['NCOR']);
    SPINS = 2; # NOTE: for collecting all spins, symmetrize them later if neccessary
    if len(aWeiss) == 1: aWeiss = r_[aWeiss, aWeiss]; # SPINS = 1 case
    dmft_id = system.getDMFTCorrIndex(parms);

    if not os.path.isfile(tmph5filename): print >> sys.stderr, 'File %s not found'%tmph5filename; return None;
    tmph5 = h5py.File(tmph5filename, 'r');
    if tmph5['L'][1] < N_LAYERS: print >> sys.stderr, 'Unfinish solving the impurity model'; return None;

    # save data from temporary file
    h5solver = h5['SolverData'];
    it = tmph5['L'][0];
    MEASURE_freq = True if 'Gw' in tmph5 else False;

    for s in tmph5['Observables']:
        new_group_str = 'Observables/%d/%s'%(it, s);
        for k in tmph5['Observables/%s'%s]:
            v = tmph5['Observables/%s/%s'%(s, k)];
            try: h5solver.create_dataset(new_group_str+"/"+k, v.shape, dtype = v.dtype, data = v);
            except: h5solver[new_group_str+"/"+k][:] = v;

    Gmat = zeros((SPINS, int(parms['N_MAX_FREQ']), NCOR), dtype = complex);
    Smat = zeros((SPINS, int(parms['N_MAX_FREQ']), NCOR), dtype = complex);
    Ntau = max(int(parms['N_TAU'])/20, 400) + 1;
    Htau = tmph5['Hybtau'][:, ::20, :];

    # the updated density: for DMFT bands, get from Gtau, for inert bands, get from Gavg of previous iteration
    nf = h5['log_density'][0 if int(val_def(parms, 'FIXED_HARTREE', 0)) > 0 else it-1, 4:].reshape(-1, NCOR+1);
    if len(nf) == 1: nf = r_[nf, nf];
    nf = nf[:, :NCOR];
    nf[:, dmft_id] = -assign(tmph5['Gtau'][-1, :], N_LAYERS);

    # get raw Gmat and Smat
    for f in range(size(tmph5['Gtau'], 1)):
        g = cppext.FT_tau2mat(tmph5['Gtau'][:, f].copy(), float(parms['BETA']), int(parms['N_MAX_FREQ']))
        try: tmp = c_[tmp, g];
        except: tmp = g.copy();
    Gmat[:, :, dmft_id] = assign(tmp, N_LAYERS);
    Smat[:, :, dmft_id] = aWeiss[:, :, dmft_id] - 1/Gmat[:, :, dmft_id];
    if MEASURE_freq:
        nfreq = size(tmph5['Gw'][:], 0);
        Gmat[:, :nfreq, dmft_id] = assign(tmph5['Gw'], N_LAYERS);
        Stmp = assign(tmph5['Sw'], N_LAYERS);
        # adjust self energy measured using improved estimator 
        # with contribution from inertial d-bands
        for L in range(N_LAYERS):
            SE_inert = get_inert_band_HF(parms, nf[:, L::N_LAYERS]);
            Stmp[0, :, L::N_LAYERS] += SE_inert[0]; 
            Stmp[1, :, L::N_LAYERS] += SE_inert[1];
        Smat[:, :nfreq, dmft_id] = Stmp;

    # symmetrize orbital and spin if necessary
    paraorb = [int(s) for s in val_def(parms, 'PARAORBITAL', '').split()];
    if len(paraorb) == 1:
        if paraorb[0] > 0: 
            if parms['DTYPE'] == '3bands': paraorb = [[0, 1, 2]];  # t2g only HARD CODE
            else: paraorb = [[0,3], [1,2,4]];   # t2g and eg HARD CODE
        else: paraorb = [];
    if len(paraorb) > 0:
        if type(paraorb[0]) != list: paraorb = [paraorb];
        print 'Symmetrize over orbital ', paraorb;
        for L in range(N_LAYERS):
            for s in range(SPINS):
                for sym_bands in paraorb:
                    gm = zeros(size(Gmat, 1), dtype = complex);
                    sm = zeros(size(Smat, 1), dtype = complex);
                    nf_tmp = 0.;
                    for f in sym_bands:
                        gm += Gmat[s, :, L + f*N_LAYERS];
                        sm += Smat[s, :, L + f*N_LAYERS];
                        nf_tmp += nf[s,  L + f*N_LAYERS];
                    for f in sym_bands:
                        Gmat[s, :, L + f*N_LAYERS] = gm / float(len(sym_bands));
                        Smat[s, :, L + f*N_LAYERS] = sm / float(len(sym_bands));
                        nf[s, L + f*N_LAYERS]  = nf_tmp / float(len(sym_bands));
    if int(parms['SPINS']) == 1:
        print 'Symmetrize over spins';
        Gmat = array([mean(Gmat, 0)]);
        Smat = array([mean(Smat, 0)]);
        nf   = array([mean(nf, 0)]);

    # smooth Gmat and Smat
    SPINS = int(parms['SPINS']);
    Smat = smooth_selfenergy(it, h5, Smat, nf);
    NCutoff = int(parms['N_CUTOFF']);
    Gmat[:, NCutoff:, :] = 1. / (aWeiss[:SPINS, NCutoff:, :] - Smat[:, NCutoff:, :]);

    # calculate Gtau from Gmat (after symmtrization)
    Gtau = zeros((SPINS, Ntau, NCOR), dtype = float);
    S0 = zeros((SPINS, NCOR));
    for L in range(N_LAYERS): S0[:, L::N_LAYERS] = get_asymp_selfenergy(parms, nf[:, L::N_LAYERS])[:,0,:];
    for s in range(SPINS):
        for f in range(NCOR):
            if f not in dmft_id:
                Gmat[s, :, f] = 1. / (aWeiss[s, :, f] - S0[s, f]);
                Smat[s, :, f] = S0[s, f];
            Gtau[s, :, f] = cppext.IFT_mat2tau(Gmat[s, :, f].copy(), Ntau, float(parms['BETA']), 1.0, 0.0);

    Gtau[:, 0, :] = -(1.-nf);
    Gtau[:,-1, :] = -nf;

    # saving data
    dT = 5; Nb2 = size(tmph5['Gtau'], 0) / 2;
    Gb2 = array([mean(tmph5['Gtau'][Nb2-dT:Nb2+dT, f], 0) for f in range(size(tmph5['Gtau'], 1))]);
    log_data(h5solver, 'log_Gbeta2', it, Gb2.flatten(), data_type = float);
    log_data(h5solver, 'log_nsolve', it, -tmph5['Gtau'][-1, :].flatten(), data_type = float);
    log_data(h5solver, 'hyb_asym_coeffs', it, tmph5['hyb_asym_coeffs'][:].flatten(), data_type = float);
    save_data(h5solver, it, ('Gtau', 'Hybtau', 'Hybmat'), (Gtau, Htau, tmph5['Hybmat'][:]));
    tmph5.close(); del tmph5;
    os.system('rm %s'%tmph5filename);
    return Gmat, Smat;


def getSymmetricLayers(tmph5, parms):
    if int(val_def(parms, 'USE_LAYER_SYMMETRY', 0)) == 0: return None;
    if not 'sym_layers' in tmph5:
        sym_layers = system.calc_sym_layers(parms);
        if len(sym_layers) > 0: tmph5.create_dataset("sym_layers", sym_layers.shape, dtype = sym_layers.dtype, data = sym_layers);
        else: return None;
    else: sym_layers = tmph5['sym_layers'][:];
    return sym_layers;


def get_inert_band_HF(parms, nf):
    FLAVORS = int(parms['FLAVORS']); SPINS = 2;
    assert(size(nf, 1) == FLAVORS);
    dmft_id = system.getDMFTCorrIndex(parms, all = False);
    inert_id = array([s for s in range(FLAVORS) if s not in dmft_id]);
    U = float(parms['U']); J = float(parms['J']);
    if len(nf) == 1: nf = r_[nf, nf];
    ret = zeros(SPINS);
    for s in range(SPINS):
        for f in inert_id: ret[s] += (U-2*J)*nf[not s, f] + (U-3*J)*nf[s, f];

    if int(val_def(parms, 'MEAN_FIELD_UNPOLARIZED', 0)) > 0: ret = ones(SPINS)*mean(ret);
    return ret;

