#!/usr/bin/env python

import h5py, sys, re, system_dependence as system;

from numpy import *;
from matplotlib.pyplot import *;
from functions import *;
from share_fun import *;
from average_green import averageGreen;


def plot_data(h5, st, iter = None, L = None, f = None, 
        noshow = True, part = None, style = '', 
        legend_loc = 0, is_subplot = True, 
        all_spins = True, average_spin = False,
        xlimit = None):
    if iter == None: iter = h5['iter'][0];
    parms = gget(h5, 'parms', iter);
    N_LAYERS = int(parms['N_LAYERS']); SPINS = int(parms['SPINS']);
    y = gget(h5, st, iter, L, f);
    is_matsubara = True;
    if average_spin: 
        all_spins = False
        y = array([mean(y, 0)])


    if st == 'Gtau' or st == 'hybtau':
        x = linspace(0, 1, size(y, 1));
        is_matsubara = False;
    else: x = gget(h5, 'wn')[:size(y,1)];

    idx = gget(h5, 'idx', L = L, f = f);
    legend_str = [];
    for i in idx:
        layer = int(i) % N_LAYERS;
        flavor = int(i) / N_LAYERS;
        legend_str.append('L=' + str(layer) + ';F=' + str(flavor) + ';it=' + str(iter));

    xlimit = None;
    if is_matsubara:
        xlb = 'Frequencies';
        if 'CUTOFF_FREQ' not in parms: parms['CUTOFF_FREQ'] = (2*int(parms['N_CUTOFF'])+1)*pi/float(parms['BETA'])
        if 'MAX_FREQ' not in parms: parms['MAX_FREQ'] = (2*int(parms['N_MAX_FREQ'])+1)*pi/float(parms['BETA'])
        if xlimit is None:
            xlimit = [0, min(float(parms['CUTOFF_FREQ'])*5/4., float(parms['MAX_FREQ']))];
        if part == None:
            if is_subplot: subplot(1,2,1);
            else: figure(0);
            plot(x, y[0].real, style+'-');
            if SPINS > 1 and all_spins: plot(x, y[1].real, style + '--');
            xlim(xlimit);
            title('Real part');
            xlabel(xlb); ylabel(st);
            legend(legend_str, loc = legend_loc);
            if is_subplot: subplot(1,2,2);
            else: figure(1);
            plot(x, y[0].imag, style+'-');
            if SPINS > 1 and all_spins: plot(x, y[1].imag, style + '--');
            xlim(xlimit);
            title('Imaginary part');
            xlabel(xlb); ylabel(st);
            # legend(legend_str, loc = legend_loc);
            if not noshow: show();
            return;
        if part == 'real' or part == 0: y = y.real;
        if part == 'imag' or part == 1: y = y.imag;
    else: xlb = 'Tau';

    plot(x, y[0], style);
    if SPINS > 1 and all_spins: plot(x, y[1], style + '--');
    xlabel(xlb); ylabel(st);
    legend(legend_str, loc = legend_loc);

    if xlimit is not None: xlim(xlimit);
    if not noshow: show();    


def view_log(h5, id):
    parms = gget(h5, 'parms');
    print "U = " + parms["U"] + "; J = " + val_def(parms, 'J', 0) + "; BETA = " + parms["BETA"];
    print "N_LAYERS = " + parms["N_LAYERS"]
    log = gget(h5, 'log');
    print log[id];

        
def load_data(datafile, ID = None):
    h5file = h5py.File(datafile, 'r');
    if ID is None: h5 = h5file[h5file.keys()[0]];
    else: h5 = h5file[str(ID)];
    return h5, h5file;


def getRenormFactor(h5, npoint = 2, norder = 1, it = None, show_p = False):
    if it == None: it = h5['iter'][0];
    se = gget(h5, 'se', it)[:, :npoint, :];
    wn = gget(h5, 'wn')[:npoint];
    z = zeros((size(se, 0), size(se, 2)), dtype = 'f8');
    imse0 = zeros((size(se, 0), size(se, 2)), dtype = float);
    for f in range(size(se, 2)):
        for s in range(size(se, 0)):
            if npoint >= 2:
                p = polyfit(wn, imag(se[s, :, f]), norder);
                z[s, f] = 1. / (1 - p[-2]);
                imse0[s, f] = p[-1];
                if show_p: print 'f=%d, s=%d'%(f,s), p;
            else: z[s, f] =  1. / (1-imag(se[s, 0, f])/wn[0]);
#    print imse0[:, ::4];
#    print mean(imse0[:, ::4], 0);
    return z;


def getFermiDOS_old(h5, it = None):
    if it is None: it = h5['iter'][0];
    parms = gget(h5, 'parms', it);
    BETA = float(parms['BETA']);
    dT = 5;
    Gtau = gget(h5, 'Gtau', it);
    Ntau = size(Gtau, 1); 
    bG = zeros((size(Gtau, 0), size(Gtau, 2)));
    for f in range(size(Gtau, 2)):
        for s in range(size(Gtau, 0)):
            bG[s, f]  = -BETA/pi*mean(Gtau[s, Ntau/2-dT:Ntau/2+dT, f]);
    return bG;


def getFermiDOS(h5, it = None):
    if it is None: it = h5['iter'][0];
    parms = gget(h5, 'parms', it);
    BETA = float(parms['BETA']);
    N_LAYERS = int(parms['N_LAYERS']);

    Giwn = h5['avgGreen/%d'%it][:];
    bG = zeros((size(Giwn, 0), size(Giwn, 2)));
    for f in range(size(Giwn, 2)):
        for s in range(size(Giwn, 0)):
            tau = BETA/2.; c0 = 1.;
            wn  = (2*arange(size(Giwn, 1))+1)*pi/BETA;
            Gwn = Giwn[s, :, f] - c0/(1j*wn);
            Gt  = sum(2./BETA * (cos(wn*tau)*Gwn.real + sin(wn*tau)*Gwn.imag)) - c0/2.;
            bG[s, f]  = -BETA/pi*Gt;
    return bG;


def show_DOS_results(h5file, filter = '', plot_what = None, L = None, f = None, Navg = 1):
    print '# BETA  T    U    Delta  N_LAYERS Ndtot DOStot               Nd                   DOS';
    for s in h5file:
        if filter not in s: continue;
        it = h5file[s+'/iter'][0];
        if str(it+1) in h5file[s+'/avgGreen'].keys(): added = 1;
        else: added = 0;
        try: parms = gget(h5file[s], 'parms');
        except: print 'ID %s not work'%s; continue;
        N_LAYERS = int(parms['N_LAYERS']); SPINS = int(parms['SPINS']);
        if L is None: L = arange(N_LAYERS);
        corr_id = system.getCorrIndex(parms);
        nf = None;
        for i in range(Navg):
            if nf is None:
                nf = getDensity(h5file[s], it-i);
                bG = getFermiDOS(h5file[s], it+added-i);
            else:
                nf += getDensity(h5file[s], it-i);
                bG += getFermiDOS(h5file[s], it+added-i);
        nf /= Navg; bG /= Navg;
        if SPINS == 1: nf = r_[nf, nf]; bG = r_[bG, bG];
        nn = sum(nf[:, corr_id][:, ::N_LAYERS], 0);
        bbG = sum(bG[:, corr_id][:, ::N_LAYERS], 0);

        print '%s: %s  %.4f  %.2f  %.4f   %d  %.4f   %.4f  '%(s, parms['BETA'], 1/float(parms['BETA']), float(parms['U']), float(parms['DELTA']), N_LAYERS,\
                sum(nf[:,corr_id])/N_LAYERS, sum(bG)/N_LAYERS), \
                (len(nn)*'  %.4f' + ' ' + len(bbG)*'  %.4f')%tuple(r_[nn, bbG]);

    if plot_what is not None:
        for s in h5file:
            if filter in s: plot_data(h5file[s], plot_what, f = f, L = L);


def show_mag_results(h5file, filter = '', plot_what = None, L = 0, Navg = 1):
    print '# BETA  T  Nd  magnetization  inv_susceptibility';
    for s in h5file:
        if filter not in s: continue;
        try: parms = gget(h5file[s], 'parms');
        except: print 'ID %s not work'%s; continue;
        tmp = gget(h5file[s], 'log');
        nd = sum(mean(tmp[1][-Navg:, :], 0));
        magnetization = sum(mean(tmp[5][-Navg:, :], 0))/int(parms['N_LAYERS']);
        susceptibility = magnetization / float(parms['H']);
        print '%s: %s  %.4f  %.4f'%(s, parms['BETA'], 1/float(parms['BETA']), sum(nd)/int(parms['N_LAYERS'])), magnetization, 1./susceptibility;

    if plot_what is not None:
        corr_id = system.getCorrIndex(parms);
        for s in h5file:
            if filter in s: plot_data(h5file[s], plot_what, f = corr_id, L = L);


def show_mass_results(h5file, filter = '', NMat = 6, NOrder = 4, Navg = 1):
    print '# U   J    BETA   mass...'; 
    for s in h5file:
        if re.search(filter, s) is None: continue;
        try: parms = gget(h5file[s], 'parms');
        except: print 'ID %s not work'%s; continue;
        it = h5file[s+'/iter'][0];
        mass = None;
        for n in range(Navg):
            tmp = 1./getRenormFactor(h5file[s], npoint = NMat, norder = NOrder, it = it-n);
            if mass is None: mass = tmp;
            else: mass += tmp;
        mass /= Navg;
        mass = mass[:, ::int(parms['N_LAYERS'])];
        mass = mass.flatten();

        print '%s   %s   %s    '%(parms['U'], parms['J'], parms['BETA']), len(mass)*'%.4f   '%tuple(mass); 


def getWilsonRatio(h5file, filter = '', Navg = 1, NMat = 6, NOrder = 4):
    print '# BETA  T  susceptibility    specific heat   Wilson ratio';
    for s in h5file:
        if filter not in s: continue;
        parms = gget(h5file[s], 'parms');
        N_LAYERS = int(parms['N_LAYERS']);
        tmp = gget(h5file[s], 'log');
        nd = sum(mean(tmp[1][-Navg:, :], 0));
        magnetization = sum(mean(tmp[5][-Navg:, :], 0))/N_LAYERS;
        susceptibility = magnetization / float(parms['H']);

        bG = None;
        for i in range(Navg):
            it = h5file[s+'/iter'][0];
            if str(it+1) in h5file[s+'/avgGreen'].keys(): added = 1;
            else: added = 0;

            t1 = getFermiDOS(h5file[s], it+added-i);
            t2 = getRenormFactor(h5file[s], NMat, NOrder, it-i);
            if bG is None: bG = t1; Z = t2; 
            else: bG += t1; Z += t2;
        bG = bG[:, ::N_LAYERS];
        Z  = Z[:,::N_LAYERS];
        gamma = sum(bG/Z);
        print '%s   %.4f    %.8f    %.8f    %.8f'%(parms['BETA'], 1/float(parms['BETA']), susceptibility, gamma, susceptibility/gamma);

 


def gget(data, st, iter = None, L = None, f = None):
    if iter == None: iter = str(data['iter'][0]);
    iter = str(iter);
    parms = load_parms(data, int(iter));
    N_LAYERS = int(parms['N_LAYERS']); FLAVORS = int(parms['FLAVORS']); SPINS = int(parms['SPINS']);
    NORB = int(parms['NORB']); NCOR = int(parms['NCOR']);
    corr_id = system.getCorrIndex(parms);
    dmft_id = system.getDMFTCorrIndex(parms);
    dmft_site_id = system.getDMFTCorrIndex(parms, all = False);
    
    if st == 'wn':
        NMaxFreq = int(parms['N_MAX_FREQ']);
        BETA = float(parms['BETA']);
        return (2*arange(0, NMaxFreq)+1)*pi/BETA;
    if st == 'parms': return parms;
    if st == 'log':
        log = data['log_density'][:];
        num_iter = len(log);
        nraw = log[:,4:];
        orbital = zeros((num_iter, FLAVORS, N_LAYERS));
        orbital_abs = zeros((num_iter, FLAVORS, N_LAYERS));
        spinup = zeros((num_iter, N_LAYERS));
        spindn = zeros((num_iter, N_LAYERS));
        density = zeros((num_iter, N_LAYERS));
        magnetization = zeros((num_iter, N_LAYERS));

        for n in range(num_iter):
            nf = nraw[n].reshape(SPINS, -1)[:, :NCOR];
            noxy = nraw[n].reshape(SPINS, -1)[:, -1]/N_LAYERS;
#            if n > 0: 
#                nf_gtau = -data['SolverData/Gtau/'+str(n)][:, -1, :];
#                nf[:, dmft_id] = nf_gtau[:, dmft_id];
            nf = nf.reshape(SPINS, FLAVORS, N_LAYERS);
            if SPINS == 1: nf = r_[nf, nf]; noxy = r_[noxy, noxy];

            density[n] = array([sum(nf[:,:,i]) for i in range(N_LAYERS)]);
            orbital[n] = (nf[0] + nf[1])/density[n];
            orbital_abs[n] = (nf[0] + nf[1]);
            spinup[n] = sum(nf[0], 0)/density[n] - 1./2;
            spindn[n] = sum(nf[1], 0)/density[n] - 1./2;
#            magnetization[n] = sum(nf[0, dmft_site_id, :] - nf[1, dmft_site_id], 0);
            magnetization[n] = sum(nf[0] - nf[1], 0) + noxy[0] - noxy[1];
#            magnetization[n] = sum(nf[0] - nf[1], 0);

        out = (log[:,:4], density, spinup, spindn, orbital, magnetization, orbital_abs);
        return out;
    
    if L == None: L = arange(N_LAYERS);
    if f == None: f = arange(FLAVORS);
    L = array([L]).flatten(); f = asarray([f]).flatten();
    idx = array([], dtype = 'i4');
    for i in range(0,len(f)): idx = r_[idx, L + f[i]*N_LAYERS];
    idx = sort(idx);

    if st == 'idx': return idx;
    if st == 'Gimp': return data['ImpurityGreen/' + iter][:,:,idx];
    if st == 'se': return data['SelfEnergy/' + iter][:,:,idx];
    if st == 'as': return data['WeissField/' + iter][:,:,idx];
    if st == 'Gavg': return data['avgGreen/' + iter][:,:, corr_id][:,:,idx];
    if st == 'hybmat': return data['SolverData/Hybmat/' + iter][:,:,idx];
    if st == 'hybtau': return data['SolverData/Hybtau/' + iter][:,:,idx];
    if st == 'Gtau': return data['SolverData/Gtau/' + iter][:,:,idx];
    if st == 'G0': return 1. / data['WeissField/' + iter][:,:,idx];
    if st == 'bG': return getFermiDOS(data, iter);
    if st == 'Gtot': return getTotalGtau(data, iter);
    if st == 'Z'   : return getRenormFactor(data, npoint = 2, it = iter);


from solver import solver_input_data;
def solver_prepare(data, it = None, L = 0):
    if it is None: it = data['iter'][0];
    print 'Getting data from iteration ', it
    parms = gget(data, 'parms', it);
    SPINS = int(parms['SPINS']); N_LAYERS = int(parms['N_LAYERS']); FLAVORS = int(parms['FLAVORS']);
    wn = gget(data, 'wn', it);

    htau = gget(data, 'hybtau', it);
    hmat = gget(data, 'hybmat', it);
    htail = data['SolverData/hyb_asym_coeffs'][it-1, 1:].reshape(SPINS, -1, N_LAYERS*FLAVORS);
    hyb_data_all = [htau, hmat, htail];

    AvgDispersion = data['SolverData/AvgDispersion'][:];
    nf = getDensity(h5, it-1);
    VCoulomb = h5['StaticCoulomb/%d'%it][:];

    out = solver_input_data(parms, L, hyb_data_all, AvgDispersion, VCoulomb, nf);

    savetxt('hybmat.real', c_[wn, out['hybmat'].real]);
    savetxt('hybmat.imag', c_[wn, out['hybmat'].imag]);
    savetxt('hybmat.tail', out['hybtail']);
    print out['MU'];




if __name__ == '__main__':
    set_printoptions(suppress=True, precision=4,linewidth=150);

    if len(sys.argv) == 2: h5, h5file = load_data(sys.argv[1]);
    else: h5, h5file = load_data(sys.argv[1], sys.argv[2]);
    print h5, h5["iter"][0];

