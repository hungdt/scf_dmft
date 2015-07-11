#!/usr/bin/env python

from numpy import *;
from share_fun import *;
import functions;
import average_green;
import system_dependence as system;
import os;


def getDensity(gmat, p):
    X = p['X'];
    W = p['W'];
    BETA = p['BETA'];
    n_f = zeros((gmat.shape[1],), dtype = 'f8');
    for i in range(0, gmat.shape[1]):
        n_f[i] = 0.5 + 2./BETA*real(sum(W/X * gmat[:, i]));
    return n_f;


def generateGaussPoints(Nm):
    Jn = zeros((Nm,Nm));
    n = arange(1,Nm, dtype='f8');

    # for Matsubara summation
    a0 = 1. / 12.;
    an = 1. / (32.*n**2 + 16.*n - 6.);
    bn = 1. / (4096.*n**4 - 4096.*n**3 + 512.*n**2 + 256.*n - 48);

    Jn[0,0] = a0;
    for i in n:
        Jn[i, i] = an[i-1];
        Jn[i-1,i] = Jn[i,i-1] = sqrt(bn[i-1]);

    d, v = linalg.eig(Jn);
    xi = asarray(d);
    wi = abs(v[0,:])**2/8;

    ind = argsort(xi);
    xi = xi[ind];
    wi = wi[ind];

    return asarray(xi), asarray(wi)


def averageGreen(delta0, mu0, w, SelfEnergy, parms, Nd, Ntot, tuneup, extra):
    BETA     = float(parms["BETA"]);
    N_LAYERS = int(parms['N_LAYERS']); FLAVORS = int(parms['FLAVORS']); SPINS = int(parms['SPINS']);
    rot_mat = extra['rot_mat'];
    
    # calculate intersite Coulomb energy here
    Vc = zeros(N_LAYERS, dtype = 'f8');

    # convert self energy to the C++ form
    SelfEnergy_rot = array([functions.irotate(SelfEnergy[s], extra['rot_mat']) for s in range(SPINS)]);
    SE = [array([s.flatten() for s in SelfEnergy_rot[n]]) for n in range(SPINS)];

    v_delta = empty(0, 'f8');
    v_nd    = empty(0, 'f8');
    ddelta = 0.; delta_step = 1.;
    dmu = 0.; mu_step = 5; 
    tol = 0.005;
    # Delta loop
    while True:
        delta = delta0 + ddelta;

        v_mu = empty(0, 'f8');
        v_n = empty(0, 'f8');
        # mu loop
        while True:
            mu = mu0 + dmu;
            
            Gavg = average_green.integrate(w, delta, mu, SE, parms, extra);
            Gavg_diag = array([Gavg[0, :, i, i] for i in range(int(parms['NORB']))]).T;
            nf = getDensity(Gavg_diag, parms); 
            my_ntot = 2*sum(nf); # factor of 2 due to spin
    
            if tuneup: print "   adjust mu: " + str(mu) + " " + str(dmu) + " " + str(my_ntot);
            if Ntot < 0 or abs(Ntot - my_ntot) < tol or not tuneup: break;
     
            v_mu = r_[v_mu, dmu];
            v_n = r_[v_n, my_ntot];
            if v_n.min() < Ntot and v_n.max() > Ntot: dmu = interp_root(v_mu, v_n, Ntot);
            else: dmu += (1. if my_ntot < Ntot else -1.)*mu_step;
           
        my_nd = 2*sum(nf[:N_LAYERS*FLAVORS]);

        if tuneup: print "adjust double counting: " + str(delta) + " " + str(ddelta) + " " + str(my_nd) + " " + str(my_nd/N_LAYERS);
        if Nd < 0 or abs(Nd - my_nd) < tol or not tuneup: break;

        v_delta = r_[v_delta, ddelta];
        v_nd = r_[v_nd, my_nd];
        if v_nd.min() < Nd and v_nd.max() > Nd: ddelta = interp_root(v_delta, v_nd, Nd);
        else: ddelta += (1. if my_nd < Nd else -1.)*delta_step;

    Gavg = array([functions.rotate_all(Gavg[s], rot_mat) for s in range(SPINS)]);
    return Gavg, delta, mu, Vc;


def HartreeRun(parms, extra):
    print "Initialization using Hartree approximation\n"
    N_LAYERS = int(parms['N_LAYERS']);
    FLAVORS  = int(parms['FLAVORS']);
    SPINS    = 1;

    p = dict({
        'MU' : float(val_def(parms, 'MU', 0)), 
        'N_LAYERS': N_LAYERS,
        'NORB' : int(parms['NORB']),
        'U'  : float(parms['U']),
        'J'  : float(parms['J']),
        'DELTA': float(val_def(parms, 'DELTA', 0)),
        'ND'   : N_LAYERS*float(val_def(parms, 'ND', -1)),
    
        'DENSITY' : N_LAYERS*float(parms['DENSITY']),
        'FLAVORS' : FLAVORS,
        'SPINS'   : 1,
    
        'OUTPUT' : '.' + parms['DATA_FILE'] + '_HartreeInit',
        'NN'     : None,
        'N_MAX_FREQ'  : 30,
        'BETA'        : float(parms['BETA']),
        'NUMK' : int(val_def(parms, 'INIT_NUMK', 8)),
        'TUNEUP' : int(val_def(parms, 'NO_TUNEUP', 0)) == 0,
        'MAX_ITER' : 15,
        'ALPHA'  : 0.5, # pay attention at this parm sometimes
        'DTYPE'  : parms['DTYPE'],
        'INTEGRATE_MOD' : val_def(parms, 'INTEGRATE_MOD', 'integrate'),
        'np' : parms['np']
        });
    
    for k, v in p.iteritems(): print k + ': ', v;
    
    bp, wf = grule(p['NUMK']);
    X, W = generateGaussPoints(p['N_MAX_FREQ']);
    wn = 1/sqrt(X)/p['BETA'];
    p.update({
        'X'  : X,
        'W'  : W,
        'w'  : wn
        });
    if p['NN'] is None and os.path.isfile(p['OUTPUT']+'.nn'): p['NN'] = p['OUTPUT'];
       
    # running
    TOL = 1e-2;
    if p['NN'] is None: 
        nn = ones(N_LAYERS*FLAVORS, dtype = 'f8') * p['DENSITY']/p['NORB']/2;  # 2 for spin
        mu = p['MU'];
        delta = p['DELTA'];
    else:
        print 'Continue from '+p['NN'];
        nn = genfromtxt(p['NN']+'.nn')[2:];
        mu = genfromtxt(p['NN']+'.nn')[1];
        delta = genfromtxt(p['NN']+'.nn')[0];
    Gavg = zeros((p['N_MAX_FREQ'], p['NORB']), dtype = 'c16');
    se = zeros((SPINS, p['N_MAX_FREQ'], N_LAYERS*FLAVORS), dtype = 'c16');
    stop = False;
    count = 0;
    ALPHA = p['ALPHA'];
    corr1 = system.getDMFTCorrIndex(parms, all = False);
    corr2 = array([i for i in range(FLAVORS) if i not in corr1]);   # index for eg bands
    old_GaussianData = extra['GaussianData'];
    extra['GaussianData'] = [bp, wf];
    while not stop:
        count += 1;
        nn_old = nn.copy();
        p['MU'] = mu;
        p['DELTA'] = delta;
        Gavg_old = Gavg.copy();
    
        for L in range(N_LAYERS):
            se_coef = functions.get_asymp_selfenergy(p, array([nn[L:N_LAYERS*FLAVORS:N_LAYERS]]))[0, 0, :];
            for s in range(SPINS): 
                for f in range(len(se_coef)): se[s, :, f*N_LAYERS+L] = se_coef[f];
        
        Gavg, delta, mu, Vc = averageGreen(delta, mu, 1j*wn, se, p, p['ND'], p['DENSITY'], p['TUNEUP'], extra);
        Gavg  = mean(Gavg, 0);
        nn = getDensity(Gavg, p);

        # no spin/orbital polarization, no charge order
        for L in range(N_LAYERS):
            nf1 = nn[0:N_LAYERS*FLAVORS:N_LAYERS];
            for id in range(FLAVORS):
                if id in corr1: nf1[id] = mean(nf1[corr1]);
                else: nf1[id] = mean(nf1[corr2]);
            nn[L:N_LAYERS*FLAVORS:N_LAYERS] = nf1;
   
        err = linalg.norm(r_[delta, mu, nn] - r_[p['DELTA'], p['MU'], nn_old]);
        savetxt(p['OUTPUT']+'.nn', r_[delta, mu, nn]);
        print 'Step %d: %.5f'%(count, err);
        if (err < TOL): stop = True; print 'converged';
        if count > p['MAX_ITER']: break;

        mu = ALPHA*mu + (1-ALPHA)*p['MU'];
        delta = ALPHA*delta + (1-ALPHA)*p['DELTA'];
        nn = ALPHA*nn + (1-ALPHA)*nn_old;

 
    # DOS
    NFREQ = 500;
    BROADENING = 0.03;
    extra['GaussianData'] = old_GaussianData;
    parms_tmp = parms.copy(); parms_tmp['DELTA'] = delta;
    Eav = system.getAvgDispersion(parms_tmp, 3, extra)[0,0,:];
    Ed  = mean(Eav[:N_LAYERS*FLAVORS][corr1]);
    Ep  = mean(Eav[N_LAYERS*FLAVORS:]) if N_LAYERS*FLAVORS < p['NORB'] else Ed;
    emax = min(4, p['U']);
    emin = -(Ed - Ep + max(se_coef) + min(4, p['U']));
    print "Energy range for HF DOS: ", emin, emax
    w = linspace(emin, emax, NFREQ) + 1j*BROADENING;
    se = zeros((SPINS, NFREQ, N_LAYERS*FLAVORS), dtype = 'c16');
    for L in range(N_LAYERS):
        se_coef = functions.get_asymp_selfenergy(p, array([nn[L:N_LAYERS*FLAVORS:N_LAYERS]]))[0, 0, :];
        for s in range(SPINS): 
            for f in range(len(se_coef)): se[s, :, f*N_LAYERS+L] = se_coef[f];
    Gr = average_green.averageGreen(delta, mu, w, se, p,p['ND'], p['DENSITY'], 0, extra)[1][0];
    savetxt(parms['ID']+'.dos', c_[w.real, -1/pi*Gr.imag], fmt = '%.6f');

    print ('End Hartree approx.:%d   Ntot=%.2f  Nd=%.2f  Delta=%.4f   '
           'Delta_eff=%.4f')%(count, 2*sum(nn)/N_LAYERS,
                              2*sum(nn[:N_LAYERS*FLAVORS]/N_LAYERS),
                              delta, delta-mean(se_coef[corr1])), ': \n', \
                              nn[:N_LAYERS*FLAVORS].reshape(-1, N_LAYERS),\
                              '\n\n'
#    os.system('rm ' + p['OUTPUT']+'.nn');
    return delta, mu, array([nn for s in range(int(parms['SPINS']))]), Vc;
