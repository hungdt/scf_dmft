import h5py, system_dependence as system;
from numpy import *;
from share_fun import load_parms, val_def, log_data;
from scipy import interpolate;



def getDensity(h5, it = None):
    if it is None: it = h5['iter'][0];
    parms    = load_parms(h5, it);
    N_LAYERS = int(parms['N_LAYERS']);
    SPINS    = int(parms['SPINS']);
    NCOR     = int(parms['NCOR']);
    U        = float(parms['U']);
    n_f = h5['log_density'][0 if it == 0 else it-1, 4:].reshape(SPINS, -1)[:, :NCOR];
    if U != 0 and it > 0:
        if int(val_def(parms, 'FIXED_HARTREE', 0)) > 0:
            n_f = h5['log_density'][0, 4:].reshape(SPINS, -1)[:, :NCOR];
        dmft_id = system.getDMFTCorrIndex(parms);
        gtau = h5['SolverData/Gtau/'+str(it)][:]
        n_f[:, dmft_id]  = -gtau[:, -1, dmft_id];
    return n_f;


def getDensityFromGmat(gmat, beta, extra):
    SPINS = len(gmat);
    wn = (2*arange(size(gmat, 1))+1)*pi/beta;
    C = extra['G_asymp_coefs'] if 'G_asymp_coefs' in extra else zeros(SPINS);
    correction = extra['correction'] if 'correction' in extra else zeros(SPINS);
    density = array([2./beta*real((sum(gmat[s], 0)) + C[s]*sum(1./wn**2)) + 0.5 - beta*C[s]/4 + correction[s] for s in range(SPINS)]);
    return density;


def getFermiDOS(Giwn, BETA):
    bG = zeros((size(Giwn, 0), size(Giwn, 2)));
    for f in range(size(Giwn, 2)):
        for s in range(size(Giwn, 0)):
            tau = BETA/2.; c0 = 1.;
            wn  = (2*arange(size(Giwn, 1))+1)*pi/BETA;
            Gwn = Giwn[s, :, f] - c0/(1j*wn);
            Gt  = sum(2./BETA * (cos(wn*tau)*Gwn.real + sin(wn*tau)*Gwn.imag)) - c0/2.;
            bG[s, f]  = -BETA/pi*Gt;
    return bG;


def smooth(hyb, C, NMaxFreq, BETA, list_NCutoff, minorder = 1):
    w = 1j*(2*arange(NMaxFreq)+1)*pi/float(BETA);
    ret_hyb = zeros((size(hyb, 0), NMaxFreq, size(hyb, 2)), complex);
    for s in range(size(hyb, 0)):
        for f in range(size(hyb, 2)):
            NCutoff = list_NCutoff[s, f];
            ret_hyb[s, :NCutoff, f] = hyb[s, :NCutoff,f];
            ret_hyb[s, NCutoff:, f] = 0;
            for norder in range(size(C, 1)): ret_hyb[s, NCutoff:, f] += C[s, norder, f]/w[NCutoff:]**(norder+minorder);
    return ret_hyb;


def smooth_selfenergy(it, h5, SelfEnergy, nf):
    parms = load_parms(h5, it);
    N_LAYERS = int(parms['N_LAYERS']); FLAVORS = int(parms['FLAVORS']); 
    SPINS = size(nf, 0);   # this SPINS may be different from parms['SPINS']
    NCOR = int(parms['NCOR']);

    # calculate asymptotic coeffs and smooth
    se_coefs = zeros((SPINS, 2, NCOR), dtype = float);
    for L in range(N_LAYERS): 
        st='SolverData/Observables/%d/L%d'%(it, L);
        try: nn = h5[st+'/nn'][:];
        except: nn = None;
        se_coefs[:, :, L::N_LAYERS] = get_asymp_selfenergy(parms, nf[:, L::N_LAYERS], nn);
    log_data(h5['SolverData'], 'selfenergy_asymp_coeffs', it, se_coefs.flatten(), data_type = float);

    list_NCutoff = ones((SPINS, NCOR), dtype = int)*int(parms['N_CUTOFF']);
    ind = SelfEnergy.imag > 0;
    SelfEnergy[ind] = real(SelfEnergy[ind]);
    return smooth(SelfEnergy, se_coefs, int(parms['N_MAX_FREQ']), float(parms['BETA']), list_NCutoff, minorder = 0);


def get_asymp_hybmat(parms, nf, MU, Eav):
    SPINS = int(parms['SPINS']);
    S = get_asymp_selfenergy(parms, nf);
    G = zeros((SPINS, 3, int(parms['FLAVORS'])), dtype = float);
    epsav = Eav[:, 0, :];
    epssqav = Eav[:, 1, :];
    epscubeav = Eav[:, 2, :];
    for s in range(SPINS):
        G[s, 0, :] = epssqav[s] - epsav[s]**2;
        G[s, 1, :] = (epssqav[s] - epsav[s]**2)*(S[s, 0, :] - 2*epsav[s] - MU) + (epscubeav[s] - epsav[s]**3);
    return G;


def get_asymp_selfenergy(parms, nf_in, nn_in = None):
    U = generate_Umatrix(float(parms['U']), float(parms['J']), 
            int(parms['FLAVORS']), val_def(parms, 'INTERACTION_TYPE', 'SlaterKanamori'));
    dmft_id = system.getDMFTCorrIndex(parms, all = False);
    FLAVORS = int(parms['FLAVORS']);
    SPINS = 2;

    nf = zeros(SPINS*FLAVORS);
    nf[::2] = nf[1::2] = nf_in[0];
    if int(parms['SPINS']) == 2: nf[1::2] = nf_in[1];

    nn = zeros((FLAVORS*SPINS, FLAVORS*SPINS));
    pos = 0;
    for i in range(FLAVORS*SPINS):
        for j in range(i+1):
            f1 = i/SPINS;
            f2 = j/SPINS;
            if f1 in dmft_id and f2 in dmft_id and nn_in is not None:
                nn[i,j] = nn[j,i] = nn_in[pos];
                pos += 1;
            else: nn[i,j] = nn[j,i] = nf[i]*nf[j];
        if f1 in dmft_id: nn[i,i] = nf[i];

    S = zeros((2, SPINS*FLAVORS)); # 2: expansion orders: (iwn)^0, (iwn)^{-1}
    for f in range(SPINS*FLAVORS):
        # zeroth order is easy: \Sigma^0_f = U_{f, f'} * <n_f'>
        S[0, f] = sum(U[f, :]*nf);

        # first order is harder: \Sigma^1_f = U_{f,f1}*U_{f,f2}*<n_f1 n_f2> - (\Sigma^0_f)^2
        for f1 in range(SPINS*FLAVORS):
            for f2 in range(SPINS*FLAVORS):
                S[1, f] += U[f, f1]*U[f,f2]*nn[f1,f2];
        S[1,f] -= S[0,f]**2;
    ret = array([S[:,::2], S[:,1::2]]);

    # for mean field, there is only \Sigma^0, other terms vanish
    # so I set \Sigma^1 to be zero
    for f in range(FLAVORS):
        if f not in dmft_id: ret[:, 1, f] = 0;
    if int(parms['SPINS']) == 1: ret = array([ret[0]]);
    return ret;


def get_asymp_selfenergy_old(parms, nf_in, nn_in = None):
    dmft_id = system.getDMFTCorrIndex(parms, all = False);
    FLAVORS = 2*len(dmft_id);

    nf = zeros(FLAVORS);
    nf[::2] = nf[1::2] = nf_in[0, dmft_id];
    if int(parms['SPINS']) == 2: nf[1::2] = nf_in[1, dmft_id];
    nn = None;
    if nn_in is not None: 
        nn = zeros((FLAVORS, FLAVORS));
        pos = 0;
        for i in range(FLAVORS):
            for j in range(i+1):
                if pos > len(nn_in): nn = None; break;
                nn[i,j] = nn[j,i] = nn_in[pos]; 
                pos += 1;

    U = float(parms['U']);
    J = float(parms['J']);
    U1= U-2*J;
    U2= U1-J;
    S = zeros((2, FLAVORS));
    for f in range(FLAVORS):
        i = int(f / 2); s = f % 2;
        S[0, f] = U*nf[2*i+(not s)];
        for j in range(FLAVORS/2):
            if j != i: S[0, f] += U1*nf[j*2+(not s)] + U2*nf[j*2+s];
        S[1, f] = 0;
        if nn is not None: # tedious formula, be careful of being wrong
            tmp = U**2*nf[2*i+(not s)];
            for j in range(FLAVORS/2): 
                if j != i: 
                    tmp += U1**2*nf[2*j+(not s)] + U2**2*nf[2*j+s] \
                        +2*U*U1*nn[2*i+(not s),2*j+(not s)]+2*U*U2*nn[2*i+(not s),2*j+s]+2*U1*U2*nn[2*j+(not s),2*j+s];
                    for j1 in range(j+1,FLAVORS/2):
                        if j1 != i: tmp += 2*U1**2*nn[2*j+(not s),2*j1+(not s)] + 2*U2**2*nn[2*j+s,2*j1+s] \
                                + 2*U1*U2*(nn[2*j+(not s),2*j1+s]+nn[2*j+s,2*j1+(not s)]);
            tmp -= S[0,f]**2;
            S[1,f] = tmp;

    # for full d-bands
    FLAVORS = 2*int(parms['FLAVORS']);
    nf = zeros(FLAVORS);
    nf[::2] = nf[1::2] = nf_in[0];
    if int(parms['SPINS']) == 2: nf[1::2] = nf_in[1];
    S1 = zeros((2, FLAVORS));
    for f in range(FLAVORS):
        i = int(f / 2); s = f % 2;
        S1[0, f] = U*nf[2*i+(not s)];
        for j in range(FLAVORS/2):
            if j != i: S1[0, f] += U1*nf[j*2+(not s)] + U2*nf[j*2+s];
    for n in range(len(dmft_id)): 
        S1[1, 2*dmft_id[n]] = S[1, 2*n];
        S1[1, 2*dmft_id[n]+1] = S[1,2*n+1];

    ret = array([S1[:,::2], S1[:,1::2]]) if int(parms['SPINS']) == 2 else array([S1[:, ::2]]);
    return ret;


def extrapolateSelfEnergy(h5, nparms, nwn):
    NCOR = int(nparms['NCOR']); SPINS = int(nparms['SPINS']);
    oit = h5['iter'][0];
    oparms  = load_parms(h5, oit);
    try: MU  = float(h5['parms/%d/MU'%(oit+1)][...]);
    except: MU  = float(h5['parms/%d/MU'%oit][...]);
    eMU = MU - h5['StaticCoulomb/%d'%oit][:];
    ose = h5['SelfEnergy/%d'%oit][:];
    own = (2*arange(size(ose, 1))+1)*pi/float(oparms['BETA']);
    oSPINS = int(oparms['SPINS']);
    assert NCOR == int(oparms['NCOR']);

    tail = h5['SolverData/selfenergy_asymp_coeffs'][-1, 1:].reshape(oSPINS, 2, -1);

    tck_real = [];
    tck_imag = [];
    for s in range(oSPINS):
        tck_real.append([]);
        tck_imag.append([]);
        for f in range(NCOR):
            tck_real[s].append(interpolate.splrep(own, ose[s, :, f].real));
            tck_imag[s].append(interpolate.splrep(own, ose[s, :, f].imag));

    ret = zeros((oSPINS, len(nwn), NCOR), dtype = 'c16');
    for s in range(oSPINS):
        for n in range(len(nwn)):
            if nwn[n] < own[0]:
                # linear extrapolate;
                ret[s, n, :] = (nwn[n]-own[0])/(own[0] - own[1])*(ose[s, 0,:] - ose[s, 1,:]) + ose[s, 0,:];
            if nwn[n] >= own[0] and nwn[n] <= own[-1]:
                for f in range(NCOR):
                    ret[s, n, f] = interpolate.splev(nwn[n], tck_real[s][f]) + 1j*interpolate.splev(nwn[n], tck_imag[s][f]);
            if nwn[n] > own[-1]:
                for k in range(size(tail, 1)):
                    ret[s, n, :] += tail[s, k,:]/(1j*nwn[n])**k;

    ind = ret.imag > 0;
    ret[ind] = ret[ind].real;
    if SPINS < oSPINS: ret = array([mean(ret, 0)]);
    if SPINS > oSPINS: ret = array([ret[0], ret[0]]);
    return ret;


def assign(data, N_LAYERS, s = [0, 1]): # s: spin index
    data_shape = data.shape;
    s = sort(array(s).flatten());
    SPINS = 2;
    FLAVORS = size(data, len(data_shape) - 1) / N_LAYERS / SPINS;
    if len(data_shape) == 2: ret = zeros((len(s), size(data, 0), N_LAYERS*FLAVORS), dtype = data.dtype);
    elif len(data_shape) == 1: ret = zeros((len(s), N_LAYERS*FLAVORS), dtype = data.dtype);
    else: print "Data shape not supported"; return None;
    for n in range(len(s)):
        for L in range(N_LAYERS):
            for f in range(FLAVORS): 
                if len(data_shape) == 2: ret[n, :, f*N_LAYERS+L] = data[:, (2*f+s[n])*N_LAYERS+L];
                if len(data_shape) == 1: ret[n, f*N_LAYERS+L] = data[(2*f+s[n])*N_LAYERS+L];
    return ret;


# matrix rotation
import fort_rot;
def irotate(fin, rot_mat):
    N_LAYERS = len(rot_mat);
    FLAVORS  = len(rot_mat[0]);
    assert(size(fin,1) == N_LAYERS*FLAVORS);

    fout = zeros((len(fin), N_LAYERS, FLAVORS, FLAVORS), dtype = fin.dtype);
    for L in range(N_LAYERS):
        tmp = fort_rot.irotate(fin[:,L::N_LAYERS], rot_mat[L]);
        fout[:,L,:,:] = tmp;
    return fout;

def rotate(fin, rot_mat):
    N_LAYERS = len(rot_mat);
    FLAVORS  = len(rot_mat[0]);
    assert(size(fin,1) == N_LAYERS and size(fin, 2) == FLAVORS);

    fout = zeros((len(fin), N_LAYERS*FLAVORS), dtype = fin.dtype);
    for L in range(N_LAYERS):
        fout[:, L::N_LAYERS] = fort_rot.rotate(fin[:,L,:,:], rot_mat[L]);
    return fout;

def rotate_all(mat, rot_mat, need_extra = False):
    Nm = len(mat);
    N  = len(rot_mat);
    L2 = len(rot_mat[0]);

    out = zeros((Nm, N*L2), dtype = mat.dtype);
    for L in range(N):
        mat_tmp = mat[:, L:L2*N:N, L:L2*N:N];
        tmp = fort_rot.rotate(mat_tmp, rot_mat[L]);
        out[:, L::N] = tmp;

    if need_extra: 
        for n in range(N*L2, size(mat, 2)): out = c_[out, mat[:, n, n]];
    return out;


def generate_Umatrix(U, J, FLAVORS, Utype):
    Umatrix = zeros((2*FLAVORS, 2*FLAVORS));
    if Utype == 'SlaterKanamori':
        U1 = U-2*J; U2 = U1-J;
        for f1 in range(2*FLAVORS):
            for f2 in range(2*FLAVORS):
                s1 = f1 % 2; a1 = f1 / 2;
                s2 = f2 % 2; a2 = f2 / 2;
                if a1 == a2:
                    if s1 != s2: Umatrix[f1, f2] = U;
                else:
                    if s1 != s2: Umatrix[f1, f2] = U1;
                    else: Umatrix[f1, f2] = U2;
    elif Utype == 'SlaterIntegrals':
        assert FLAVORS == 5, 'only accept FLAVORS=5 for d bands';
        F0 = U; F2 = 70*J/13.; F4 = 112*J/13.;
        U0 = F0 + 8/7.*(F2+F4)/14.;
        J1 = 3/49.*F2 + 20/9.*1/49.*F4;
        J2 =-2*5/7.*(F2+F4)/14. + 3*J1;
        J3 = 6*5/7.*(F2+F4)/14. - 5*J1;
        J4 = 4*5/7.*(F2+F4)/14. - 3*J1;
        # row(column) info: xy, yz, 3z^2, xz, x^2-y^2
        UPavarini = array([
            [ U0, U0-2*J1, U0-2*J2, U0-2*J1, U0-2*J3 ],
            [ U0-2*J1, U0, U0-2*J4, U0-2*J1, U0-2*J1 ],
            [ U0-2*J2, U0-2*J4, U0, U0-2*J4, U0-2*J2 ],
            [ U0-2*J1, U0-2*J1, U0-2*J4, U0, U0-2*J1 ],
            [ U0-2*J3, U0-2*J1, U0-2*J2, U0-2*J1, U0 ]
            ]);
        JPavarini = array([
            [ U0, J1, J2, J1, J3 ],
            [ J1, U0, J4, J1, J1 ],
            [ J2, J4, U0, J4, J2 ],
            [ J1, J1, J4, U0, J1 ],
            [ J3, J1, J2, J1, U0 ]
            ]);

        # swap to the order of wannier90: 3z^2, xz, yz, x^2-y^2, xy
        dmap = array([
            [0, 1, 2, 3, 4],
            [2, 3, 1, 4, 0]
            ]);
        UPavarini[:, dmap[0]] = UPavarini[:, dmap[1]];
        UPavarini[dmap[0], :] = UPavarini[dmap[1], :];
        JPavarini[:, dmap[0]] = JPavarini[:, dmap[1]];
        JPavarini[dmap[0], :] = JPavarini[dmap[1], :];

        for f1 in range(2*FLAVORS):
            for f2 in range(2*FLAVORS):
                s1 = f1 % 2; a1 = f1 / 2;
                s2 = f2 % 2; a2 = f2 / 2;
                if s1 == s2: Umatrix[f1, f2] = UPavarini[a1, a2] - JPavarini[a1, a2];
                else: Umatrix[f1, f2] = UPavarini[a1, a2];
    else: exit('Unknown interaction type');
    return Umatrix;


