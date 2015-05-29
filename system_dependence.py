from numpy import *;
import functions as fun;
import share_fun;


def getAvgDispersion(parms, NthOrder_in, extra):
    int_module = __import__(share_fun.val_def(parms, 'INTEGRATE_MOD', 'integrate'), fromlist=[]);
    NthOrder = 3;
    rot_mat = extra['rot_mat'];
    NORB = int(parms['NORB']); N = int(parms['N_LAYERS']); F = int(parms['FLAVORS']); S = int(parms['SPINS']);
    bp,wf = extra['GaussianData'];
    if 'HR' in extra:
        out = array([int_module.calc_Havg(NthOrder, extra['HR'], extra['R'], 
            float(share_fun.val_def(parms, 'H', 0))*(-1)**s, float(parms['DELTA']), bp, wf).reshape(NthOrder, NORB, NORB) for s in range(S)]);
    else: 
        out = array([int_module.calc_Havg(NthOrder, extra['tight_binding_parms'], 
            float(share_fun.val_def(parms, 'H', 0))*(-1)**s, float(parms['DELTA']), bp, wf).reshape(NthOrder, NORB, NORB) for s in range(S)]);
    ret   = zeros((NthOrder, NORB));

    rret = zeros((S, NthOrder, NORB));
    for s in range(S):
        for n in range(NthOrder):
            for L in range(N):
                tmp = mat(out[s, n, L*F:(L+1)*F, L*F:(L+1)*F]);
                dtmp = diag(rot_mat[L]*tmp*rot_mat[L].H);
                if linalg.norm(dtmp.imag) > 1e-10: print 'getAvgDispersion: imaginary part is rather large: %g'%linalg.norm(dtmp.imag);
                ret[n, L*F:(L+1)*F] = dtmp.real; 
        swap_vec = zeros((2, N*F), dtype = int);
        for L in range(N):
            for f in range(F): swap_vec[:,f*N+L] = array([f*N+L, L*F+f]);
        ret[:, swap_vec[0]] = ret[:, swap_vec[1]];
        rret[s] = ret;

    # avg dispersion of uncorrelated orbitals (no need for matrix rotation)
    for s in range(S):
        for n in range(NthOrder):
            for f in range(N*F,NORB): rret[s, n, f] = out[s, n, f, f];
    return rret[:, :NthOrder_in, :];


def calc_sym_layers(parms):
    return array([
        [0, 1],
        [0, 2],
        [0, 3]]);


def getCorrIndex(parms):
    return arange(int(parms['N_LAYERS'])*int(parms['FLAVORS']));

def getDMFTCorrIndex(parms, all = True):
    N_LAYERS = int(parms['N_LAYERS']); FLAVORS = int(parms['FLAVORS']);
    DTYPE = parms['DTYPE'];
    if   DTYPE == 'eg':  ind = array([0, 3]);
    elif DTYPE == 't2g': ind = array([1, 2, 4]);
    elif DTYPE == 'd'  : ind = array([0, 1, 2, 3, 4]);
    elif DTYPE == '3bands' : ind = array([0, 1, 2]);
    else: 
        if 'CORR_ID' in parms: ind = array([int(s) for s in parms['CORR_ID'].split()]);
        else: ind = arange(FLAVORS);
    if not all: return ind;

    ret = zeros(N_LAYERS*len(ind), dtype = int);
    for L in range(N_LAYERS):
        for n, f in enumerate(ind): ret[n*N_LAYERS+L] = f*N_LAYERS+L;
    return ret;


def getInfoForDensityCalculation(h5, it):
    p = share_fun.load_parms(h5, it);
    corr_id = getCorrIndex(p);
    dmft_id = getDMFTCorrIndex(p);
    N_LAYERS = int(p['N_LAYERS']); FLAVORS = int(p['FLAVORS']); SPINS = int(p['SPINS']); 
    NORB = int(p['NORB']); NCOR = int(p['NCOR']);
    se = h5['SolverData/selfenergy_asymp_coeffs'][:];
    se = se[se[:,0] == it][0, 1:].reshape(SPINS,2,-1);
    mu = float(p['MU']);
    Gcoefs = h5['SolverData/AvgDispersion'][:, 0, :] - mu;
    Gcoefs[:, corr_id] += se[:, 0, :];
    ret = dict({
        'G_asymp_coefs' : Gcoefs,
        'correction'    : zeros(SPINS)
        });
    ret1 = dict({
        'G_asymp_coefs' : Gcoefs[:,corr_id],
        'correction'    : zeros(SPINS)
        });

    if float(p['U']) != 0 and it > 0:
        approx_dens = fun.getDensityFromGmat(h5['ImpurityGreen/'+str(it)][:], float(p['BETA']), ret1);
        try: correct_dens = -h5['SolverData/Gtau/%d'%it][:, -1, :];
        except: correct_dens = None;
        if correct_dens is None: d = 0;
        else:
            d = zeros((SPINS, NCOR));
            for L in range(N_LAYERS): d[:, dmft_id] = correct_dens[:, dmft_id] - approx_dens[:, dmft_id];
    else: d = 0;
    ret['correction'] = zeros((SPINS, NORB));
#    ret['correction'][:, corr_id] = d; # correction not needed
    return ret;

