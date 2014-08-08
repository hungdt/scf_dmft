import h5py, re, argparse, os;
import scipy.interpolate;
import scipy.optimize;
from numpy import *;


# functions for share

def getOptions(argv):
    parser = argparse.ArgumentParser(description='DMFT process.');
    parser.add_argument('-i', dest='ID', type = str, default = None, help='ID for this running');
    parser.add_argument('-n', dest='np', type = int, default = 1, help='number of processes');
    parser.add_argument('-p', '--parms', type = str, default = None, help='parameter file');
    parser.add_argument('-d', '--data', type = str, default = None, help='data file');
    parser.add_argument('opts', metavar = 'O', type = str, nargs = '*', default = None, help='more options');
    args = parser.parse_args(argv);

    np = args.np;
    
    parms = dict();
            
    if not args.parms is None:
        parms = readParameters(args.parms);
        if not args.ID is None:
            ID = args.ID;
        else:
            ID = str(val_def(parms, "ID", 0));

        if not args.data is None:
            DATA_FILE = args.data;
        elif "DATA_FILE" in parms.keys():
            DATA_FILE = parms["DATA_FILE"];
        else:
            DATA_FILE = "data_" + str(ID) + ".h5";
    else:
        ID = args.ID;
        DATA_FILE = args.data;
        if DATA_FILE is None:
            DATA_FILE = "data_" + str(ID) + ".h5";
        
    if not args.opts is None:
        for s in args.opts:
            k, v = s.split('=', 1);
            parms[k] = v;
    parms['DATA_FILE'] = DATA_FILE;
    parms['ID'] = str(ID);

    if ID is None:
        print "Please give me the ID!"
        exit();

    return parms, np, args.parms;

def readParameters(str_parms):
    comment_symbol = '(!|#|%|//)'
    propFile= open(str_parms, "rU")
    propDict= dict()
    for propLine in propFile:
        propDef= propLine.strip();
        if re.match(comment_symbol, propDef) is not None or re.search('=', propDef) is None: continue;
        tmp = re.split('=', propDef);
        name = tmp[0].strip();
        value = re.split(';', re.split(comment_symbol, tmp[1])[0])[0].strip();
        propDict[name]= value;
    propFile.close();
    return propDict;


def check_convergence(h5, iter, new, old, abstol = None, reltol = None, log = True, mixing = 1., Ntime = 3):
    if new is None or old is None: return False;
    if abstol is None and reltol is None: return False;

    if 'log_convergence' in h5.keys():
        log_conv = h5['log_convergence'][:, 2];
        count = 0;
        for i in range(1, len(log_conv)):
            if log_conv[i] > log_conv[i-1]: count += 1;
        if count >= 3: 
            print 'Cannot have a better converged result!';
#            return True; 
    v = new - old;
    if reltol is not None: tol = reltol; err = linalg.norm(v)/linalg.norm(new);
    else: tol = abstol; err = linalg.norm(v);
    ret = r_[tol, err, mixing];
    if log: 
        log_data(h5, 'log_convergence', iter, ret);
        log_conv = h5['log_convergence'][:];
        if len(log_conv) < Ntime: return False;
        log_conv = log_conv[-3:, :];
        if all(log_conv[:,2] < tol): return True;
        else: return False;
    if err < tol: return True;
    else: return False;


def log_data(h5, log_name, it, data, data_type = 'f8'):
    dt = r_[it, data];
    length = len(dt);
    if not log_name in h5.keys():
        log = h5.create_dataset(log_name, (1, length), dtype = data_type, maxshape = (None, length));
        new_pos = 0;
    else:
        log = h5[log_name];
        new_pos = log.shape[0];
        if length != log.shape[1]:
            return False;
        length = log.shape[1];
        log.resize((new_pos+1, length));
    log[new_pos, :] = dt;
    return True;

def log_density(h5, iter, parms, nf):
    N_LAYERS = int(parms['N_LAYERS']);
    MU = float(parms["MU"]);
    data = array([MU, sum(nf), sum(nf)/float(N_LAYERS)]);
    data = r_[data, nf.flatten()];
    log_data(h5, 'log_density', iter, data);

    reportfile = 'dmft.out';
    fileexist = os.path.isfile(reportfile);
    out = open(reportfile, 'a');
    if not fileexist: 
        if int(val_def(parms, 'AFM', 0)) > 0: out.write('# AFM calculation\n');
        out.write('# Iter  MU   Ntot   Ntot/site   Nd    nf\n');
    if iter == 0: out.write('\n# ID: %s\n'%parms['ID']);
    nf1 = c_[nf[:, :-1:N_LAYERS], nf[:,-1]];
    pattern = '%d   %.5f   %.2f   %.4f   %.4f   ' + len(nf1.flatten())*'  %.4f' + '\n';
    data1 =r_[MU, sum(nf), sum(nf)/float(N_LAYERS), sum(nf1[:,:-1]), nf1.flatten()];
    out.write(pattern%tuple(r_[iter, data1]));
    out.close();

def save_data(h5, it, list_str, list_data):
    assert(len(list_str) == len(list_data));
    for i in range(0, len(list_str)):
        h5.create_dataset(list_str[i]+'/'+str(it), list_data[i].shape, list_data[i].dtype, data = list_data[i]);

def save_parms(h5, iter, parms):
    grp = h5.create_group("parms/" + str(iter));
    for k, v in parms.iteritems():
        grp.create_dataset(k, dtype = h5py.new_vlen(str), data = str(v)); 

def load_parms(h5, iter):
    p = dict();
    grp = h5["parms/" + str(iter)];
    for k, v in grp.iteritems():
        p[k] = str(v[...]);
    return p;

def val_def(parms, item, default):
    if item in parms.keys(): return parms[item];
    else: return default;
    
def grule(n):
    # return : 
    #   bp : based points
    #   wf : corresponding weights
    #
    # [bp,wf]=grule(n)
    # This function computes Gauss base points and weight factors
    # using the algorithm given by Davis and Rabinowitz in 'Methods
    # of Numerical Integration', page 365, Academic Press, 1975.

    bp=zeros((n,));
    wf=zeros((n,)); 
    iter=2; 
    m=int(fix((n+1)/2));
    e1=n*(n+1);
    mm=4*m-1;
    t=(pi/(4*n+2))*arange(3,mm+1,4); 
    nn=(1-(1-1/n)/(8*n*n));
    xo=nn*cos(t);
    for j in range(1,iter+1):
        pkm1=1; 
        pk=xo;
        for k in range(2,n+1):
            t1=xo*pk; 
            pkp1=t1-pkm1-(t1-pkm1)/k+t1;
            pkm1=pk; pk=pkp1;
        den=1.-xo*xo; 
        d1=n*(pkm1-xo*pk); 
        dpn=d1/den;
        d2pn=(2*xo*dpn-e1*pk)/den;
        d3pn=(4*xo*d2pn+(2-e1)*dpn)/den;
        d4pn=(6*xo*d3pn+(6-e1)*d2pn)/den;
        u=pk/dpn;
        v=d2pn/dpn;
        h=-u*(1+(.5*u)*(v+u*(v*v-u*d3pn/(3*dpn))));
        p=pk+h*(dpn+(.5*h)*(d2pn+(h/3)*(d3pn+.25*h*d4pn)));
        dp=dpn+h*(d2pn+(.5*h)*(d3pn+h*d4pn/3));
        h=h-p/dp; 
        xo=xo+h;

    for i in range(0, xo.size):
        bp[i]=-xo[i]-h[i];
    fx=d1-h*e1*(pk+(h/2)*(dpn+(h/3)*(d2pn+(h/4)*(d3pn+(.2*h)*d4pn))));
    for i in range(0, xo.size):
        wf[i]=2*(1-bp[i]**2)/(fx[i]**2);
    if ( (m+m) > n ):
        bp[m-1]=0;
    if ( not ((m+m) == n) ):
        m=m-1;
    jj=arange(1,m+1); 
    n1j=(n+1-jj);
    bp[n1j-1]=-bp[jj-1];
    wf[n1j-1]=wf[jj-1];

    return bp, wf

def interp_root(xi, yi, y0):
    ind = argsort(xi);
    x = array(xi)[ind];
    y = array(yi)[ind] - y0;

    order = 1; # linear interpolation only, no need to be fancy
    tck = scipy.interpolate.splrep(x, y, k = order);
    fun = lambda x: scipy.interpolate.splev(x, tck);
    return scipy.optimize.bisect(fun, x[argmin(y)], x[argmax(y)]);

def divideTasks(np, nt):
    remain = nt % np;
    ret = zeros((np, ), dtype = 'i4');
    dpl = zeros((np, ), dtype = 'i4');
    ret[0] = nt/np;
    dpl[0] = 0;
    for i in range(1, np):
        ret[i] = nt/np;
        if remain > 0:
            ret[i] = ret[i] + 1;
            remain = remain - 1;
        dpl[i] = dpl[i-1] + ret[i - 1];
    return ret, dpl;


#-----MIXING-----#
class Mixing:
    def __init__(self, mix, mixing_fixed = 1):
        self.mix = mix;
        self.mixing_data = dict({
               'Vi'      : None,
               'lastF'   : None,
               'lastVi'  : None,
               'dF'      : [],
               'dV'      : [],
               'F'       : [],
               'Vo'      : [],
               'mixing_length' : 3,
        
               'linmix'        : mix - floor(mix) if floor(mix) != mix else 1.,
               'error_change'  : 0,
               'error_inc'     : 0,
               'error_incmax'  : 3,
               'error_dec'     : 0,
               'error_decmax'  : 3,
               'mix_factor'    : 2/3.,
               'last_error'    : inf
               });
        if mixing_fixed > 0: del self.mixing_data['linmix'];


    def rmm_diis_mixing(self, R):
        N = len(R);
        A = ones((N+1,N+1), dtype=float);
        A[-1,-1] = 0;
        for i in xrange(0, N):
            for j in xrange(0, N): A[i,j] = dot(R[i], R[j]);
        B = zeros(N+1); B[-1] = 1.;
        alpha = linalg.solve(A, B)[:N];
        print alpha.transpose();
        return alpha;


    def broyden_mixing(self, Fm, dV, dF, alpha = 0.7, w0 = 0.01, w = None, start = True):
        M = len(dV);
        correction = alpha*Fm;
        if M == 0 or not start: return correction;
    
        if w is None: w = ones(M);
        u = [alpha*dF[n] + dV[n] for n in range(0,M)];
        a  = array([[w[k]*w[n]*dot(conjugate(dF[n]), dF[k]) for n in range(0,M)] for k in range(0,M)]);
        cm = array([w[k]*dot(conjugate(dF[k]), Fm) for k in range(0,M)]);
        beta = linalg.inv(w0**2*identity(M) + a);
        gm = dot(cm, beta);
        print gm.transpose();
    
        for n in range(0, M): correction -= w[n]*gm[n]*u[n];
        return correction;

    def get_mixing_value(self):
        return self.mix if 'linmix' not in self.mixing_data else self.mixing_data['linmix'];

    def do_task(self, so):
        mix = self.mix;
        data = self.mixing_data;
        mixing_length = 5 if 'mixing_length' not in data else data['mixing_length'];
        if 'it' not in data: data['it'] = 0;
        else: data['it'] += 1;
        Vi = so.flatten();
        Vo = so.flatten();
        if data['Vi'] is None: 
            data['Vi'] = Vo.copy();
            return Vi;
        else: 
            this_error = linalg.norm(Vo - data['Vi'])/linalg.norm(Vo);
            data['error_change'] = this_error - data['last_error'];
            data['last_error']   = this_error;
        
        if 'start_other_mixing' not in data: data['start_other_mixing'] = False;
    
        crit0 = 1e-1;
        crit = 5*crit0 if data['start_other_mixing'] else crit0;
        if linalg.norm(Vo - data['Vi'])/linalg.norm(Vo) > crit or (mix > 0 and mix <= 1): data['start_other_mixing'] = False;
        else: data['start_other_mixing'] = True;
    
        if not data['start_other_mixing']: # linear mixing
            linmix = mix-floor(mix) if floor(mix) != mix else 1.;
            mixmin = 0.01;
            mixmax = 1.00;
            dmix   = 0.1;
            if 'linmix' in data:
                if data['error_change'] > 0:
                    data['error_dec'] = 0;
                    data['error_inc'] += 1;
                    if data['error_inc'] > data['error_incmax']:
                        data['error_inc'] = 0;
                        data['linmix'] *= data['mix_factor'];
                elif data['error_change'] < 0:
                    data['error_dec'] += 1;
                    if data['error_dec'] > data['error_decmax']:
                        data['linmix'] += dmix;
                        data['error_dec'] = 0;
                if data['linmix'] < mixmin: data['linmix'] = mixmin;
                if data['linmix'] > mixmax: data['linmix'] = mixmax;
                linmix = data['linmix'];
    
            print "Linear mixing ... mixing number = %.2f"%linmix;
            if linmix < 0.01:
                print "Mixing number is too small, probably it cannot converge";
                return None;
            Vi = Vo*linmix + (1.0-linmix)*data['Vi'];
    
        start_other_mixing = data['start_other_mixing'];
        if mix > 1 and mix <= 2 and start_other_mixing: # Broyden mixing
            alpha = mix - 1;
            print "Broyden mixing, mixing length = %d, alpha = %.2f ..."%(mixing_length, alpha);
            Vi = data['Vi'].copy();
            Fm = Vo - Vi;
            if data['lastF'] is not None:
                data['dF'].append((Fm - data['lastF'])/linalg.norm(Fm - data['lastF']));
                data['dV'].append((Vi - data['lastVi'])/linalg.norm(Fm - data['lastF']));
            data['lastVi'] = Vi; data['lastF'] = Fm;
            if len(data['dF']) > mixing_length: data['dF'].pop(0); data['dV'].pop(0);
            Vi += self.broyden_mixing(Fm, data['dV'], data['dF'], alpha, start = start_other_mixing);
    
        if mix > 2: # RMM-DIIS mixing
            data['F'].append(Vo-data['Vi']);
            data['Vo'].append(Vo);
            if len(data['F']) > mixing_length: 
                data['F'].pop(0);
                data['Vo'].pop(0);
            if start_other_mixing:
                print "RMM-DIIS mixing, mixing_length = %d ..."%mixing_length;
                alpha = self.rmm_diis_mixing(data['F']);
                Vi = zeros(Vo.shape, dtype = Vo.dtype);
                for i in xrange(0, len(alpha)):  Vi += alpha[i]*data['Vo'][i];
    
        data['Vi'] = Vi.copy();
        return Vi;
#-----END MIXING-----#
