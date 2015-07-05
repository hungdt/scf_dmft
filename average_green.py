import os, sys;
from socket import gethostname;
from uuid import uuid1;
from mpi4py import MPI;
from numpy import *;
from functions import getDensityFromGmat, rotate_all, irotate;
from share_fun import val_def, interp_root, divideTasks;


def averageGreen(delta0, mu0, w, SelfEnergy, parms, Nd, Ntot, tuneup, extra):
    BETA     = float(parms["BETA"]);
    N_LAYERS = int(parms['N_LAYERS']); FLAVORS = int(parms['FLAVORS']); SPINS = int(parms['SPINS']);
    rot_mat = extra['rot_mat'];
    
    # calculate intersite Coulomb energy here
    Vc = zeros(N_LAYERS, dtype = float);

    # convert self energy to the C++ form
    SelfEnergy_rot = array([irotate(SelfEnergy[s], rot_mat) for s in range(SPINS)]);
    SE = array([array([s.flatten() for s in SelfEnergy_rot[n]]) for n in range(SPINS)]);

    v_delta = empty(0, 'f8');
    v_nd    = empty(0, 'f8');
    ddelta = 0.; delta_step = 1.;
    dmu = 0.; mu_step = 0.5; 
    tol = 0.003; firsttime = True;
    initial_Gasymp = extra['G_asymp_coefs'] if 'G_asymp_coefs' in extra.keys() else None;
    starting_error = 0.;
    # Delta loop
    while True:
        delta = delta0 + ddelta;
        if initial_Gasymp is not None: extra['G_asymp_coefs'][:N_LAYERS*FLAVORS] = initial_Gasymp[:N_LAYERS*FLAVORS] - ddelta;

        v_mu = empty(0, 'f8');
        v_n = empty(0, 'f8');
        # mu loop
        while True:
            mu = mu0 + dmu;
            if initial_Gasymp is not None: extra['G_asymp_coefs'] = initial_Gasymp - dmu;
            
            Gavg = integrate(w, delta, mu, SE, parms, extra, parallel = True if int(val_def(parms, 'KINT_PARALLEL', 1)) > 0 else False);
            Gavg_diag = array([[diag(Gavg[s, n]) for n in range(size(Gavg,1))] for s in range(SPINS)]);
            nf = getDensityFromGmat(Gavg_diag, BETA, extra);
            my_ntot = sum(nf) if SPINS == 2 else 2*sum(nf); # factor of 2 due to spin
            
            print "   adjust mu: " + str(mu) + " " + str(dmu) + " " + str(my_ntot);
            if firsttime: starting_error = abs(Ntot - my_ntot)/N_LAYERS; Gavg0 = Gavg.copy(); firsttime = False;
            if Ntot < 0 or abs(Ntot - my_ntot)/N_LAYERS < tol or not tuneup: break;
    
            v_mu = r_[v_mu, dmu];
            v_n = r_[v_n, my_ntot];
            if v_n.min() < Ntot and v_n.max() > Ntot: dmu = interp_root(v_mu, v_n, Ntot);
            else: dmu += (1. if my_ntot < Ntot else -1.)*mu_step;

        my_nd = sum(nf[:, :N_LAYERS*FLAVORS]);
        if tuneup: print "adjust double counting: " + str(delta) + " " + str(ddelta) + " " + str(my_nd) + " " + str(my_nd/N_LAYERS);
        if Nd < 0 or abs(Nd - my_nd)/N_LAYERS < tol or not tuneup: break;

        v_delta = r_[v_delta, ddelta];
        v_nd = r_[v_nd, my_nd];
        if v_nd.min() < Nd and v_nd.max() > Nd: ddelta = interp_root(v_delta, v_nd, Nd);
        else: ddelta += (1. if my_nd < Nd else -1.)*delta_step;

    # adjusted Gavg with mu_new = mu_0 + N*dmu and delta_new = delta_0 + N*ddelta;
    N = float(val_def(parms, 'TUNEUP_FACTOR', 1));
    if N != 1. and (ddelta != 0. or dmu != 0.) and starting_error < 50*tol:
        mu = mu0 + N*dmu;
        delta = delta0 + N*ddelta;
        Gavg = integrate(w, delta, mu, SE, parms, extra, parallel = True if int(val_def(parms, 'KINT_PARALLEL', 1)) > 0 else False);
        print 'TUNEUP_FACTOR =', N, ', final adjustment: mu = ', mu, ', dmu = ', N*dmu, '; delta = ', delta, ', ddelta = ', N*ddelta;


    Gavg = array([rotate_all(Gavg[s], rot_mat) for s in range(SPINS)]);
    Gavg0 = array([rotate_all(Gavg0[s], rot_mat, need_extra = True) for s in range(SPINS)]);
    if initial_Gasymp is not None: extra['G_asymp_coefs'] = initial_Gasymp;

    return Gavg, Gavg0, delta, mu, Vc;


# parent task
def integrate(w, DELTA, MU, SelfEnergy, parms, extra, parallel = False):
    data = {
            'w' : w,
            'DELTA' : DELTA,
            'MU'    : MU,
            'SelfEnergy': SelfEnergy,
            'Hf'    : float(val_def(parms, 'H', 0)),
            'N_LAYERS' : int(parms['N_LAYERS']),
            'FLAVORS'  : int(parms['FLAVORS']),
            'SPINS'    : int(parms['SPINS']),
            'NORB'     : int(parms['NORB']),
            'nthreads' : -1,
            'integrate_mod': val_def(parms, 'INTEGRATE_MOD', 'integrate'),
            'extra'    : extra
            };

    job_hostfile = 'LSB_DJOB_HOSTFILE';  # this should be changed depending on queuing system
    if job_hostfile not in os.environ or not parallel: return run_task(**data);

    # prepare for spawning
    parent_hostname = gethostname().split('.')[0];
    f = open(os.environ[job_hostfile]);
    node_list = f.read().split();
    f.close();

    host_list = {};
    for s in node_list:
        if s not in host_list: host_list[s] = { 'np' : 1 };
        else: host_list[s]['np'] += 1;
    data['nthreads'] = host_list[parent_hostname]['np']; 
    
    myhostfile = "child_nodefile."+uuid1().get_hex();
    f = open(myhostfile, 'w');
    for s in host_list: 
        if s != parent_hostname: f.write(s+'\n');
    f.close();

    myinfo = MPI.Info.Create();
    myinfo.Set("hostfile", myhostfile);
    np = len(host_list.keys());
    if np == 1: 
        os.system("rm %s"%myhostfile);
        return run_task(**data);    # only parent node
    running_script = os.path.abspath(os.path.split(sys.argv[0])[0]) + '/average_green.py'
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=[running_script, 'child'], maxprocs=np-1, info = myinfo);
    os.system("rm %s"%myhostfile);

    # get rank information
    rank_info = comm.gather(None, root=MPI.ROOT);
#    print >> sys.stderr, rank_info;
    for s in rank_info: host_list[s[0]]['rank'] = s[1];
    host_list[parent_hostname]['rank'] = -1;

    # prepare data
    w = data['w'];
    se = data['SelfEnergy'];
    del data['w'];
    del data['SelfEnergy'];

    # broadcast data
    ntasks_atom, displs_atom = divideTasks(len(node_list), len(w));
    pos = 0;
    for k, v in host_list.iteritems():
        host_list[k]['displs'] = displs_atom[pos];
        nt = 0;
        for n in range(v['np']): nt += ntasks_atom[pos]; pos += 1;
        host_list[k]['ntasks'] = nt;

#    print >> sys.stderr, host_list

    comm.bcast(data, root = MPI.ROOT);
    for k, v in host_list.iteritems():
        if k == parent_hostname: continue;
        displs = v['displs']; ntasks = v['ntasks']; n = v['rank'];
        comm.send( w[displs:displs+ntasks], dest = n, tag = 100*n);
        comm.send(se[:, displs:displs+ntasks, :], dest = n, tag = 100*n+1);
        comm.send(v['np'], dest = n, tag = 100*n+2);

    # run task
    displs = host_list[parent_hostname]['displs']; ntasks = host_list[parent_hostname]['ntasks']; n = host_list[parent_hostname]['rank'];
    data['w']  =  w[displs:displs+ntasks];
    data['SelfEnergy'] = se[:, displs:displs+ntasks, :];
    Gout_parent = run_task(**data);

    # collect data
    Gout = None;
    for k, v in host_list.iteritems():
        if k == parent_hostname: results = Gout_parent;
        else: n = v['rank']; results = comm.recv(source = n, tag = 100*n+3);
        if Gout is None: Gout = results;
        else: Gout = r_['1', Gout, results];

    comm.Disconnect();
    return Gout;


def run_child_task():
    comm = MPI.Comm.Get_parent();
    size = comm.Get_size();
    rank = comm.Get_rank();

    comm.gather([gethostname().split('.')[0], rank], root=0);

    data = comm.bcast(None, root = 0);
    data['w']  = comm.recv(source = 0, tag = 100*rank);
    data['SelfEnergy'] = comm.recv(source = 0, tag = 100*rank+1);
    data['nthreads'] = comm.recv(source = 0, tag = 100*rank+2);

#    print >> sys.stderr, gethostname(), rank, data['nthreads'], len(data['w']);

    out = run_task(**data);
    comm.send(out, dest = 0, tag = 100*rank+3);

    comm.Disconnect();


# task run by both parent and children
def run_task(w, DELTA, MU, SelfEnergy, Hf, N_LAYERS, FLAVORS, SPINS, NORB, nthreads, integrate_mod, extra):
#    print >> sys.stderr, gethostname(), len(w);
    int_module = __import__(integrate_mod, fromlist=[]);
    nbin = len(w);
    bp, wf = extra['GaussianData'];
    if 'HR' in extra:
        Gavg = array([int_module.calc_Gavg(w, DELTA, MU, SelfEnergy[n].copy(), extra['HR'], extra['R'], Hf*(-1)**n, bp, wf, nthreads).reshape(nbin, NORB, NORB) 
            for n in range(SPINS)]);
        # swap the Gavg to the format of my code
        swap_vec = zeros((2, N_LAYERS*FLAVORS), dtype = int);
        for L in range(N_LAYERS):
            for f in range(FLAVORS): swap_vec[:,f*N_LAYERS+L] = array([f*N_LAYERS+L, L*FLAVORS+f]);
        for s in range(SPINS):
            for n in range(len(Gavg[s])):
                Gavg[s, n,:,swap_vec[0]] = Gavg[s, n, :, swap_vec[1]];
                Gavg[s, n,swap_vec[0],:] = Gavg[s, n, swap_vec[1], :];
    else:
        Gavg = array([int_module.calc_Gavg(w, DELTA, MU, SelfEnergy[n].copy(), extra['tight_binding_parms'], Hf*(-1)**n, bp, wf, nthreads).reshape(nbin, NORB, NORB) 
            for n in range(SPINS)]);
    return Gavg;


# main part just for child only
if __name__ == '__main__':
    if sys.argv[1] == 'child': run_child_task();
    else: print >> sys.stderr, 'This is a child process for MPI integrate. Improper running.'; sys.exit();

