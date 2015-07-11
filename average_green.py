import os
import sys
from socket import gethostname
from uuid import uuid1
from mpi4py import MPI
from numpy import *
from functions import getDensityFromGmat, rotate_all, irotate
from share_fun import interp_root, divideTasks


def averageGreen(delta0, mu0, w, SelfEnergy, parms, Nd, Ntot, tuneup, extra):
    N_LAYERS = int(parms['N_LAYERS'])
    FLAVORS = int(parms['FLAVORS'])
    SPINS = int(parms['SPINS'])
    rot_mat = extra['rot_mat']
    parallel = int(parms.get('KINT_PARALLEL', 2))
    
    # calculate intersite Coulomb energy here
    Vc = zeros(N_LAYERS, dtype=float)

    # convert self energy to the C++ form
    SelfEnergy_rot = array([irotate(SelfEnergy[s], rot_mat)
                           for s in range(SPINS)])
    SE = array([array([s.flatten() for s in SelfEnergy_rot[n]])
                for n in range(SPINS)])

    v_delta = array([])
    ddelta = 0.
    delta_step = 1.
    v_nd = array([])
    dmu = 0.
    mu_step = 0.5

    tol = 0.003
    firsttime = True
    initial_Gasymp = extra['G_asymp_coefs'] if 'G_asymp_coefs' in extra.keys()\
                     else None
    starting_error = 0.
    # Delta loop
    while True:
        delta = delta0 + ddelta
        if initial_Gasymp is not None:
            extra['G_asymp_coefs'][:N_LAYERS*FLAVORS] = initial_Gasymp[:N_LAYERS*FLAVORS] - ddelta
        v_mu = array([]) 
        v_n = array([])

        # mu loop
        while True:
            mu = mu0 + dmu
            if initial_Gasymp is not None:
                extra['G_asymp_coefs'] = initial_Gasymp - dmu
            
            Gavg = integrate(w, delta, mu, SE, parms, extra, parallel)
            Gavg_diag = array([[diag(Gavg[s, n]) for n in range(size(Gavg,1))]
                               for s in range(SPINS)])
            nf = getDensityFromGmat(Gavg_diag, float(parms['BETA']), extra)
            my_ntot = sum(nf) if SPINS == 2 else 2*sum(nf)

            print "   adjust mu: %.5f  %.5f  %.5f"%(mu, dmu, my_ntot)
            if firsttime:
                starting_error = abs(Ntot - my_ntot)/N_LAYERS
                Gavg0 = Gavg.copy()
                firsttime = False
            if Ntot < 0 or abs(Ntot - my_ntot)/N_LAYERS < tol or not tuneup:
                break
    
            v_mu = r_[v_mu, dmu]
            v_n = r_[v_n, my_ntot]
            if v_n.min() < Ntot and v_n.max() > Ntot:
                dmu = interp_root(v_mu, v_n, Ntot)
            else:
                dmu += (1. if my_ntot < Ntot else -1.)*mu_step

        my_nd = sum(nf[:, :N_LAYERS*FLAVORS])
        if tuneup:
            print ('adjust double counting: %.5f  %.5f  '
                   '%.5f  %.5f')%(delta, ddelta, my_nd, my_nd/N_LAYERS)
        if Nd < 0 or abs(Nd - my_nd)/N_LAYERS < tol or not tuneup:
            break

        v_delta = r_[v_delta, ddelta]
        v_nd = r_[v_nd, my_nd]
        if v_nd.min() < Nd and v_nd.max() > Nd:
            ddelta = interp_root(v_delta, v_nd, Nd)
        else:
            ddelta += (1. if my_nd < Nd else -1.)*delta_step

    # adjusted Gavg with mu_new = mu_0 + N*dmu and
    # delta_new = delta_0 + N*ddelta;
    N = float(parms.get('TUNEUP_FACTOR', 1))
    if N != 1. and (ddelta != 0. or dmu != 0.) and starting_error < 50*tol:
        mu = mu0 + N*dmu
        delta = delta0 + N*ddelta
        Gavg = integrate(w, delta, mu, SE, parms, extra, parallel)
        print ('TUNEUP_FACTOR = %d final adjustment: mu = %.4f, dmu = %.4f, '
               'delta = %.4f, ddelta = %.4f')%(N, mu, N*dmu, delta, N*ddelta)

    Gavg = array([rotate_all(Gavg[s], rot_mat) for s in range(SPINS)])
    Gavg0 = array([rotate_all(Gavg0[s], rot_mat, need_extra = True)
                   for s in range(SPINS)])
    if initial_Gasymp is not None:
        extra['G_asymp_coefs'] = initial_Gasymp
    return Gavg, Gavg0, delta, mu, Vc


def integrate(w, DELTA, MU, SelfEnergy, parms, extra, parallel=0):
    """
    Main function for k-integration

    Input argument: parallel
        0: no parallel computation (nthreads = 1)
        1: OpenMP computation (nthreads = -1, use OMP_NUM_THREADS)
        2: MPI computation (but no OpenMP, nthreads = 1)
    """
    data = {
            'w' : w,
            'DELTA' : DELTA,
            'MU' : MU,
            'SelfEnergy': SelfEnergy,
            'Hf' : float(parms.get('H', 0)),
            'N_LAYERS' : int(parms['N_LAYERS']),
            'FLAVORS' : int(parms['FLAVORS']),
            'SPINS' : int(parms['SPINS']),
            'NORB' : int(parms['NORB']),
            'nthreads' : 1,
            'integrate_mod': parms.get('INTEGRATE_MOD', 'integrate'),
            'extra' : extra,
            }

    # this should be changed depending on queuing system
    if parallel < 2 or parms['np'] == 1:
        if parallel == 1: 
            data['nthreads'] = -1
#            print 'average_green.integrate: OpenMP parallelization'
#        print 'average_green.integrate: no MPI parallelization'
        return run_task(**data)
#    else: print 'average_green.integrate: MPI parallelization'

    # prepare for spawning
    nprocs = parms['np'] - 1
    myinfo = MPI.Info.Create()
    for job_hostfile in ('LSB_DJOB_HOSTFILE', 'PBS_NODEFILE'):
        if job_hostfile in os.environ:
            myinfo.Set("hostfile", os.environ[job_hostfile])
            print 'Found MPI hostfile: %s'%job_hostfile
    running_script = __file__                     
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=[running_script, 'child'],
                               maxprocs=nprocs, info=myinfo)

    # prepare data
    w = data['w']
    se = data['SelfEnergy']
    del data['w']
    del data['SelfEnergy']

    # broadcast data
    comm.bcast(data, root=MPI.ROOT)
    ntasks, displs = divideTasks(nprocs+1, len(w))  # also count the parent

    # this is data for parent process
    data['w'] = w[displs[0]:displs[0] + ntasks[0]]
    data['SelfEnergy'] = se[:, displs[0]:displs[0]+ntasks[0]]

    ntasks = ntasks[1:]
    displs = displs[1:]
    for n in range(nprocs):
        comm.send(w[displs[n]:displs[n]+ntasks[n]], dest=n, tag=100*n)
        comm.send(se[:, displs[n]:displs[n]+ntasks[n]], dest=n, tag=100*n+1)

    # run task from parent process
    Gout = run_task(**data)
    
    # collect data
    for n in range(nprocs):
        results = comm.recv(source=n, tag=100*n+3)
        Gout = r_['1', Gout, results]

    comm.Disconnect()
    return Gout


def run_child_task():
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
#    print 'child task rank %d/%d at %s'%(rank, comm.Get_size(), gethostname())

    data = comm.bcast(None, root=0)
    data['w']  = comm.recv(source=0, tag=100*rank)
    data['SelfEnergy'] = comm.recv(source=0, tag=100*rank+1)

    out = run_task(**data)
    comm.send(out, dest=0, tag=100*rank+3)
    comm.Disconnect()


# task run by both parent and children
def run_task(w, DELTA, MU, SelfEnergy, Hf, N_LAYERS, FLAVORS, SPINS, NORB,
             nthreads, integrate_mod, extra):
#    print >> sys.stderr, gethostname(), len(w);
    int_module = __import__(integrate_mod, fromlist=[])
    nbin = len(w)
    bp, wf = extra['GaussianData']
    if 'HR' in extra:
        Gavg = array([int_module.calc_Gavg(w, DELTA, MU, SelfEnergy[n].copy(),
                                           extra['HR'], extra['R'],
                                           Hf*(-1)**n, bp, wf,
                                           nthreads).reshape(nbin, NORB, NORB) 
                     for n in range(SPINS)])
        # swap the Gavg to the format of my code
        swap_vec = zeros((2, N_LAYERS*FLAVORS), dtype=int)
        for L in range(N_LAYERS):
            for f in range(FLAVORS):
                swap_vec[:,f*N_LAYERS+L] = array([f*N_LAYERS+L, L*FLAVORS+f])
        for s in range(SPINS):
            for n in range(len(Gavg[s])):
                Gavg[s, n,:,swap_vec[0]] = Gavg[s, n, :, swap_vec[1]]
                Gavg[s, n,swap_vec[0],:] = Gavg[s, n, swap_vec[1], :]
    else:
        Gavg = array([int_module.calc_Gavg(w, DELTA, MU, SelfEnergy[n].copy(),
                                           extra['tight_binding_parms'],
                                           Hf*(-1)**n, bp, wf,
                                           nthreads).reshape(nbin, NORB, NORB) 
                     for n in range(SPINS)])
    return Gavg


# main part just for child only
if __name__ == '__main__':
    if sys.argv[1] == 'child':
        run_child_task();
    else:
        print >> sys.stderr, ('This is a child process for MPI integrate. '
                              'Improper running.')
        sys.exit()
