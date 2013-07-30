import os, sys;
import h5py, user_config, cppext;
from numpy import *;
from share_fun import val_def;
from functions import generate_Umatrix;


def init_solver(parms, np):
    solver_type = parms['SOLVER_TYPE'];
    print '%s solver is used...'%solver_type;
    input_args = {
            'solver_path' : val_def(parms, 'SOLVER_EXE_PATH', user_config.solver_matrix),
            'mpirun_path' : val_def(parms, 'SOLVER_MPIRUN_PATH', user_config.mpirun),
            'np'          : np
            }

    if solver_type == 'CTHYB_Matrix': 
        input_args['parm2xml'] = val_def(parms, 'PARMS2XML', user_config.parm2xml);
        input_args['solver_path'] = user_config.solver_matrix;
        solver = HybridizationMatrixSolver(input_args);
    elif solver_type == 'CTHYB_Segment': 
        input_args['solver_path'] = user_config.solver_segment;
        solver = HybridizationSegmentSolver(input_args);
    elif solver_type == 'TRIQS':
        input_args['solver_path'] = user_config.solver_triqs;
        solver = TRIQSSolver(input_args);

    else: print 'Solver %s unknown'%solver_type;
    return solver;



class HybridizationMatrixSolver:
    def __init__(self, input_args):
        self.args = input_args;

    def prepare(self, prefix, in_data):
        # prepare hybtau file for CTQMC
        print 'Prepare running solver for ' + prefix;
        self.prefix = prefix;
        parms   = in_data['parms'];
        hyb_tau = in_data['hybtau'];
        FLAVORS  = int(parms['FLAVORS']);
        for f in range(FLAVORS): hyb_tau[:, f] = -hyb_tau[::-1, f];
        hyb_tau = c_[linspace(0, float(parms['BETA']), int(parms['N_TAU']) + 1), hyb_tau];
        savetxt(prefix+'.hybtau', hyb_tau);
   
        if FLAVORS/2 == 3: Lattice = '"t2g system"';
        if FLAVORS/2 == 2: Lattice = '"eg system"';
        if FLAVORS/2 == 1: Lattice = '"site"';
        # prepare parms file for CTQMC
        green_only = 1; self.list_obs = None;
        if int(parms['MEASURE']) > 0: green_only = 0; self.list_obs = parms['OBSERVABLES'].split();
        QMC_parms = {
                'LATTICE_LIBRARY'         : user_config.LatticeLibrary,
                'LATTICE'                 : Lattice,
                'MODEL_LIBRARY'           : user_config.ModelLibrary,
                'MODEL'                   : user_config.Model,
                'L'                       : FLAVORS/2,
                'SITES'                   : FLAVORS/2,
                'GREEN_ONLY'              : green_only,
    
                'SEED'                    : random.random_integers(10000),
                'SWEEPS'                  : val_def(parms, 'SWEEPS', 500000),
                'THERMALIZATION'          : val_def(parms, 'THERMALIZATION', 300),
                'N'                       : parms['N_TAU'],
                'N_ORDER'                 : val_def(parms, 'N_ORDER', 50),
                'N_MEAS'                  : val_def(parms, 'N_MEAS', 200),
                'N_SHIFT'                 : val_def(parms, 'N_SHIFT', 0),
                'N_SWAP'                  : val_def(parms, 'N_SWAP', 0),
    
                'BETA'                    : parms['BETA'],
                'U'                       : parms['U'],
                "U'"                      : float(parms['U']) - 2*float(parms['J']),
                'J'                       : parms['J'],
                'SPINS'                   : 2,
                'CONSERVED_QUANTUMNUMBERS': '"Nup, Ndown"',
                'F'                       : prefix + '.hybtau'
                };
          
        for f in range(FLAVORS/2):  
            QMC_parms['MUUP'+str(f)] = in_data['MU'][2*f];
            QMC_parms['MUDOWN'+str(f)] = in_data['MU'][2*f+1];
    
        solver_parms_file = open(prefix + '.parms', 'w');
        for k, v in QMC_parms.iteritems():
            solver_parms_file.write(k + ' = ' + str(v) + ';\n');
        solver_parms_file.write('{}');
        solver_parms_file.close();

    def run(self):
        cmd = '%s %s.parms %s 1>&2'%(self.args['parm2xml'], self.prefix,self.prefix);
        print cmd; os.system(cmd);
        cmd = '%s -n %d %s %s.in.xml'%(self.args['mpirun_path'], self.args['np'], self.args['solver_path'], self.prefix);
        print cmd; return os.system(cmd);

    def collect(self):
        print 'Collect data from ' + self.prefix;
        measure = 0;
        collect_error = False;
        if self.list_obs is not None:
            print 'also collect data for observables ', self.list_obs;
            if 'error' in self.list_obs: 
                collect_error = True;
                self.list_obs.pop(self.list_obs.index('error'));
            measure = 1;
        Gtau, Gerr, obs = cppext.get_raw_data(self.prefix, measure, self.list_obs);
        if collect_error: obs.update({'GreenError' : mean(Gerr, 0) });
        return Gtau, obs;



class HybridizationSegmentSolver:
    def __init__(self, input_args):
        self.args = input_args;

    def prepare(self, prefix, in_data):
        # prepare hybtau file for CTQMC
        print 'Prepare running solver for ' + prefix;
        self.prefix = prefix;
        self.list_obs = None;
        self.parms = in_data['parms'];
        self.MEASURE_freq = int(val_def(in_data['parms'], 'MEASURE_freq', 1));

        parms   = in_data['parms'];
        FLAVORS  = int(parms['FLAVORS']);

        # prepare parms file for CTQMC
        QMC_parms = {
                'SEED'                    : random.random_integers(10000),
                'SWEEPS'                  : int(val_def(parms, 'SWEEPS', 500000)),
                'THERMALIZATION'          : int(val_def(parms, 'THERMALIZATION', 300)),
                'N_TAU'                   : int(parms['N_TAU']),
                'N_HISTOGRAM_ORDERS'      : int(val_def(parms, 'N_ORDER', 50)),
                'N_MEAS'                  : int(val_def(parms, 'N_MEAS', 100)),
                'N_CYCLES'                : int(val_def(parms, 'N_CYCLES', 30)),

                'BETA'                    : float(parms['BETA']),
                'U_MATRIX'                : self.prefix+'.Umatrix',
                'MU_VECTOR'               : self.prefix+'.MUvector',

                'BASENAME'                : prefix,
                'DELTA'                   : prefix + '.hybtau',
                'N_ORBITALS'              : FLAVORS,

                'MEASURE_freq'            : self.MEASURE_freq,
                'N_MATSUBARA'             : int(parms['N_CUTOFF']),
                };

        self.Norder = QMC_parms['N_HISTOGRAM_ORDERS'];
        solver_parms_file = open(prefix + '.parms', 'w');
        for k, v in QMC_parms.iteritems(): solver_parms_file.write(k + ' = ' + str(v) + ';\n');

        # Umatrix: either Slater-Kanamori form or using Slater integrals
        Umatrix = generate_Umatrix(float(parms['U']), float(parms['J']), 
                FLAVORS/2, val_def(parms, 'INTERACTION_TYPE', 'SlaterKanamori'));
        hyb_tau = in_data['hybtau'];
        hyb_tau = c_[linspace(0, float(parms['BETA']), int(parms['N_TAU']) + 1), hyb_tau];
        savetxt(prefix+'.hybtau', hyb_tau);
        savetxt(self.prefix+'.Umatrix', Umatrix);
        savetxt(self.prefix+'.MUvector', in_data['MU']);


    def run(self):
        FLAVORS = int(self.parms['FLAVORS']);
        cmd = '%s -n %d %s %s.parms 1>&2'%(self.args['mpirun_path'], self.args['np'], self.args['solver_path'], self.prefix);
        print cmd; 
        retval = os.system(cmd);
        gh5 = h5py.File('%s.out.h5'%self.prefix, 'r');
        sign = gh5['/simulation/results/Sign/mean/value'][...];
        if sign < 0.99: print >> sys.stderr, 'sign = %.4f: Run QMC again for %s!'%(sign, self.prefix); retval = 1;
        for i in range(FLAVORS):
            norder = gh5['/simulation/results/order_%d/mean/value'%i][...];
            if norder > self.Norder: 
                print sys.stderr >> "mean Norder of flavor %d > Norder = %d"%(norder, self.Norder);
                retval = 1;
        gh5.close(); del gh5;
        return retval;


    def collect(self):
        print 'Collect data from ' + self.prefix;
        FLAVORS = int(self.parms['FLAVORS']);
        obs = None;
        gh5 = h5py.File('%s.out.h5'%self.prefix, 'r');

        Gtau = array([gh5['/G_tau/%d/mean/value'%f][:] for f in range(FLAVORS)]).T;
        Serr = None;
        if self.MEASURE_freq:
            Giwn = array([gh5['/G_omega/%d/mean/value'%f][:, 0] + 1j*gh5['/G_omega/%d/mean/value'%f][:, 1] for f in range(FLAVORS)]).T;
            Siwn = array([gh5['/S_omega/%d/mean/value'%f][:, 0] + 1j*gh5['/S_omega/%d/mean/value'%f][:, 1] for f in range(FLAVORS)]).T;
            if int(self.parms['MEASURE']) > 0:
                if 'error' in self.parms['OBSERVABLES']:
                    Serr = zeros((len(Siwn), FLAVORS));
                    for f in range(FLAVORS):
                        Fval = gh5['simulation/results/fw_re_%d/mean/value'%f][:] + 1j*gh5['simulation/results/fw_im_%d/mean/value'%f][:];
                        Ferr = gh5['simulation/results/fw_re_%d/mean/error'%f][:] + 1j*gh5['simulation/results/fw_im_%d/mean/error'%f][:];
                        Gval = gh5['simulation/results/gw_re_%d/mean/value'%f][:] + 1j*gh5['simulation/results/gw_im_%d/mean/value'%f][:];
                        Gerr = gh5['simulation/results/gw_re_%d/mean/error'%f][:] + 1j*gh5['simulation/results/gw_im_%d/mean/error'%f][:];
                        Serr[:, f] = abs(Fval/Gval) * sqrt(abs(Ferr/Fval)**2 + abs(Gerr/Gval)**2);

        nn = array([]);
        nf = -Gtau[-1, :];
        for i in range(FLAVORS):
            for j in range(i+1):
                if i == j: tmp = nf[i];
                else: tmp = gh5['/simulation/results/nn_%d_%d/mean/value'%(i,j)][...];
                nn = r_[nn, tmp];

        gh5.close();
        obs = { 'nn' : nn };
        if Serr is not None: obs.update({'SelfEnergyError': Serr});

        if self.MEASURE_freq: return Gtau, obs, Giwn, Siwn;
        else: return Gtau, obs;



class TRIQSSolver:
    def __init__(self, input_args):
        self.args = input_args;

    def prepare(self, prefix, in_data):
        print 'Prepare running solver for ' + prefix;
        self.prefix = prefix;

        parms = in_data['parms'];
        BETA = float(parms['BETA']);
        NCOR = int(parms['FLAVORS']) / 2;
        self.Ntau = int(parms['N_TAU']) + 1;
        self.Ncor = NCOR;

        hyb_mat = in_data['hybmat'];
        hyb_tail = in_data['hybtail'];
        wn = (2*arange(size(hyb_mat, 0))+1)*pi/BETA;
        savetxt(prefix+'.hybmat.real', c_[wn, hyb_mat.real]);
        savetxt(prefix+'.hybmat.imag', c_[wn, hyb_mat.imag]);
        savetxt(prefix+'.hybmat.tail', hyb_tail);
        savetxt(prefix+'.MUvector', in_data['MU']);

        Umatrix = generate_Umatrix(float(parms['U']), float(parms['J']), 
                NCOR, val_def(parms, 'INTERACTION_TYPE', 'SlaterKanamori'));
        savetxt(prefix+'.Umatrix', Umatrix);

        # prepare parms file for CTQMC
        QMC_parms = {
                'SWEEPS_EACH_NODE' : int(val_def(parms, 'SWEEPS', 500000))/self.args['np'],
                'THERMALIZATION'   : val_def(parms, 'THERMALIZATION', 300),
                'N_MEAS'           : val_def(parms, 'N_MEAS', 50),

                'BETA'             : parms['BETA'],
                'U_MATRIX'         : prefix+'.Umatrix',
                'MU_VECTOR'        : prefix + '.MUvector',

                'HYB_MAT'          : prefix + '.hybmat',
                'NCOR'             : NCOR,
                'HDF5_OUTPUT'      : prefix + '.solution.h5',

                'N_LEGENDRE'       : val_def(parms, 'TRIQS_N_LEGENDRE', 50),
                'ACCUMULATION'     : val_def(parms, 'TRIQS_ACCUMULATION', 'legendre'),
                'SPINFLIP'         : val_def(parms, 'TRIQS_SPINFLIP', 1),
                };

        solver_parms_file = open(prefix + '.parms', 'w');
        for k, v in QMC_parms.iteritems(): solver_parms_file.write(k + ' = ' + str(v) + ';\n');


    def run(self):
        cmd = '%s -n %d %s %s.parms 1>&2'%(self.args['mpirun_path'], self.args['np'], self.args['solver_path'], self.prefix);
        print cmd; 
        retval = os.system(cmd);
        return retval;


    def collect(self):
        print 'Collect data from ' + self.prefix;
        R = h5py.File(self.prefix+'.solution.h5', 'r');
        BETA = R['G/up0/Mesh/Beta'][...];
        SPINS = 2; spins = ('up', 'dn');
        NCOR = self.Ncor;
        G = []; S = []; nf = []; Gl = [];
        is_legendre = True if 'G_Legendre' in R else False;
        for f in range(NCOR):
            for sp in spins:
                G.append(R['G/%s%d/Data'%(sp,f)][0, 0, :, 0] + 1j*R['G/%s%d/Data'%(sp,f)][0, 0, :, 1]);
                if is_legendre: Gl.append(R['G_Legendre/%s%d/Data'%(sp,f)][0, 0, :]);
                S.append(R['Sigma/%s%d/Data'%(sp,f)][0, 0, :, 0] + 1j*R['Sigma/%s%d/Data'%(sp,f)][0, 0, :, 1]);
                nf.append(R['Observables/N_%s%d'%(sp,f)][...]);
        Giwn = array(G).T;
        Siwn = array(S).T;
        nf   = array(nf);

        nn = array([]);
        for i in range(SPINS*NCOR):
            for j in range(i+1):
                if i == j: tmp = nf[i];
                else: tmp = R['Observables/nn_%d_%d'%(i,j)][...];
                nn = r_[nn, tmp];
        obs = { 'nn' : nn };

        Gtau = zeros((self.Ntau, SPINS*NCOR), dtype = float);
        for f in range(SPINS*NCOR):
            Gtau[:, f] = cppext.IFT_mat2tau(Giwn[:, f].copy(), self.Ntau, BETA, 1.0, 0.0);
        Gtau[-1, :] = -nf;
        Gtau[0,  :] = -(1-nf);
        
        Stail = zeros((len(R['Sigma/up0/Tail/array'][:]), NCOR*SPINS), dtype = complex);
        order_min = R['Sigma/up0/Tail/OrderMinMIN'][...];
        order_max = R['Sigma/up0/Tail/OrderMaxMAX'][...];
        for f in range(NCOR):
            for s, sp in enumerate(spins):
                for n in range(len(Stail)):
                    Stail[n, 2*f+s] = R['Sigma/%s%d/Tail/array'%(sp,f)][n, 0, 0, 0] + 1j*R['Sigma/%s%d/Tail/array'%(sp,f)][n, 0, 0, 1];
        Stail = r_[order_min*ones((1, NCOR*SPINS)), Stail, order_max*ones((1, NCOR*SPINS))];
        if is_legendre: 
            GLegendre = array(Gl).T;
            obs = { 'SelfEnergyTail' : Stail, 'GLegendre' : GLegendre , 'nn' : nn };
        else: obs = { 'SelfEnergyTail' : Stail, 'nn' : nn };
        return Gtau, obs, Giwn, Siwn;
