#!/usr/bin/env python

import h5py, time, sys, os;
import re
import init, solver, system_dependence as system;

from numpy import *;
from average_green import averageGreen;
from functions import *;
from share_fun import *;


# 
# INIT: for the details of variables, see return of init.py
#

# read input arguments
parms, np, parms_file = getOptions(sys.argv[1:]);

# create or open database (HDF5)
h5file = h5py.File(parms['DATA_FILE'], 'a');
vars_dict = init.initialize(h5file, parms);
for k, v in vars_dict.iteritems(): exec('%s=v'%k);


#
# DMFT, DMFT, DMFT !!!
#
set_printoptions(suppress=True, precision=4, linewidth=150);
for k, v in parms.iteritems(): print k + " = " + str(v) + ";";
print "\n\n"

parms['MEASURE'] = int(val_def(parms, 'MEASURE', -1));
if 'OBSERVABLES' in parms and parms['MEASURE'] < 0: parms['MEASURE'] = 0;
if int(val_def(parms, 'ADJUST_DELTA_ONCE', 1)) > 0: Nd = -1; print "Only adjust double counting at the beginning";

it = h5["iter"][0];
SelfEnergy_out = h5['SelfEnergy/%d'%it][:];
mu_out = mu_in = float(parms['MU']);
delta_out = delta_in = float(parms['DELTA']);
while True:
    # start a new iteration
    MaxIter = int(val_def(parms, "MAX_ITER", 20));
    if it >= MaxIter and parms['MEASURE'] < 0: break;
    it += 1; 
    if parms['MEASURE'] > 0:
        print 'Final loop for measuring'
        for k, v in parms.iteritems():
            if re.match('^FINAL_', k):
                key_to_change = re.split('^FINAL_', k)[1]
                parms[key_to_change] = v
                print '  %s=%s'%(key_to_change, str(v))
                if key_to_change == 'CUTOFF_FREQ':
                    parms['N_CUTOFF'] = int(round((float(parms['BETA'])/pi*float(v) - 1)/2.));
                    print '  %s=%s'%('N_CUTOFF', str(parms['N_CUTOFF']))
    time_spent = time.time();
    print "ITERATION: %d\n"%it;

    # average over k-space for Gavg then produce Weiss field
    # use Gauss quadrature with C++ extension code
    # mixing self energy
    tmp = mixer_SE.do_task(r_[SelfEnergy_out.real.flatten(), SelfEnergy_out.imag.flatten()]).reshape(2, -1);
    SelfEnergy_in = (tmp[0] + 1j*tmp[1]).reshape(SelfEnergy_out.shape);
    mu_orig = mu_in; delta_orig = delta_in;
    if str(it) not in h5['WeissField']:
        print "Tedious part: averaging Green function";
        extra.update(system.getInfoForDensityCalculation(h5, it-1));
        print 'Density correction = \n', extra['correction'][:, :NCOR];
        
        Gavg, Gavg0, delta_out, mu_out, VCoulomb = averageGreen(delta_in, mu_in, 1j*wn, SelfEnergy_in, parms, Nd, DENSITY, int(val_def(parms, 'NO_TUNEUP', 0)) == 0, extra);
        delta_in, mu_in = mixer.do_task(r_[delta_out, mu_out]);
        if mu_in != mu_out or delta_in != delta_out:
            print 'There is mixing, average Green function once more';
            extra['G_asymp_coefs'][:N_LAYERS*int(parms['FLAVORS'])] -= delta_out - delta_in;                                                           
            extra['G_asymp_coefs'] -= mu_out - mu_in;
            Gavg, Gavg0, delta_in, mu_in, VCoulomb = averageGreen(delta_in, mu_in, 1j*wn, SelfEnergy_in, parms, Nd, DENSITY, False, extra);
            parms["DELTA"] = delta_in;
            parms["MU"] = mu_in;    # just want MU shown in log_density corresponding to nf

        nf = getDensityFromGmat(Gavg0, float(parms['BETA']), extra);
        nf = c_[nf[:,:NCOR], sum(nf[:,NCOR:], 1)];
        log_density(h5, it-1, parms, nf);

        parms["DELTA"] = delta_in;
        parms["MU"] = mu_in;
        h5['parms/%d/DELTA'%it][...] = str(delta_in);
        h5['parms/%d/MU'%it][...] = str(mu_in);

        aWeiss = 1./Gavg[:, :, corr_id] + SelfEnergy_in;
        save_data(h5, it, ['avgGreen', 'WeissField', 'StaticCoulomb'], [Gavg0, aWeiss, VCoulomb]);

        NthOrder = 3;
        dispersion_avg = system.getAvgDispersion(parms, NthOrder, extra);
        h5['SolverData/AvgDispersion'][:] = dispersion_avg;

    else: 
        Gavg0 = h5['avgGreen/%d'%it][:];
        aWeiss = h5['WeissField/%d'%it][:];
        VCoulomb = h5['StaticCoulomb/%d'%it][:];

    time_spent = r_[time_spent, time.time()];
 
  
    # run the solver here and get Gimp
    # need: path for data file, iteration number, layer index
    if str(it) not in h5['SelfEnergy']:
        print "Tedious part: Running impurity solver %d times"%N_LAYERS;
        dispersion_avg = h5['SolverData/AvgDispersion'][:];
        nf = getDensity(h5, it-1);
        h5file.close(); del h5;
        tmph5filename = solver.run_solver(dispersion_avg, nf, 1j*wn, it, parms, aWeiss, np, VCoulomb);
        if tmph5filename is None: print >> sys.stderr, "Something wrong while running the solver"; break;
        h5file = h5py.File(parms['DATA_FILE'], 'a');
        h5 = h5file[parms['ID']];
        Gimp, SelfEnergy_out = solver.solver_post_process(parms, aWeiss, h5, tmph5filename);
        if SelfEnergy_out is None: break;
        save_data(h5, it, ['ImpurityGreen', 'SelfEnergy'], [Gimp, SelfEnergy_out]);
    else: SelfEnergy_out = h5['SelfEnergy/%d'%it][:];
    time_spent = r_[time_spent, time.time()];

    # finish the iteration
    time_spent = array(diff(time_spent), dtype = int);
    log_data(h5, 'log_time', it, r_[time_spent, sum(time_spent)], data_type = int);

    # check if needs to adjust parms
    new_parms = parms_file; 
    if os.path.isfile(new_parms):
        parms_new = readParameters(new_parms);
        print 'Check for updating parameters';
        updated = False;
        for k, v in parms_new.iteritems(): 
            if k not in parms or str(parms[k]) != str(v): 
                print k, ' = ', v
                parms[k] = v
                updated = True;
                if k == 'MU': 
                    mu_in = float(parms_new['MU'])
                    print '  chemical potential is forced to be %s'%parms_new['MU']
        if not updated: print 'no new parameters.';
    save_parms(h5, it+1, parms);

    h5["iter"][...] = it; # this is the mark that iteration 'it' is done
    print "Time for iteration %d: %d, %d, %d\n"%(it, time_spent[0], time_spent[1], sum(time_spent));

    # check stop condition
    # generate criterion for convergence: DOS at Fermi level
#    DOS_in  = getFermiDOS(Gavg0, float(parms['BETA']));
#    Gavg0 = averageGreen(delta_in, mu_in, 1j*wn, SelfEnergy_out, parms, Nd, DENSITY, False, extra)[1];
#    DOS_out = getFermiDOS(Gavg0, float(parms['BETA']));
#    DOS_in = c_[DOS_in[:,:NCOR:N_LAYERS], sum(DOS_in[:, NCOR:], 1)/N_LAYERS];
#    DOS_out = c_[DOS_out[:,:NCOR:N_LAYERS], sum(DOS_out[:, NCOR:], 1)/N_LAYERS];
    print 'End iteration %d\n\n'%it;

    if check_convergence(h5, it, r_[mu_orig, delta_orig, SelfEnergy_in.flatten()], r_[mu_in, delta_in, SelfEnergy_out.flatten()], 
            abstol = float(val_def(parms, 'TOLERANCE', 0.001*int(parms['SPINS']))), mixing = mixer.get_mixing_value(), Ntime=3):
        print 'CONVERGE!'
        if parms['MEASURE'] == 0: parms['MEASURE'] = 1; 
        else: break;
    # check if it goes beyond max iter, measure and stop
    elif parms['MEASURE'] > 0: break;
    if it >= MaxIter:
        if parms['MEASURE'] < 0: break;
        else: parms['MEASURE'] = 1;

# the end
h5file.close();
