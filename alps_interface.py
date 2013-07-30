import pyalps.cthyb as cthyb
import pyalps.mpi as mpi 
import sys, re

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

parms = readParameters(sys.argv[1]);

int_entry = [
        'SEED', 'SWEEPS', 'THERMALIZATION', 'N_TAU', 'N_HISTOGRAM_ORDERS', 'N_MEAS', 'N_CYCLES', 'N_ORBITALS', 'N_MATSUBARA'
        ];
float_entry = ['BETA',];

for s in int_entry: parms[s] = int(parms[s]);
for s in float_entry: parms[s] = float(parms[s]);

parms.update({
    'VERBOSE'   : 1,
    'TEXT_OUTPUT' : 1,
    'MAX_TIME'  : 80000,
    'MEASURE_nn': 1,
    'SPINFLIP'  : 1
    });

if mpi.rank==0:
    for k, v in parms.iteritems(): print k, ' = ', v;

cthyb.solve(parms)
