import pyalps.cthyb as cthyb
import pyalps.mpi as mpi 
import sys, re

def readParameters(str_parms):
    comment_symbol = '(!|#|%|//)'
    propFile= open(str_parms, "rU")
    propDict= dict()
    for propLine in propFile:
        propDef= propLine.strip();
        if re.match(comment_symbol, propDef) is not None \
           or re.search('=', propDef) is None: continue;
        tmp = re.split('=', propDef);
        name = tmp[0].strip();
        value = re.split(';', re.split(comment_symbol, tmp[1])[0])[0].strip();
        propDict[name]= value;
    propFile.close();
    return propDict;

parms = readParameters(sys.argv[1]);
parms.update({
    'VERBOSE'   : 0,
    'TEXT_OUTPUT' : 1,
    'MEASURE_nn': 1,
    'SPINFLIP'  : 1,
    'GLOBALFLIP' : 1,
    });
if mpi.rank==0: 
    for k, v in parms.iteritems(): print k, ' = ', v;
cthyb.solve(parms)
