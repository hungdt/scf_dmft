import sys;
sys.path.append('/home/hungdt/works1/devel/scf_dmft');

from int_2dhopping import calc_Gavg;
from numpy import *;
from share_fun import grule;
from functions import rotate_all;
from scipy.interpolate import *;
from scipy.optimize import brentq;
import init;



def get_chem(w, dos, den):
    tck = splrep(w, dos);
    den_fun = lambda x: splint(w[0], x, tck) - den;
    return brentq(den_fun, w[0], w[-1]);


Erange = [-2, 2];
Npoints = 500;
delta = 0.005;
mu = 0.;
den = 1.5;

CF = array([0., 0., 0.]);
H  = 0.0;
bp, wf = grule(120);

HR, R, rot_mat = init.getHamiltonian(1, 'rham.py');
rot_mat = {};
for l in [2]:
    rot_mat[l] = [];
    for c in range(1): rot_mat[l].append(mat(eye(3)));


w = linspace(Erange[0], Erange[1], Npoints) + 1j*delta;
SE = zeros((Npoints, 3*3), dtype = 'c16');

t = 0.264;
ratio = [0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3];
for r in ratio:
    t1 = t*r;
    HR[0, 0, 0] = t;
    HR[0, 1, 1] = t1;
    
    Gavg = calc_Gavg(w, 0., mu, SE, HR, R, H, bp, wf, 0).reshape(Npoints, 3, 3);
    G = rotate_all(Gavg, rot_mat);

    dos = sum(-1/pi*G.imag, 1)/3;
    den1 = den/3.; mu1 = get_chem(w, dos, den1);
    den2 = den/6.; mu2 = get_chem(w, dos, den2);
    print "%.2f     %.4f    %.4f    %.4f"%(r, 4*t1, mu1, mu2);

    savetxt('dos%.1f.dat'%r, c_[w, dos]);
