#!/usr/bin/env python

from numpy import *;
from cppext import *;
from time import time;

NLa = random.random_integers(0,10);
CLa = 2.;
NSr = random.random_integers(0,10);
CSr = 1.;

nV = random.rand(NLa+NSr);
Nxy = 10;
Nz = 5;

start_time = time();
print ewald_sum(NLa, CLa, NSr, CSr, nV, Nxy, Nz);
print 'Time spent = ', -start_time + time();

print ('Another version');
start_time = time();
print new_ewald_sum(NLa, CLa, NSr, CSr, nV, Nxy, Nz);
print 'Time spent = ', -start_time + time();
