#!/bin/bash

NN=32

# the path to dmft.py in the main code
PYDMFT=../../dmft.py

# run dmft.py --help for some more info
# here -p for parameter file
# and  -n for the number of cores in use
$PYDMFT -p parms -n $NN
