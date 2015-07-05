#!/bin/bash

#BSUB -J cro_u2.3j0.4b60
#BSUB -n 128
#BSUB -a "bcs openmpi"
#BSUB -M 512
#BSUB -W 24:00
#BSUB -o output_%J.out
#BSUB -e output_%J.err
#BSUB -P jara0112

source $HOME/.zshrc
NN=$LSB_DJOB_NUMPROC

/home/td120143/codes/scf_dmft/dmft.py -p parms -n $NN
