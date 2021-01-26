#!/bin/bash
#SBATCH --account=weis
#SBATCH --time=1:00:00
#SBATCH --job-name=floating
#SBATCH --nodes=1             # This should be nC/36 (36 cores on eagle)
#SBATCH --ntasks-per-node=36
#SBATCH --mail-user john.jasa@nrel.gov
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output=output.%j.out
#SBATCH --partition=debug


nDV=1 # Number of design variables (x2 for central difference)
nOF=1  # Number of openfast runs per finite-difference evaluation
nC=$((nDV + nDV * nOF)) # Number of cores needed. Make sure to request an appropriate number of nodes = N / 36

source activate weis-env 

mpirun -v -np $nC python weis_driver.py
