#!/bin/bash
#SBATCH -n 1 
#SBATCH --mem=8GB
#SBATCH -t 1:00:00
#SBATCH --array=0-45

#SBATCH -o OXUX-ADUM-%a.out
#SBATCH -e OXUX-ADUM-%a.out

#nCladding       : 1.2
#nCore           : 1.5
#coreRadius      : 0.9
#free_wavelength : 0.532 um
#nUpper          : 1.0
#numModes        : 46

cd /users/jlizaraz/CEM/wavesight
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_platform.py config_dict-OXUX-ADUM.pkl 100 $SLURM_ARRAY_TASK_ID

