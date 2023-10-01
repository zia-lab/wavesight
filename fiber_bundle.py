#!/usr/bin/env python3
# fiber_bundle.py

import numpy as np
from matplotlib import pyplot as plt
import wavesight as ws
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
import argparse
import warnings
from matplotlib import style
style.use('dark_background')

warnings.filterwarnings('ignore', 'invalid value encountered in scalar add')
warnings.filterwarnings('ignore', 'invalid value encountered in add')
warnings.filterwarnings('ignore', 'invalid value encountered in scalar subtract')
warnings.filterwarnings('ignore', 'invalid value encountered in subtract')

show_plot = False
send_to_slack = True
make_streamplots =  False
grab_fields  = True # whether to import the h5 files that contain the monotired fields
wavesight_dir = '/users/jlizaraz/CEM/wavesight'
num_time_slices = 100 # how many time samples of fields

batch_template = '''#!/bin/bash
#SBATCH -n 1 
#SBATCH --mem=8GB
#SBATCH -t 1:00:00
#SBATCH --array=0-{num_modes}

#SBATCH -o {config_root}-%a.out
#SBATCH -e {config_root}-%a.out

#nCladding       : {nCladding}
#nCore           : {nCore}
#coreRadius      : {coreRadius}
#free_wavelength : {wavelength} um
#nUpper          : {nUpper}
#numModes        : {numModes}

cd {wavesight_dir}
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_platform.py {config_fname} {num_time_slices} $SLURM_ARRAY_TASK_ID
'''

def approx_time(sim_cell, spatial_resolution, run_time, kappa=3.06e-6):
    rtime = (kappa * sim_cell.x * sim_cell.y * sim_cell.z
             * run_time * spatial_resolution**3)
    return rtime

def fan_out(nCladding, nCore, coreRadius, λFree, nUpper):
    fiber_spec = {'nCladding': nCladding,
                'nCore': nCore,
                'coreRadius': coreRadius,
                'grid_divider': 4,
                'nUpper': nUpper,
                'λFree': λFree}
    fiber_sol = ws.multisolver(fiber_spec,
                            solve_modes = 'all',
                            drawPlots=False,
                            verbose=True)
    numModes = fiber_sol['totalModes']
    fiber_sol = ws.calculate_numerical_basis(fiber_sol, verbose=False)
    a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask, numSamples = fiber_sol['coord_layout']
    nUpper = fiber_sol['nUpper']
    λUpper = λFree / nUpper
    sample_resolution = 10
    MEEP_resolution  = 20
    slack_channel = 'nvs_and_metalenses'
    distance_to_monitor = 1.5 * λUpper
    fiber_alpha = np.arcsin(np.sqrt(nCore**2-nCladding**2))
    config_dict = {}
    config_dict['ρrange'] = ρrange
    config_dict['Xg'] = Xg
    config_dict['Yg'] = Yg
    config_dict['λUpper'] = λUpper
    config_dict['sample_resolution'] = sample_resolution
    config_dict['MEEP_resolution'] = MEEP_resolution
    config_dict['slack_channel']   = slack_channel
    config_dict['num_time_slices'] = num_time_slices
    config_dict['distance_to_monitor'] = distance_to_monitor
    config_dict['fiber_alpha'] = fiber_alpha
    config_dict['eigennums']   = fiber_sol['eigenbasis_nums']
    config_dict['fiber_sol'] = fiber_sol
    config_dict['nUpper'] = nUpper
    config_dict['numModes'] = numModes
    print("There are %d modes to solve." % numModes)
    batch_rid = ws.rando_id()
    config_fname = 'config_dict-'+batch_rid+'.pkl'
    with open(config_fname,'wb') as file:
        print("Saving parameters to %s" % config_fname)
        pickle.dump(config_dict, file)
    batch_fname = 'sbatch-'+batch_rid+'.sh'
    batch_cont = batch_template.format(wavesight_dir=wavesight_dir,
                    config_fname = config_fname,
                    config_root  = batch_rid,
                    coreRadius   = coreRadius,
                    nCladding    = nCladding,
                    nCore        = nCore,
                    nUpper       = nUpper,
                    wavelength   = λFree,
                    numModes     = numModes,
                    num_time_slices = num_time_slices,
                    num_modes=(numModes-1))
    with open(batch_fname, 'w') as file:
        print("Saving sbatch file to %s" % batch_fname)
        file.write(batch_cont+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple CLI that accepts four parameters.')
    parser.add_argument('nCladding', type=float, help='The refractive index of the cladding.')
    parser.add_argument('nCore', type=float, help='The refractive index of the core.')
    parser.add_argument('coreRadius', type=float, help='The radius of the core.')
    parser.add_argument('free_space_wavelength', type=float, help='The free space wavelength.')
    parser.add_argument('nUpper', type=float, help='The refrective index of the upper medium.')
    args = parser.parse_args()
    fan_out(args.nCladding, args.nCore, args.coreRadius, args.free_space_wavelength, args.nUpper)
