#!/usr/bin/env python3

# ┌──────────────────────────────────────────────────────────┐
# │              _   _   _   _   _   _   _   _   _           │
# │             / \ / \ / \ / \ / \ / \ / \ / \ / \          │
# │            ( w | a | v | e | s | i | g | h | t )         │
# │             \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/          │
# │                                                          │
# │      This script takes the parameters that define a      │
# │     step-index fiber. It solves for the propagation      │
# │     constants and it creates a set of slurm scripts      │
# │     that launch the simulation of the refraction of      │
# │      the waveguide modes across an end face of the       │
# │                          fiber.                          │
# │                                                          │
# └──────────────────────────────────────────────────────────┘
#fiberModes-Calc

import os
import shutil
import pickle
import argparse
import warnings
import subprocess
import numpy as np
import wavesight as ws
from printech import *

ignored_warnings = ['add','subtract','scalar add','scalar subtract']
for ignored in ignored_warnings:
    warnings.filterwarnings('ignore', 'invalid value encountered in '+ignored)

data_dir = '/users/jlizaraz/data/jlizaraz/CEM/wavesight/'
wavesight_dir = '/users/jlizaraz/CEM/wavesight'
num_time_slices = 60 # approx how many time samples of fields
# sample about this amount and post to Slack as sims progress
sample_posts = 3

# load the templates for the first and second bash scripts
bash_template_1 = ws.bash_template_1
bash_template_2 = ws.bash_template_2

def approx_time(sim_cell, spatial_resolution, run_time, kappa=3.06e-6):
    rtime = (kappa * sim_cell.x * sim_cell.y * sim_cell.z
             * run_time * spatial_resolution**3)
    return rtime

def fan_out(sim_params):
    '''
    Given  the  geometry  and  composition of a step-index fiber
    this   function   solves   for   the  waveguide  propagation
    constants. It saves these to a dictionary that is pickled to
    disk.  It  also  creates  two bash scripts, the first one of
    which  when  executed  triggers  the  following  sequence of
    events:

    1. A single job is submitted to estimate the memory and time
    requirements for each simulation. The requirements are saved
    a single line in a .req file.

    2.  Once  this  job  is  completed,  a second bash script is
    executed  automatically  by  Slurm,  since  it  has  as  its
    dependency  the  first  script. This second script submits a
    job  array  to  Slurm. Each job corresponding to each of the
    propagating modes of the waveguide. It also schedule another
    job,  that  is  dependent  on the job array. This job is the
    analysis  job, which takes all the files produced by the job
    array and creates a set of summary plots.

    Parameters
    ----------
    nCladding : float
        The refractive index of the cladding.
    nCore : float
        The refractive index of the core.
    coreRadius : float
        The radius of the core.
    λFree : float
        The free space wavelength.
    nBetween : float
        The refrective index of the medium to which the
        output of the fiber refracts to from an end face.
        Not used in this script but needed for the 
        simulations where the propagating modes are
        refracted out into this medium.
    
    Returns
    -------
    waveguide_dir : str
        The path of the directory where results will be saved.

    '''
    nCladding = sim_params['nCladding']
    nCore = sim_params['nCore']
    coreRadius = sim_params['coreRadius']
    λFree = sim_params['λFree']
    nBetween = sim_params['nBetween']
    python_bin = sim_params['python_bin']
    python_bin_MEEP = sim_params['python_bin_MEEP']
    code_dir = sim_params['code_dir']
    MEEP_num_cores = str(sim_params['MEEP_num_cores'])

    autorun = sim_params['autorun']
    zProp = sim_params['zProp']
    MEEP_resolution = sim_params['MEEP_resolution']
    req_run_mem_in_GB = sim_params['req_run_mem_in_GB']
    num_zProps = sim_params['num_zProps']
    
    slack_channel = sim_params['slack_channel']
    λBetween = sim_params['λBetween']
    nBetween = sim_params['nBetween']
    config_fname = sim_params['expanded_fname']

    fiber_spec = {'nCladding': nCladding,
                'nCore': nCore,
                'coreRadius': coreRadius,
                'grid_divider': 4,
                'nBetween': nBetween,
                'λFree': λFree}
    fiber_sol = ws.multisolver(fiber_spec,
                            solve_modes='all',
                            drawPlots=False,
                            verbose=True)

    numModes = fiber_sol['totalModes']
    fiber_sol = ws.calculate_numerical_basis(fiber_sol, verbose=False)
    (a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask, numSamples) = fiber_sol['coord_layout']

    sample_resolution   = 10
    distance_to_monitor = sim_params['wg_to_EH2']
    fiber_alpha         = np.arcsin(np.sqrt(nCore**2-nCladding**2))

    waveguide_sol = {}
    waveguide_sol['Xg'] = Xg
    waveguide_sol['Yg'] = Yg
    waveguide_sol['ρrange'] = ρrange
    waveguide_sol['λBetween'] = λBetween
    waveguide_sol['nBetween'] = nBetween
    waveguide_sol['numModes'] = numModes
    waveguide_sol['fiber_sol'] = fiber_sol
    waveguide_sol['fiber_alpha'] = fiber_alpha
    waveguide_sol['sample_posts'] = sample_posts
    waveguide_sol['slack_channel'] = slack_channel
    waveguide_sol['MEEP_resolution'] = MEEP_resolution
    waveguide_sol['num_time_slices'] = num_time_slices
    waveguide_sol['sample_resolution'] = sample_resolution
    waveguide_sol['eigennums'] = fiber_sol['eigenbasis_nums']
    waveguide_sol['distance_to_monitor'] = distance_to_monitor

    printer("There are %d modes to solve." % numModes)
    waveguide_id = ws.rando_id()
    waveguide_dir = os.path.join(data_dir, waveguide_id)
    if not os.path.exists(waveguide_dir):
        os.mkdir(waveguide_dir)
        printer(f"Directory {waveguide_dir} created.")
    else:
        printer(f"Directory {waveguide_dir} already exists.")
    printer('saving a copy of the config file to the data directory')
    config_fname_dest = os.path.join(waveguide_dir, 'config.json')
    shutil.copy(config_fname, config_fname_dest)
    waveguide_sol['config_file_fname'] = config_fname_dest
    config_dict = ws.load_from_json(config_fname)
    design_id   = config_dict['sim_id']
    printer('moving the metalens design file to the data dir')
    design_fname = 'metalens-design-%s.h5' % design_id
    shutil.move(design_fname, os.path.join(waveguide_dir, 'metalens-design.h5'))
    waveguide_sol_fname = 'waveguide_sol-' + waveguide_id + '.pkl'
    waveguide_sol_fname = os.path.join(waveguide_dir, waveguide_sol_fname)
    with open(waveguide_sol_fname,'wb') as file:
        printer("Saving configuration parameters to %s" % waveguide_sol_fname)
        pickle.dump(waveguide_sol, file)
    bash_script_fname_1 = waveguide_id + '-1.sh'
    bash_script_fname_2 = waveguide_id + '-2.sh'
    batch_script_1 = bash_template_1.format(
                    MEEP_num_cores = MEEP_num_cores,
                    code_dir = code_dir,
                    python_bin = python_bin,
                    python_bin_MEEP = python_bin_MEEP,
                    waveguide_sol_fname = waveguide_sol_fname,
                    wavesight_dir = wavesight_dir,
                    waveguide_id  = waveguide_id,
                    coreRadius    = coreRadius,
                    nCladding     = nCladding,
                    waveguide_dir = waveguide_dir,
                    nCore         = nCore,
                    nBetween      = nBetween,
                    wavelength    = λFree,
                    numModes      = numModes,
                    num_time_slices = num_time_slices,
                    MEEP_resolution = MEEP_resolution,
                    req_run_mem_in_GB = req_run_mem_in_GB,
                    num_modes       = (numModes-1),
                    num_zProps       = (num_zProps-1))
    batch_script_2 = bash_template_2.format(
                    MEEP_num_cores = MEEP_num_cores,
                    code_dir = code_dir,
                    python_bin = python_bin,
                    python_bin_MEEP = python_bin_MEEP,
                    wavesight_dir=wavesight_dir,
                    waveguide_sol_fname = waveguide_sol_fname,
                    waveguide_id  = waveguide_id,
                    waveguide_dir = waveguide_dir,
                    coreRadius   = coreRadius,
                    nCladding    = nCladding,
                    nCore        = nCore,
                    nBetween     = nBetween,
                    wavelength   = λFree,
                    numModes     = numModes,
                    num_time_slices = num_time_slices,
                    MEEP_resolution = MEEP_resolution,
                    zProp        = zProp,
                    nProp        = nBetween,
                    num_modes    = (numModes-1),
                    num_zProps    = (num_zProps-1))
    rule()
    with open(bash_script_fname_1, 'w') as file:
        printer("Saving bash script to %s" % bash_script_fname_1)
        file.write(batch_script_1+'\n')
    extra_copy_fname = os.path.join(waveguide_dir, bash_script_fname_1)
    shutil.copy(bash_script_fname_1, extra_copy_fname)
    rule()
    code_print(batch_script_1)
    rule()
    with open(bash_script_fname_2, 'w') as file:
        printer("Saving bash script to %s" % bash_script_fname_2)
        file.write(batch_script_2+'\n')
    extra_copy_fname = os.path.join(waveguide_dir, bash_script_fname_2)
    shutil.copy(bash_script_fname_2, extra_copy_fname)
    rule()
    code_print(batch_script_2)
    rule()
    if autorun:
        printer('submitting the first job to Slurm')
        subprocess.run(['bash', bash_script_fname_1])
    return waveguide_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launching the propagating modes of a fiber.')
    parser.add_argument('configfile', help='Name of the json config file')
    args = parser.parse_args()
    # read the config file
    sim_params = ws.load_from_json(args.configfile)
    waveguide_dir = fan_out(sim_params)
