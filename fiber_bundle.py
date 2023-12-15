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
# │      the waveguide modes across the end face of the      │
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
exclude_dirs  = ['moovies', 'moovies-EH4', 'err', 'out', 'figs']
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

def fan_out(config_params):
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
    # if the config file includes the key variant_of
    # assumption is that launching the modes from the waveguide
    # is not necessary
    # in this case all that should be done here is to copy
    # the req file and the waveguide_sol file to the data dir
    # of this new simulation
    take_shortcut = ('variant_of' in config_params)

    if take_shortcut:
        variant_of = config_params['variant_of']
    else:
        variant_of = '----|----'

    nCladding       = config_params['nCladding']
    nCore           = config_params['nCore']
    coreRadius      = config_params['coreRadius']
    λFree           = config_params['λFree']
    nBetween        = config_params['nBetween']
    python_bin      = config_params['python_bin']
    python_bin_MEEP = config_params['python_bin_MEEP']
    code_dir        = config_params['code_dir']

    if 'req_run_time_in_hours' in config_params:
        req_run_time_in_hours = config_params['req_run_time_in_hours']
    else:
        req_run_time_in_hours = 2
    MEEP_num_cores = str(config_params['MEEP_num_cores'])

    autorun           = config_params['autorun']
    EH2_to_EH3        = config_params['EH2_to_EH3']
    MEEP_resolution   = config_params['MEEP_resolution']
    req_run_mem_in_GB = config_params['req_run_mem_in_GB']
    num_zProps        = config_params['num_zProps']
    
    slack_channel = config_params['slack_channel']
    λBetween      = config_params['λBetween']
    nBetween      = config_params['nBetween']
    config_fname  = config_params['expanded_fname']

    waveguide_id = ws.rando_id()
    waveguide_dir = os.path.join(data_dir, waveguide_id)
    if not os.path.exists(waveguide_dir):
        os.mkdir(waveguide_dir)
        printer(f"Directory {waveguide_dir} created.")
    else:
        printer(f"Directory {waveguide_dir} already exists.")

    fiber_spec = {'nCladding': nCladding,
                'nCore': nCore,
                'coreRadius': coreRadius,
                'grid_divider': 4,
                'nBetween': nBetween,
                'λFree': λFree}
    if take_shortcut:
        # copy the modes solution to the waveguide dir
        printer("getting the mode solutions from the master waveguide")
        variant_of = config_params['variant_of']
        master_waveguide_dir = os.path.join(data_dir, variant_of)
        master_waveguide_sol_fname = os.path.join(master_waveguide_dir, 'waveguide_sol-%s.pkl' % variant_of)
        waveguide_sol_fname = 'waveguide_sol-' + waveguide_id + '.pkl'
        waveguide_sol_fname = os.path.join(waveguide_dir, waveguide_sol_fname)
        shutil.copy(master_waveguide_sol_fname, waveguide_sol_fname)
        # copy the req file to the waveguide dir
        printer("getting the resource requirements from the master waveguide")
        master_req_fname = os.path.join(master_waveguide_dir, variant_of + '.req')
        req_fname = waveguide_id + '.req'
        req_fname = os.path.join(waveguide_dir, req_fname)
        shutil.copy(master_req_fname, req_fname)
        # read how many modes were solved for already
        with open(master_waveguide_sol_fname,'rb') as file:
            master_waveguide_sol = pickle.load(file)
        numModes = master_waveguide_sol['numModes']
        # get the directories from the master waveguide
        job_dir_contents = os.listdir(master_waveguide_dir)
        job_dir_contents = [a_dir for a_dir in job_dir_contents if a_dir not in exclude_dirs]
        master_mode_dirs = [os.path.join(master_waveguide_dir, a_dir) for a_dir in job_dir_contents]
        master_mode_dirs = [a_dir for a_dir in master_mode_dirs if os.path.isdir(a_dir)]
        def wave_sorter(x):
            idx = int(x.split('-')[-1])
            return idx
        master_mode_dirs = list(sorted(master_mode_dirs, key = wave_sorter))
        master_mode_dict = {idx: a_dir for idx, a_dir in enumerate(master_mode_dirs)}
        print("mmds", master_mode_dict)
        # create the folders for each mode in the waveguide directory
        # and copy files from the master waveguide folder
        dest_folders = {}
        for mode_idx in range(numModes):
            mode_id = ws.rando_id()
            mode_folder = '%s-%d' % (mode_id, mode_idx)
            mode_folder = os.path.join(waveguide_dir, mode_folder)
            dest_folders[mode_idx] = mode_folder
            if not os.path.exists(mode_folder):
                os.mkdir(mode_folder)
                printer(f"Directory {mode_folder} created.")
            else:
                printer(f"Directory {mode_folder} already exists.")
            if mode_idx == 0:
                files_to_copy = ['EH2.h5',
                                 'EH3.h5', 
                                 'e-field-xy-slices.h5',
                                 'e-field-yz-slices.h5',
                                 'e-field-xz-slices.h5']
            else:
                files_to_copy = ['EH2.h5',
                                 'EH3.h5',
                                 'e-h-fields.h5']
            for file_to_copy in files_to_copy:
                source_file = os.path.join(master_mode_dict[mode_idx], file_to_copy)
                dest_file = os.path.join(mode_folder, file_to_copy)
                shutil.copy(source_file, dest_file)
    else:
        fiber_sol = ws.multisolver(fiber_spec,
                                solve_modes='all',
                                drawPlots=False,
                                verbose=True)
        numModes = fiber_sol['totalModes']
        fiber_sol = ws.calculate_numerical_basis(fiber_sol, verbose=False)
        (a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask, numSamples) = fiber_sol['coord_layout']

        sample_resolution   = 10
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
        printer("There are %d modes to solve." % numModes)
        waveguide_sol_fname = 'waveguide_sol-' + waveguide_id + '.pkl'
        waveguide_sol_fname = os.path.join(waveguide_dir, waveguide_sol_fname)
        with open(waveguide_sol_fname,'wb') as file:
            printer("Saving configuration parameters to %s" % waveguide_sol_fname)
            pickle.dump(waveguide_sol, file)
    printer('saving a copy of the config file to the data directory')
    config_fname_dest = os.path.join(waveguide_dir, 'config.json')
    shutil.copy(config_fname, config_fname_dest)
    design_id   = config_params['sim_id']
    printer('moving the metalens design file to the data dir')
    design_fname = 'metalens-design-%s.h5' % design_id
    shutil.move(design_fname, os.path.join(waveguide_dir, 'metalens-design.h5'))
    printer("configuring the two necessary bash scripts")
    bash_script_fname_1 = waveguide_id + '-1.sh'
    bash_script_fname_2 = waveguide_id + '-2.sh'
    if take_shortcut:
        master_shortcut = 'true'
    else:
        master_shortcut = 'false'
    batch_script_1 = bash_template_1.format(
                    variant_of = variant_of,
                    master_shortcut = master_shortcut,
                    MEEP_num_cores = MEEP_num_cores,
                    req_run_time_in_hours = req_run_time_in_hours,
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
                    variant_of = variant_of,
                    master_shortcut = master_shortcut,
                    MEEP_num_cores = MEEP_num_cores,
                    req_run_time_in_hours = req_run_time_in_hours,
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
                    EH2_to_EH3      = EH2_to_EH3,
                    nProp        = nBetween,
                    num_modes    = (numModes-1),
                    num_zProps   = (num_zProps-1))
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
    config_params = ws.load_from_json(args.configfile)
    waveguide_dir = fan_out(config_params)
