#!/usr/bin/env python3
# fiber_bundle.py

import pickle
import argparse
import warnings
import subprocess
import numpy as np
import wavesight as ws

ignored_warnings = ['add','subtract','scalar add','scalar subtract']
for ignored in ignored_warnings:
    warnings.filterwarnings('ignore', 'invalid value encountered in '+ignored)

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

def fan_out(nCladding, nCore, coreRadius, λFree, nUpper, autorun, MEEP_resolution = 20):
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
    nUpper : float
        The refrective index of the medium to which the
        output of the fiber refracts to from an end face.
        Not used in this script but needed for the 
        simulations where the propagating modes are
        refracted out into this medium.
    
    Returns
    -------
    None

    '''
    fiber_spec = {'nCladding': nCladding,
                'nCore': nCore,
                'coreRadius': coreRadius,
                'grid_divider': 4,
                'nUpper': nUpper,
                'λFree': λFree}
    fiber_sol = ws.multisolver(fiber_spec,
                            solve_modes='all',
                            drawPlots=False,
                            verbose=True)
    numModes = fiber_sol['totalModes']
    fiber_sol = ws.calculate_numerical_basis(fiber_sol, verbose=False)
    (a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask, numSamples) = fiber_sol['coord_layout']
    nUpper = fiber_sol['nUpper']
    λUpper = λFree / nUpper

    sample_resolution   = 10
    slack_channel       = 'nvs_and_metalenses'
    distance_to_monitor = 1.5 * λUpper
    fiber_alpha         = np.arcsin(np.sqrt(nCore**2-nCladding**2))

    config_dict = {}
    config_dict['Xg'] = Xg
    config_dict['Yg'] = Yg
    config_dict['ρrange'] = ρrange
    config_dict['λUpper'] = λUpper
    config_dict['nUpper'] = nUpper
    config_dict['numModes'] = numModes
    config_dict['fiber_sol'] = fiber_sol
    config_dict['fiber_alpha'] = fiber_alpha
    config_dict['sample_posts'] = sample_posts
    config_dict['slack_channel'] = slack_channel
    config_dict['MEEP_resolution'] = MEEP_resolution
    config_dict['num_time_slices'] = num_time_slices
    config_dict['sample_resolution'] = sample_resolution
    config_dict['eigennums'] = fiber_sol['eigenbasis_nums']
    config_dict['distance_to_monitor'] = distance_to_monitor

    print("There are %d modes to solve." % numModes)
    batch_rid = ws.rando_id()
    config_fname = 'config_dict-'+batch_rid+'.pkl'
    with open(config_fname,'wb') as file:
        print("Saving configuration parameters to %s" % config_fname)
        pickle.dump(config_dict, file)
    bash_script_fname_1 = batch_rid+'-1.sh'
    bash_script_fname_2 = batch_rid+'-2.sh'
    batch_script_1 = bash_template_1.format(wavesight_dir=wavesight_dir,
                    config_fname = config_fname,
                    config_root  = batch_rid,
                    coreRadius   = coreRadius,
                    nCladding    = nCladding,
                    nCore        = nCore,
                    nUpper       = nUpper,
                    wavelength   = λFree,
                    numModes     = numModes,
                    num_time_slices = num_time_slices,
                    MEEP_resolution = MEEP_resolution,
                    num_modes=(numModes-1))
    batch_script_2 = bash_template_2.format(wavesight_dir=wavesight_dir,
                    config_fname = config_fname,
                    config_root  = batch_rid,
                    coreRadius   = coreRadius,
                    nCladding    = nCladding,
                    nCore        = nCore,
                    nUpper       = nUpper,
                    wavelength   = λFree,
                    numModes     = numModes,
                    num_time_slices = num_time_slices,
                    MEEP_resolution = MEEP_resolution,
                    num_modes=(numModes-1))
    with open(bash_script_fname_1, 'w') as file:
        print("Saving bash script to %s" % bash_script_fname_1)
        file.write(batch_script_1+'\n')
    with open(bash_script_fname_2, 'w') as file:
        print("Saving bash script to %s" % bash_script_fname_2)
        file.write(batch_script_2+'\n')
    if autorun:
        print('Submitting the first job to Slurm ...')
        subprocess.run(['bash', bash_script_fname_1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple CLI that accepts four parameters.')
    parser.add_argument('--nCladding', type=float, help='The refractive index of the cladding.')
    parser.add_argument('--nCore', type=float, help='The refractive index of the core.')
    parser.add_argument('--coreRadius', type=float, help='The radius of the core.')
    parser.add_argument('--free_space_wavelength', type=float, help='The free space wavelength.')
    parser.add_argument('--nUpper', type=float, help='The refrective index of the upper medium.')
    parser.add_argument('--autorun', action='store_true', help='Automatically submit the Slurm scripts.')
    parser.add_argument('--MEEP_resolution', nargs='?', type=int, const=20, help='The resolution at which the MEEP simulations will be run.')
    
    args = parser.parse_args()
    fan_out(args.nCladding, args.nCore, args.coreRadius, 
            args.free_space_wavelength, args.nUpper, args.autorun, args.MEEP_resolution)
