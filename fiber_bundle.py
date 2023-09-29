#!/usr/bin/env python3

import meep as mp
import numpy as np
import h5py as h5pie
import cmasher as cm
import os
from matplotlib import pyplot as plt
import wavesight as ws
import time
import cmasher as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
import argparse
from matplotlib import style
style.use('dark_background')

show_plot = False
send_to_slack = True
make_streamplots =  False
grab_fields  = True # whether to import the h5 files that contain the monotired fields


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
    num_time_slices = 150 # how many time samples of fields
    distance_to_monitor = 1.5 * λUpper
    fiber_alpha = np.arcsin(np.sqrt(nCore**2-nCladding**2))
    config_dict = {}
    config_dict['ρrange'] = ρrange
    config_dict['Xg'] = Xg
    config_dict['Yg'] = Yg
    config_dict['λUpper'] = λUpper
    config_dict['sample_resolution'] = sample_resolution
    config_dict['MEEP_resolution'] = MEEP_resolution
    # fiber_sol['run_time_fun']    = run_time_fun
    # fiber_sol['sim_height_fun']  = sim_height_fun
    config_dict['slack_channel']   = slack_channel
    config_dict['num_time_slices'] = num_time_slices
    config_dict['distance_to_monitor'] = distance_to_monitor
    config_dict['fiber_alpha'] = fiber_alpha
    config_dict['eigennums']   = fiber_sol['eigenbasis_nums']
    config_dict['fiber_sol'] = fiber_sol
    config_dict['nUpper'] = nUpper

    with open('config_dict.pkl','wb') as file:
        pickle.dump(config_dict, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple CLI that accepts four parameters.')
    parser.add_argument('nCladding', type=float, help='The refractive index of the cladding.')
    parser.add_argument('nCore', type=float, help='The refractive index of the core.')
    parser.add_argument('coreRadius', type=float, help='The radius of the core.')
    parser.add_argument('free_space_wavelength', type=float, help='The free space wavelength.')
    parser.add_argument('nUpper', type=float, help='The refrective index of the upper medium.')
    args = parser.parse_args()
    fan_out(args.nCladding, args.nCore, args.coreRadius, args.free_space_wavelength, args.nUpper)
