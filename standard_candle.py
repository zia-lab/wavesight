#!/usr/bin/env python
# usage: ./standard_candle.py num_cores resolution save_to_ht(0 or 1)
# for standard use: 6 40 1 1
                # ┌──────────────────────────────────────────┐ #
                # │ ┌──────────────────────────────────────┐ │ #
                # │ │                                      │ │ #
                # │ │                                      │ │ #
                # │ │                                      │ │ #
                # │ │                                      │ │ #
                # │ │                                      │ │ #
                # │ │                                      │ │ #
                # │ │                                      │ │ #
                # │ │                                      │ │ #
                # │ ├───────$───────────$──────────$───────┤ │ #
                # │ │######################################│ │ #
                # │ │######################################│ │ #
                # │ │######################################│ │ #
                # │ │###################*##################│ │ #
                # │ │######################################│ │ #
                # │ │######################################│ │ #
                # │ │######################################│ │ #
                # │ │######################################│ │ #
                # │ └──────────────────────────────────────┘ │ #
                # └──────────────────────────────────────────┘ #

# This script runs a standard simulation to serve as a comparison of
# performance between different systems. The simulation consists of a cube
# 8 um on the side, with a 0.5 PML boundary layer. Half the cube is filled
# with a uniform dielectric and the other half consists of vaccum. In the
# interface there are five protruding cylinders.
# In this structure there is a point dipole in the middle of the region
# with a dielectric. The fields are propagating from this source for
# 10 units of time.

# The script always outputs a pkl file which contains a dictionary with
# metadata for the simulation. This file may also include the
# fields at the end, or the fields may instead be saved in an h5 file.

# The output files are labeled candle-{epoch_time}.pkl
# or candle-{epoch_time}.h5.
# The h5 file has one group called fields, inside of which there are
# 6 datasets: Ex, Ey, Ez, Hx, Hy, Hz.
# The script takes four positional arguments:
# - num_cores
# - resolution
# - save_to_h5: 1-> True, 0->False
# - run_time_multiplier

import meep as mp
import numpy as np
from itertools import product
from time import time, sleep
import resource
from matplotlib import pyplot as plt

import h5py
import os
import psutil
import pickle
import sys

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
except:
    rank = None


def standard(metalens):
    metalens['run_date'] = int(time())
    matter = mp.Medium(epsilon=metalens['epsilon'])

    pillar_locs = list(product([-metalens['pillar_separation'],metalens['pillar_separation']],
                      [-metalens['pillar_separation'],metalens['pillar_separation']]))
    pillar_locs.append((0,0))

    pillars = [mp.Cylinder(radius=metalens['pillar_radius'],
                             height=metalens['pillar_height'],
                             center=mp.Vector3(x,y,0),
                             material=matter) for x,y in pillar_locs]

    substrate = [mp.Block(size=mp.Vector3(metalens['sim_cell_width'],metalens['sim_cell_width'],metalens['sim_cell_width']/2),
                          center=mp.Vector3(0,0,-metalens['sim_cell_width']/4),
                          material=matter)]
    geometry = substrate + pillars
    cell = mp.Vector3(metalens['sim_cell_width'],
                      metalens['sim_cell_width'],
                      metalens['sim_cell_width'])
    pml = [mp.PML(metalens['PML_width'])]
    source = [mp.Source(src=mp.ContinuousSource(
                        wavelength=metalens['wavelength']),
                        component=mp.Ex,
                        center=mp.Vector3(0, 0, -metalens['sim_cell_width']/4),
                        size=mp.Vector3(0,0,0))]
    symmetries = []
    for symmetry in metalens['symmetries']:
        if symmetry == 'x':
            symmetries.append(mp.Mirror(mp.X))
        if symmetry == 'y':
            symmetries.append(mp.Mirror(mp.Y))
    sim = mp.Simulation(cell_size=cell,
                       boundary_layers = pml,
                       geometry=geometry,
                       force_complex_fields = metalens['complex_fields'],
                       symmetries=symmetries,
                       sources=source,
                       resolution=metalens['resolution'])
    start_time = time()
    mp.quiet(metalens['quiet'])
    # mp.verbosity(3)
    sim.init_sim()
    metalens['time_for_init'] = (time() - start_time)
    start_time = time()
    #sim.run(mp.at_end(mp.output_efield(sim)),until=metalens['sim_time'])
    sim.run(until=metalens['sim_time'])
    metalens['time_for_running'] = time() - start_time
    start_time = time()
    sim.filename_prefix = 'standard_candle-%d' % metalens['run_date']
    print(sim.filename_prefix)
    mp.output_efield(sim)
    mp.output_hfield(sim)

    # print("collecting fields...")
    # h5_fname = 'candle-{run_date}.h5'.format(run_date = metalens['run_date'])
    # if metalens['save_fields_to_h5'] and rank ==0:#(not os.path.exists(h5_fname)):
    #     metalens['h5_fname'] = h5_fname
    #     h5_file = h5py.File(h5_fname,'w', driver='mpio', comm=MPI.COMM_WORLD)
    #     fields = h5_file.create_group('fields')
    #     Ex = np.transpose(sim.get_array(component=mp.Ex),(2,1,0))
    #     metalens['voxels'] = Ex.shape[0]**3
    #     metalens['voxels'] = Ex.shape[0]**3
    #     metalens['above_pillar_index'] = int(Ex.shape[0]/2. + 3*metalens['pillar_height']*metalens['resolution']/2)
    #     mp.output_efield('test')
    #     fields.create_dataset('Ex',
    #            data=Ex)
    #     del Ex
    #     fields.create_dataset('Ey',
    #            data=np.transpose(sim.get_array(component=mp.Ey),(2,1,0)))
    #     fields.create_dataset('Ez',
    #            data=np.transpose(sim.get_array(component=mp.Ez),(2,1,0)))
    #     fields.create_dataset('Hx',
    #            data=np.transpose(sim.get_array(component=mp.Hx),(2,1,0)))
    #     fields.create_dataset('Hy',
    #            data=np.transpose(sim.get_array(component=mp.Hy),(2,1,0)))
    #     fields.create_dataset('Hz',
    #            data=np.transpose(sim.get_array(component=mp.Hz),(2,1,0)))
    #     h5_file.close()
    # else:
    #     metalens['fields'] = {
    #       'Ex': np.transpose(sim.get_array(component=mp.Ex),(2,1,0)),
    #       'Ey': np.transpose(sim.get_array(component=mp.Ey),(2,1,0)),
    #       'Ez': np.transpose(sim.get_array(component=mp.Ez),(2,1,0)),
    #       'Hx': np.transpose(sim.get_array(component=mp.Hx),(2,1,0)),
    #       'Hy': np.transpose(sim.get_array(component=mp.Hy),(2,1,0)),
    #       'Hz': np.transpose(sim.get_array(component=mp.Hz),(2,1,0))
    #       }
    #     metalens['voxels'] = metalens['fields']['Ex'].shape[0]**3
    #     metalens['above_pillar_index'] = int(metalens['fields']['Ex'].shape[0]/2. + 3*metalens['pillar_height']*metalens['resolution']/2)
    metalens['voxels'] = int((8*metalens['resolution'])**3)
    metalens['time_for_saving_fields'] = round(time() - start_time,0)
    metalens['max_mem_usage_in_Gb'] = metalens['num_cores']*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1E6
    metalens['max_mem_usage_in_Gb'] = round(metalens['max_mem_usage_in_Gb'],2)
    metalens['summary'] = '''init time = {time_for_init:.1f} s
    running simulation = {time_for_running:.1f} s
    saving fields = {time_for_saving_fields:.1f} s
    num voxels = {voxels}
    max memory usage = {max_mem_usage_in_Gb:.2f} Gb'''.format(**metalens)
    metalens['pkl_fname'] = 'standard-candle-%d.pkl' % metalens['run_date']
    return metalens

# process = psutil.Process(os.getpid())

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: standard_candle.py num_cores resolution duration_multiplier")
        exit()
    num_cores = int(sys.argv[1])
    resolution = int(sys.argv[2])
    # save_to_h5 = int(sys.argv[3])
    run_time_multiplier = float(sys.argv[3])
    # if save_to_h5 == 1:
    #     save_to_h5 = True
    # else:
    #     save_to_h5 = False
    # process = psutil.Process(os.getpid())
    metalens = {'epsilon': 5.8418,
              'pillar_radius': 0.25,
              'pillar_height' : 0.5,
              'pillar_separation': 1,
              'sim_cell_width': 8,
              'PML_width': 0.5,
              'wavelength': 0.5,
              'resolution': resolution,
              'symmetries': [],
              'sim_time': 10. * run_time_multiplier,
              'complex_fields': True,
              'num_cores': num_cores,
              'quiet':False}
            #   'save_fields_to_h5': save_to_h5}
    metalens = standard(metalens)

    param_strings_and_nums = ["{one}, {two}".format(one=k,two=metalens[k]) for k in metalens if type(metalens[k]) in [str,float,int]]
    print('\n'.join(param_strings_and_nums))

    # if not os.path.exists(metalens['pkl_fname']):
    #     pickle.dump(metalens, open(metalens['pkl_fname'],'wb'))
    if (rank == 0):
        pickle.dump(metalens, open(metalens['pkl_fname'],'wb'))
