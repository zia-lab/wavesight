#!/usr/bin/env python3

# ┌──────────────────────────────────────────────────────────┐
# │              _   _   _   _   _   _   _   _   _           │
# │             / \ / \ / \ / \ / \ / \ / \ / \ / \          │
# │            ( w | a | v | e | s | i | g | h | t )         │
# │             \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/          │
# │                                                          │
# │       Given the waveguide ID, an axial propagation       │
# │    distance, and a refractive index this script takes    │
# │        all the transverse refracted fields of the        │
# │     waveguide modes, at the position where they were     │
# │     monitored, and calculates the propagated fields.     │
# │    The propagated fields are saved in a single array     │
# │     whose first index corresponds to the mode index,     │
# │      whose second index picks between E or H, whose      │
# │    third index picks between the x-y-z components of     │
# │      the field, on whose fourth and fifth index are      │
# │    indexed to the spatial positions at the propagated    │
# │                          plane.                          │
# │                                                          │
# └──────────────────────────────────────────────────────────┘

import os
import h5py
import argparse
import numpy as np
import wavesight as ws
from contextlib import contextmanager

data_dir        = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
exclude_dirs    = ['moovies']
overwrite       = True

@contextmanager
def h5_handler(filename, mode):
    try:
        file = h5py.File(filename, mode)
        yield file
    except Exception as e:
        os.remove(filename)
        print(f"An error occurred: {e}. File {filename} has been removed.")
    finally:
        if file:
            file.close()

def wave_jumper(waveguide_id, zProp, nProp):
    # first find all the subdirs that correspond to the waveguide_id
    waveguide_dir = os.path.join(data_dir, waveguide_id)
    job_dir_contents = os.listdir(waveguide_dir)
    job_dir_contents = [a_dir for a_dir in job_dir_contents if a_dir not in exclude_dirs]
    job_dirs = [os.path.join(waveguide_dir, a_dir) for a_dir in job_dir_contents]
    job_dirs = [a_dir for a_dir in job_dirs if os.path.isdir(a_dir) ]
    def wave_sorter(x):
        idx = int(x.split('-')[-1])
        return idx
    job_dirs = list(sorted(job_dirs, key = wave_sorter))
    # go through each folder and propagate the fields across the given gap
    for job_idx, job_dir in enumerate(job_dirs):
        mode_id = job_dir.split('/')[-1][:9]
        print('>> %d/%d' % (job_idx, len(job_dirs) - 1))
        refracted_h5_fname = 'EH2.h5'
        refracted_h5_fname = os.path.join(job_dir, refracted_h5_fname)
        propagated_h5_fname = 'EH3.h5'
        propagated_h5_fname = os.path.join(job_dir, propagated_h5_fname)
        mode_sol, _ = ws.load_from_h5(refracted_h5_fname)
        mode_idx, kz, m, modeType, parity = (mode_sol['mode_idx'], mode_sol['kz'],
                                             mode_sol['m'], mode_sol['modeType'],
                                             mode_sol['parity'])
        kFree      = mode_sol['kFree']
        λFree      = 2*np.pi/kFree
        coreRadius = mode_sol['coreRadius']
        nCladding  = mode_sol['nCladding']
        nCore      = mode_sol['nCore']
        fiber_NA   = np.sqrt(nCore**2 - nCladding**2)
        fiber_β    = np.arcsin(fiber_NA)
        # the spatial extent of the fields will be expanded to about this dimension
        xCoords    = mode_sol['coords']['xCoordsMonxy']
        yCoords    = mode_sol['coords']['yCoordsMonxy']
        N_samples  = len(xCoords)
        current_width = xCoords[-1] - xCoords[0]
        print(">> The source field spans is defined over a distance of %.2f μm ..." % current_width)
        prop_plane_width = 2 * (current_width/2 + 1.1 * zProp * np.tan(fiber_β))
        print(">> Given the NA of the fiber and the propagation distance the field will be propagated to an extended domain with width of %.2f μm ..." % prop_plane_width)
        (_, xCoords) = ws.array_stretcher(xCoords, prop_plane_width)
        (_, yCoords) = ws.array_stretcher(yCoords, prop_plane_width)
        total_pad_width = len(xCoords) - N_samples
        h5_keys = ['h_xy_slices_fname_h5', 'e_xy_slices_fname_h5']
        data = {}
        propagated_h5_fname = os.path.join(job_dir, propagated_h5_fname)
        if os.path.exists(propagated_h5_fname) and not overwrite:
            print(">> Done already, continue to next.")
            continue
        else:
            with h5_handler(propagated_h5_fname, 'w') as prop_h5_file:
                # the data is saved differently in the case of time-resolved
                # and steady state simulations
                if mode_sol['time_resolved']:
                    for h5_key in h5_keys:
                        refracted_h5_fname = mode_sol[h5_key]
                        refracted_h5_fname = os.path.join(job_dir, mode_sol[h5_key])
                        field = h5_key[0]
                        plane = h5_key.split('_')[1]
                        field_list = []
                        data_key_root = '%s-%s' % (field, plane)
                        with h5py.File(refracted_h5_fname, 'r') as h5f:
                            for cartesian_component in 'xyz':
                                data_key_r = '%s%s.r' % (field, cartesian_component)
                                data_key_i = '%s%s.i' % (field, cartesian_component)
                                field_data = 1j*np.array(h5f[data_key_i][:,:,-1])
                                field_data += np.array(h5f[data_key_r][:,:,-1])
                                field_data = field_data.T
                                field_list.append(field_data)
                        field_list = np.array(field_list)
                        data[data_key_root] = field_list
                else:
                    refracted_h5_fname = mode_sol['eh_monitors_fname_h5']
                    refracted_h5_fname = os.path.join(job_dir, refracted_h5_fname)
                    with h5py.File(refracted_h5_fname, 'r') as h5f:
                        for plane in ['xy']:
                            plane_slice = h5f[plane]
                            for field in 'eh':
                                field_list = []
                                for cartesian_component in 'xyz':
                                    data_key_r = '%s%s.r' % (field, cartesian_component)
                                    data_key_i = '%s%s.i' % (field, cartesian_component)
                                    field_data = 1j*np.array(plane_slice[data_key_i])
                                    field_data += np.array(plane_slice[data_key_r])
                                    field_data = field_data.T
                                    field_list.append(field_data)
                                field_list = np.array(field_list)
                                data_key_root = '%s-%s' % (field, plane)
                                data[data_key_root] = field_list
                transverse_fields = np.array([data['e-xy'], data['h-xy']])
                transverse_fields = ws.sym_pad(transverse_fields, total_pad_width, mode='constant')
                prop_E_field = ws.electric_vectorial_diffraction(zProp, transverse_fields[0], xCoords, yCoords, λFree, nProp)
                prop_H_field = ws.magnetic_vectorial_diffraction(zProp, transverse_fields[0], xCoords, yCoords, λFree, nProp)

                dx = xCoords[1] - xCoords[0]
                integrand = np.sum(np.abs(transverse_fields[0])**2, axis=0)
                E_squared_int_source = ws.simpson_quad_ND(integrand, [dx,dx])
                integrand = np.sum(np.abs(prop_E_field)**2, axis=0)
                E_squared_int_prop   = ws.simpson_quad_ND(integrand, [dx,dx])
                print('∫E^2_source dxdy = %.3f' % E_squared_int_source)
                print('∫E^2_propag dxdy = %.3f' % E_squared_int_prop)

                integrand = np.sum(np.abs(transverse_fields[1])**2, axis=0)
                H_squared_int_source = ws.simpson_quad_ND(integrand, [dx,dx])
                integrand = np.sum(np.abs(prop_H_field)**2, axis=0)
                H_squared_int_prop   = ws.simpson_quad_ND(integrand, [dx,dx])
                print('∫H^2_source dxdy = %.3f' % H_squared_int_source)
                print('∫H^2_propag dxdy = %.3f' % H_squared_int_prop)

                propagated_field = np.array([prop_E_field, prop_H_field])
                prop_h5_file.create_dataset('EH_field', data = propagated_field)
                prop_h5_file.create_dataset('xCoords',  data = xCoords)
                prop_h5_file.create_dataset('yCoords',  data = yCoords)
                prop_h5_file.create_dataset('zProp',    data = zProp)
                prop_h5_file.create_dataset('nProp',    data = nProp)
                prop_h5_file.create_dataset('E^2_integral_source', data = E_squared_int_source)
                prop_h5_file.create_dataset('E^2_integral_propagated', data = E_squared_int_prop)
                prop_h5_file.create_dataset('H^2_integral_source', data = H_squared_int_source)
                prop_h5_file.create_dataset('H^2_integral_propagated', data = H_squared_int_prop)
                prop_h5_file.create_dataset('waveguide_id', data=waveguide_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Job plotter.')
    parser.add_argument('waveguide_id', type=str, help='The label for the job.')
    parser.add_argument('zProp', type=float, help='z-distance for propagation.')
    parser.add_argument('nProp', type=float, help='Refractive index of propagating medium.')
    args = parser.parse_args()
    wave_jumper(args.waveguide_id, args.zProp, args.nProp)