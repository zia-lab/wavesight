#!/usr/bin/env python3

import os
import h5py
import argparse
import numpy as np
from printech import *
import wavesight as ws
from contextlib import contextmanager

data_dir        = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
exclude_dirs    = ['moovies', 'moovies-EH4', 'err', 'out', 'figs']
overwrite       = False
num_batches     = 5 # this should match the number of jobs in the EH4_to_EH5 job array

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

def wave_to_vol(waveguide_id, zPropindex):
    '''
    This function takes the id for a given simulation and 
    takes the fields that have been propagated across the
    metalens and propagate them along the z-axis by   the
    provided distance.

    The distance is provided as an index in the the zProps
    array which is determined from the start of the 
    simulation.

    The result is save into a new folder called EH5 in 
    h5 format.

    Parameters
    ----------
    waveguide_id : str
        The label for the job.
    zPropindex : int
         
    Returns
    -------
    None
    '''
    # first find all the subdirs that correspond to the waveguide_id
    waveguide_dir = os.path.join(data_dir, waveguide_id)
    printer("loading the configuration parameters")
    config_params = ws.load_from_json(os.path.join(waveguide_dir, 'config.json'))
    job_dir_contents = os.listdir(waveguide_dir)
    job_dir_contents = [a_dir for a_dir in job_dir_contents if a_dir not in exclude_dirs]
    job_dirs = [os.path.join(waveguide_dir, a_dir) for a_dir in job_dir_contents]
    job_dirs = [a_dir for a_dir in job_dirs if os.path.isdir(a_dir) ]
    def wave_sorter(x):
        idx = int(x.split('-')[-1])
        return idx
    job_dirs = list(sorted(job_dirs, key = wave_sorter))
    # read in necessary configuration parameters
    λFree  = config_params['λFree']
    nProp  = config_params['nHost']
    zProps = config_params['zProps']
    xymin  = config_params['xymin']
    xymax  = config_params['xymax']
    # go through each folder and propagate the fields across the given gap
    for job_idx, job_dir in enumerate(job_dirs):
        printer('slice %d of %d' % (job_idx, len(job_dirs) - 1))
        source_h5_fname = os.path.join(job_dir, 'EH4.h5')
        (EH4_fields, _) = ws.load_from_h5(source_h5_fname)
        xCoords         = EH4_fields['xCoordsMonxy']
        yCoords         = EH4_fields['yCoordsMonxy']
        xCoords_snip    = xCoords[(xCoords <= xymax) & (xCoords >= xymin)]
        yCoords_snip    = yCoords[(yCoords <= xymax) & (yCoords >= xymin)]
        xG, yG          = np.meshgrid(xCoords, yCoords)
        cross_mask      = (xG <= xymax) & (xG >= xymin) & (yG <= xymax) & (yG >= xymin)
        cross_mask      = np.broadcast_to(cross_mask, (3, cross_mask.shape[0], cross_mask.shape[1]))
        current_width   = (xCoords[-1] - xCoords[0])
        printer("the source field is defined over a square cross section with a side of %.2f μm" % current_width)
        prop_idx = zPropindex
        if prop_idx == -1:
            zProps_job = zProps
            prop_indices = range(len(zProps))
        else:
            zProps_partition = ws.index_divider(zProps, num_batches)
            zProps_partition_indices = ws.index_divider(zProps, num_batches, return_indices=True)
            zProps_job = zProps_partition[prop_idx]
            prop_indices = zProps_partition_indices[prop_idx]
        for prop_idx, zProp in zip(prop_indices, zProps_job):
            data = {}
            propagated_h5_fname = 'EH5-%d.h5' % prop_idx
            eh5_dir = os.path.join(job_dir, 'EH5')
            if not os.path.exists(eh5_dir):
                os.makedirs(eh5_dir, exist_ok=True)
            propagated_h5_fname = os.path.join(eh5_dir, propagated_h5_fname)
            if os.path.exists(propagated_h5_fname) and not overwrite:
                printer("done already, or being computed elsewhere, continuing to next.")
                continue
            else:
                with h5_handler(propagated_h5_fname, 'w') as prop_h5_file:
                    for plane in ['xy']:
                        plane_slice = EH4_fields[plane]
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
                    prop_E_field = ws.electric_vectorial_diffraction(zProp, transverse_fields[0], 
                                                                    xCoords, yCoords, λFree, nProp)
                    prop_H_field = ws.magnetic_vectorial_diffraction(zProp, transverse_fields[0],
                                                                    xCoords, yCoords, λFree, nProp)
                    dx = xCoords[1] - xCoords[0]
                    integrand = np.sum(np.abs(transverse_fields[0])**2, axis=0)
                    E_squared_int_source = ws.simpson_quad_ND(integrand, [dx,dx])
                    integrand = np.sum(np.abs(prop_E_field)**2, axis=0)
                    E_squared_int_prop   = ws.simpson_quad_ND(integrand, [dx,dx])
                    printer('∫E^2_source dxdy = %.3f' % E_squared_int_source)
                    printer('∫E^2_propag dxdy = %.3f' % E_squared_int_prop)

                    integrand = np.sum(np.abs(transverse_fields[1])**2, axis=0)
                    H_squared_int_source = ws.simpson_quad_ND(integrand, [dx,dx])
                    integrand = np.sum(np.abs(prop_H_field)**2, axis=0)
                    H_squared_int_prop   = ws.simpson_quad_ND(integrand, [dx,dx])
                    printer('∫H^2_source dxdy = %.3f' % H_squared_int_source)
                    printer('∫H^2_propag dxdy = %.3f' % H_squared_int_prop)
                    prop_E_field = prop_E_field[cross_mask].reshape(3, xCoords_snip.size, yCoords_snip.size)
                    prop_H_field = prop_H_field[cross_mask].reshape(3, xCoords_snip.size, yCoords_snip.size)
                    propagated_field = np.array([prop_E_field, prop_H_field])
                    prop_h5_file.create_dataset('EH_field', data = propagated_field)
                    prop_h5_file.create_dataset('xCoords',  data = xCoords_snip)
                    prop_h5_file.create_dataset('yCoords',  data = yCoords_snip)
                    prop_h5_file.create_dataset('zProp',    data = zProp)
                    prop_h5_file.create_dataset('nProp',    data = nProp)
                    prop_h5_file.create_dataset('E^2_integral_source', data = E_squared_int_source)
                    prop_h5_file.create_dataset('E^2_integral_propagated', data = E_squared_int_prop)
                    prop_h5_file.create_dataset('H^2_integral_source', data = H_squared_int_source)
                    prop_h5_file.create_dataset('H^2_integral_propagated', data = H_squared_int_prop)
                    prop_h5_file.create_dataset('waveguide_id', data=waveguide_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EH4 to EH5')
    parser.add_argument('-waveguide_id', '--waveguide_id', type=str, help='The label for the job.')
    parser.add_argument('-zPropindex', '--zPropindex', type=int, default=-1, help='index for zProps', required=False)
    args = parser.parse_args()
    wave_to_vol(args.waveguide_id, int(args.zPropindex))

