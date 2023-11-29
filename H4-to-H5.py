#!/usr/bin/env python3

import os
import h5py
import argparse
import numpy as np
from printech import *
import wavesight as ws
from contextlib import contextmanager

data_dir        = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
exclude_dirs    = ['moovies','moovies-EH4']
overwrite       = False

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

def wave_to_vol(waveguide_id, zmin, zmax, deltaxy):
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
        source_h5_fname = 'EH4.h5'
        source_h5_fname = os.path.join(job_dir, source_h5_fname)
        mode_sol, _ = ws.load_from_h5(refracted_h5_fname)
        kFree      = mode_sol['kFree']
        λFree      = 2*np.pi/kFree
        nProp      = 2.41
        MEEP_resolution = mode_sol['MEEP_resolution']
        zProps = np.linspace(zmin, zmax, int((zmax - zmin)*MEEP_resolution)+1)
        (EH4_fields, _) = ws.load_from_h5(source_h5_fname)
        xCoords    = EH4_fields['xCoordsMonxy']
        yCoords    = EH4_fields['yCoordsMonxy']
        xCoords_snip = xCoords[(np.abs(xCoords) <= deltaxy)]
        yCoords_snip = yCoords[(np.abs(yCoords) <= deltaxy)]
        xG, yG = np.meshgrid(xCoords, yCoords)
        cross_mask = (np.abs(xG) <= deltaxy) & (np.abs(yG) <= deltaxy)
        cross_mask = np.broadcast_to(cross_mask, (3, cross_mask.shape[0], cross_mask.shape[1]))
        current_width = xCoords[-1] - xCoords[0]
        print(">> The source field spans is defined over a distance of %.2f μm ..." % current_width)
        for prop_idx, zProp in enumerate(zProps):
            print(prop_idx)
            data = {}
            propagated_h5_fname = 'EH5-%d.h5' % prop_idx
            propagated_h5_fname = os.path.join(job_dir, propagated_h5_fname)
            if os.path.exists(propagated_h5_fname) and not overwrite:
                print(">> Done already, continue to next.")
                continue
            else:
                with h5_handler(propagated_h5_fname, 'w') as prop_h5_file:
                    refracted_h5_fname = mode_sol['eh_monitors_fname_h5']
                    refracted_h5_fname = os.path.join(job_dir, refracted_h5_fname)
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
                    prop_E_field = ws.electric_vectorial_diffraction(zProp, transverse_fields[0], xCoords, yCoords, λFree, nProp)
                    prop_H_field = ws.magnetic_vectorial_diffraction(zProp, transverse_fields[0], xCoords, yCoords, λFree, nProp)

                    dx = xCoords[1] - xCoords[0]
                    integrand = np.sum(np.abs(transverse_fields[0])**2, axis=0)
                    E_squared_int_source = ws.simpson_quad_ND(integrand, [dx,dx])
                    integrand = np.sum(np.abs(prop_E_field)**2, axis=0)
                    E_squared_int_prop   = ws.simpson_quad_ND(integrand, [dx,dx])
                    # print('∫E^2_source dxdy = %.3f' % E_squared_int_source)
                    # print('∫E^2_propag dxdy = %.3f' % E_squared_int_prop)

                    integrand = np.sum(np.abs(transverse_fields[1])**2, axis=0)
                    H_squared_int_source = ws.simpson_quad_ND(integrand, [dx,dx])
                    integrand = np.sum(np.abs(prop_H_field)**2, axis=0)
                    H_squared_int_prop   = ws.simpson_quad_ND(integrand, [dx,dx])
                    # print('∫H^2_source dxdy = %.3f' % H_squared_int_source)
                    # print('∫H^2_propag dxdy = %.3f' % H_squared_int_prop)
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
    parser.add_argument('waveguide_id', type=str, help='The label for the job.')
    parser.add_argument('zmin', type=float, help='min prop distance')
    parser.add_argument('zmax', type=float, help='max prop distance')
    parser.add_argument('deltaxy', type=float, help='width of the cross section')
    args = parser.parse_args()
    wave_to_vol(args.waveguide_id, float(args.zmin), float(args.zmax), float(args.deltaxy))

