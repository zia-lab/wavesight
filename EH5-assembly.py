#!/usr/bin/env python3

import os
import argparse
import wavesight as ws
from datapipes import *
from contextlib import contextmanager

data_dir        = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
exclude_dirs    = ['moovies', 'moovies-EH4', 'err', 'out', 'figs']

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

def vol_assembler(waveguide_id):
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
        all_slices = []
        all_zs     = []
        eh5_dir = os.path.join(job_dir, 'EH5')
        EH5_fnames = os.listdir(eh5_dir)
        EH5_fnames = [fname for fname in EH5_fnames if fname.startswith('EH5-')]
        EH5_fnames = sorted(EH5_fnames, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        for eh5fname in EH5_fnames:
            eh5fname = os.path.join(eh5_dir, eh5fname)
            (EH5_field, _) = ws.load_from_h5(eh5fname)
            EH_field = EH5_field['EH_field']
            zProp    =  EH5_field['zProp']
            all_zs.append(zProp)
            all_slices.append(EH_field)
        all_slices = np.array(all_slices)
        all_slices = np.transpose(all_slices,(1, 2, 3, 4, 0))
        all_zs     = np.array(all_zs)
        assembled_fname = 'EH5.h5'
        assembled_fname = os.path.join(job_dir, assembled_fname)
        print("Saving data to %s" % assembled_fname)
        with h5_handler(assembled_fname, 'w') as assembled_h5_file:
            assembled_h5_file.create_dataset('EH_field_r', data = np.real(all_slices))
            assembled_h5_file.create_dataset('EH_field_i', data = np.imag(all_slices))
            assembled_h5_file.create_dataset('zProp',    data = all_zs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='From slices to complete array')
    parser.add_argument('waveguide_id', type=str, help='The waveguide id.')
    args = parser.parse_args()
    vol_assembler(args.waveguide_id)