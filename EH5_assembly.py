#!/usr/bin/env python3

import os
import argparse
import wavesight as ws
from datapipes import *
from contextlib import contextmanager
from printech import *

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


def prod(nums):
    pr = 1
    for num in nums:
        pr *= num
    return pr

def vol_assembler(waveguide_id):
    '''
    This function will take the slices of the EH5 field and will assemble
    them together in a single array containing the fields for E and H and
    all their cartesian components.

    It assumes that the slices have been generated by the EH4_to_EH5.py
    script.

    Parameters
    ----------
    waveguide_id : str
        The label for the job.
    
    Returns
    -------
    None
    '''
    # first find all the subdirs that correspond to the waveguide_id
    max_chunk_size_in_MB = 1024
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
        printer("Job %d of %d" % (job_idx+1, len(job_dirs)))
        eh5_dir = os.path.join(job_dir, 'EH5')
        EH5_fnames = os.listdir(eh5_dir)
        EH5_fnames = [fname for fname in EH5_fnames if fname.startswith('EH5-')]
        EH5_fnames = sorted(EH5_fnames, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        assembled_fname = 'EH5.h5'
        assembled_fname = os.path.join(job_dir, assembled_fname)
        first_slice_fname = os.path.join(eh5_dir, EH5_fnames[0])
        (EH5_field, _) = ws.load_from_h5(first_slice_fname)
        slice_shape = EH5_field['EH_field'].shape
        (_, _, N, M) = slice_shape
        num_slices = len(EH5_fnames)
        size_per_slice_in_bytes = prod(slice_shape) * 48
        size_per_slice_in_Mbytes = size_per_slice_in_bytes / 1024**2
        chunk_size = int(np.ceil(max_chunk_size_in_MB/size_per_slice_in_Mbytes))
        printer("aggregating write operations to %d slices" % chunk_size)
        with h5py.File(assembled_fname,'w') as assembled_h5_file:
            zProps = []
            chunks = []
            entire_H5 = assembled_h5_file.create_dataset('EH_field', (2, 3, N, M, num_slices), dtype=np.complex128)
            for slice_index, eh5fname in enumerate(EH5_fnames):
                printer("Slicing %d of %d" % (slice_index+1, num_slices))
                eh5fname = os.path.join(eh5_dir, eh5fname)
                (EH5_field, _) = ws.load_from_h5(eh5fname)
                EH_field = EH5_field['EH_field']
                chunks.append(EH_field)
                zProp    = EH5_field['zProp']
                zProps.append(zProp)
                if slice_index % chunk_size == chunk_size - 1:
                    chunks = np.array(chunks)
                    chunks = np.transpose(chunks, (1,2,3,4,0))
                    idx_start = slice_index - chunk_size + 1
                    idx_end   = slice_index + 1
                    entire_H5[:, :, :, :, idx_start: idx_end] = chunks
                    chunks = []
                elif slice_index == num_slices - 1:
                    num_chunks = len(chunks)
                    chunks = np.array(chunks)
                    chunks = np.transpose(chunks, (1,2,3,4,0))
                    idx_start = slice_index - num_chunks + 1
                    idx_end   = slice_index + 1
                    entire_H5[:, :, :, :, idx_start: idx_end] = chunks
            zProps = np.array(zProps)
            assembled_h5_file.create_dataset('zProp', data = zProps)

def vol_assembler_single_big_one_directly(waveguide_id):
    '''
    This function will take the slices of the EH5 field and will assemble
    them together in a single array containing the fields for E and H and
    all their cartesian components.

    It assumes that the slices have been generated by the EH4_to_EH5.py
    script.

    Parameters
    ----------
    waveguide_id : str
        The label for the job.
    
    Returns
    -------
    None
    '''
    # first find all the subdirs that correspond to the waveguide_id
    max_chunk_size_in_MB = 1024
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
        printer("Job %d of %d" % (job_idx+1, len(job_dirs)))
        eh5_dir = os.path.join(job_dir, 'EH5')
        EH5_fnames = os.listdir(eh5_dir)
        EH5_fnames = [fname for fname in EH5_fnames if fname.startswith('EH5-')]
        EH5_fnames = sorted(EH5_fnames, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        assembled_fname = 'EH5-alt.h5'
        assembled_fname = os.path.join(job_dir, assembled_fname)
        first_slice_fname = os.path.join(eh5_dir, EH5_fnames[0])
        (EH5_field, _) = ws.load_from_h5(first_slice_fname)
        slice_shape = EH5_field['EH_field'].shape
        (_, _, N, M) = slice_shape
        num_slices = len(EH5_fnames)
        size_per_slice_in_bytes = prod(slice_shape) * 48
        size_per_slice_in_Mbytes = size_per_slice_in_bytes / 1024**2
        chunk_size = int(np.ceil(max_chunk_size_in_MB/size_per_slice_in_Mbytes))
        with h5py.File(assembled_fname,'w') as assembled_h5_file:
            zProps = []
            chunks = []
            entire_H5 = assembled_h5_file.create_dataset('EH_field', (2, 3, N, M, num_slices), dtype=np.complex128)
            for slice_index, eh5fname in enumerate(EH5_fnames):
                printer("Slicing %d of %d" % (slice_index+1, num_slices))
                eh5fname = os.path.join(eh5_dir, eh5fname)
                (EH5_field, _) = ws.load_from_h5(eh5fname)
                EH_field = EH5_field['EH_field']
                chunks.append(EH_field)
                zProp    = EH5_field['zProp']
                zProps.append(zProp)
                if slice_index % chunk_size == chunk_size - 1:
                    chunks = np.array(chunks)
                    chunks = np.transpose(chunks, (1,2,3,4,0))
                    chunks = np.ascontiguousarray(chunks)
                    print("cshape", chunks.shape)
                    idx_start = slice_index - chunk_size + 1
                    idx_end   = slice_index + 1
                    # entire_H5[:, :, :, :, idx_start: idx_end] = chunks
                    entire_H5.write_direct(chunks, dest_sel=np.s_[:, :, :, :, idx_start: idx_end])
                    chunks = []
                elif slice_index == num_slices - 1:
                    num_chunks = len(chunks)
                    chunks = np.array(chunks)
                    chunks = np.transpose(chunks, (1,2,3,4,0))
                    chunks = np.ascontiguousarray(chunks)
                    idx_start = slice_index - num_chunks + 1
                    idx_end   = slice_index + 1
                    # entire_H5[:, :, :, :, idx_start: idx_end] = chunks
                    entire_H5.write_direct(chunks, dest_sel=np.s_[:, :, :, :, idx_start: idx_end])
            zProps = np.array(zProps)
            assembled_h5_file.create_dataset('zProp', data = zProps)

# def vol_assembler_slices_together_but_apart(waveguide_id):
#     '''
#     This function will take the slices of the EH5 field and will assemble
#     them together in a single array containing the fields for E and H and
#     all their cartesian components.

#     It assumes that the slices have been generated by the EH4_to_EH5.py
#     script.

#     Parameters
#     ----------
#     waveguide_id : str
#         The label for the job.
    
#     Returns
#     -------
#     None
#     '''
#     # first find all the subdirs that correspond to the waveguide_id
#     waveguide_dir = os.path.join(data_dir, waveguide_id)
#     job_dir_contents = os.listdir(waveguide_dir)
#     job_dir_contents = [a_dir for a_dir in job_dir_contents if a_dir not in exclude_dirs]
#     job_dirs = [os.path.join(waveguide_dir, a_dir) for a_dir in job_dir_contents]
#     job_dirs = [a_dir for a_dir in job_dirs if os.path.isdir(a_dir) ]
#     def wave_sorter(x):
#         idx = int(x.split('-')[-1])
#         return idx
#     job_dirs = list(sorted(job_dirs, key = wave_sorter))
#     # go through each folder and propagate the fields across the given gap
#     for job_idx, job_dir in enumerate(job_dirs):
#         printer("Job %d of %d" % (job_idx+1, len(job_dirs)))
#         eh5_dir = os.path.join(job_dir, 'EH5')
#         EH5_fnames = os.listdir(eh5_dir)
#         EH5_fnames = [fname for fname in EH5_fnames if fname.startswith('EH5-')]
#         EH5_fnames = sorted(EH5_fnames, key=lambda x: int(x.split('-')[-1].split('.')[0]))
#         assembled_fname = 'EH5.h5'
#         assembled_fname = os.path.join(job_dir, assembled_fname)
#         num_slices = len(EH5_fnames)
#         zeroth_slice_fname = os.path.join(eh5_dir, EH5_fnames[0])
#         (EH5_field, _) = ws.load_from_h5(zeroth_slice_fname)
#         EH5_field = EH5_field['EH_field']
#         with h5py.File(assembled_fname,'w') as assembled_h5_file:
#             zProps = []
#             for slice_idx, eh5fname in enumerate(EH5_fnames):
#                 eh5fname = os.path.join(eh5_dir, eh5fname)
#                 (EH5_field, _) = ws.load_from_h5(eh5fname)
#                 EH_field = EH5_field['EH_field']
#                 printer("Slicing %d of %d" % (slice_idx+1, num_slices))
#                 assembled_h5_file.create_dataset('EH_field-%d' % slice_idx, data = EH_field,  dtype=np.complex128)
#                 zProp    = EH5_field['zProp']
#                 zProps.append(zProp)
#             zProps = np.array(zProps)
#             assembled_h5_file.create_dataset('zProp', data = zProps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='From slices to complete array')
    parser.add_argument('waveguide_id', type=str, help='The waveguide id.')
    args = parser.parse_args()
    vol_assembler(args.waveguide_id)