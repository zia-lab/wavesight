#!/usr/bin/env python3

import os
import pickle
import h5py
import cmasher as cm
from matplotlib import style
import numpy as np
from matplotlib import pyplot as plt
# style.use('default')
style.use('dark_background')
from matplotlib.backends.backend_pdf import PdfPages
import wavesight as ws
import argparse

# this script can be used to generate a compendium of the fields in the xz yz sagittal planes
# it takes as simple argument equal to the job id for the entire simulation

parser = argparse.ArgumentParser(description='Job plotter.')
parser.add_argument('big_job_id', type=str, help='The label for the job.')
args = parser.parse_args()

cmap         = cm.watermelon
data_dir     = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
alt_indexing = False
post_to_slack = True

def wave_plotter(big_job_id):
    data_dir_contents = os.listdir(data_dir)
    job_dirs = [os.path.join(data_dir, dir) for dir in data_dir_contents if (dir.startswith(big_job_id) 
                                                                            and os.path.isdir(os.path.join(data_dir,dir)))]
    pkl_filter = lambda x: x.startswith('sim') and x.endswith('.pkl')
    if alt_indexing:
        print("Retrieving the mode ordering ...")
        job_dirs = np.array(job_dirs)
        indices = []
        for job_dir in job_dirs:
            # grab the pickle
            job_dir_files = os.listdir(str(job_dir))
            pkl_fname = os.path.join(job_dir, list(filter(pkl_filter, job_dir_files))[0])
            # load the pickle
            with open(pkl_fname, 'rb') as file:
                mode_sol = pickle.load(file)
            indices.append(mode_sol['mode_idx'])
        indices = np.array(indices)
        sorter = np.argsort(indices)
        job_dirs = job_dirs[sorter]
        job_dirs = list(map(str, job_dirs))
    else:
        job_dirs = list(sorted(job_dirs))

    pdf_sink = PdfPages('%s-sagittal.pdf' % big_job_id)
    for job_idx, job_dir in enumerate(job_dirs):
        print('%d/%d' % (job_idx, len(job_dirs)))
        # grab the pickle
        job_dir_files = os.listdir(str(job_dir))
        pkl_fname = os.path.join(job_dir, list(filter(pkl_filter, job_dir_files))[0])
        # load the pickle
        with open(pkl_fname, 'rb') as file:
            mode_sol = pickle.load(file)
        mode_idx, kz, m, modeType, parity = mode_sol['mode_idx'], mode_sol['kz'], mode_sol['m'], mode_sol['modeType'], mode_sol['parity']
        if parity == 'EVEN':
            suptitle = 'Mode #%s | kz = %.2f 1/μm | m = %d | %s+' % (mode_idx, kz, m, modeType)
        elif parity == 'ODD':
            suptitle = 'Mode #%s | kz = %.2f 1/μm | m = %d | %s-' % (mode_idx, kz, m, modeType)
        else:
            suptitle = 'Mode #%s | kz = %.2f 1/μm | m = %d | %s'  % (mode_idx, kz, m, modeType)
        # get the data from the h5 files
        h5_keys = ['h_xy_slices_fname_h5', 'e_xy_slices_fname_h5',
                'h_xz_slices_fname_h5', 'e_xz_slices_fname_h5',
                'h_yz_slices_fname_h5', 'e_yz_slices_fname_h5']
        data = {}
        for h5_key in h5_keys:
            h5_fname = 'fiber_platform-' + mode_sol[h5_key]
            field = h5_key[0]
            plane = h5_key.split('_')[1]
            field_list = []
            h5_fname = os.path.join(job_dir, h5_fname)
            with h5py.File(h5_fname, 'r') as h5f:
                data_key_root = '%s-%s' % (field, plane)
                for cartesian_component in 'xyz':
                    data_key_r = '%s%s.r' % (field, cartesian_component)
                    data_key_i = '%s%s.i' % (field, cartesian_component)
                    field_data = 1j*np.array(h5f[data_key_i][:,:,-1])
                    field_data += np.array(h5f[data_key_r][:,:,-1])
                    field_list.append(field_data)
            field_list = np.array(field_list)
            data['%s_%s' %(field, plane)] = field_list

        monitor_fields = {}
        for plane in ['xy','xz','yz']:
            field_arrays = []
            for idx, field_name in enumerate(['e','h']):
                field_data = {}
                h5_full_name = os.path.join(job_dir, 'fiber_platform-' + mode_sol[f'{field_name}_{plane}_slices_fname_h5'])
                with h5py.File(h5_full_name,'r') as h5_file:
                    h5_keys = list(h5_file.keys())
                    for h5_key in h5_keys:
                        datum = np.array(h5_file[h5_key][:,:,-1]).T
                        # datum = np.transpose(datum,(1,0,2))
                        field_data[h5_key] = datum
                field_array = np.zeros((3,)+datum.shape, dtype=np.complex_)
                field_parts  = f'{field_name}x {field_name}y {field_name}z'.split(' ')
                for idx, field_component in enumerate(field_parts):
                    field_array[idx] = 1j*np.array(field_data[field_component+'.i'])
                    field_array[idx] += np.array(field_data[field_component+'.r'])
                field_arrays.append(field_array)
            field_arrays = np.array(field_arrays)
            monitor_fields[plane] = field_arrays
        simWidth = mode_sol['simWidth']
        fiber_height = mode_sol['fiber_height']
        coreRadius = mode_sol['coreRadius']

        for sagplane in ['xz','yz']:
            sagdir = sagplane[0]
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(3*3, 2*3 * fiber_height / simWidth))
            for field_idx, field in enumerate(['E','H']):
                for cartesian_idx, cartesian_component in enumerate('xyz'):
                    final_field = monitor_fields[sagplane][field_idx,cartesian_idx,:,:]
                    extent = [-simWidth/2, simWidth/2, -fiber_height/2, fiber_height/2]
                    plotField = np.real(final_field)
                    cmrange = max(np.max(plotField), np.max(-plotField))
                    ax = axes[field_idx, cartesian_idx]
                    ax.add_patch(plt.Rectangle((-simWidth/2,-fiber_height/2), simWidth, fiber_height/2, color='w', alpha=0.05))
                    ax.add_patch(plt.Rectangle((-coreRadius,-fiber_height/2), 2*coreRadius, fiber_height/2, color='b', alpha=0.1))
                    ax.imshow(plotField, 
                            cmap=cmap, 
                            origin='lower',
                            vmin=-cmrange,
                            vmax=cmrange,
                            extent=extent,
                            interpolation='none')
                    ax.set_xlabel('%s/μm' % sagplane[0])
                    ax.set_ylabel('z/μm')
                    title = 'Re($%s_%s$)' % (field, cartesian_component)
                    ax.set_title(title)
            fig.suptitle("%s | %s plane" % (suptitle, sagplane))
            plt.tight_layout()
            pdf_sink.savefig()
            plt.close()
    pdf_sink.close()
    if post_to_slack:
        msg = ws.post_file_to_slack(big_job_id,
                                    'OXUX-ADUM-sagittal.pdf',
                                    open('OXUX-ADUM-sagittal.pdf','rb'),
                                    'pdf',
                                    big_job_id+'.pdf',
                                    slack_channel='nvs_and_metalenses')

if __name__ == '__main__':
    wave_plotter(args.big_job_id)