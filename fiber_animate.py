#!/usr/bin/env python3

import os
import h5py
import pickle
import argparse
import numpy as np
import cmasher as cm
import wavesight as ws

from matplotlib import style
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages

style.use('dark_background')
# this script can be used to generate a compendium of the fields in the xz yz sagittal planes
# it takes as simple argument equal to the job id for the entire simulation

parser = argparse.ArgumentParser(description='Job plotter.')
parser.add_argument('big_job_id', type=str, help='The label for the job.')
args = parser.parse_args()

cmap          = cm.watermelon
data_dir      = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
alt_indexing  = False
post_to_slack = True
num_frames    = 50
phase_min     = 0
phase_max     = 2*np.pi
phases        = np.linspace(phase_min, phase_max, num_frames, endpoint=False)

def wave_plotter(big_job_id, max_plots=np.inf, extra_fname = ''):
    '''
    This function takes the job id for a given batch simulation
    and creates plots for the fields in the saggital and xy
    monitors.
    A pdf is created for each monitor. Thees pdfs are saved in
    the same directory as the data and also posted to Slack.
    Parameters
    ----------
    big_job_id : str
        The label for the job.
    max_plots : int, optional
        The maximum number of plots to generate. The default is np.inf.
    extra_fname : str, optional
        Extra string to append to the filename. The default is ''.
    Returns
    -------
    None
    '''
    # first find all the subdirs that correspond to the big_job_id
    data_dir_contents = os.listdir(data_dir)
    def job_dir_filter(a_dir):
        return (a_dir.startswith(big_job_id) 
                and os.path.isdir(os.path.join(data_dir, a_dir)))
    job_dirs = [os.path.join(data_dir, a_dir) for a_dir in data_dir_contents if job_dir_filter(a_dir)]
    pkl_filter = lambda x: (x.startswith('sim') and x.endswith('.pkl'))
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
        def wave_sorter(x):
            idx = int(x.split('-')[-1])
            return idx
        job_dirs = list(sorted(job_dirs, key = wave_sorter))

    # saggital_pdf_fname = '%s-sagittal%s.pdf' % (big_job_id, extra_fname)
    # xy_pdf_fname = '%s-xy%s.pdf' % (big_job_id, extra_fname)
    # these are the objects in which the figures will be saved
    # pdf_sink_xy_plots = PdfPages(xy_pdf_fname)
    # pdf_sink_sag_plots = PdfPages(saggital_pdf_fname)
    # go through each folder and make the sagittal and xy plots
    for job_idx, job_dir in enumerate(job_dirs):
        # this is useful for debugging purposes
        if job_idx >= max_plots:
            continue
        print('%d/%d' % (job_idx, len(job_dirs) - 1))
        # grab the pickle filename
        job_dir_files = os.listdir(str(job_dir))
        print(job_dir)
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
        # these are only used if the data is time-resolved
        h5_keys = ['h_xy_slices_fname_h5', 'e_xy_slices_fname_h5',
                   'h_xz_slices_fname_h5', 'e_xz_slices_fname_h5',
                   'h_yz_slices_fname_h5', 'e_yz_slices_fname_h5']
        data = {}
        # if mode_idx != 1:
        #     continue
        # the data is saved differently in the case of time-resolved
        # and steady state simulations
        if mode_sol['time_resolved']:
            for h5_key in h5_keys:
                h5_fname = 'fiber_platform-' + mode_sol[h5_key]
                field = h5_key[0]
                plane = h5_key.split('_')[1]
                field_list = []
                h5_fname = os.path.join(job_dir, h5_fname)
                data_key_root = '%s-%s' % (field, plane)
                with h5py.File(h5_fname, 'r') as h5f:
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
            h5_fname = mode_sol['eh_monitors_fname_h5']
            with h5py.File(h5_fname, 'r') as h5f:
                for plane in ['xy','yz','xz']:
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
        # These dictionaries are a bit more useful
        monitor_fields       = {}
        monitor_fields['xy'] = np.array([data['e-xy'], data['h-xy']])
        monitor_fields['xz'] = np.array([data['e-xz'], data['h-xz']])
        monitor_fields['yz'] = np.array([data['e-yz'], data['h-yz']])
        sim_width            = mode_sol['sim_width']
        full_sim_height      = mode_sol['full_sim_height']
        coreRadius           = mode_sol['coreRadius']

        extent = [-sim_width/2, sim_width/2, -full_sim_height/2, full_sim_height/2]
        for sagidx, sagplane in enumerate(['xz','yz']):
            # get a good figsize
            if sagidx == 0:
                dummyField = monitor_fields['xz'][0,0,:,:]
                dummyField = np.real(dummyField)
                fig, ax = plt.subplots()
                ax.imshow(dummyField, extent=extent, cmap=cm.ember)
                ax.set_xlabel('x/um')
                ax.set_ylabel('y/um')
                ax.set_title('Re(Ex) | [1.0]')
                plt.tight_layout()
                fig.canvas.draw()
                bbox = fig.get_tightbbox(fig.canvas.get_renderer())
                plt.close()
                figsize = (3 * 4 * bbox.width/bbox.height, 2 * 4 * 1.1)
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
            cmranges = {}
            animation_fname = './moovies/fiber_animate-%s-%s-%s.mp4' % (big_job_id, mode_idx, sagplane)
            if os.path.exists(animation_fname):
                print("Animation already exists. Skipping ...")
                continue
            def update(phase):
                for field_idx, field in enumerate(['E','H']):
                    for cartesian_idx, cartesian_component in enumerate('xyz'):
                        final_field = monitor_fields[sagplane][field_idx,cartesian_idx,:,:]
                        plotField = final_field * np.exp(-1j*phase)
                        if (field, cartesian_component) in cmranges:
                            cmrange = cmranges[(field, cartesian_component)]
                        else:
                            cmrange = np.max(np.abs(plotField))
                        plotField = np.real(plotField)
                        ax = axes[field_idx, cartesian_idx]
                        if phase == 0:
                            ax.add_patch(plt.Rectangle((-sim_width/2,-full_sim_height/2), sim_width, full_sim_height/2, color='w', alpha=0.05))
                            ax.add_patch(plt.Rectangle((-coreRadius,-full_sim_height/2), 2*coreRadius, full_sim_height/2, color='b', alpha=0.1))
                        ax.imshow(plotField, 
                                cmap=cmap, 
                                origin='lower',
                                vmin=-cmrange,
                                vmax=cmrange,
                                extent=extent,
                                interpolation='none')
                        if phase == 0:
                            ax.set_xlabel('%s/μm' % sagplane[0])
                            ax.set_ylabel('z/μm')
                        pretty_range = '$%s$' % ws.num2tex(cmrange, 2)
                        title = 'Re($%s_%s$) | [%s]' % (field, cartesian_component, pretty_range)
                        ax.set_title(title)
                fig.suptitle("%s | %s plane" % (suptitle, sagplane))
                if phase == 0:
                    plt.tight_layout()
            ani = FuncAnimation(fig, update, frames=phases)
            ani.save(animation_fname, writer='ffmpeg', fps=20)
            plt.close(fig)
        for plane in ['xy']:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 9 * 2 / 3))
            cmranges = {}
            animation_fname = './moovies/fiber_animate-%s-%s-%s.mp4' % (big_job_id, mode_idx, plane)
            if os.path.exists(animation_fname):
                print("Animation already exists. Skipping ...")
                continue
            def update(phase):
                # print(phase)
                for field_idx, field in enumerate(['E','H']):
                    for cartesian_idx, cartesian_component in enumerate('xyz'):
                        final_field = monitor_fields[plane][field_idx,cartesian_idx,:,:]
                        extent = [-sim_width/2, sim_width/2, -sim_width/2, sim_width/2]
                        plotField = final_field * np.exp(-1j*phase)
                        if (field, cartesian_component) in cmranges:
                            cmrange = cmranges[(field, cartesian_component)]
                        else:
                            cmrange = np.max(np.abs(plotField))
                        plotField = np.real(plotField)
                        ax = axes[field_idx, cartesian_idx]
                        if phase == 0:
                            ax.add_patch(plt.Circle((0,0), radius=coreRadius, fill=False))
                        ax.imshow(plotField, 
                                cmap=cmap, 
                                origin='lower',
                                vmin=-cmrange,
                                vmax=cmrange,
                                extent=extent,
                                interpolation='none')
                        if phase ==0:
                            ax.set_xlabel('x/μm')
                            ax.set_ylabel('y/μm')
                        if (field, cartesian_component) in cmranges:
                            pretty_range = '$%s$' % ws.num2tex(cmranges[(field, cartesian_component)], 2)
                        else:
                            pretty_range = '$%s$' % ws.num2tex(cmrange, 2)
                            cmranges[(field, cartesian_component)] = cmrange
                        title = 'Re($%s_%s$) | [%s]' % (field, cartesian_component, pretty_range)
                        ax.set_title(title)
                fig.suptitle("%s | %s plane" % (suptitle, plane))
                if phase == 0:
                    plt.tight_layout()
            ani = FuncAnimation(fig, update, frames=phases)
            ani.save(animation_fname, writer='ffmpeg', fps=20)
            plt.close(fig)

    # if post_to_slack:
    #     _ = ws.post_file_to_slack(big_job_id,
    #                                 saggital_pdf_fname,
    #                                 open(saggital_pdf_fname,'rb'),
    #                                 'pdf',
    #                                 saggital_pdf_fname,
    #                                 slack_channel='nvs_and_metalenses')
    #     _ = ws.post_file_to_slack(big_job_id,
    #                                 xy_pdf_fname,
    #                                 open(xy_pdf_fname,'rb'),
    #                                 'pdf',
    #                                 xy_pdf_fname,
    #                                 slack_channel='nvs_and_metalenses')

if __name__ == '__main__':
    wave_plotter(args.big_job_id)