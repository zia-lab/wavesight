#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cmasher as cm
import wavesight as ws
from printech import *
from matplotlib import style
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

style.use('dark_background')

cmap          = cm.watermelon
data_dir      = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
exclude_dirs  = ['moovies', 'err', 'figs', 'out']


def EH4_plot(waveguide_id, max_plots=np.inf, extra_fname = ''):
    '''
    This function takes the job id for a given batch simulation
    and creates plots for the fields in the saggital and xy
    monitors.

    A pdf is created for each monitor. Thees pdfs are saved in
    the same directory as the data and also posted to Slack.

    Parameters
    ----------
    waveguide_id : str
        The label for the job.
    max_plots : int, optional
        The maximum number of plots to generate. The default is np.inf.
    extra_fname : str, optional
        Extra string to append to the filename. The default is ''.
    
    Returns
    -------
    None
    '''
    # first find all the subdirs that correspond to the waveguide_id
    waveguide_dir = os.path.join(data_dir, waveguide_id)
    config_params = ws.load_from_json(os.path.join(waveguide_dir, 'config.json'))
    send_to_slack = config_params['send_to_slack']
    job_dir_contents = os.listdir(waveguide_dir)
    job_dir_contents = [a_dir for a_dir in job_dir_contents if a_dir not in exclude_dirs]
    job_dirs = [os.path.join(waveguide_dir, a_dir) for a_dir in job_dir_contents]
    job_dirs = [a_dir for a_dir in job_dirs if os.path.isdir(a_dir)]
    def wave_sorter(x):
        idx = int(x.split('-')[-1])
        return idx
    job_dirs = list(sorted(job_dirs, key = wave_sorter))

    saggital_pdf_fname = '%s-sagittal%s.pdf' % (waveguide_id, extra_fname)
    saggital_pdf_fname = os.path.join(waveguide_dir, saggital_pdf_fname)

    xy_pdf_fname = '%s-xy%s.pdf' % (waveguide_id, extra_fname)
    xy_pdf_fname = os.path.join(waveguide_dir, xy_pdf_fname)
    # these are the objects in which the figures will be saved
    pdf_sink_xy_plots = PdfPages(xy_pdf_fname)
    pdf_sink_sag_plots = PdfPages(saggital_pdf_fname)
    # go through each folder and make the sagittal and xy plots
    for job_idx, job_dir in enumerate(job_dirs):
        mode_id = job_dir.split('/')[-1][:9]
        if job_idx >= max_plots:
            continue
        print('%d/%d' % (job_idx, len(job_dirs) - 1))
        h5_fname_EH2 = 'EH2.h5'
        h5_fname_EH2 = os.path.join(job_dir, h5_fname_EH2)
        try:
            mode_sol, _ = ws.load_from_h5(h5_fname_EH2)
        except:
            printer('Could not load %s' % h5_fname_EH2)
            continue
        h5_fname_EH4 = 'EH4.h5'
        h5_fname_EH4 = os.path.join(job_dir, h5_fname_EH4)
        try:
            EH4_sol,  _ = ws.load_from_h5(h5_fname_EH4)
        except:
            continue
        mode_idx, kz, m, modeType, parity = mode_sol['mode_idx'], mode_sol['kz'], mode_sol['m'], mode_sol['modeType'], mode_sol['parity']
        if parity == 'EVEN':
            suptitle = 'Mode #%s | kz = %.2f 1/μm | m = %d | %s+' % (mode_idx, kz, m, modeType)
        elif parity == 'ODD':
            suptitle = 'Mode #%s | kz = %.2f 1/μm | m = %d | %s-' % (mode_idx, kz, m, modeType)
        else:
            suptitle = 'Mode #%s | kz = %.2f 1/μm | m = %d | %s'  % (mode_idx, kz, m, modeType)
        # get the data from the h5 files
        data = {}
        # the data is saved differently in the case of time-resolved
        # and steady state simulations
        for plane in ['xy','yz','xz']:
            plane_slice = EH4_sol[plane]
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
        monitor_fields['xy'] = np.array([data.pop('e-xy'), data.pop('h-xy')])
        monitor_fields['xz'] = np.array([data.pop('e-xz'), data.pop('h-xz')])
        monitor_fields['yz'] = np.array([data.pop('e-yz'), data.pop('h-yz')])
        xCoordsMonxy            = EH4_sol['xCoordsMonxy']
        zCoordsMonxz = EH4_sol['zCoordsMonxz']

        full_sim_height      = zCoordsMonxz[-1] - zCoordsMonxz[0]
        sim_width            = xCoordsMonxy[-1] - xCoordsMonxy[0]

        xy_extent =  [-sim_width/2, sim_width/2, -sim_width/2, sim_width/2]
        sag_extent = [-sim_width/2, sim_width/2, -full_sim_height/2, full_sim_height/2]
        for sagidx, sagplane in enumerate(['xz','yz']):
            # get a good figsize
            if sagidx == 0:
                dummyField = monitor_fields['xz'][0,0,:,:]
                dummyField = np.real(dummyField)
                fig, ax = plt.subplots()
                ax.imshow(dummyField, extent=sag_extent, cmap=cm.ember)
                ax.set_xlabel('x/um')
                ax.set_ylabel('y/um')
                ax.set_title('Re(Ex) | [1.0]')
                plt.tight_layout()
                fig.canvas.draw()
                bbox = fig.get_tightbbox(fig.canvas.get_renderer())
                plt.close()
                figsize = (3 * 4 * bbox.width/bbox.height, 2 * 4 * 1.1)
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
            for field_idx, field in enumerate(['E','H']):
                for cartesian_idx, cartesian_component in enumerate('xyz'):
                    final_field = monitor_fields[sagplane][field_idx,cartesian_idx,:,:]
                    plotField = np.real(final_field)
                    cmrange = max(np.max(plotField), np.max(-plotField))
                    ax = axes[field_idx, cartesian_idx]
                    ax.imshow(plotField, 
                            cmap=cmap, 
                            origin='lower',
                            vmin=-cmrange,
                            vmax=cmrange,
                            extent=sag_extent,
                            interpolation='none')
                    ax.set_xlabel('%s/μm' % sagplane[0])
                    ax.set_ylabel('z/μm')
                    pretty_range = '$%s$' % ws.num2tex(cmrange, 2)
                    title = 'Re($%s_%s$) | [%s]' % (field, cartesian_component, pretty_range)
                    ax.set_title(title)
            fig.suptitle("%s | %s plane" % (suptitle, sagplane))
            plt.tight_layout()
            pdf_sink_sag_plots.savefig()
            plt.close()
        for plane in ['xy']:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 9 * 2 / 3))
            for field_idx, field in enumerate(['E','H']):
                for cartesian_idx, cartesian_component in enumerate('xyz'):
                    final_field = monitor_fields[plane][field_idx,cartesian_idx,:,:]
                    plotField = np.real(final_field)
                    cmrange = max(np.max(plotField), np.max(-plotField))
                    ax = axes[field_idx, cartesian_idx]
                    ax.imshow(plotField, 
                            cmap=cmap, 
                            origin='lower',
                            vmin=-cmrange,
                            vmax=cmrange,
                            extent=xy_extent,
                            interpolation='none')
                    ax.set_xlabel('x/μm')
                    ax.set_ylabel('y/μm')
                    pretty_range = '$%s$' % ws.num2tex(cmrange, 2)
                    title = 'Re($%s_%s$) | [%s]' % (field, cartesian_component, pretty_range)
                    ax.set_title(title)
            fig.suptitle("%s | %s plane" % (suptitle, plane))
            plt.tight_layout()
            pdf_sink_xy_plots.savefig()
            plt.close()
    pdf_sink_sag_plots.close()
    pdf_sink_xy_plots.close()

    if send_to_slack:
        _ = ws.post_file_to_slack('EH3-EH4 : %s (sagittal)' % waveguide_id,
                                    saggital_pdf_fname,
                                    open(saggital_pdf_fname,'rb'),
                                    'pdf',
                                    os.path.split(saggital_pdf_fname)[-1],
                                    slack_channel='nvs_and_metalenses')
        _ = ws.post_file_to_slack('EH3-EH4 : %s (xy)' % waveguide_id,
                                    xy_pdf_fname,
                                    open(xy_pdf_fname,'rb'),
                                    'pdf',
                                    os.path.split(xy_pdf_fname)[-1],
                                    slack_channel='nvs_and_metalenses')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EH4 plotter.')
    parser.add_argument('waveguide_id', type=str, help='The ID for the waveguide.')
    args = parser.parse_args()
    EH4_plot(args.waveguide_id)
