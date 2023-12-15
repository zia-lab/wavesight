#!/usr/bin/env python3

# ┌──────────────────────────────────────────────────────────┐
# │              _   _   _   _   _   _   _   _   _           │
# │             / \ / \ / \ / \ / \ / \ / \ / \ / \          │
# │            ( w | a | v | e | s | i | g | h | t )         │
# │             \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/          │
# │         Given a waveguide ID this script generates       │
# │    figures for the transverse fields incident on the     │
# │    metasurface after being propagated across the gap     │
# │      between the end face of the waveguide and the       │
# │               aperture of the metasurface.               │
# │                                                          │
# └──────────────────────────────────────────────────────────┘

import os
import h5py
import argparse
import numpy as np
import cmasher as cm
import wavesight as ws

from matplotlib import style
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
style.use('dark_background')

# this script can be used to generate plots for the fields incident on the metasurface
# it takes as simple argument equal to the job id for the entire simulation

cmap          = cm.watermelon
data_dir      = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
post_to_slack = True
exclude_dirs  = ['moovies']

def wave_plotter(waveguide_id, max_plots=np.inf, extra_fname = ''):
    '''
    This function takes the job id for a given batch simulation
    and creates plots for the fields incident on the metasurface.
    These pdfs are saved in the same directory as the data and 
    are also posted to Slack.
    
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
    job_dir_contents = os.listdir(waveguide_dir)
    job_dir_contents = [a_dir for a_dir in job_dir_contents if a_dir not in exclude_dirs]
    job_dirs = [os.path.join(waveguide_dir, a_dir) for a_dir in job_dir_contents]
    job_dirs = [a_dir for a_dir in job_dirs if os.path.isdir(a_dir)]
    def wave_sorter(x):
        idx = int(x.split('-')[-1])
        return idx
    job_dirs = list(sorted(job_dirs, key = wave_sorter))

    xy_pdf_fname = 'ML-IN-%s%s.pdf' % (waveguide_id, extra_fname)
    xy_pdf_fname = os.path.join(waveguide_dir, xy_pdf_fname)
    # these are the objects in which the figures will be saved
    pdf_sink_xy_plots = PdfPages(xy_pdf_fname)
    # go through each folder and make the sagittal and xy plots
    for job_idx, job_dir in enumerate(job_dirs):
        mode_id = job_dir.split('/')[-1][:9]
        if job_idx >= max_plots:
            continue
        print('%d/%d' % (job_idx, len(job_dirs) - 1))
        h5_fname = 'EH2.h5'
        h5_fname = os.path.join(job_dir, h5_fname)
        mode_sol, _ = ws.load_from_h5(h5_fname)
        mode_idx, kz, m, modeType, parity = mode_sol['mode_idx'], mode_sol['kz'], mode_sol['m'], mode_sol['modeType'], mode_sol['parity']
        coreRadius           = mode_sol['coreRadius']
        kFree                = mode_sol['kFree']
        λFree                = 2*np.pi/kFree

        # get the data from the h5 files
        # these are only used if the data is time-resolved

        # These dictionaries are a bit more useful
        propagated_h5_fname = 'EH3.h5'
        propagated_h5_fname = os.path.join(job_dir, propagated_h5_fname)
        with h5py.File(propagated_h5_fname, 'r') as h5f:
            plotting_fields     = np.array(h5f['EH_field'])
            xCoords             = np.array(h5f['xCoords'])
            yCoords             = np.array(h5f['yCoords'])
            nProp               = np.array(h5f['nProp'])
            zProp               = np.array(h5f['zProp'])

        if parity == 'EVEN':
            suptitle = 'Mode #%s | m = %d | %s+ | λfree = %.1f nm | n = %.2f | Δz = %.1f μm' % (mode_idx, m, modeType, λFree*1000, nProp, zProp)
        elif parity == 'ODD':
            suptitle = 'Mode #%s | m = %d | %s- | λfree = %.1f nm | n = %.2f | Δz = %.1f μm' % (mode_idx, m, modeType, λFree*1000, nProp, zProp)
        else:
            suptitle = 'Mode #%s | m = %d | %s | λfree = %.1f nm | n = %.2f | Δz = %.1f μm'  % (mode_idx, m, modeType, λFree*1000, nProp, zProp)

        extent = [xCoords[0], xCoords[-1], yCoords[0], yCoords[-1]]

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 9 * 2 / 3))
        for field_idx, field in enumerate(['E','H']):
            for cartesian_idx, cartesian_component in enumerate('xyz'):
                final_field = plotting_fields[field_idx,cartesian_idx,:,:]
                plotField = np.real(final_field)
                cmrange = max(np.max(plotField), np.max(-plotField))
                ax = axes[field_idx, cartesian_idx]
                ax.add_patch(plt.Circle((0,0), radius=coreRadius, fill=False))
                ax.imshow(plotField, 
                        cmap=cmap, 
                        origin='lower',
                        vmin=-cmrange,
                        vmax=cmrange,
                        extent=extent,
                        interpolation='none')
                ax.set_xlabel('x/μm')
                ax.set_ylabel('y/μm')
                pretty_range = '$%s$' % ws.num2tex(cmrange, 2)
                title = 'Re($%s_%s$) | [%s]' % (field, cartesian_component, pretty_range)
                ax.set_title(title)
        fig.suptitle(suptitle)
        plt.tight_layout()
        pdf_sink_xy_plots.savefig()
        plt.close()
    pdf_sink_xy_plots.close()

    if post_to_slack:
        _ = ws.post_file_to_slack(waveguide_id,
                                    xy_pdf_fname,
                                    open(xy_pdf_fname,'rb'),
                                    'pdf',
                                    os.path.split(xy_pdf_fname)[-1],
                                    slack_channel='nvs_and_metalenses')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot the fields incident of the metasurface.')
    parser.add_argument('waveguide_id', type=str, help='The label for the job.')
    args = parser.parse_args()
    wave_plotter(args.waveguide_id)