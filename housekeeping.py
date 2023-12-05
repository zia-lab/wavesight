#!/usr/bin/env python3

import os
import time
import shutil
import argparse
from datapipes import *
from printech import *
from misc import format_time
from hail_david import send_message

data_dir = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
code_dir = '/users/jlizaraz/CEM/wavesight'
exclude_dirs = ['moovies', 'moovies-EH4']

def house_keeper(waveguide_id):
    try:
        # first find all the subdirs that correspond to the waveguide_id
        waveguide_dir = os.path.join(data_dir, waveguide_id)
        err_dir = os.path.join(waveguide_dir, 'err')
        out_dir = os.path.join(waveguide_dir, 'out')
        fig_dir = os.path.join(waveguide_dir, 'figs')
        config_params               = load_from_json(os.path.join(waveguide_dir, 'config.json'))
        config_params['end_time']   = time.time()
        time_taken = config_params['end_time'] - config_params['start_time']
        config_params['time_taken'] = time_taken
        time_taken_str = format_time(time_taken)
        save_to_json(os.path.join(waveguide_dir, 'config.json'), config_params)
        if not os.path.isdir(err_dir):
            os.makedirs(err_dir, exist_ok=True)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)
        wavedir_files = os.listdir(waveguide_dir)
        err_files = [fname for fname in wavedir_files if fname.endswith('.err')]
        out_files = [fname for fname in wavedir_files if fname.endswith('.out')]
        fig_files = [fname for fname in wavedir_files if fname.endswith(('.png','.pdf','.jpg','jpeg'))]
        # move files to folders
        printer("moving err, out, and fig files to their separate folders")
        for err_file in err_files:
            err_file = os.path.join(waveguide_dir, err_file)
            shutil.move(err_file, err_dir)
        for out_file in out_files:
            out_file = os.path.join(waveguide_dir, out_file)
            shutil.move(out_file, out_dir)
        for fig_file in fig_files:
            fig_file = os.path.join(waveguide_dir, fig_file)
            shutil.move(fig_file, fig_dir)
        # delete the bash scripts from the code dir
        # they are kept in the data dir
        bash_slurm_scripts = ['%s-1.sh' % waveguide_id, '%s-2.sh' % waveguide_id]
        printer("removing slurm scripts from code directory")
        for bss in bash_slurm_scripts:
            bss = os.path.join(code_dir, bss)
            os.remove(bss)
        message = "Simulation %s has ended, total time taken %s" % (waveguide_id, time_taken_str)
        send_message(message)
    except Exception as e:
        message = "Housekeeping %s has failed with the following error: %s" % (waveguide_id, e)
        send_message(message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Putting the house in order.')
    parser.add_argument('waveguide_id', help='waveguide id')
    args = parser.parse_args()
    house_keeper(args.waveguide_id)
