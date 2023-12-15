#!/usr/bin/env python3

import os
import time
import argparse
from printech import *
from hail_david import send_message

error_flags = ['BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES',
               'AssertionError',
               'MPI_Abort',
               'FileNotFoundError'
               ]

def output_vigilante(monitor_folder, sleep_time=2):
    '''
    This function takes the path to a folder. It detects all the files with 
    extensions .out and .err and it monitors them for new lines that have 
    been added to them.

    Parameters
    ----------
    monitor_folder: str
        The path to a folder
    
    Returns
    -------
    None
    '''
    if '/' == monitor_folder[-1]:
        monitor_stem = os.path.basename(monitor_folder[:-1])
    else:
        monitor_stem = os.path.basename(monitor_folder)
    def update_output_files(mon_folder):
        out_folder = os.path.join(mon_folder, 'out')
        err_folder = os.path.join(mon_folder, 'err')
        outs_and_errs = [os.path.join(mon_folder, f) for f in os.listdir(mon_folder) if (f.endswith('.out') or f.endswith('.err'))]
        if os.path.exists(out_folder):
            outs_and_errs += [os.path.join(out_folder, f) for f in os.listdir(out_folder) if (f.endswith('.out') or f.endswith('.err'))]
        if os.path.exists(err_folder):
            outs_and_errs += [os.path.join(err_folder, f) for f in os.listdir(err_folder) if (f.endswith('.out') or f.endswith('.err'))]
        return outs_and_errs

    def read_new_lines(file, last_read_line):
        with open(file, 'r') as f:
            lines = f.readlines()
            new_lines = [line.strip() for line in lines[last_read_line:]]
            return new_lines, len(lines)
    
    line_numbers = {}
    try:
        while True:
            output_files = update_output_files(monitor_folder)
            for file in output_files:
                if not os.path.exists(file):
                    continue
                file_stem = os.path.basename(file)
                last_read_line = line_numbers.get(file_stem, 0)
                new_lines, total_lines = read_new_lines(file, last_read_line)
                if new_lines:
                    line_block = '\n'.join(new_lines)
                    printer(line_block)
                    error_checks = [error_flag in line_block for error_flag in error_flags]
                    if any(error_checks):
                        send_message('Error detected in %s for %s!' % (monitor_stem, file_stem))
                line_numbers[file_stem] = total_lines
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        printer('\nExiting vigilante ...')
        exit(0)


if __name__ == '__main__':
    # use argparse to take a string as a parameter
    parser = argparse.ArgumentParser(description='Monitor output files for new lines')
    parser.add_argument('monitor_folder', help='The path to a folder')
    args = parser.parse_args()
    output_vigilante(args.monitor_folder)
