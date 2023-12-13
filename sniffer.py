#!/usr/bin/env python3

import os
import time
import argparse
from printech import *
from hail_david import send_message

error_flag = 'BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES'

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
    def update_output_files(mon_folder):
        return [os.path.join(mon_folder, f) for f in os.listdir(mon_folder) if (f.endswith('.out') or f.endswith('.err'))]

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
                last_read_line = line_numbers.get(file, 0)
                new_lines, total_lines = read_new_lines(file, last_read_line)
                if new_lines:
                    line_block = '\n'.join(new_lines)
                    printer(line_block)
                    if error_flag in line_block:
                        send_message('Bad termination detected in %s' % file.split('/')[-1])
                line_numbers[file] = total_lines
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
