#!/usr/bin/env python3

import os
import time
import argparse
from printech import *

def output_vigilante(monitor_folder, sleep_time=2):
    '''
    This function takes the path to a folder. It detects all the files with 
    extension .out and it monitors them for new lines that have been included
    in them.
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
    
    while True:
        output_files = update_output_files(monitor_folder)
        for file in output_files:
            last_read_line = line_numbers.get(file, 0)
            new_lines, total_lines = read_new_lines(file, last_read_line)
            if new_lines:
                printer('\n'.join(new_lines))
            line_numbers[file] = total_lines
        time.sleep(sleep_time)

if __name__ == '__main__':
    # use argparse to take a string as a parameter
    parser = argparse.ArgumentParser(description='Monitor output files for new lines')
    parser.add_argument('monitor_folder', help='The path to a folder')
    args = parser.parse_args()
    output_vigilante(args.monitor_folder)
