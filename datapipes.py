#!/usr/bin/env python3
import os
import re
import h5py
import json
import inspect
import subprocess
import numpy as np
from functools import wraps
from printech import *

def load_from_json(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            clear_line = line.split('//')[0].strip()
            if clear_line:
                lines.append(clear_line)
    clear_lines = '\n'.join(lines)
    the_dictionary = json.loads(clear_lines)
    return the_dictionary


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Convert numpy arrays to lists
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def save_to_json(filename, source_dict, header=''):
    '''
    This function saves a dictionary to a JSON file with the given filename.
    It admits values to be int, float, str, list, dict, or numpy arrays.
    '''
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(source_dict, file, ensure_ascii=False, cls=CustomJSONEncoder, indent=4)
    return None

def save_to_h5(data, filename, group=None, comments='', overwrite=False):
    '''
    Save numeric values, numpy arrays, strings, and nested dictionaries
    from the provided dictionary to an H5 file.
    If a value of the provided dictionary is a list, then
    the list is converted to a dictionary with keys equal
    to the indices of the lists (as strings).    

    Parameters
    ----------
    data: dict
        The input dictionary.
    filename: str
        The path to the H5 file to save.
    group: (h5py.Group, optional)
        Current group for recursion. Defaults to None.
    
    Returns
    -------
    None

    Example
    -------
    data_dict = {
        'a': [1,2,3],
        'b': {
            'c': "hello",
            'd': np.array([1, 2, 3]),
            'e': {
                'f': 3.14,
                'g': "world"
            }
        },
        'h': np.array([[1, 2], [3, 4]])
    }
    save_to_h5(data_dict, 'output.h5')
    '''
    if group == None and os.path.exists(filename):
        if overwrite:
            print("File already exists, overwriting ...")
        else:
            print("File already exists, doing nothing ...")
            return None
    
    # Initialize the file/group on the first call
    if group is None:
        with h5py.File(filename, 'w') as h5f:
            if comments:
                h5f.attrs['comments'] = comments.encode('utf-8')
            save_to_h5(data, filename, group=h5f)
    else:
        for key, value in data.items():
            # If the value is a dictionary, recurse into it
            if isinstance(value, dict):
                if isinstance(key, int):
                    key = str(key)
                subgroup = group.create_group(key)
                save_to_h5(value, filename, group=subgroup)
                
            # Convert lists to dictionaries
            elif isinstance(value, list):
                list_dict = {str(i): v for i, v in enumerate(value)}
                subgroup = group.create_group(key)
                save_to_h5(list_dict, filename, group=subgroup)
            
            else:
                # If the value is a string, encode it to UTF-8
                if isinstance(value, str):
                    value = value.encode('utf-8')
                group.create_dataset(key, data=value)

def load_from_h5(filename, keys=None, only_keys=False, return_comments=True):
    '''
    Load an H5 file and return its contents or its keys.

    Parameters
    ----------
    filename: str
        The path to the H5 file to load.
    keys: (list, str, or None, optional)
        A list of keys, a single key, or None to load all from the H5 file.
        If a single key is given, then the function does not return
        a dictionary with a comment but it simply returns the value of
        that single key.
    group: (h5py.Group, optional)
        Current group for recursion. Defaults to None.
    only_keys: (bool)
        If set to True, returns only the keys without loading any data. Defaults to False.

    Returns
    -------
    dict, value, list or None
        Depending on the parameters, returns a dictionary, a value, a list of keys, or None.
    Returns:
    tuple: A tuple containing a dictionary with the loaded data and the comments string.
    '''

    def retrieve_group(group):
        data = {}
        for key in group:
            if isinstance(group[key], h5py.Group):
                data[key] = retrieve_group(group[key])
            else:
                data[key] = group[key][()]
                if isinstance(data[key], bytes):
                    data[key] = data[key].decode('utf-8')
        return data

    with h5py.File(filename, 'r') as h5f:
        if only_keys:
            comment = h5f.attrs.get('comments')
            return list(h5f.keys()), comment

        comment = h5f.attrs.get('comments')
        
        if keys:
            data = {}
            for key in keys:
                if key in h5f:
                    if isinstance(h5f[key], h5py.Group):
                        data[key] = retrieve_group(h5f[key])
                    else:
                        data[key] = h5f[key][()]
                        if isinstance(data[key], bytes):
                            data[key] = data[key].decode('utf-8')
        else:
            data = retrieve_group(h5f)
    if len(data) == 1:
        return list(data.values())[0]
    else:
        if return_comments:
            return data, comment
        else:
            return data

def run_script_and_get_json(script_path, args_dict):
    '''
    This function takes the path to a script with a command-line
    interface,   and   a  dictionary  with  values  to  all  the
    parameters that the CLI receives.

    This  assumes  that  the  names  of  the arguments are known
    beforehand so that args_dict may be well formed.

    For an alternative that automatically analyzes the script to
    be run, see extract_cli_arguments_and_run.

    Using  the given values for the arguments this function runs
    the script.

    The  script  must  save  its results to a .json file and the
    last line that it prints must be the name of such file.

    When  the  script  has  finished running, this function will
    then  grab the filename of that json file, read its contents
    and return them as a dictionary.

    Parameters
    ----------
    script_path: str
        Path to Python script with CLI and json output.
    args_dict: dict
        Dictionary with keys equal to the named arguments of the
        CLI and values equal to the values one wishes to input.
    
    Returns
    -------
    result_data: dict
        Dictionary with the parsed results of the function as
        saved in the json file by the Python script.

    '''
    # Construct the command with named arguments
    command = [script_path]
    for arg, value in args_dict.items():
        command.append(f"--{arg}")
        command.append(str(value))
    
    # Run the command and capture the output

    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True)
    stdout_lines = []
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            clean_output = output.strip()
            print(clean_output)
            stdout_lines.append(clean_output)
    
    # Parse the last line of the standard output to get the JSON filename
    json_output_filename = stdout_lines[-1]
    
    # Read the result from the JSON file
    with open(json_output_filename, 'r') as file:
        result_data = json.load(file)
    result_data['stdout'] = '\n'.join(stdout_lines)
    return result_data

def extract_cli_arguments_and_run(script_path):
    '''
    Given  the  path  to  a  python  script  with a command-line
    interface.   This   function   inspects   the  command  line
    interface, which must be built with argparse and it define a
    function  that will take in the CLI parameters as positional
    arguments, feed them to the script via the CLI, and load the
    output of the script.

    Eventhough the CLI might allow for optional  parameters  the 
    function  returned  here  requires  that  every parameter is
    given.

    The  script  needs  to output its results to a json file and
    the  last  line  that it prints must contain the name of the
    file to which the function saved its results to.

    Parameters
    ----------
    script_path : str
        The filepath to the script to be wrapped.
    
    Returns
    -------
    cli_function_wrapper : fun
        A  function  that takes as aruments the arguments of the
        CLI  and  which  returns  a  dictionary with the results
        parse from json output.
    '''
    # Regular expression to match argparse argument definitions
    arg_pattern = re.compile(r'\.add_argument\([\'"]--(\w+)')
    
    # List to store argument names
    arg_names = []
    
    # Read the script and search for argparse definitions
    with open(script_path, 'r') as file:
        for line in file:
            match = arg_pattern.search(line)
            if match:
                # Extract the argument name and add it to the list
                arg_names.append(match.group(1))

    # Function that takes CLI arguments, wraps them into a dictionary,
    # and feeds them to run_script_and_get_json
    def cli_function_template(*args):
        # Construct the args dictionary from positional arguments
        args_dict = dict(zip(arg_names, args))
        
        # Call the run_script_and_get_json function with the script path and args dictionary
        return run_script_and_get_json(script_path, args_dict)
    
    # Set the __name__ attribute of the function to a generic name like 'cli_function'
    cli_function_template.__name__ = 'cli_function'
    
    # Use functools.wraps to preserve the metadata of the template function
    @wraps(cli_function_template)
    def cli_function_wrapper(*args):
        return cli_function_template(*args)
    
    # Update the signature of the wrapper function to match the argument names
    cli_function_wrapper.__signature__ = inspect.signature(cli_function_template).replace(parameters=[
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in arg_names
    ])
    
    return cli_function_wrapper
