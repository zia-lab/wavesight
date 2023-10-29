#!/usr/bin/env python3

import h5py
import numpy as np
import os

def save_to_h5(data, filename, group=None, comments='', overwrite=False):
    '''
    Save numeric values, numpy arrays, strings, and nested dictionaries from the provided dictionary to an H5 file.
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
                print(key, type(key))
                group.create_dataset(key, data=value)

def load_from_h5(filename, keys=None, only_keys=False):
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
        return data, comment
