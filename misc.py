#!/usr/bin/env python3

import os
import psutil
import random
import tempfile
import requests
import subprocess
import numpy as np
from math import ceil
from slacking import *
import http.client, urllib
from time import time, sleep
from scipy.signal import find_peaks

try:
    from dave import secrets
    pushover_user = secrets['pushover_user']
    pushover_token = secrets['pushover_token']
    slack_token = secrets['slack_token']
except:
    print("Ask David about secrets.")
    pass

def send_message(message):
    '''
    This function sends a notification message via Pushover.

    Parameters
    ----------
    message : str
        The message to be sent.
    
    Returns
    -------
    None
    '''
    conn = http.client.HTTPSConnection("api.pushover.net",443)
    conn.request("POST", "/1/messages.json",
        urllib.parse.urlencode({
        "token": pushover_token,
        "user": pushover_user,
        "message": message,
        }), { "Content-type": "application/x-www-form-urlencoded" })
    conn.getresponse()
    return None

def send_count(count):
    '''
    This function sends a notification count via Pushover.

    Parameters
    ----------
    count : int
        The count to be sent.
    
    Returns
    -------
    None
    '''
    conn = http.client.HTTPSConnection("api.pushover.net",443)
    conn.request("POST", "/1/glances.json",
        urllib.parse.urlencode({
        "token": pushover_token,
        "user": pushover_user,
        "count": count,
        }), { "Content-type": "application/x-www-form-urlencoded" })
    conn.getresponse()
    return None

def send_image(image_fname,message=''):
    '''
    This function sends images via Pushover.

    Parameters
    ----------
    image_fname : str
        The file name of the image to be sent
    
    Returns
    -------
    None
    '''
    if image_fname.lower().endswith('.jpg'):
        mime = 'image/jpeg'
    elif image_fname.lower().endswith('.png'):
        mime = 'image/png'
    elif image_fname.lower().endswith('.jpeg'):
        mime = 'image/jpeg'
    else:
        return "send jpg, jpeg, or png only"
    r = requests.post("https://api.pushover.net/1/messages.json", data = {
        "token": pushover_token,
        "user": pushover_user,
        "message": message
    },
    files = {
        "attachment": ("image.jpg",
                        open(image_fname, "rb"), "image/jpeg")
    })
    return None

def get_total_size_of_directory(dir_path):
    '''
    Calculate the total size of files in a given directory.
    Only at the first level.

    Parameters
    ----------
    dir_path : str
        Path to the directory whose size is to be calculated.
    
    Returns
    -------
    dir_size_in_MB : float
        The size of the files in the given directory, in MB. 
        This does not include the size of files that may be
        in subdirectories of the given directory.
    '''
    filenames = os.listdir(dir_path)
    fsizes = [os.path.getsize(os.path.join(dir_path, f)) for f in filenames if os.path.isfile(os.path.join(dir_path, f))]
    dir_size_in_MB = sum(fsizes)/1024**2
    return dir_size_in_MB

def dict_summary(adict, header, prekey='', bullet='', aslist=False, split=False):
    '''
    This function takes a dictionary and returns a summary of all the keys
    and values in the dictionary for which the values are either strings or
    numbers. If the value is a dictionary, then the function calls itself
    recursively.

    Parameters
    ----------
    adict : dict
        Dictionary.
    header : str
        something to append to the top of the summary
    prekey : str, optional
        What to prepend to the key in the final format, by default ''
    aslist : bool, optional
        Whether to return as a string of a list with the rows, by default False

    Returns
    -------
    txt_summary : str or list
        A string or list summarizing the contents of the dictionary.
'''
    float_types = (float, np.float16, np.float32, np.float64)
    int_types   = (int, np.int8, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32, np.uint64)
    if header:
        txt_summary = [header]
        txt_summary.append('-'*len(header))
    else:
        txt_summary = []
    for key, val in adict.items():
        if type(val) == str:
            if prekey:
                txt_summary.append(prekey + bullet + key + ' : ' + val)
            else:
                txt_summary.append(bullet + key + ' : ' + val)
        elif type(val) == bool:
            if prekey:
                txt_summary.append(prekey + bullet + key + ' : ' + str(val))
            else:
                txt_summary.append(bullet + key + ' : ' + str(val))
        elif isinstance(val, int_types):
            if prekey:
                txt_summary.append(prekey + bullet + key + ' : ' + str(val))
            else:
                txt_summary.append(bullet + key + ' : ' + str(val))
        elif isinstance(val, float_types):
            txt_summary.append('%s%s : %.3e' % (prekey, key, val))
        elif type(val) == dict:
            nested_dict = dict_summary(val, header='', prekey= prekey + '-' + key, aslist = True)
            txt_summary.extend(nested_dict)
    if aslist:
        if split:
            txt_summary = [(s.split(' : ')[0], ''.join(s.split(' : ')[1:])) for s in txt_summary]
        return txt_summary
    else:
        txt_summary = insert_string_at_nth_position(txt_summary, '', 5)
        return '\n'.join(txt_summary)

def rando_id(num_groups=2, sep='-', verbose=False):
    '''
    Returns a random string formed from pairs of vowels
    and consonants. By default the string returned  has
    27-bit entropy, which makes each string to be about
    10^-8 likely to be chosen.

    Parameters
    ----------
    num_groups: int
        How many 4-char groups the random string has.
    sep: str
        The string that separates the 4-char groups.
    verbose: bool
        Whether to print the entropy of the generated strings.
    
    Returns
    -------
    rid : str
        A random string in a somewhat readable format.
    '''
    vowels = list('AEIOU')
    consonants = list('BCDFGHJKLMNPQRSTVWXYZ')
    rid = [(random.choice(vowels) + random.choice(consonants)
            + random.choice(vowels) + random.choice(consonants)) 
                 for i in range(num_groups)]
    if verbose:
        ent = round((num_groups*2)*np.log2(len(vowels))
                    +(num_groups*2)*np.log2(len(consonants)))
        print('Generated string has %d-bit entropy.' % ent)
    rid = sep.join(rid)
    return rid

def latex_eqn_to_png(tex_code, figname, timed=True, outfolder=os.getcwd()):
    '''
    Compile a given bit of LaTeX into png and pdf formats. Best for creating equations, since
    it creates a tight bounding box around the given equation.

    The pdf might have white space around the equation, but the png is very nicely trimmed.
    
    Requires a local installation of pdflatex and convert.
    
    Parameters
    ----------
    tex_code : str
        An equation, including $$, \[\] or LaTeX equation environment.
    time : bool
        If True then the filename includes a timestamp.
    figname : str
        Filenames for png and pdf have that root.
    outfolder : str
        Directory name where the images will be stored.

    Returns
    -------
    None
    
    Example
    -------
    
    .. code-block:: python

        tex_code = r"""\begin{equation}
        x+y
        \end{equation}"""
        
        latex_eqn_to_png(tex_code, True, 'simple_eqn', outfolder = '/Users/juan/')
        
        --> creates /Users/juan/simple_eqn.pdf and /Users/juan/simple_eqn.npg

    '''
    now = int(time())
    percentage_margin = 10.
    # this header could be modify to add additional packages
    header = r'''\documentclass[border=2pt]{standalone}
    \usepackage{amsmath}
    \usepackage{mathtools}
    \usepackage{amssymb}
    \usepackage{varwidth}
    \begin{document}
    \begin{varwidth}{\linewidth}'''
    footer = '''\end{varwidth}
    \end{document}'''
    texcode = "%s\n%s\n%s" % (header, tex_code, footer)
    with tempfile.TemporaryDirectory() as temp_folder:
        if outfolder == None:
            outfolder = temp_folder
        temp_latex = os.path.join(temp_folder,'texeqn.tex')
        with open(temp_latex,'w') as f:
            f.write(texcode)
        try:
            subprocess.run(['pdflatex', '-halt-on-error', temp_latex],
                           check=True,
                           stdout=subprocess.DEVNULL,
                           cwd=temp_folder)
            subprocess.run(['convert', '-density', '500', 'texeqn.pdf', '-quality', '100', 'texeqn.png'], 
                           check=True,
                           stdout=subprocess.DEVNULL,
                           cwd=temp_folder)
            subprocess.run(['convert', 'texeqn.png', '-fuzz', '1%', '-trim', '+repage', 'texeqn.png'], 
                           check=True,
                           stdout=subprocess.DEVNULL,
                           cwd=temp_folder)
            subprocess.run(['convert', 'texeqn.png', '-fuzz', '1%', '-trim', '+repage', 'texeqn.jpg'], 
                           check=True,
                           stdout=subprocess.DEVNULL,
                           cwd=temp_folder)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting latex to image: {e}")
            return
        if timed and figname == None:
            out_pdf_fname = os.path.join(outfolder,"texeqn-%d.pdf" % now)
            out_png_fname = os.path.join(outfolder,"texeqn-%d.png" % now)
            out_jpg_fname = os.path.join(outfolder,"texeqn-%d.jpg" % now)
        elif timed and figname != None:
            out_pdf_fname = os.path.join(outfolder,"%s-%d.pdf" % (figname,now))
            out_png_fname = os.path.join(outfolder,"%s-%d.png" % (figname,now))
            out_jpg_fname = os.path.join(outfolder,"%s-%d.jpg" % (figname,now))
        elif not timed and figname == None:
            out_pdf_fname = os.path.join(outfolder,"texeqn.pdf")
            out_png_fname = os.path.join(outfolder,"texeqn.png")
            out_jpg_fname = os.path.join(outfolder,"texeqn.jpg")
        elif not timed and figname != None:
            out_pdf_fname = os.path.join(outfolder,"%s.pdf" % (figname))
            out_png_fname = os.path.join(outfolder,"%s.png" % (figname))
            out_jpg_fname = os.path.join(outfolder,"%s.jpg" % (figname))
        os.system('cd "%s"; mv texeqn.pdf "%s"' % (temp_folder, out_pdf_fname))
        os.system('cd "%s"; mv texeqn.png "%s"' % (temp_folder, out_png_fname))
        os.system('cd "%s"; mv texeqn.jpg "%s"' % (temp_folder, out_jpg_fname))
    return None

def sig_figs_in(num):
    '''
    Count how many significant figures are in a given number.
    This might fail if the number has been calculated with.
    Only good for numbers one has input oneself.
    '''
    return len(str(num).replace('.',''))

def rounder(num, n):
    '''
    Given a number, round it to n significant figures.

    Parameters
    ----------
    num : float

    n : int
        how many sig figs to return in the rounded num.

    Returns
    -------
    rounded : int
        the rounded number.
    '''
    rounded = round(num, -int(np.floor(np.log10(num))) + (n - 1))
    return rounded

def num2tex(num, sig_figs):
    '''
    Given a number this function formats it in LaTeX scientific notation.
    The output has the number of significant figures specified by the user, padding
    with zeroes if necessary.

    Parameters
    ----------
    num : float
        A number.
    sig_figs : int
        Number of significant figures.

    Returns
    -------
    str: A string in LaTeX scientific notation.
    '''
    fstring = '%.' + str(sig_figs) + 'g'
    strNum = fstring % num
    if 'e' not in strNum:
        lennum = len(strNum)
        if '.' in strNum:
            lennum -= 1
        else:
            strNum += '.'
            strNum += '0'*(sig_figs-lennum)
            return strNum
        if lennum < sig_figs:
            strNum += '0'*(sig_figs-lennum)
        return strNum
    mant, expo = strNum.split('e')
    lenmant = len(mant)
    if '.' in mant:
        lenmant -= 1
    else:
        mant += '.'
    if lenmant < sig_figs:
        mant += '0'*(sig_figs-lenmant)
    expo = int(expo)
    texForm = '%s \\times 10^{%s}' % (mant, expo)
    return texForm

def var_collisions(varname, these_globals):
    '''
    Given a variable name, return a list of all variables in the
    global namespace that contain that variable name.

    This  is  useful  to  see  if  a possible variable name will
    collide with an existing variable name.

    A  variable  name  is  a  good  variable name if it is not a
    substring  of any other variable name, and no other variable
    name is a substring of it.
    
    Parameters
    ----------
    varname : str
        variable name to check.
    these_globals : dict
        dictionary of global variables.

    Returns
    -------
    commonvars : list
        list of variable names that collide with varname.
    '''
    globalvars = list(filter(lambda x: '_' != x[0], these_globals))
    commonvars = [v for v in globalvars if (varname in v) or (v in varname)]
    if len(commonvars) == 0:
        print("No variable names collide with '%s'." % varname)
    else:
        print("The following variable names collide with '%s':" % varname)
        print(commonvars)

def random_in_range(xmin, xmax, shape):
    '''
    Parameters
    ----------
    xmin : float
        lower bound of random numbers
    xmax : float
        upper bound of random numbers
    shape : tuple or int
        shape of array of random numbers
    
    Returns
    -------
    rando_array : np.array (shape)
        pseudo-random real numbers in given range
    '''
    rando_array = np.random.random(shape)*(xmax-xmin) + xmin
    return rando_array

def transient_scope_wiggler(times, time_signal, tol=0.01):
    '''
    This   function   finds  the  time  at  which  the  relative
    oscillations  of  a  signal that is heading towards a steady
    value  have  dropped  to  a given tolerance. This is done by
    finding  the times at which the signal has local minimas and
    then checking the relative difference between them.

    If  the relative difference is below the tolerance, then the
    first  time  is returned. If not, then the signal is flipped
    and  the  same procedure is applied. If the tolerance is not
    reached,  then  None is returned. This function assumes that
    the  given  signal  is  non-negative, and that the transient
    decays  in  an  oscillatory  manner, after having reached an
    initial peak.

    ..code-block:: python

        ▲                                                                                
        │                                                                        
        │                                                                        
        │       *****                                                         
        │      **   **                                                        
        │     **      **         *******                                      
        │     *        *         *     ***        ******       *****          
        │    **         **      *        **      **    ***    **   *** 
        │    *           **    **         **** ***       ******               
        │    *            ******             ***                              
        │   *               **                                                
        │   *                                                                 
        │   *                                                                 
        │  **                                                                                                                             
        │                                                                        
    ──┼─────────────────────────────────────────────────────────────▶
        │                                                                        

    Parameters
    ----------
    times : np.array
        Time axis.
    time_signal : np.array
        Time-dependent signal.
    tol : float, optional

    Returns
    -------
    good_time: float
        Time at which the signal relative oscillations have 
        dropped below the tolerance.

    '''
    valley_indices = find_peaks(-time_signal)[0]
    valley_times = times[valley_indices]
    relative_diffs = np.abs(np.diff(time_signal[valley_indices]))
    relative_diffs /= time_signal[valley_indices][:-1]
    good_times = valley_times[:-1][relative_diffs<tol]
    if len(good_times)>1:
        return (good_times[0])
    else:
        valley_indices = find_peaks(time_signal)[0]
        valley_times = times[valley_indices]
        relative_diffs = np.abs(np.diff(time_signal[valley_indices])/time_signal[valley_indices][:-1])
        good_times = valley_times[:-1][relative_diffs<tol]
        if len(good_times)>1:
            return (good_times[0])
        else:
            print('No steady state time found within given tolerance.')
            return None

def transient_scope_sigmoid(times, signal, tol=0.01):
    '''
    This  function finds the time at which a given signal with a
    sigmoid-like  behaviour  has  reached  its  steady  state to
    within a given relative tolerance.

    The signal is assumed to be non-negative.

    ..code-block:: python

        ▲                                  ***************  
        │                                ***                
        │                             ***                   
        │                            **                     
        │                          ***                      
        │                         **                        
        │                        **                         
        │                       **                          
        │                      **                           
        │                     **                            
        │                   ***                             
        │                 ***                               
        │               ***                                 
        │             ***                                   
        │         ****                                      
        │**********                                         
        ────────────────────────────────────────────────────▶
        │                                                                                                                     

    Parameters
    ----------
    times : np.array
        the time
    signal : np.array
        the time-dependent signal
    tol : float, optional
        the tolerance, default is 0.01
    Returns
    -------
    good_time: float
        Time at which the signal relative change have
        dropped below the tolerance.

    '''
    signal = signal - signal[0]
    # avoid getting stuck on the initial values
    mean_value = np.mean(signal)
    good_value_indices = np.where(signal >= mean_value)[0]
    relative_diffs = np.abs(np.diff(signal[good_value_indices]))
    relative_diffs /= signal[good_value_indices][:-1]
    good_times = times[good_value_indices][:-1][relative_diffs<tol]
    if len(good_times)>1:
        return (good_times[0])
    else:
        return None

def format_time(seconds):
    '''
    Given a time in seconds this formats a string
    as HH:MM:SS.

    Parameters
    ----------
    seconds : float
        Time in seconds.
    
    Returns
    -------
    fmt_time : str
        Time formatted as HH:MM:SS.
    '''
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    fmt_time = "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
    return fmt_time

def ceil_to_multiple(number, mult = 1):
    '''
    Given a number this function returns the smallest multiple
    of mult that is larger than the given number.

    Parameters
    ----------
    number : float
        The number to be ceiled.
    multiple : float
        The multiple to which the number should be ceiled.
    
    Return
    ------
    ceiled_num : float
        The ceiled number.
    '''
    ceiled_num = ceil(number / mult) * mult
    return ceiled_num

def array_stretcher(arr, min_range_new):
    '''
    Given  a  new range and a 1D array of evenly sampled numbers
    spanning  a certain range. This function returns a new array
    that is sampled to the same resolution but spanning across a
    range  that  is at least as large as the provided range. The
    new range may not be the one provided to the function, since
    this  might conflict with sampling with the same resolution.
    The  elements  of the original array will be a subset of the
    elements of the new array.

    The difference between the length of the new_array and the
    arr is an even number.

    Parameters
    ----------
    arr : np.array
        An array of evenly sampled numbers.
    min_range_new : float
        The minimum span that the new array will have.
    
    Returns
    -------
    (new_range, new_array) : tuple
        2-tuple containing
    new_range : float
        The actual range of the returned array
    new_array : np.array
        A new array with same mean as the original one, but such 
        that new_array[-1]-new_array[0] is within the  requested
        new  range  to  within one sampling unit of the original 
        array.
    '''
    the_mean = np.mean(arr)
    if np.isclose(the_mean, 0):
        the_mean = 0
    arr = arr - the_mean
    N_old = len(arr)
    # determine the spacing of the element in the array
    # it should be sufficient to simply do one difference
    # but because of numerical precision this is more precise
    spacing = np.mean(np.diff(arr))
    # determine the extent of numbers represented in the array
    width_old = arr[-1] - arr[0]
    # determine how many cells need to be added to the right
    add_to_right = ceil_to_multiple(min_range_new/2 - width_old/2, spacing)
    add_to_right = round(add_to_right/spacing)
    # add the same to the left
    N_new = N_old + 2 * add_to_right
    new_range = spacing * (N_new - 1)
    new_array = np.linspace(-new_range/2, new_range/2, N_new)
    # put back the mean
    new_array += the_mean
    return (new_range, new_array)

def sym_pad(arr, pad_width, mode='constant', **kwargs):
    '''
    Given n (which must be even) this function will pad on all sides
    at the last two dimensions of the provided array.

    Parameters
    ----------
    arr : np.array (..., N, N)
        An array with at least two dimensions.
    n : int
        An even number, corresponding to the total width of the pad.
    mode: str
        The mode to use for padding: 'constant', 'mean', etc. Same 
        as those for np.pad.
    kwargs:
        Same as thos that may be provided to np.pad.
    
    Return
    ------
    padded_array : np.array (..., N + n, N + n)
        Same as arr but padded.
    
    References
    ----------
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    '''
    assert pad_width % 2 == 0
    num_dims = arr.ndim
    half_pad = (pad_width//2, pad_width//2)
    padding = [(0, 0)] * (num_dims - 2) + [half_pad, half_pad]
    padded_array = np.pad(arr, padding, mode=mode, **kwargs)
    return padded_array

def insert_string_at_nth_position(str_list, to_insert, n):
    '''
    Given a list of strings, return the same list where
    in the original n-th positions of the list the to_insert
    string has been inserted.

    Parameters
    ----------
    str_list : list of str
        A list of strings.
    to_insert : str
        The string to insert.
    n         : int
        Where should it be.

    Returns
    -------
    riffled_strings : list of str
        The new list of strings.
    '''
    # Check if n is valid
    if n < 1:
        raise ValueError("n should be a positive integer")
    riffled_strings = []  # Initialize a new list
    count = 1  # Initialize a counter
    for s in str_list:
        riffled_strings.append(s)  # Append the current string from the original list
        if count % n == 0 and count != len(str_list):  # Avoid inserting after the last element
            riffled_strings.append(to_insert)  # Insert to_insert after every n-th position
        count += 1  # Increment the counter
    return riffled_strings

def get_memory_usage(caller_frame, string_ouput=True):
    '''
    This function returns the memory ussage of the given process. It also
    returns the line in the code where the call was made.
    Useful for debugging memory usage.

    Parameters
    ----------
    caller_frame : frame
        The frame of the caller function.
    string_output : bool, optional
        Whether to return a string or a float, by default True.
    
    Returns
    -------
    memory_usage, line_no : tuple
        A tuple containing the memory usage and the line number.
    memory_usage : float or str
        Memory usage in MB either as formatted string or a float.
    line_no : int
        The line number of the caller function.
    '''
    line_no = caller_frame.f_lineno
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    if string_ouput:
        return f"(MEM USAGE) #{line_no} : {memory_usage:.2f} MB"
    else:
        return memory_usage, line_no

def get_global_memory_usage(caller_frame, job_id = None, string_output=True):
    '''
    This function returns the memory ussage of the given process. It also
    returns the line in the code where the call was made.
    Useful for debugging memory usage.

    Parameters
    ----------
    caller_frame : frame
        The frame of the caller function.
    string_output : bool, optional
        Whether to return a string or a float, by default True.
    
    Returns
    -------
    memory_usage, line_no : float, int
    memory_usage : float or str
        Memory usage in MB either as formatted string or a float.
    line_no : int
        The line number of the caller function.
    '''
    line_no = caller_frame.f_lineno
    if job_id is None:
        return get_memory_usage(caller_frame)
    else:
        memory_usage_in_GB = get_max_RSS(job_id)
        if memory_usage_in_GB == None:
            memory_usage_in_GB = 0.
            memory_usage = 0
            if string_output:
                return f"(MEM USAGE) #{line_no} : nan MB"
            else:
                return memory_usage, line_no
        else:
            memory_usage = memory_usage_in_GB*1024
            if string_output:
                return f"(MEM USAGE) #{line_no} : {memory_usage:.2f} MB"
            else:
                return memory_usage, line_no

def get_max_RSS(job_id, return_as_string = False, max_retries = 5):
    '''
    Get the max RSS of a given job id.

    Parameters
    ----------
    job_id : int
        The job id at slurm.
    return_as_string : bool, optional
        Whether to return the max RSS as a string, by default False.
    
    Returns
    -------
    max_rss_in_GB : float or str
        The max RSS in GB.
    '''
    try:
        # Execute sacct command to get MaxRSS for the specified job
        job_id = str(job_id)
        job_id = job_id.replace('[','')
        job_id = job_id.replace(']','')
        cmd = ['sstat', 
                '-j',
                str(job_id),
                '--format=MaxRSS',
                '--noheader']
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        # Parse the output to get MaxRSS
        max_rss = (result.strip().split('\n')[0])  # Take the first line of output
        # Handling cases where MaxRSS is not available or the job is not found
        if max_rss == "" or "not found" in max_rss.lower():
            print("job %s not found for maxRSS query" % job_id)
            if return_as_string:
                return ""
            else:
                return None
        max_rss = max_rss.strip()
        if 'No steps running for' in max_rss:
            print("No steps running for job %s" % job_id)
            if return_as_string:
                return ""
            else:
                return None
        elif max_rss.endswith('K'):
            max_rss = max_rss.replace('K', '')
            max_rss = float(max_rss)
            max_rss_in_GB = max_rss / 1024 / 1024
        elif max_rss.endswith('M'):
            max_rss = max_rss.replace('M', '')
            max_rss = float(max_rss)
            max_rss_in_GB = max_rss / 1024
        elif max_rss.endswith('G'):
            max_rss = max_rss.replace('G', '')
            max_rss_in_GB = float(max_rss)
        else:
            max_rss_in_GB = 0.
        if return_as_string:
            max_rss_in_GB = "%.2f GB" % max_rss_in_GB
        return max_rss_in_GB

    except subprocess.CalledProcessError as e:
        # Handle errors in subprocess execution
        print("Error in subprocess execution: %s" % e)
        return ""

slurm_things=r"%j | %T | %i | %C | %l | %L | %r | %S | %m | %D | %E | %k | %N"

def get_squeue_data(user='jlizaraz', get_RSS = True, return_as_dict = True):
    '''
    This function can be use to get the squeue data for a given user.

    Parameters
    ----------
    user : str, optional
        The user for which to get the squeue data, by default 'jlizaraz'.
    get_RSS : bool, optional
        Whether to get the max RSS for each job, by default True.
    return_as_dict : bool, optional
        Whether to return the data as a dictionary, by default True.
    
    Returns
    -------
    jobs : list or dict
        The squeue data for the given user parsed as a list of a dictionary of
        dictionaries.
    '''
    # Define a more comprehensive squeue format string
    cmd = ["squeue", "-u", user, "--format", slurm_things]
    # Execute the command
    try:
        output = subprocess.check_output(cmd).decode()
    except subprocess.CalledProcessError as e:
        print(f"Error executing squeue: {e}")
        return []
    # Parse the output
    jobs = output.splitlines()
    jobs = [job.split(' | ') for job in jobs]
    headers = jobs[0]
    data = list(sorted(jobs[1:], key=lambda x: int(x[2].split('_')[0])))
    jobs = [headers] + data
    if get_RSS:
        header = jobs[0] + ['MaxRSS']
        job_data = []
        for job in jobs[1:]:
            max_rss = get_max_RSS(job[2])
            job_data.append(job + [max_rss])
        jobs = [header] + job_data
    if return_as_dict:
        job_dict = {}
        for job_row in job_data:
            job_dict[job_row[0]] = dict(zip(header, job_row))
        return job_dict
    else:
        return jobs

def index_divider(the_sequence, num_sub_groups, return_indices=False):
    '''
    Given a sequence of elements, divide it into num_sub_groups sub-groups.
    If return_indices is True, return the indices of the elements in the
    original sequence. Otherwise, return the elements themselves.

    Parameters
    ----------
    the_sequence : list, tuple, np.ndarray
        The sequence to be divided.
    num_sub_groups : int
        The number of sub-groups to divide the sequence into.

    Returns
    -------
    index_groups : list of lists
        The indices of the elements in the original sequence, divided into
        at most num_sub_groups sub-groups.
    or iter_groups : list of (same type as the_sequence)
        The elements of the original sequence, divided into at most
        num_sub_groups sub-groups.
    '''
    assert num_sub_groups > 0
    assert isinstance(num_sub_groups, int)
    assert isinstance(the_sequence, (list, tuple, np.ndarray))
    if num_sub_groups == 1:
        if return_indices:
            return [[0]]
        else:
            return [the_sequence]
    range_size = int(len(the_sequence))
    indices = list(range(range_size))
    group_size = int(np.ceil(range_size/num_sub_groups))
    index_groups = []
    iter_groups = []
    for group_idx in range(num_sub_groups):
        idx_start = group_idx * group_size
        idx_end   = min((group_idx + 1) * group_size, range_size)
        group_indices = indices[idx_start: idx_end]
        group_elements = [the_sequence[idx] for idx in group_indices]
        if len(group_elements) == 0:
            break
        index_groups.append(group_indices)
        iter_groups.append(group_elements)
    if return_indices:
        return index_groups
    else:
        if isinstance(the_sequence, np.ndarray):
            return list(map(np.array,iter_groups))
        elif isinstance(the_sequence, list):
            return list(map(list,iter_groups))
        elif isinstance(the_sequence, tuple):
            return list(map(tuple,iter_groups))
        return iter_groups
