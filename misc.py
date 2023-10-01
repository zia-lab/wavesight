#!/usr/bin/env python3

import os
from time import time, sleep
import tempfile
import subprocess
import numpy as np
import http.client, urllib
import requests
import json
import io
import random

try:
    from dave import secrets
    pushover_user = secrets['pushover_user']
    pushover_token = secrets['pushover_token']
    slack_token = secrets['slack_token']
except:
    print("Ask David about secrets.")
    pass

def send_message(message):
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
    if '.jpg' in image_fname.lower():
        mime = 'image/jpeg'
    elif '.png' in image_fname.lower():
        mime = 'image/png'
    elif '.jpeg' in image_fname.lower():
        mime = 'image/jpeg'
    else:
        return "send jpg of jpeg only"
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

default_slack_channel = '#datahose'
slack_icon_emoji = ':see_no_evil:'
slack_user_name = 'labbot'

def post_message_to_slack(text, blocks = None, thread_ts = None, slack_channel = default_slack_channel, max_retries = 20):
    dt = 1
    ok = False
    num_tries = 0
    max_wait_time = 60
    try:
        if thread_ts == None:
            while (not ok) and (num_tries <= max_retries):
                req = requests.post('https://slack.com/api/chat.postMessage', {
                    'token': slack_token,
                    'channel': slack_channel,
                    'text': text,
                    'icon_emoji': slack_icon_emoji,
                    'username': slack_user_name,
                    'blocks': json.dumps(blocks) if blocks else None
                }).json()
                ok = 'error' not in req
                if not ok:
                    if req['error'] == 'ratelimited':
                        print("rate limited, waiting")
                        sleep(dt)
                        dt = dt*2
                        if dt >= max_wait_time:
                            dt = max_wait_time
                        num_tries += 1
                    else:
                        print("Error in request")
                        return req
                if num_tries >= max_retries:
                    print("Max retries reached.")
                    return req
            return req
        else:
            while (not ok) and (num_tries <= max_retries):
                req = requests.post('https://slack.com/api/chat.postMessage', {
                    'token': slack_token,
                    'channel': slack_channel,
                    'text': text,
                    'icon_emoji': slack_icon_emoji,
                    'username': slack_user_name,
                    'thread_ts': thread_ts,
                    'blocks': json.dumps(blocks) if blocks else None
                }).json()
                ok = 'error' not in req
                if not ok:
                    if req['error'] == 'ratelimited':
                        print("rate limited, waiting")
                        sleep(dt)
                        dt = dt*2
                        if dt >= max_wait_time:
                            dt = max_wait_time
                        num_tries += 1
                    else:
                        print("Error in request")
                        return req
                if num_tries >= max_retries:
                    print("Max retries reached.")
                    return req
            return req
    except:
        pass

def post_file_to_slack(text, file_name, file_bytes, file_type=None, title=None, thread_ts = None, slack_channel=default_slack_channel, max_retries=20):
    dt = 1
    ok = False
    num_tries = 0
    max_wait_time = 60
    try:
        if thread_ts == None:
            while (not ok) and (num_tries <= max_retries):
                req = requests.post(
                'https://slack.com/api/files.upload',
                {
                    'token': slack_token,
                    'filename': file_name,
                    'channels': slack_channel,
                    'filetype': file_type,
                    'initial_comment': text,
                    'title': title
                },files = { 'file': file_bytes }).json()
                ok = 'error' not in req
                if not ok:
                    if req['error'] == 'ratelimited':
                        print("rate limited, waiting")
                        sleep(dt)
                        dt = dt*2
                        if dt >= max_wait_time:
                            dt = max_wait_time
                        num_tries += 1
                    else:
                        print("Error in request")
                        return req
                if num_tries >= max_retries:
                    print("Max retries reached.")
                    return req
            return req
        else:
            while (not ok) and (num_tries <= max_retries):
                req = requests.post(
                'https://slack.com/api/files.upload',
                {
                    'token': slack_token,
                    'filename': file_name,
                    'channels': slack_channel,
                    'filetype': file_type,
                    'initial_comment': text,
                    'thread_ts': thread_ts,
                    'title': title
                },files = { 'file': file_bytes }).json()
                ok = 'error' not in req
                if not ok:
                    if req['error'] == 'ratelimited':
                        print("rate limited, waiting")
                        sleep(dt)
                        dt = dt*2
                        if dt >= max_wait_time:
                            dt = max_wait_time
                        num_tries += 1
                    else:
                        print("Error in request")
                        return req
                if num_tries >= max_retries:
                    print("Max retries reached.")
                    return req
            return req
    except:
        pass

def send_fig_to_slack(fig, slack_channel, info_msg, shortfname, thread_ts = None):
    '''
    Use to send a matplotlib figure to Slack.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A figure object from matplotlib.
    slack_channel : str
        Name of Slack channel to send to.
    info_msg : str
        A string message to be send together with the figure.
    shortfname : str
        The string that is used in Slack to refer to the image, with no extension.
    thread_ts : str, optional
        The timestamp of the thread to which to post to inside of given channel. The default is None,
        which means that the message will be posted to the channel directly.
    
    Returns
    -------
    None.
    '''
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='jpg')
        buf.seek(0)
        post_file_to_slack(info_msg, shortfname, buf.read(), slack_channel=slack_channel, thread_ts = thread_ts)
    except:
        pass

def insert_string_at_nth_position(str_list, to_insert, n):
    '''
    Given a list of strings, return the same list where
    in the original n-th positions of the list the to_insert
    string has been inserted.

    Parameters
    ----------
    str_list : list of str
    to_insert : str
    n         : int

    Returns
    -------

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


def dict_summary(adict, header, prekey='', aslist=False):
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
    if header:
        txt_summary = [header]
        txt_summary.append('-'*len(header))
    else:
        txt_summary = []
    for key, val in adict.items():
        if type(val) == str:
            if prekey:
                txt_summary.append(prekey+'-'+key+' : '+val)
            else:
                txt_summary.append('-'+key+' : '+val)
        elif type(val) in [int]:
            if prekey:
                txt_summary.append(prekey+'-'+key+' : ' + str(val))
            else:
                txt_summary.append('-'+key+' : '+str(val))
        elif type(val) in [float]:
            if prekey:
                txt_summary.append('%s-%s : %.3f' % (prekey, key, val))
            else:
                txt_summary.append('%s-%s : %.3f' % (prekey, key, val))
        elif type(val) == dict:
            nested_dict = dict_summary(val, header='', prekey= prekey + '-' + key, aslist = True)
            txt_summary.extend(nested_dict)
    if aslist:
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
    
    tex_code = r"""\begin{equation}
    x+y
    \end{equation}"""
    
    latex_eqn_to_png(tex_code, True, 'simple_eqn', outfolder = '/Users/juan/')
    
    --> creates /Users/juan/simple_eqn.pdf and /Users/juan/simple_eqn.npg
    
    Nov-24 2021-11-24 14:39:28
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