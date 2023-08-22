#!/usr/bin/env python3

import os
from time import time
import tempfile
import subprocess
import numpy as np

def latex_eqn_to_png(tex_code, figname, timed=True, outfolder=os.getcwd()):
    '''
    Compile a given bit of LaTeX into png and pdf formats. Best for creating equations, since
    it creates a tight bounding box around the given equation.

    The pdf might have white space around the equation, but the png is very nicely trimmed.
    
    Requires a local installation of pdflatex and convert.
    
    Parameters
    ----------
    tex_code  (str): An equation, including $$, \[\] or LaTeX equation environment.
    time     (bool): If True then the filename includes a timestamp.
    figname   (str): Filenames for png and pdf have that root.
    outfolder (str): Directory name where the images will be stored.

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
    Only good for number one has input oneself.
    '''
    return len(str(num).replace('.',''))

def rounder(num, n):
    '''
    Given a number, round it to n significant figures.
    Parameters
    ----------
    num (float)
    n     (int): how many sig figs to return in the rounded num.
    Returns
    -------
    rounded: the rounded number.
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
    num     (float): A number.
    sig_figs  (int): Number of significant figures.

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