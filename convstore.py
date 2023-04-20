#!/usr/bin/env python

import numpy as np
from scipy import special
from scipy.optimize import root_scalar

def zerocrossings(numarray):
    '''
    Return the indices of the zero crossings in the given numpy array.
    '''
    return np.where(np.diff(np.sign(numarray)))[0]

def findallroots(fun, xmin, xmax, dx, xtol=1e-9, 
                 max_iter=10000, num_sigfigs=6,
                 razorsharp=0.00001,
                 method='secant',verbose=False):
    '''
    Find  all  roots of a function in a given interval. To accomplish this
    the  function  is  sampled  with  the  given  resolution dx within the
    interval  (xmin,  xmax)  and  the  zero  crossings are found. The zero
    crossings  are  then  used  as  initial  guesses  for the root finding
    algorithm.

    At  the  end of the search, the roots are disambiguated by rounding to
    the  specified  number  of  significant figures. This is done to avoid
    returning multiple roots that are very close to each other.

    When  a  method  is  given  which  requires a bracket, this bracket is
    estimated  by  slightly increasing the position for the zero crossing.
    This  might  not  give  an  adequate  bracket  for  the  root  finding
    algorithm,  so  it  it expanded until the function effectively changes
    sign.  If the function does not change after max_fixes iterations, the
    root is skipped.

    The roots are returned in a sorted array.

    Parameters
    ----------
    fun : function
        The function to find the roots of, must be univariate.
    xmin : float
        The lower bound of the interval to search for roots.
    xmax : float
        The upper bound of the interval to search for roots.
    dx : float
        The step size to use when searching for roots.
    xtol : float, optional
        The tolerance to use when searching for roots.
    max_iter : int, optional
        The maximum number of iterations to use when searching for roots.
    num_sigfigs : int, optional
        The number of significant figures to use when disambiguating similar roots.
    razorsharp : float, optional
        The amount to expand the bracket for the root finding algorithm.
    method : str, optional
        The method to use when searching for roots.  Must be one of the following:
        'bisect', 'brentq', 'brenth', 'ridder', 'toms748', 'secant'.
    verbose : bool, optional
        If True, print information about the roots as they are found.
    
    Returns
    -------
    zeros : array

    '''
    rightzor = 1+razorsharp
    leftzor = 1-razorsharp
    numSamples = int(np.ceil((xmax-xmin)/dx))
    x = np.linspace(xmin, xmax, numSamples)
    y = fun(x)
    zcs = zerocrossings(y)
    zcs = zcs[~np.isnan(y[zcs])]
    if verbose:
        print(x[zcs])
    zeros = []
    for zc in zcs:
        if verbose:
            print("Evaluating root at x = {:.6f}".format(x[zc]))
        bracket = (leftzor*x[zc], rightzor*x[zc])
        if method in 'bisect brentq brenth ridder toms748'.split(' '):
            l, r = bracket
            fl, fr = fun(l), fun(r)
            max_fixes = 100
            if np.isnan(fl) or np.isnan(fr):
                continue
            nfix = 0
            while np.sign(fun(l) * fun(r)) != -1:
                l = l*leftzor
                r = r*rightzor
                nfix += 1
                if nfix > max_fixes:
                    break
                if np.isnan(fun(l)) or np.isnan(fun(r)):
                    break
            bracket = (l,r)
            if verbose:
                print("bracket", bracket, nfix)
            zerofind = root_scalar(fun,
                                   bracket=bracket,
                                   method=method,
                                   xtol=xtol,
                                   maxiter=max_iter)
        else:
            zerofind = root_scalar(fun,
                                x0=bracket[0],
                                x1=bracket[1], 
                                method='secant',
                                xtol=xtol,
                                maxiter=max_iter)
        the_root = zerofind.root
        the_check = fun(the_root)
        if verbose:
            print(the_root, the_check)
        if (zerofind.converged
            and (the_root <= xmax) 
            and (the_root >= xmin)):
            zeros.append(the_root)
    zeros = np.array(zeros)
    # remove zeros that are equal to the given number of digits
    zeros = np.unique(round2sigfigs(zeros,num_sigfigs))
    zeros = np.sort(zeros)
    return zeros

def find_layout_rectangle(ar, N):
    w_est = np.sqrt(N / ar)
    h_est = w_est * ar

    w_candidates = np.array([np.floor(w_est), np.ceil(w_est)], dtype=int)
    h_candidates = np.array([np.floor(h_est), np.ceil(h_est)], dtype=int)

    valid_pairs = np.array([(w, h) for w in w_candidates for h in h_candidates if w * h >= N])

    ar_diffs = np.abs(valid_pairs[:, 0] / valid_pairs[:, 1] - ar)
    best_pair = valid_pairs[np.argmin(ar_diffs)]

    return best_pair

def round2sigfigs(x, p): 
    '''
    Round x to p significant figures.

    REF: https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy
    '''
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags