#!/usr/bin/env python3

import numpy as np
from scipy import special
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt

def round2sigfigs(x, p):
    '''
    Round x to p significant figures.

    REF: https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy
    '''
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def tmfungen(λfree, n0, n1, a):
    '''
    This function returns the eigenvalue function for TM(0,m) modes.

    The expressions for the eigenvalue functions are generated in the
    accompanying Mathematica notebook "wavesight.nb".

    Parameters
    ----------
    λfree : float
        Free space wavelength in microns.
    n0 : float
        Cladding refractive index.
    n1 : float
        Core refractive index.
    a : float
        Core radius in microns.

    Returns
    -------
    tm : function
    '''
    n = 0
    def tm(γ):
        return  (n1**2*(special.jv(-1 + n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2))))/(2.*a*n0**2*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2))) + (-special.kn(-1 + n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)) - special.kn(1 + n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)))/(2.*a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)*special.kn(n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)))
    return tm

def tefungen(λfree, n0, n1, a):
    '''
    This function returns the eigenvalue function for TE(0,m) modes.

    The expressions for the eigenvalue functions are generated in the
    accompanying Mathematica notebook "wavesight.nb".

    Parameters
    ----------
    λfree : float
        Free space wavelength in microns.
    n0 : float
        Cladding refractive index.
    n1 : float
        Core refractive index.
    a : float
        Core radius in microns.

    Returns
    -------
    te : function
    '''
    n = 0
    def te(γ):
        return (special.jv(-1 + n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)))/(2.*a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2))) + (-special.kn(-1 + n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)) - special.kn(1 + n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)))/(2.*a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)*special.kn(n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)))
    return te

def hefungen(λfree, n, n0, n1, a):
    '''
    This function returns the eigenvalue function for HE(n,m) modes.

    The expressions for the eigenvalue functions are generated in the
    accompanying Mathematica notebook "wavesight.nb".

    Parameters
    ----------
    λfree : float
        Free space wavelength in microns.
    n0 : float
        Cladding refractive index.
    n1 : float
        Core refractive index.
    a : float
        Core radius in microns.
    n : int
        Order of the HE mode.

    Returns
    -------
    he : function
    '''
    def he(γ):
        return  -((n**2*(1/(γ**2 - (4*n0**2*np.pi**2)/λfree**2) + 1/(-γ**2 + (4*n1**2*np.pi**2)/λfree**2))*(1/(a**2*(γ**2 - (4*n0**2*np.pi**2)/λfree**2)) + n1**2/(a**2*n0**2*(-γ**2 + (4*n1**2*np.pi**2)/λfree**2))))/a**2) + ((special.jv(-1 + n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)))/(2.*a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2))) + (-special.kn(-1 + n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)) - special.kn(1 + n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)))/(2.*a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)*special.kn(n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2))))*((n1**2*(special.jv(-1 + n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2))))/(2.*a*n0**2*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(n,a*np.sqrt(-γ**2 + (4*n1**2*np.pi**2)/λfree**2))) + (-special.kn(-1 + n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)) - special.kn(1 + n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)))/(2.*a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2)*special.kn(n,a*np.sqrt(γ**2 - (4*n0**2*np.pi**2)/λfree**2))))
    return he

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

def multisolver(fiber_spec, drawPlots=False, verbose=False):
    '''
    Solves for the propagation constants of a step-index fiber with the given specifications.

    Parameters
    ----------
    fiber_spec : dict with the following keys:
        nCore : float
            The refractive index of the core.
        nCladding : float
            The refractive index of the cladding.
        coreRadius : float
            The radius of the core in microns.
        free_space_wavelength : float
            The wavelength of the light in free space in microns.
    drawPlots : bool, optional
        Whether to draw plots of the mode profiles. The default is False.
    verbose : bool, optional
        Whether to print out extra information. The default is False.
    
    Returns
    -------
    sol : dict with the following keys:
        TMβ : array
            The propagation constants of the TM(0,n) modes.
        TEβ : array
            The propagation constants of the TE(0,n) modes.
        HEβ : dict
            The propagation constants of the HE(n,m) modes. The keys are the values of n (n>=1).
    '''
    nCore = fiber_spec['nCore']
    if 'NA' in fiber_spec:
        NA = fiber_spec['NA']
        nCladding = np.sqrt(nCore**2 - NA**2)
    else:
        nCladding = fiber_spec['nCladding']
        NA =  np.sqrt(nCore**2 - nCladding**2)
    coreRadius = fiber_spec['coreRadius']
    wavelength = fiber_spec['free_space_wavelength']
    γmax = nCore*2*np.pi/wavelength
    γmin = nCladding*2*np.pi/wavelength
    γspan = γmax - γmin
    γmax = γmax - 0.001*γspan
    γmin = γmin + 0.001*γspan
    # split the solution domain in at least 100 parts
    dγ = (γmax-γmin)/300
    sol = fiber_spec
    Vnum = 2*np.pi * coreRadius * NA / wavelength
    sol['Vnum'] = Vnum
    numModes = int(Vnum**2/2)
    numModesTE = int(Vnum/2)
    numModesTM = int(Vnum/2)
    numModesHE = int(Vnum*(Vnum-2)/4)
    sol['numModesFromVnum'] = numModes
    sol['nCladding'] = nCladding
    if verbose:
        print("Approx number of complex HE modes: ", numModesHE)
        print("Approx number of TE modes: ", numModesTE)
        print("Approx number of TE modes: ", numModesTE)
        print("Approx number of total modes: ", numModes)
        print("="*80)


    sol['γmax'] = γmax
    sol['γmin'] = γmin

    tmfun = tmfungen(λfree=wavelength, 
                    n0=nCladding, 
                    n1=nCore, 
                    a=coreRadius)

    sol['tmfun'] = tmfun

    tefun = tefungen(λfree=wavelength, 
                    n0=nCladding, 
                    n1=nCore, 
                    a=coreRadius)
    
    sol['tefun'] = tefun

    print("Calculting TE(0,n) propagation constants...")
    dγprime = dγ/numModesTE
    temodes = findallroots(tefun, γmin, γmax, dγprime, method='brentq', num_sigfigs=6)

    print("Calculting TM(0,n) propagation constants...")
    tmmodes = findallroots(tmfun, γmin, γmax, dγprime, method='brentq', num_sigfigs=6)
    γrange = np.linspace(γmin, γmax, 1000)
    
    if drawPlots:
        tmvals = tmfun(γrange)
        tevals = tefun(γrange)
        tmmodes = findallroots(tmfun, γmin, γmax, dγ, method='bisect', num_sigfigs=6, verbose=False)
        tmzerocheck = tmfun(tmmodes)
        tezerocheck = tefun(temodes)

        plt.figure(figsize=(10,5))
        plt.plot(γrange, tmvals, 'r')
        plt.scatter(tmmodes,tmzerocheck, c='b')
        plt.plot([γmin, γmax], [0,0], "w:")
        plt.ylim(-1,1)
        plt.title('TM eigenvalues')
        plt.show()

        zerocrossindices = zerocrossings(tevals)
        zcvals = tevals[zerocrossindices]
        good_crossings = np.where(~np.isnan(zcvals))
        zcvals = zcvals[good_crossings]
        zerocrossindices = zerocrossindices[good_crossings]

        plt.figure(figsize=(10,5))
        plt.plot(γrange, tevals, 'r')
        plt.scatter(temodes,tezerocheck, c='b')
        plt.plot([γmin, γmax], [0,0], "w:")
        plt.ylim(-1,1)
        plt.title('TE eigenvalues')
        plt.show()

    n = 1
    hemodes = {}
    print("Calculting HE(n,m) propagation constants...")
    sol['hefuns'] = {}
    maxHEmodes = int(np.sqrt(2*numModesHE))
    while True:
        approxModes = maxHEmodes - n
        approxModes = max(2, approxModes)
        dγprime = dγ / approxModes
        print(f'n={n}',end='|')
        hefun = hefungen(λfree=wavelength, 
                    n=n, 
                    n0=nCladding, 
                    n1=nCore, 
                    a=coreRadius)
        sol['hefuns'][n] = hefun
        hevals = hefun(γrange)
        hezeros = findallroots(hefun, γmin, γmax, dγprime, method='secant', num_sigfigs=10, verbose=False)
        if len(hezeros) == 0:
            break
        hemodes[n] = hezeros
        hezerocheck = hefun(hezeros)
        if drawPlots:
            plt.figure(figsize=(15,5))
            plt.plot(γrange, hevals, 'r')
            plt.plot([γmin, γmax], [0,0], "w:")
            plt.scatter(hezeros, hezerocheck, c='g')
            plt.ylim(-0.01,0.04)
            plt.title('HE(%d, n) eigenvalues' % n)
            plt.show()
        # if n == 1:
        #     break
        n = n + 1

    numCalcModes = (2 * sum(list(map(len,hemodes.values()))), len(temodes), len(tmmodes))
    print("\nHE = %s, TE = %d, TM = %d, TOTAL = %d, FROM_Vnum = %d" % (*numCalcModes, sum(numCalcModes), numModes))
    sol['TEβ'] = temodes
    sol['TMβ'] = tmmodes
    sol['HEβ'] = hemodes
    return sol
