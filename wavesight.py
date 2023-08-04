#!/usr/bin/env python3

import numpy as np
from scipy import special 
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt
from fieldgen import * 
from convstore import * 
from tqdm.notebook import tqdm
from scipy.interpolate import RegularGridInterpolator
from scipy.fftpack import fft2, ifft2
import cmasher as cmr
import warnings
from matplotlib.patches import Rectangle
from misc import *

real_dtype = np.float64
complex_dtype = np.complex128  

def tmfungen(λfree, n1, n2, a):
    '''
    This function returns the eigenvalue function for TM(0,m) modes.

    The expressions for the eigenvalue functions are generated in the
    accompanying Mathematica notebook "wavesight.nb".

    Parameters
    ----------
    λfree : float
        Free space wavelength in μm.
    n1 : float
        Core refractive index.
    n2 : float
        Cladding refractive index.
    a : float
        Core radius in μm.

    Returns
    -------
    tm : function
    '''
    m = 0
    def tm(kz):
        return (n1**2*(special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))))/(2.*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) + (n2**2*(-special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) - special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))/(2.*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))
    return tm

def tefungen(λfree, n1, n2, a):
    '''
    This function returns the eigenvalue function for TE(0,m) modes.

    The expressions for the eigenvalue functions are generated in the
    accompanying Mathematica notebook "wavesight.nb".

    Parameters
    ----------
    λfree : float
        Free space wavelength in μm.
    n1 : float
        Core refractive index.
    n2 : float
        Cladding refractive index.
    a : float
        Core radius in μm.

    Returns
    -------
    te : function
    '''
    m = 0
    def te(kz):
        return (special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)))/(2.*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) + (-special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) - special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))/(2.*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))
    return te

def hefungen(λfree, m, n1, n2, a):
    '''
    This function returns the eigenvalue function for HE(n,m) modes.

    The expressions for the eigenvalue functions are generated in the
    accompanying Mathematica notebook "wavesight.nb".

    Parameters
    ----------
    λfree : float
        Free space wavelength in μm.
    n1 : float
        Core refractive index.
    n2 : float
        Cladding refractive index.
    a : float
        Core radius in μm.
    m : int
        Order of the HE mode.

    Returns
    -------
    he : function
    '''
    def he(kz):
        return  -((m**2*(1/(-kz**2 + (4*n1**2*np.pi**2)/λfree**2) + 1/(kz**2 - (4*n2**2*np.pi**2)/λfree**2))*(n1**2/(-kz**2 + (4*n1**2*np.pi**2)/λfree**2) + n2**2/(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))/a**2) + ((special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)))/(2.*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) + (-special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) - special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))/(2.*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))*((n1**2*(special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))))/(2.*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) + (n2**2*(-special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) - special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))/(2.*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))
    return he

def multisolver(fiber_spec, solve_modes = 'all', drawPlots=False, verbose=False, tm_te_funcs=False):
    '''
    This function solves the propagation constants of a step-index fiber with the given specifications. This assuming that the cladding is so big that it is effectively infinite.

    The propagation constants determine the z-dependence of the fields along the z-direction. The propagation constants are always bounded by what would be the plane wave wavenumbers in the cladding or in the core (assuming they were homogeneous).

    Parameters
    ----------
    fiber_spec : dict with the following keys:
        nCore : float
            The refractive index of the core.
        nCladding : float
            The refractive index of the cladding. If NA is given then
            nCladding is computed to be consistent with the given nCore
            and given NA.
        coreRadius : float
            The radius of the core in μm.
        λFree : float
            The wavelength of the light in free space in μm.
        grid_divider: int, not necessary here but when later
            used in the layout generator, this is used to determine
            the fineness of the grid by making it equal to
            λfree / max(nCore, nCladding, nFree) / grid_divider
    solve_modes: str either 'all' or 'transvserse'
    drawPlots : bool, optional
        Whether to draw plots of the mode profiles. The default is False.
    verbose : bool, optional
        Whether to print out extra information. The default is False.
    tm_te_funcs : bool, optional
        If True then the return dictionary includes the keys tmfun and tefun. The default is False.
    
    Returns
    -------
    sol : dict with all the keys included in fiber_spec plus these following others:
        kzmax : float
            2π/λfree * nCladding (no kz larger than this)
        kzmin : float
            2π/λfree * nCore (no kz smaller than this)
        Vnum : float
            The V number of the fiber.
        numModesFromVnum: float
            The number of modes according to the V number.
        totalNumModes : int
            The total number of modes that were found.
        tmfun : function
            The eigenvalue function for the TM modes.
        tefun : function
            The eigenvalue function for the TE modes.
        hefuns : dict
            The eigenvalue functions for the HE modes. The keys are values of m (m ≥ 1).
        TMkz : array
            The propagation constants of the TM(0,n) modes.
        TEkz : array
            The propagation constants of the TE(0,n) modes.
        HEkz : dict
            The propagation constants of the HE(m,n) modes. The keys are values of m (m ≥ 1). The values are arrays with the kz(m,n) propagation constants.
    '''
    assert solve_modes in ['transverse', 'all'], "solve_modes must be either all or transverse."
    warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
    nCore = fiber_spec['nCore']
    if 'NA' in fiber_spec:
        print("Estimating nCladding from nCore and NA ...")
        NA        = fiber_spec['NA']
        nCladding = np.sqrt(nCore**2 - NA**2)
    else:
        nCladding = fiber_spec['nCladding']
        NA        =  np.sqrt(nCore**2 - nCladding**2)
    separator     = "="*40
    coreRadius    = fiber_spec['coreRadius']
    wavelength    = fiber_spec['λFree']
    kzmax         = nCore * 2 * np.pi / wavelength
    kzmin         = nCladding * 2 *np.pi / wavelength
    kzspan        = kzmax - kzmin
    kzmax         = kzmax - 0.001 * kzspan
    kzmin         = kzmin + 0.001 * kzspan
    # split the solution domain in at least 300 parts
    dkz           = (kzmax - kzmin) / 300
    sol           = fiber_spec
    Vnum          = 2 * np.pi * coreRadius * NA / wavelength
    sol['Vnum']   = Vnum
    numModes      = int(Vnum**2/2)
    numModesTE    = int(Vnum/2)
    numModesTM    = int(Vnum/2)
    numModesHE    = int(Vnum*(Vnum-2)/4)
    sol['numModesFromVnum'] = numModes
    sol['nCladding'] = nCladding
    maxHEmodes = int(np.sqrt(2*numModesHE))
    if verbose:
        print(separator)
        print("Approx number of complex HE modes: ", numModesHE)
        print("Approx number of TE modes: ", numModesTE)
        print("Approx number of TM modes: ", numModesTM)
        print("Approx number of total modes: ", numModes)
        print("Approx Max n for HE modes: ", maxHEmodes)
        print(separator)
    
    sol['kzmax'] = kzmax
    sol['kzmin'] = kzmin

    # secular equation for TM modes
    tmfun = tmfungen(λfree = wavelength, 
                        n2 = nCladding, 
                        n1 = nCore, 
                         a = coreRadius)

    # secular equation function for TE modes
    tefun = tefungen(λfree = wavelength,  
                        n2 = nCladding, 
                        n1 = nCore, 
                         a = coreRadius)
    
    if tm_te_funcs:
        sol['tmfun'] = tmfun
        sol['tefun'] = tefun

    if solve_modes in ['transverse', 'all']:
        print("Calculating TE(0,n) propagation constants ...")
        dkzprime = dkz/numModesTE
        temodes = findallroots(tefun, kzmin, kzmax, dkzprime, 
                            dtype=real_dtype, method='brentq', num_sigfigs=6)

        print("Calculating TM(0,n) propagation constants ...")
        tmmodes = findallroots(tmfun, kzmin, kzmax, dkzprime, 
                            dtype=real_dtype, method='brentq', num_sigfigs=6)
        kzrange = np.linspace(kzmin, kzmax, 1000, dtype=real_dtype)
    
    if drawPlots:
        tmvals = tmfun(kzrange)
        tevals = tefun(kzrange)
        tmmodes = findallroots(tmfun, kzmin, kzmax, dkz, 
                               dtype=real_dtype, method='bisect', 
                               num_sigfigs=6, verbose = False)
        tmzerocheck = tmfun(tmmodes)
        tezerocheck = tefun(temodes)

        plt.figure(figsize=(10,5))
        plt.plot(kzrange, tmvals, 'r')
        plt.scatter(tmmodes,tmzerocheck, c='b')
        plt.plot([kzmin, kzmax], [0,0], "w:")
        plt.ylim(-1,1)
        plt.title('TM eigenvalues')
        plt.show()

        zerocrossindices = zerocrossings(tevals)
        zcvals = tevals[zerocrossindices]
        good_crossings = np.where(~np.isnan(zcvals))
        zcvals = zcvals[good_crossings]
        zerocrossindices = zerocrossindices[good_crossings]

        plt.figure(figsize=(10,5))
        plt.plot(kzrange, tevals, 'r')
        plt.scatter(temodes,tezerocheck, c='b')
        plt.plot([kzmin, kzmax], [0,0], "w:")
        plt.ylim(-1,1)
        plt.title('TE eigenvalues')
        plt.show()

    hemodes = {}
    if solve_modes == 'all':
        m = 1
        print("Calculating HE(m,n) propagation constants ...")
        while True:
            approxModes = maxHEmodes - m
            approxModes = max(2, approxModes)
            dkzprime = dkz / approxModes
            print(f'm={m}',end='\r')
            hefun = hefungen(λfree=wavelength, 
                        m=m, 
                        n2=nCladding, 
                        n1=nCore, 
                        a=coreRadius)
            # sol['hefuns'][m] = hefun
            hevals = hefun(kzrange)
            hezeros = findallroots(hefun, kzmin, kzmax, dkzprime, 
                                dtype=real_dtype, method='secant', num_sigfigs=10, verbose=False)
            if len(hezeros) == 0:
                break
            hemodes[m] = hezeros
            hezerocheck = hefun(hezeros)
            if drawPlots:
                plt.figure(figsize=(15,5))
                plt.plot(kzrange, hevals, 'r')
                plt.plot([kzmin, kzmax], [0,0], "w:")
                plt.scatter(hezeros, hezerocheck, c='g')
                plt.ylim(-0.01,0.04)
                plt.title('HE(%d, n) eigenvalues' % m)
                plt.show()
            m = m + 1  

    numCalcModes = (2 * sum(list(map(len,hemodes.values()))), len(temodes), len(tmmodes))
    print("")
    print(separator)
    print("HE modes = %s\nTE modes = %d\nTM modes = %d\nTOTAL modes = %d\nFROM_Vnum = %d" % (*numCalcModes, sum(numCalcModes), numModes))
    print(separator)
    # put the modes in the solution dictionary
    sol['TEkz'] = {0: temodes}
    sol['TMkz'] = {0: tmmodes}
    sol['HEkz'] = hemodes
    totalModes = sum(list(map(len, sol['HEkz'].values())))
    totalModes += len(sol['TMkz'][0])
    totalModes += len(sol['TEkz'][0])
    sol['totalModes'] = totalModes
    return sol

def Ae(modetype, a, n1, n2, λfree, m, kz):
    if modetype == "HE":
        return 1
    elif modetype == "TE":
        return 1
    elif modetype == 'TM':
        return 0

def Ah(modetype, a, n1, n2, λfree, m, kz):
    if modetype == "HE":
        return ((-1j)*a**2*n1**2*np.pi*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*λfree**2*(4*n2**2*np.pi**2 - kz**2*λfree**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)))*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))*(special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))) + 1j*np.pi*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*(a**2*n2**2*(4*n1**2*np.pi**2 - kz**2*λfree**2)*(4*n2**2*np.pi**2 - kz**2*λfree**2)*special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))**2 + 4*kz**2*m**2*(n1**2 - n2**2)*λfree**4*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))**2 + 2*a**2*n2**2*(4*n1**2*np.pi**2 - kz**2*λfree**2)*(4*n2**2*np.pi**2 - kz**2*λfree**2)*special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))*special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + a**2*n2**2*(4*n1**2*np.pi**2 - kz**2*λfree**2)*(4*n2**2*np.pi**2 - kz**2*λfree**2)*special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))**2))/(a*kz*m*λfree**3*(-4*n2**2*np.pi**2 + kz**2*λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))*(np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)))*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*(special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))))
    elif modetype == 'TE':
        return 0
    elif modetype == 'TM':
        return 1

def Be(modetype, a, n1, n2, λfree, m, kz):
    if modetype == "HE":
        return special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))/special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))
    elif modetype == 'TE':
        return special.jv(0, a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))/special.kn(0,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))
    elif modetype == 'TM':
        return 0
    
def Bh(modetype, a, n1, n2, λfree, m, kz):
    if modetype == 'HE':
        return (1j*a**2*n1**2*np.pi*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*λfree**2*(4*n1**2*np.pi**2 - kz**2*λfree**2)*(4*n2**2*np.pi**2 - kz**2*λfree**2)*special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))**2*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + 4j*kz**2*m**2*(n1**2 - n2**2)*np.pi*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*λfree**6*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))**2*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + 1j*a**2*n1**2*np.pi*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*λfree**2*(4*n1**2*np.pi**2 - kz**2*λfree**2)*(4*n2**2*np.pi**2 - kz**2*λfree**2)*special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))**2*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + 1j*a**2*n2**2*np.pi*(4*n2**2*np.pi**2 - kz**2*λfree**2)*(-4*n1**2*np.pi**2 + kz**2*λfree**2)**2*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*(special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))) - 1j*a**2*np.pi*(4*n1**2*np.pi**2 - kz**2*λfree**2)*(4*n2**2*np.pi**2 - kz**2*λfree**2)*special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*(2*n1**2*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*λfree**2*special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + n2**2*(4*n1**2*np.pi**2 - kz**2*λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*(special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))))/(a*kz*m*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*λfree**5*(-4*n1**2*np.pi**2 + kz**2*λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))*(np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)))*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*(special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) + special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))))
    elif modetype == "TE":
        return 0.
    elif modetype == 'TM':
        return -((np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.jv(1,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)))/(np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.kn(1,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))
    
def AeAhBeBh(modetype, a, n1, n2, λfree, m, kz):
    return (Ae(modetype, a, n1, n2, λfree, m, kz),
            Ah(modetype, a, n1, n2, λfree, m, kz),
            Be(modetype, a, n1, n2, λfree, m, kz),
            Bh(modetype, a, n1, n2, λfree, m, kz))

def coordinate_layout(fiber_sol):
    '''
    Given a fiber solution, return the coordinate arrays for plotting
    and coordinating the numerical analysis.
    Parameters
    ----------
    fiber_sol : dict
        A dictionary containing the fiber solution. It needs to have
        at least the following keys:
        - 'coreRadius' : float
            The radius of the core.
        - 'claddingIndex' : float  
            The refractive index of the cladding.
        - 'coreIndex' : float
            The refractive index of the core.
        - 'free_space_wavelength' : float
            The wavelength of the light in free space.
    
    Returns
    -------
    a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask : tuple
    a : float
        The radius of the core.
    b : float
        The side of the computational domain.
    Δs : float
        The sampling resolution in the x-y direction. Assumed to be half the 
        wavelength in the core.
    xrange, yrange : 1D arrays
        The coordinate arrays for the x-y directions.
    ρrange, φrange : 1D arrays
    `   The coordinate arrays for the ρ-φ directions in the cylindrical system.
    Xg, Yg : 2D arrays
        The coordinate arrays for x-y.
    ρg, φg : 2D arrays
        ρg has the values for the radial coordinate, φg has the values for the
        azimuthal coordinate.
    nxy : 2D array
        The refractive index profile of the waveguide.
    crossMask : 2D array
        A mask that is True where the core is and False where the cladding is.
    
    '''
    a = fiber_sol['coreRadius']
    b = a*1.5 # assumed, ideally the cladding should extend to infinity
    nCore = fiber_sol['nCore']
    nCladding = fiber_sol['nCladding']
    nFree = fiber_sol['nFree']
    λfree = fiber_sol['λFree']
    grid_divider = fiber_sol['grid_divider']
    maxIndex = max(nCore, nCladding, nFree)
    Δs = λfree / maxIndex / grid_divider
    # calculate the coordinates arrays
    numSamples = int(2 * b / Δs)
    xrange     = np.linspace(-b, b, numSamples, dtype=real_dtype)
    yrange     = np.linspace(-b, b, numSamples, dtype=real_dtype)
    ρrange     = np.linspace(0 , np.sqrt(2)*b, numSamples, dtype=real_dtype)
    φrange     = np.linspace(-np.pi, np.pi, numSamples, dtype=real_dtype)
    Xg, Yg     = np.meshgrid(xrange, yrange)
    ρg         = np.sqrt(Xg**2 + Yg**2)
    φg         = np.arctan2(Yg, Xg)

    crossMask          = np.zeros((numSamples, numSamples)).astype(np.bool8)
    crossMask[ρg <= a] = True
    crossMask[ρg > a]  = False

    # #Coords-Calc
    nxy             = np.zeros((numSamples, numSamples))
    nxy[crossMask]  =  nCore
    nxy[~crossMask] =  nCladding

    return a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask, numSamples

def calculate_numerical_basis(fiber_sol):
    '''
    Given a solution for the propagation modes of an optical waveguide, calculate a numerical basis.
    
    Parameters
    ----------
    fiber_sol : dict
        A dictionary containing the fiber solution. It needs to have
        the following keys:
        - 'coreRadius' : float
            The radius of the core.
        - 'nCladding' : float
            The refractive index of the cladding.
        - 'nCore' : float
            The refractive index of the core.
        - 'free_space_wavelength' : float
            The wavelength of the light in free space.
        - 'totalModes' : int
            The total number of calculated modes.
        - 'TEkz' : 1D dict
            A single key equal to m=0, the values are array with
            the TE modes propagation constants.
        - 'TMkz' : 1D dict
            A single key equal to m=0, the values are array with
            the TM modes propagation constants.
        - 'HEkz' : 1D dict
            Keys are m values, values are 1D arrays of kz values.
    
    Returns
    -------
    fiber_sol : dict:
        The same dictionary as the input, but with two new keys:
        - 'coord_layout' : tuple
            The tuple returned by the coordinate_layout function.
        - 'eigenbasis' : 5D np.array
            The numerical basis. The first dimension is the mode number,
            the second dimension only has two values, 0 and 1, 0 for the
            E  field, and 1 for the H field. The third dimension is used
            for  the  different components of the corresponding field in
            cylindrical  coordinates.  The  first index being the radial
            component,  the  second index being the azimuthal component,
            and  the  third  index  being  the  z component. Finally the
            fourth  and fifth dimensions are arrays that hold the values
            of   the  corresponding  field  components.  The  modes  are
            enumerated  such that first all TE modes are given, then all
            TM modes, and finally all HE modes. This same enumeration is
            the one used for the 'eigenbasis_nums' key.
        - 'eigenbasis_nums': list of tuples
            A  list  of  tuples,  each  tuple has nine values, the first
            value  is  a  string, either 'TE', 'TM', or 'HE', the second
            value is value of m, and the third value is the value of  n.
            The  order  in  this list is so that the i-th element of the
            list  corresponds  to  the  i-th mode in the eigenbasis. The
            fourth  and  fifth values are for the transverse propagation
            constants  the fourth one being the one inside the core, and
            the fifth one being the one outside the core. From the sixth
            to the ninth values are the values for Ae, Ah, Be, and Bh.
    '''
    warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
    warnings.filterwarnings('ignore', 'invalid value encountered in multiply')
    warnings.filterwarnings('ignore', 'invalid value encountered in divide')
    coord_layout = coordinate_layout(fiber_sol)
    nCore = fiber_sol['nCore']
    nCladding = fiber_sol['nCladding']
    λfree = fiber_sol['λFree']
    if 'coord_layout' in fiber_sol:
        del fiber_sol['coord_layout']
    if 'eigenbasis' in fiber_sol:
        del fiber_sol['eigenbasis']
    a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask, numSamples = coord_layout
    # determine how many modes there are in total
    totalModes = fiber_sol['totalModes']
    eigenbasis = np.zeros((totalModes, 2, 3, numSamples, numSamples),  dtype=complex_dtype)
    counter = 0
    eigenbasis_nums = []
    for modtype in ['TE','TM','HE']:
        solkey = modtype + 'kz'
        for m, kzs in fiber_sol[solkey].items():
            global_phase = np.exp(1j * m * φg)
            for kzidx, kz in enumerate(tqdm(kzs)):
                γ = np.sqrt(nCore**2*4*np.pi**2/λfree**2 - kz**2)
                β = np.sqrt(kz**2 - nCladding**2*4*np.pi**2/λfree**2)
                # #AB-Calc
                Ae, Ah, Be, Bh = AeAhBeBh(modtype, a, nCore, nCladding, λfree, m, kz)
                eigen_nums = (modtype, m, kzidx, γ, β, Ae, Ah, Be, Bh)
                eigenbasis_nums.append(eigen_nums)
                # calculate the transverse field components
                for idx, genfuncs in enumerate([
                                (Et1genρ, Et2genρ, Ht1genρ, Ht2genρ), 
                                (Et1genφ, Et2genφ, Ht1genφ, Ht2genφ)]):
                    Et1genρf, Et2genρf, Ht1genρf, Ht2genρf = genfuncs
                    Et1ρ = Et1genρf(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCore, nCladding)
                    Et2ρ = Et2genρf(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCore, nCladding)
                    Ht1ρ = Ht1genρf(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCore, nCladding)
                    Ht2ρ = Ht2genρf(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCore, nCladding)
                    Etvals1ρ             = Et1ρ(ρrange)
                    Etvals1ρ             = np.interp(ρg, ρrange, Etvals1ρ)
                    Etvals1ρ[~crossMask] = 0
                    Etvals2ρ             = Et2ρ(ρrange)
                    Etvals2ρ             = np.interp(ρg, ρrange, Etvals2ρ)
                    Etvals2ρ[crossMask]  = 0
                    Etρ                  = Etvals1ρ + Etvals2ρ
                    Etρ[np.isnan(Etρ)]   = 0
                    Etnorm               = np.sum(np.abs(Etρ[0,::])**2 + np.abs(Etρ[1,::])**2)
                    Etnorm               = np.sqrt(Etnorm)
                    Htvals1ρ             = Ht1ρ(ρrange)
                    Htvals1ρ             = np.interp(ρg, ρrange, Htvals1ρ)
                    Htvals1ρ[~crossMask] = 0
                    Htvals2ρ             = Ht2ρ(ρrange)
                    Htvals2ρ             = np.interp(ρg, ρrange, Htvals2ρ)
                    Htvals2ρ[crossMask]  = 0
                    Htρ                  = Htvals1ρ + Htvals2ρ
                    Htρ[np.isnan(Htρ)]   = 0
                    Htnorm               = np.sum(np.abs(Htρ[0,::])**2 + np.abs(Htρ[1,::])**2)
                    Htnorm               = np.sqrt(Htnorm)
                    eigenbasis[counter, 0, idx, :, :] = Etρ
                    eigenbasis[counter, 1, idx, :, :] = Htρ
                # calculate the axial field components
                for idx, genfuncs in enumerate([(Ezgen_1, Ezgen_2), 
                                                (Hzgen_1, Hzgen_2)]):
                    # Et stands here either for Ez or Hz
                    # idx stands for the index for E or H
                    Et1genρf, Et2genρf = genfuncs
                    if idx == 0:
                        Et1ρ = Et1genρf(Ae, m, γ)
                        Et2ρ = Et2genρf(Be, m, β)
                    else:
                        Et1ρ = Et1genρf(Ah, m, γ)
                        Et2ρ = Et2genρf(Bh, m, β)
                    Etvals1ρ             = Et1ρ(ρrange)
                    Etvals1ρ             = np.interp(ρg, ρrange, Etvals1ρ)
                    Etvals1ρ[~crossMask] = 0
                    Etvals2ρ             = Et2ρ(ρrange)
                    Etvals2ρ             = np.interp(ρg, ρrange, Etvals2ρ)
                    Etvals2ρ[crossMask]  = 0
                    Etρ                  = Etvals1ρ + Etvals2ρ
                    Etρ[np.isnan(Etρ)]   = 0
                    # set the third index to the calculated field
                    # remember that this needs to be multplied by the global phase
                    eigenbasis[counter, idx, 2, :, :] = Etρ
                # normalize the field
                Etnorm = np.sum(np.abs(eigenbasis[counter, 0, 0, :, :])**2 
                                + np.abs(eigenbasis[counter, 0, 1, :, :])**2
                                + np.abs(eigenbasis[counter, 0, 2, :, :])**2)
                Etnorm = np.sqrt(Etnorm)
                Htnorm = np.sum(np.abs(eigenbasis[counter, 1, 0, :, :])**2 
                                + np.abs(eigenbasis[counter, 1, 1, :, :])**2
                                + np.abs(eigenbasis[counter, 1, 2, :, :])**2)
                Htnorm = np.sqrt(Htnorm)
                eigenbasis[counter, 0, :, :, :] /= Etnorm
                eigenbasis[counter, 1, :, :, :] /= Htnorm
                eigenbasis[counter, 0, :, :, :] *= global_phase
                eigenbasis[counter, 1, :, :, :] *= global_phase
                counter += 1
    fiber_sol['eigenbasis'] = eigenbasis
    fiber_sol['coord_layout'] = coord_layout
    fiber_sol['eigenbasis_nums'] = eigenbasis_nums
    return fiber_sol

def poyntingrefractor(Efield, Hfield, nxy, nFree, verbose=False):
    '''
    Approximate  the  refracted  field across a planar interface
    using  the Poynting vector as an analog to the wavevector of
    a plane-wave.

    Any  evanescent  fields  are  ignored. All cases where there
    would  be  total  internal reflection the refracted field is
    set to zero.

    Parameters
    ----------
    Efield : np.array (3, N, M)
        The electric field incident on the interface.
    Hfield : np.array (3, N, M)
        The H-field incident on the interface.
    nxy : np.array    (N, M)
        The refractive index transverse to the interface inside the incident medium.
    nFree : float
        The refractive index of the homogeneous refractive medium.
    verbose : bool, optional
        Whether to print or not progress messages, by default False.
    
    Returns
    -------
    Eref, Href : tuple of np.array (3, N, M)
        The refracted electric and magnetic fields.
    '''
    # #EXH-Calc
    # calculate the Poynting vector
    if verbose:
        print("Calculating the Poynting vector field...")
    Sfield = 0.5*np.real(np.cross(Efield, Hfield, axis=0))

    # #normIncidentk-Calc
    # calculate the magnitude of the Poynting field
    if verbose:
        print("Calculating the magnitude of the Poynting field...")
    Sfieldmag = np.sqrt(np.sum(Sfield**2, axis=0))
    if verbose:
        print("Calculating the transverse component of the Poynting field...")
    # calculate the unit vector in the direction of the Poynting vector
    if verbose:
        print("Calculating the unit vector in the direction of the Poynting vector...")
    kfield = Sfield / Sfieldmag
    # calculate the transverse component of the Poynting vector
    Stransverse = np.sqrt(Sfield[0,:,:]**2 + Sfield[1,:,:]**2)

    # #β-Calc
    if verbose:
        print("Calculating the angle of incidence field...")
    # Assuming that the normal is pointing in the +z direction
    βfield = np.arctan2(Stransverse, Sfield[2, :, :])

    # #θ-Calc
    if verbose:
        print("Calculating the angle of refraction field...")
    θfield = np.arcsin(nxy/nFree * np.sin(βfield))

    # #FresnelS-Calc
    if verbose:
        print("Calculating the Fresnel coefficients...")
    
    fresnelS = (2 * nxy * np.cos(βfield) 
                / ( nxy * np.cos(βfield) 
                + np.sqrt(nFree**2 
                                - nxy**2 * np.sin(βfield)**2)
                    )
                )
    
    # #FresnelP-Calc
    fresnelP = (2 * nFree * nxy * np.cos(βfield) 
                / (nFree**2 * np.cos(βfield) 
                +  nxy * np.sqrt(nFree**2 
                            - nxy**2 * np.sin(βfield)**2)
                    )
                )
    
    # #ζ-Calc
    if verbose:
        print("Calculating the ζ of the local coord system...")
    # calculate the unit vector field perpendicular to the plane of incidence
    # which is basically k X z
    ζfield = np.zeros(kfield.shape)
    ζfield[0] = kfield[1]
    ζfield[1] = -kfield[0]
    # normalize it
    ζfield /= np.sqrt(np.sum(np.abs(ζfield)**2, axis=0))

    # in the case of normal incidence, the ζ is not defined
    # so it can be set to the unit vector in the first direction
    normalIncidence = (βfield == 0)
    ζfield[0][normalIncidence] = 1
    ζfield[1][normalIncidence] = 0
    ζfield[2][normalIncidence] = 0
    # #EincS-Calc
    if verbose:
        print("Calculating the S and P component of the incident electric field...")
    # decompose the field in P and S polarizations
    # first find P-pol and then use the complement to determine S-pol
    # the S-pol can be obtained by the dot product of E with ζ
    ESdot = Efield[0,:,:] * ζfield[0] + Efield[1,:,:] * ζfield[1]
    EincS = np.zeros(ζfield.shape)
    EincS[0] = ESdot[0] * ζfield[0]
    EincS[1] = ESdot[1] * ζfield[1]
    # #EincP-Calc
    EincP = Efield - EincS
    del ESdot

    # #ErefS-Calc
    if verbose:
        print("Calculating the S and P component of the refracted electric field...")
    ErefS = np.zeros(Efield.shape)
    ErefS[0] = fresnelS * EincS[0]
    ErefS[1] = fresnelS * EincS[1]
    ErefS[2] = fresnelS * EincS[2]
    ErefS[np.isnan(ErefS)] = 0

    # #ErefP-Calc
    ErefP = np.zeros(Efield.shape)
    ErefP[0] = fresnelP * EincP[0]
    ErefP[1] = fresnelP * EincP[1]
    ErefP[2] = fresnelP * EincP[2]
    ErefP[np.isnan(ErefP)] = 0
    # #Eref-Calc
    if verbose:
        print("Calculating the total refracted electric field...")
    Eref  = ErefS + ErefP

    # #kref-Calc
    if verbose:
        print("Calculating the refracted wavevector (normalized) field...")
    ξfield    = np.zeros(ζfield.shape)
    ξfield[0] = -ζfield[1]
    ξfield[1] = ζfield[0]
    kref      = np.zeros(kfield.shape)
    kref[0]   = np.sin(θfield) * ξfield[0]
    kref[1]   = np.sin(θfield) * ξfield[1]
    kref[2]   = np.cos(θfield)

    # #Href-Calc
    if verbose:
        print("Calculating the refracted H field...")
    Href = nFree * np.cross(kref, Eref, axis=0)

    return kref, Eref, Href


def from_cyl_cart_to_cart_cart(field):
    '''
    Given  a  field  in  cylindrical  coordinates, convert it to
    cartesian  coordinates.  The  given  field  is assumed to be
    anchored  to a cartesian coordinate system in the sense that
    each  of the indices in its array corresponds to a cartesian
    grid  in the usual sense but the value of the vector at that
    position is given in terms of cylindrical coordinates.

    This  function  assumes  that  the  region  described by the
    cartesian coordinates is a squar centered on the axis.

    Parameters
    ----------
    field : np.ndarray
        A  field  in  cylindrical  coordinates  with  shape  (3,
        numSamples,  numSamples) the indices being the ρ, φ, and
        z components respectively.

    Returns
    -------
    ccfield : np.ndarray
        A   field   in  cartesian  coordinates  with  shape  (3,
        numSamples,  numSamples) the indices being the x, y, and
        z components respectively of the given vector field.
    '''
    xrange = np.linspace(-1,1,field.shape[1])
    yrange = np.linspace(-1,1,field.shape[2])
    Xg, Yg = np.meshgrid(xrange, yrange)
    φg     = np.arctan2(Yg, Xg)
    ccfield = np.zeros(field.shape, dtype=field.dtype)
    ccfield[2] = field[2]
    # create the cartesian coordinates of the cylindrical unit vector fields
    # first for the ρ component
    uρ = np.zeros((2,field.shape[1],field.shape[2]))
    uρ[0] = np.cos(φg)
    uρ[1] = np.sin(φg)
    # now for the φ component
    uφ = np.zeros((2,field.shape[1],field.shape[2]))
    uφ[0] = -np.sin(φg)
    uφ[1] = np.cos(φg)
    # using these convert the cylindrical components to cartesian
    # adding the x components of what comes from the two unit vectors
    # scaled up by the field values in the cylindrical basis
    ccfield[0] = field[0]*uρ[0] + field[1]*uφ[0]
    ccfield[1] = field[0]*uρ[1] + field[1]*uφ[1]
    return ccfield


def angular_farfield_propagator(field, λFree, nMedium, Zf, Zi, si, sf, Δf = None, options = {}):
    '''This function approximates the farfield of a field given the nearfield
    and the propagation distance.

    This is done by using the angular spectrum method. In which the Fourier
    transform of the nearfield is understood to describe the coeffients that
    approximate the farfield as the superposition of plane waves.

    It assumes that the nearfield is given on a square grid perpendicular to
    the z-axis. The farfield is also given on a square grid perpendicular to
    the z-axis.

    The farfield is approximated by the following formula:
        Efar = (2π 1j / kFree) * ((Zf-Zi)/Rfsq) * np.exp(1j*kFree*Rf) * S2
         with
        S2 equal to the Fourier transform extrapolated to the position
        of the farfield.

    Parameters
    ----------
    field : np.array (3, Ni, Ni) or (Ni, Ni) or (1, Ni, Ni)
        An array describing the nearfield of the field to be propagated.
    nMedium : float
        Refractive index of the homogenous medium or propagation.
    Zf : float
        The z-coordinate of the farfield.
    Zi : float
        The z-coordinate of the nearfield.
    si : float
        The size of the nearfield.
    sf : float
        The size of the farfield.
    Δf : float (optional)
        Spatial resolution of the farfield. If None, it is set to Δi.
    options : dict (optional)
        A dictionary of options for the function. The options are:
            'return_fourier' : bool
                If True, the function also returns the Fourier transform of the
                nearfield.
            'return_as_dict' : bool
                If True, the function returns a dictionary with the following
                keys:
                    'Efar' : np.array (3, Nf, Nf)
                        The farfield of the field.
                    'Efourier' : np.array (3, Ni, Ni)
                        The Fourier transform of the nearfield.
                If False, the function returns the two arrays in the same order
                as the keys in the dictionary.
    
    Returns
    -------
    Return depends on the options dictionary, see above.
    '''
    rmin = abs(Zf - Zi)
    rmax = np.sqrt(rmin**2 + si**2/2 + sf**2/2)
    λmedium = λFree/nMedium
    if rmax - rmin > λmedium:
        print(f"ATTENTION: {(rmax-rmin)/λmedium:.1f} = (rmax - rmin) / λmedium > 1. The angular spectrum approximation is problematic.")
        print("Consider decreasing si, decreasing sf, or increasing Zf.")
    all_options = {'return_fourier': False,
                   'return_as_dict': False}
    for k, v in all_options.items():
        if k in options:
            globals()[k] = options[k]
        else:
            globals()[k] = v
    c     = 1
    κ     = 1 # 2 * np.pi
    ω     = c * 2*np.pi/λFree
    kFree = ω*nMedium/c
    far_out = kFree*rmin
    assert far_out > 100, f'kr = {round(kFree*rmin)} too small for angular-spectrum approximation.'
    if len(field.shape) == 2:
        field = np.array([field])
    num_components = field.shape[0]
    Ni = field[-1].shape[0]
    # spatial resolution in the nearfield
    Δi    = si/Ni
    if Δf == None:
        # reduces the resolution of the farfield so that both fields
        # have the same number of samples
        Δf    = Δi * (sf / si)
        Nf    = Ni
    else:
        # number of samples in each dir of farfield
        Nf    = int(np.ceil(sf/Δf))
    # coordinate layout for the nearfield
    xi    = np.linspace(-si/2, si/2, Ni)
    yi    = np.linspace(-si/2, si/2, Ni)
    # coordinate layout for the farfield
    xf    = np.linspace(-sf/2, sf/2, Nf)
    yf    = np.linspace(-sf/2, sf/2, Nf)
    # coordinate arrays in the farfield
    Xf, Yf = np.meshgrid(xf, yf)
    # the square of the distance from the center of neafield to the farfield
    Rfsq  = Xf**2 + Yf**2 + (Zf-Zi)**2
    # the distance itself
    Rf    = np.sqrt(Rfsq)
    # The direction cosines at the farfield
    XfoRf = κ * Xf/Rf
    YfoRf = κ * Yf/Rf
    # A factor that is common to all components of the farfield
    S1    = (2 * np.pi * 1j / kFree) * ((Zf-Zi)/Rfsq) * np.exp(1j*kFree*Rf)
    # Spatial frequencies of the FFT
    kx    = (np.fft.fftfreq(Ni, d=Δi))
    kx    = κ * np.fft.fftshift(kx)
    ky    = kx
    # Meshgrid of spatial frequencies
    Kx, Ky = np.meshgrid(kx, ky)
    # Initialize the farfield
    Efar  = np.zeros((num_components, Nf, Nf), dtype=np.complex128)
    if return_fourier:
        Efourier = np.zeros((num_components, Ni, Ni), dtype=np.complex128)
    for field_component in range(num_components):
        afield = field[field_component]
        Eifou = np.fft.fft2(afield)
        Eifou = np.fft.fftshift(Eifou)
        if return_fourier:
            Efourier[field_component] = Eifou
        S2interpolator = RegularGridInterpolator((kx, ky), Eifou)
        S2 = S2interpolator((XfoRf, YfoRf))
        Efar[field_component] = S1 * S2
    if return_as_dict:
        if return_fourier:
            return {'farfield': Efar, 'fourier': Efourier}
        else:
            return {'farfield': Efar}
    else:
        if return_fourier:
            return Efar, Efourier
        else:
            return Efar

def device_layout(device_design):
    '''
    This function creates a figure representing the device layout.
    
    Parameters
    ----------
    device_design (dict): with at least the following keys:
        coreRadius (float): the radius of the core in μm
        mlRadius (float): the radius of the metalens in μm
        Δ (float): the distance between the end face of the fiber and the start of the metalens in μm
        mlPitch (float): the pitch of the metalens in μm
        emDepth (float): the depth of the emitter in the crystal host in μm, measured from the base of the metalens pillars
        emΔxy (float): the lateral uncertainty (in μm) in the position of the emitter
        emΔz (float): the uncertainty in the axial position of the emitter in μm
        mlHeight (float): the height of the metalens in μm
        λFree (float): the free-space wavelength of the emitter in μm
        nCore (float): the refractive index of the core
        nHost (float): the refractive index of the host
        nClad (float): the refractive index of the cladding
        NA (float): the numerical aperture of the fiber
    
    Returns
    -------
    fig, ax: the figure and axis objects
    '''
    def CenteredRectangle(xy, width, height, **opts):
        x, y = xy
        return Rectangle((x - width/2, y - height/2), width, height, **opts)
    def BottomRectangle(xy, width, height, **opts):
        x, y = xy
        return Rectangle((x - width/2, y), width, height, **opts)
    coreRadius = device_design['coreRadius']
    mlRadius = device_design['mlRadius']
    Δ = device_design['Δ']
    mlPitch = device_design['mlPitch']
    emDepth = device_design['emDepth']
    emΔxy = device_design['emΔxy']
    emΔz = device_design['emΔz']
    mlHeight = device_design['mlHeight']
    λFree = device_design['λFree']
    nCore = device_design['nCore']
    nHost = device_design['nHost']
    wholeWidth  = 1.2*2*max(coreRadius, mlRadius)
    textframe   = wholeWidth * 0.05
    fiberTip    = emDepth* 0.75
    NA = device_design['NA']
    if 'nCladding' not in device_design:
        nCladding = np.sqrt(nCore**2 - NA**2)
    else:
        nCladding = device_design['nCladding']
    designSpec  = [f'λFree = {λFree*1000} nm',
                f'Δ = {Δ} μm',
                f'coreRad = {coreRadius} μm',
                f'mlHeight = {mlHeight} μm',
                f'emDepth = {emDepth} μm',
                f'Δxy = {emΔxy} μm',
                f'nCore = {nCore}',
                f'nHost = {nHost}',
                'nClad = %.2f' % nCladding,
                'fiberNA = %.2f' % NA,
                f'Δz = {emΔz} μm']
    designSpec = list(sorted(designSpec, key=lambda x: -len(x)))
    designSpec = '\n'.join(designSpec)
    wholeHeight = (fiberTip + Δ + mlHeight + emDepth + 4 * emΔz)
    top_left_corner = (-wholeWidth/2 + textframe, wholeHeight-fiberTip - textframe)
    finalFieldWidth  =  2*emΔxy*1
    finalFieldHeight =  2*emΔz*1
    fig, ax = plt.subplots()
    clad = BottomRectangle((0, 0-fiberTip), wholeWidth, fiberTip, color='c', alpha=0.5)
    ax.add_patch(clad)
    core = BottomRectangle((0, 0-fiberTip), coreRadius*2, fiberTip, color='r', alpha=0.5)
    ax.add_patch(core)
    ml = BottomRectangle((0, fiberTip + Δ - fiberTip), 2*mlRadius, mlHeight, color='g', alpha=0.5)
    ax.add_patch(ml)
    host = BottomRectangle((0, fiberTip + Δ + mlHeight - fiberTip), wholeWidth, wholeHeight, color='g', alpha=0.3)
    ax.add_patch(host)
    fieldBox = CenteredRectangle((0, fiberTip + Δ + mlHeight + emDepth - fiberTip), finalFieldWidth, finalFieldHeight, color='w', alpha=0.5)
    ax.add_patch(fieldBox)
    ax.set_xlim(-wholeWidth/2, wholeWidth/2)
    ax.set_ylim(-fiberTip, wholeHeight - fiberTip)
    ax.plot(([0,0],[0- fiberTip, wholeHeight- fiberTip]), 'w:', lw=1, alpha=0.2)
    ax.text(*top_left_corner, designSpec, fontsize=9, ha='left', va = 'top', fontdict={'family': 'monospace'})
    ax.set_xlabel('x/μm')
    ax.set_ylabel('z/μm')
    ax.set_aspect('equal')
    plt.close()
    return fig, ax

def scalar_field_FFT_RS_prop_func(Lobs, z, apertureFunction, λfree, nref, numSamples='auto', interpFun=False):
    '''
    scalar_field_func_prop  takes a field component in an aperture plane
    and   propagates   that   to   an  observation  plane  by  using  an
    implementation  of the direct integration of the Rayleigh-Sommerfeld
    diffraction  integral.  This  implementation  is based on the method
    described  in  Shen  and  Wang  (2006).  The field is sampled in the
    aperture  plane  and  in the obserbation plane using a uniform grid.
    The field is assumed to be zero outside of the aperture plane.

    Parameters
    ----------
    +  Lobs  (float): spatial width of the obsevation region, in μm. The
    observation  region  is  assumed to be a squared centered on (x,y) =
    (0,0),  and  extending  from  -Lobs/2  to Lobs/2 in both the x and y
    directions.

    + z (float): distance between the aperture plane and the observation
    plane, given in μm. The aperture plane is assumed to be at z=0.

    +  apertureFunction  (function):  a bi-variate function that returns
    the  complex  amplitude of the field in the aperture plane. Input to
    the function is assumed to be in cartesian coordinates x,y.  If  the
    function has an attribute "null" set to True, then the function will
    simply return a matrix of zeros.

    + λfree (float): wavelength in vacuum of field, given in μm.

    + nref (float): refractive index of the propagating medium.

    Options
    -------
    +  numSamples   (int or Automatic): number of samples to use in  the
    aperture  plane  and  the  observation  plane. The aperture plane is
    sampled  using  a  uniform  grid,  and the observation plane is also
    sampled using a uniform grid. The default is Automatic in which case
    numSamples  is  calculated  so that the sample size is equal to half
    the wavelength of the wave inside of the propagating medium.

    Returns
    -------
    (numSamples, xCoords, yCoords, field) (tuple)
    +  xCoords (np.array): x coordinates of the observation plane, given
    in μm.

    +  yCoords (np.array): y coordinates of the observation plane, given
    in μm.

    +   field   (np.array):  complex  amplitude  of  the  field  in  the
    observation  plane.  The top left corner of the array corresponds to
    the  lower  left  corner  of  the observation plane. The coordinates
    associated with each element in the given array should be taken from
    xCoords and yCoords.

    References
    ----------
    +   Shen,   Fabin,  and  Anbo  Wang.  "Fast-Fourier-transform  based
    numerical integration method for the Rayleigh-Sommerfeld diffraction
    formula." Applied optics 45, no. 6 (2006): 1102-1110.

    Example (double slit):
    ----------------------

    def doubleSlit(separation, width, height):
        def apertureFun(x, y):
            return np.where((
                            ((np.abs(x - separation/2) <= width/2) | (np.abs(x + separation/2) <= width/2))
                            & (np.abs(y) <= height/2)
            ), 1, 0)
        return apertureFun

    slitSep = 4.
    slitWidth = 1.
    slitHeight = 10.
    Lobs = 100.
    z = 100
    nref = 1
    λfree = 0.532
    numSamples = 'auto'

    apFun = doubleSlit(slitSep, slitWidth, slitHeight)

    # Estimate the diffraction pattern from the simplified formula
    diforders = range(10)
    xmaxi = []
    for diforder in diforders:
        stheta = diforder * λfree / slitSep
        if stheta > 1:
            break
        else:
            theta = np.arcsin(stheta)
            xdif = z * np.tan(theta)
            if np.abs(xdif) <= Lobs/2:
                xmaxi.append(xdif)
                xmaxi.append(-xdif)


    numSamples, xCoords, yCoords, field = scalar_field_func_prop(Lobs, z, apFun, λfree, nref, numSamples)

    extent = (xCoords[0], xCoords[-1], yCoords[0], yCoords[-1])
    pField = np.abs(field)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(pField,
            extent=extent,
            cmap=cmr.ember,
            interpolation='spline16')
    ax.scatter(xmaxi, np.zeros_like(xmaxi), s=20, facecolors='none', edgecolors='w', alpha=0.5)
    ax.set_xlabel('x/μm')
    ax.set_ylabel('y/μm')
    ax.set_title('Diffraction pattern of a double slit\ns=%.2fμm | w=%.2fμm | L=%.2fμm | Δz=%.2fμm' % (slitWidth, slitWidth, slitHeight, z))
    plt.show()
    
    '''
    assert z>=0, 'z must be non-negative'

    λ = λfree / nref
    k = 2*np.pi/λ

    def gr(r):
        return (np.exp(1j * k * r) * z) * (1./r - 1j*k) / (2*np.pi * r**2)

    if numSamples == 'auto':
        numSamples = round(2*Lobs/λ)
    else:
        numSamples = numSamples

    Lap = Lobs
    k = 2. * np.pi / λ
    Δζ = Lap / numSamples
    Δη = Δζ

    # Simpson's rule with 3/8 tail correction
    BSimpson = simpson_weights_1D(numSamples)
    BSimpson = np.matmul(BSimpson.T, BSimpson)

    # ζ, η are coordinates in the source plane
    ζCoords = np.linspace(-Lap/2, Lap/2, numSamples)
    ηCoords = np.linspace(-Lap/2, Lap/2, numSamples)
    # x, y are coordinates in the observation plane
    xCoords = np.linspace(-Lobs/2, Lobs/2, numSamples)
    yCoords = np.linspace(-Lobs/2, Lobs/2, numSamples)

    # An override to help the cases in vector field propagation
    # where some field component is identically zero

    if hasattr(apertureFunction, 'null'):
        if apertureFunction.null:
            return (numSamples, xCoords, yCoords, np.zeros((numSamples, numSamples)))
    
    ζmesh, ηmesh = np.meshgrid(ζCoords, ηCoords)
    padextra = (2*numSamples - 1) - numSamples

    # Put together the U array
    if interpFun:
        U = apertureFunction((ζmesh, ηmesh))
    else:
        U = apertureFunction(ζmesh, ηmesh)

    # if z=0 there's nothing to do and the same input field should be returned
    if z == 0:
        return (numSamples, xCoords, yCoords, U)
    
    U = BSimpson*U
    U = np.pad(U, 
            pad_width=((0,padextra),(0,padextra)),
            mode='constant',
            constant_values=0.)

    x0 = xCoords[0]
    y0 = yCoords[0]
    η0 = ηCoords[0]
    ζ0 = ζCoords[0]
    # Put together the H array
    Hx1 = np.full((numSamples-1, 2*numSamples-1), x0)
    Hx2 = np.tile(xCoords, (2*numSamples-1,1)).T
    Hx  = np.concatenate((Hx1, Hx2))

    Hζ2 = np.full((numSamples-1, 2*numSamples-1), ζ0)
    Hζ1 = np.tile(ζCoords[::-1], (2*numSamples - 1,1)).T
    Hζ  = np.concatenate((Hζ1, Hζ2)).T

    Hxζ = Hx - Hζ.T

    Hy1 = np.full((numSamples-1, 2*numSamples-1), y0)
    Hy2 = np.tile(yCoords, (2*numSamples-1,1)).T
    Hy  = np.concatenate((Hy1, Hy2))

    Hη2 = np.full((numSamples-1, 2*numSamples-1), η0)
    Hη1 = np.tile(ηCoords[::-1], (2*numSamples - 1,1)).T
    Hη  = np.concatenate((Hη1, Hη2)).T

    Hyη = (Hy - Hη.T).T

    # calculate r
    rEva = np.sqrt(Hxζ**2 + Hyη**2 + z**2)
    # evaluate gr
    H = gr(rEva)

    # compute the Fourier transforms
    FFU = fft2(U)
    FFH = fft2(H)
    # perform the convolution
    FFUH = FFU * FFH
    # invert the result
    S = ifft2(FFUH)
    # get the good parts
    field = (Δη*Δζ) * S[-numSamples::,-numSamples::]

    return (numSamples, xCoords, yCoords, field)

def vector_field_FFT_RS_prop_func(Lobs, z, apertureFunctions, λfree, nref, numSamples='auto', interpFun = False):
    '''
    vector_field_func_prop takes a field with three cartesian components
    in  an aperture plane and propagates that to an observation plane by
    using  an  implementation of the direct integration of the Rayleigh-
    Sommerfeld diffraction integral. This implementation is based on the
    method  described  in  Shen and Wang (2006). The field is sampled in
    the  aperture  plane  and  in  the obserbation plane using a uniform
    grid. The field is assumed to be zero outside of the aperture plane.
    No  checks  are  made  that  the given field components constitute a
    valid electromagnetic field. It assumes that the refractive index is
    isotropic.

    Parameters
    ----------
    +  Lobs  (float): spatial width of the obsevation region, in μm. The
    observation  region  is  assumed to be a s quare centered on (x,y) =
    (0,0),  and  extending  from  -Lobs/2  to Lobs/2 in both the x and y
    directions.

    + z (float): distance between the aperture plane and the observation
    plane, given in μm. The aperture plane is assumed to be at z=0.

    + apertureFunctions (tuple): a tuple with three bi-variate functions
    which  return  the  complex  amplitude  of  the  corresponding field
    cartesian component in the aperture plane. Input to the functions is
    assumed  to  be in cartesian coordinates x,y. If the function has an
    attribute "null" set to True, then the function will simply return a
    matrix of zeros.

    + λfree (float): wavelength in vacuum of field, given in μm.

    + nref (float): refractive index of the propagating medium.

    Options
    -------
    +  "numSamples"  (int or Automatic): number of samples to use in the
    aperture  plane  and  the  observation  plane. The aperture plane is
    sampled  using  a  uniform  grid,  and the observation plane is also
    sampled using a uniform grid. The default is Automatic in which case
    numSamples  is  calculated  so that the sample size is equal to half
    the wavelength of the wave inside of the propagating medium.

    Returns
    -------
    (numSamples, xCoords, yCoords, field) (tuple)
    +  xCoords (np.array): x coordinates of the observation plane, given
    in μm.

    +  yCoords (np.array): y coordinates of the observation plane, given
    in μm.

    +  fields  (np.array):  with shape (3, numSamples, numSamples) where
    the  first  index takes values 0, 1, 2 for the x, y, and z cartesian
    components  and  the second two indices are anchored to positions in
    the  obervation plane according to xCoords and yCoords. The top left
    corner  of  the  array  corresponds  to the lower left corner of the
    observation plane.

    References
    ----------
    +   Shen,   Fabin,  and  Anbo  Wang.  "Fast-Fourier-transform  based
    numerical integration method for the Rayleigh-Sommerfeld diffraction
    formula." Applied optics 45, no. 6 (2006): 1102-1110.
    '''
    λ = λfree / nref
    if numSamples == 'auto':
        numSamples = round(2*Lobs/λ)
    else:
        numSamples = numSamples
    fields = np.zeros((3, numSamples, numSamples), dtype=complex)
    for field_idx, apertureFunction in enumerate(apertureFunctions):
        (numSamples, xCoords, yCoords, field) = scalar_field_FFT_RS_prop_func(Lobs, z, apertureFunction, λfree, nref, numSamples, interpFun=interpFun)
        fields[field_idx] = field
    return (numSamples, xCoords, yCoords, fields)

def simpson_weights_1D(numSamples):
    '''
    simpson_weights_1D  returns  the  weights  for Simpson's rule for 1D
    numerical integration. Ff there's an even number of intervals (which
    is  the  same  as  an odd number of samples) the 1/3 Simpson rule is
    used. If there's an odd number of intervals (which is the same as an
    even number of samples) then Simpson's 1/3 is used for the first n-3
    points and the 3/8 rule is used for the remaining tail.
    Parameters
    ----------
    numSamples (int): how many evaluation points are avaiable for integration
    Returns
    -------
    BSimpson (np.array): array of weights for mixed Simpson's rule.
    '''
    if numSamples % 2 == 1:
        BSimpson = np.zeros((1,numSamples))
        BSimpson[0,1::2] = 4
        BSimpson[0,2::2] = 2
        BSimpson[0,0] = 1
        BSimpson[0,-1] = 1
        BSimpson /= 3.
    else:
        BSimpson = np.zeros((1,numSamples))
        BSimpson[0,1::2] = 4
        BSimpson[0,2::2] = 2
        BSimpson[0,0] = 1
        BSimpson[0,-4] = 1
        BSimpson /= 3.
        BSimpson[0,-4] += 3/8.
        BSimpson[0,-3::] = 3/8*np.array([3,3,1])
    return BSimpson

def scalar_field_FFT_RS_prop_array(zProp, Ufield, ζCoords, ηCoords, λfree, nref):
    '''
    scalar_field_array_prop  takes  a scalar field in a region contained
    in  a  source  plane  and  propagates  the field to a an observation
    region  contained  in  a parallel plane at a distance zProp from the
    source plane. his implementation is based on the method described in
    Shen and Wang (2006).

    It  assumes  that  array  Ufield  provided  to the function has been
    adequately  sampled and it uses the same sampling for the propagated
    field.

    Parameters
    ----------

    +   zProp  (float):  distance  between  the  source  plane  and  the
    observation plane, given in μm.

    +  Ufield  (np.array):  complex amplitude of the field in the source
    plane,  sampled according to the coordinates provided by ζCoords and
    ηCoords. Must be a square array.

    +  ζCoords  (np.array): x coordinates on the source plane indexed to
    the given Ufield.

    +  ηCoords  (np.array): y coordinates on the source plane indexed to
    the given Ufield.

    + λfree (float): wavelength in vacuum of field, given in μm.

    + nref (float): refractive index of the propagating medium.

    Returns
    -------
    (xCoords, yCoords, field) (tuple)
    +  xCoords (np.array): x coordinates on the observation plane, given
    in μm.

    +  yCoords (np.array): y coordinates on the observation plane, given
    in μm.

    +   field   (np.array):  complex  amplitude  of  the  field  in  the
    observation  plane.  The top left corner of the array corresponds to
    the  lower  left  corner  of  the observation plane. The coordinates
    associated with each element in the given array should be taken from
    xCoords and yCoords.

    References
    ----------
    +   Shen,   Fabin,  and  Anbo  Wang.  "Fast-Fourier-transform  based
    numerical integration method for the Rayleigh-Sommerfeld diffraction
    formula." Applied optics 45, no. 6 (2006): 1102-1110.

    Example (double slit):
    ----------------------
    
    def doubleSlit(separation, width, height):
        def apertureFun(x, y):
            return np.where((
                            ((np.abs(x - separation/2) <= width/2) | (np.abs(x + separation/2) <= width/2))
                            & (np.abs(y) <= height/2)
            ), 1, 0)
        return apertureFun

    slitSep = 5.
    slitWidth = 1.
    slitHeight = 10.
    zProp = 100.
    L_aperture = 100.
    z = 100
    nref = 1
    λfree = 0.532
    numSamples = 2*int(L_aperture/λfree)

    apFun = doubleSlit(slitSep, slitWidth, slitHeight)
    ζCoords = np.linspace(-L_aperture/2, L_aperture/2, numSamples)
    ηCoords = np.linspace(-L_aperture/2, L_aperture/2, numSamples)
    ζGrid, ηGrid = np.meshgrid(ζCoords, ηCoords)
    apertureField = apFun(ζGrid, ηGrid)

    # Estimate the diffraction pattern from the simplified formula
    diforders = range(10)
    difordersV = range(1,10)
    xmaxi = []
    ymaxi = []
    for diforderH in diforders:
        for diforderV in difordersV:
            stheta = diforderH * λfree / slitSep
            sthetaV = (2*diforderV+1) * λfree / slitHeight / 2
            if stheta > 1:
                continue
            if sthetaV > 1:
                continue
            else:
                theta = np.arcsin(stheta)
                xdif = z * np.tan(theta)
                thetaV = np.arcsin(sthetaV)
                ydif = z * np.tan(thetaV)
                if np.abs(xdif) <= L_aperture/2:
                    if np.abs(ydif) <= L_aperture/2:
                        xmaxi.append(xdif)
                        ymaxi.append(ydif)
                        xmaxi.append(xdif)
                        ymaxi.append(0)
                        xmaxi.append(xdif)
                        ymaxi.append(-ydif)
                        xmaxi.append(-xdif)
                        ymaxi.append(ydif)
                        xmaxi.append(-xdif)
                        ymaxi.append(0)
                        xmaxi.append(-xdif)
                        ymaxi.append(-ydif)

    difMaxima = list(zip(xmaxi, ymaxi))
    difMaxima = list(set(difMaxima))
    difMaxima = np.array(difMaxima)

    (xCoords, yCoords, numfield) = ws.scalar_field_FFT_RS_prop_array(zProp, apertureField, ζCoords, ηCoords, λfree, nref)

    extent = (xCoords[0], xCoords[-1], yCoords[0], yCoords[-1])
    pField = np.abs(numfield)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(pField,
            extent=extent,
            cmap=cmr.ember,
            interpolation='None')
    ax.scatter(difMaxima[:,0], difMaxima[:,1], s=40, marker='o',facecolors='none', edgecolors='r', alpha=0.5)
    ax.set_xlabel('x/μm')
    ax.set_ylabel('y/μm')
    ax.set_title('Diffraction pattern of a double slit\ns=%.2fμm | w=%.2fμm | L=%.2fμm | Δz=%.2fμm' % (slitWidth, slitWidth, slitHeight, z))
    plt.show()
    '''
    assert zProp>=0, 'z must be non-negative'

    λ = λfree / nref
    k = 2*np.pi / λ

    def gr(r):
        return (np.exp(1j * k * r) * zProp) * (1./r - 1j*k) / (2*np.pi * r**2)

    numSamples = len(ζCoords)
    assert len(ζCoords) == len(ηCoords), "Input must be square."
    Lap = (ζCoords[-1] - ζCoords[0])
    # The implementation requires that the size of the source
    # and observation regions be the same.
    Lobs = Lap
    k = 2. * np.pi / λ
    Δζ = Lap / numSamples
    Δη = Δζ

    # Simpson's rule with 3/8 tail correction
    BSimpson = simpson_weights_1D(numSamples)
    BSimpson = np.matmul(BSimpson.T, BSimpson)

    # ζ, η are coordinates in the source plane
    # x, y are coordinates in the observation plane
    xCoords = np.linspace(-Lobs/2, Lobs/2, numSamples)
    yCoords = np.linspace(-Lobs/2, Lobs/2, numSamples)
    
    # if zProp=0 there's nothing to do and the same input field should be returned
    if zProp == 0:
        return (numSamples, xCoords, yCoords, Ufield)
    
    padextra = (numSamples - 1)
    Ufield = BSimpson * Ufield
    Ufield = np.pad(BSimpson * Ufield, 
            pad_width=((0,padextra), (0,padextra)),
            mode='constant',
            constant_values=0.)

    x0, y0 = xCoords[0], yCoords[0] 
    ζ0, η0 = ζCoords[0], ηCoords[0]

    # Put together the H array
    Hx1 = np.full((numSamples-1, 2*numSamples-1), x0)
    Hx2 = np.tile(xCoords, (2*numSamples-1,1)).T
    Hx  = np.concatenate((Hx1, Hx2))

    Hζ2 = np.full((numSamples-1, 2*numSamples-1), ζ0)
    Hζ1 = np.tile(ζCoords[::-1], (2*numSamples - 1,1)).T
    Hζ  = np.concatenate((Hζ1, Hζ2)).T

    Hxζ = Hx - Hζ.T

    Hy1 = np.full((numSamples-1, 2*numSamples-1), y0)
    Hy2 = np.tile(yCoords, (2*numSamples-1,1)).T
    Hy  = np.concatenate((Hy1, Hy2))

    Hη2 = np.full((numSamples-1, 2*numSamples-1), η0)
    Hη1 = np.tile(ηCoords[::-1], (2*numSamples - 1,1)).T
    Hη  = np.concatenate((Hη1, Hη2)).T

    Hyη = (Hy - Hη.T).T

    # calculate r
    rEva = np.sqrt(Hxζ**2 + Hyη**2 + zProp**2)
    # evaluate gr
    Hfield = gr(rEva)

    # compute the Fourier transforms
    FFU = fft2(Ufield)
    FFH = fft2(Hfield)
    # perform the convolution
    FFUH = FFU * FFH
    # invert the result
    Sfield = ifft2(FFUH)
    # get the good parts
    field = (Δη*Δζ) * Sfield[-numSamples::,-numSamples::]

    return (xCoords, yCoords, field)

def vector_field_FFT_RS_prop_array(zProp, Ufields, ζCoords, ηCoords, λfree, nref):
    '''
    vector_field_func_prop takes a field with three cartesian components
    in  an aperture plane and propagates that to an observation plane by
    using  an  implementation of the direct integration of the Rayleigh-
    Sommerfeld diffraction integral. This implementation is based on the
    method  described  in  Shen and Wang (2006). The field is sampled in
    the  aperture  plane  and  in  the obserbation plane using a uniform
    grid. The field is assumed to be zero outside of the aperture plane.
    No  checks  are  made  that  the given field components constitute a
    valid electromagnetic field. It assumes that the refractive index is
    isotropic.

    Parameters
    ----------
    +  Lobs  (float): spatial width of the obsevation region, in μm. The
    observation  region  is  assumed to be a s quare centered on (x,y) =
    (0,0),  and  extending  from  -Lobs/2  to Lobs/2 in both the x and y
    directions.

    + z (float): distance between the aperture plane and the observation
    plane, given in μm. The aperture plane is assumed to be at z=0.

    + apertureFunctions (tuple): a tuple with three bi-variate functions
    which  return  the  complex  amplitude  of  the  corresponding field
    cartesian component in the aperture plane. Input to the functions is
    assumed  to  be in cartesian coordinates x,y. If the function has an
    attribute "null" set to True, then the function will simply return a
    matrix of zeros.

    + λfree (float): wavelength in vacuum of field, given in μm.

    + nref (float): refractive index of the propagating medium.

    Returns
    -------
    (numSamples, xCoords, yCoords, field) (tuple)
    +  xCoords (np.array): x coordinates of the observation plane, given
    in μm.

    +  yCoords (np.array): y coordinates of the observation plane, given
    in μm.

    +  fields    (np.array):   with  the  same  shape  as  Ufields where
    the  first  index takes values 0, 1, 2 for the x, y, and z cartesian
    components  and  the second two indices are anchored to positions in
    the  obervation plane according to xCoords and yCoords. The top left
    corner  of  the  array  corresponds  to the lower left corner of the
    observation plane.

    References
    ----------
    +   Shen,   Fabin,  and  Anbo  Wang.  "Fast-Fourier-transform  based
    numerical integration method for the Rayleigh-Sommerfeld diffraction
    formula." Applied optics 45, no. 6 (2006): 1102-1110.
    '''
    numSamples = len(ζCoords)
    
    fields = np.zeros(Ufields.shape, dtype=complex)
    numComponents = Ufield.shape[0]
    for field_idx in range(numComponents):
        Ufield = Ufields[field_idx]
        (xCoords, yCoords, field) = scalar_field_FFT_RS_prop_array(zProp, Ufield, ζCoords, ηCoords, λfree, nref)
        fields[field_idx] = field
    return (xCoords, yCoords, fields)
