#!/usr/bin/env python3

import numpy as np
from scipy import special 
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt
from fieldgen import * 
from convstore import * 
from tqdm.notebook import tqdm

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
        Free space wavelength in microns.
    n1 : float
        Core refractive index.
    n2 : float
        Cladding refractive index.
    a : float
        Core radius in microns.

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
        Free space wavelength in microns.
    n1 : float
        Core refractive index.
    n2 : float
        Cladding refractive index.
    a : float
        Core radius in microns.

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
        Free space wavelength in microns.
    n1 : float
        Core refractive index.
    n2 : float
        Cladding refractive index.
    a : float
        Core radius in microns.
    m : int
        Order of the HE mode.

    Returns
    -------
    he : function
    '''
    def he(kz):
        return  -((m**2*(1/(-kz**2 + (4*n1**2*np.pi**2)/λfree**2) + 1/(kz**2 - (4*n2**2*np.pi**2)/λfree**2))*(n1**2/(-kz**2 + (4*n1**2*np.pi**2)/λfree**2) + n2**2/(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))/a**2) + ((special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)))/(2.*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) + (-special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) - special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))/(2.*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))*((n1**2*(special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))))/(2.*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) + (n2**2*(-special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) - special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))/(2.*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))
    return he

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
        grid_divider: int, not necessary here but when later
            used in the layout generator, this is used to determine
            the fineness of the grid by making it equal to
            λfree / max(nCore, nCladding, nFree) / grid_divider
    drawPlots : bool, optional
        Whether to draw plots of the mode profiles. The default is False.
    verbose : bool, optional
        Whether to print out extra information. The default is False.
    
    Returns
    -------
    sol : dict with all the keys included in fiber_spec plus these following others:
        kzmax : float
            2π/λfree * nCladding
        kzmin : float
            2π/λfree * nCore
        Vnum : float
            The V number of the fiber.
        numModesFromVnum: float
            The number of modes according to the V number.
        totalNumModes : int
            The total number of found modes.
        tmfun : function
            The eigenvalue function for the TM modes.
        tefun : function
            The eigenvalue function for the TE modes.
        hefuns : dict
            The eigenvalue functions for the HE modes. The keys are the values of n (n>=1).
        TMkz : array
            The propagation constants of the TM(0,n) modes.
        TEkz : array
            The propagation constants of the TE(0,n) modes.
        HEkz : dict
            The propagation constants of the HE(n,m) modes. The keys are the values of n (n>=1).
    '''
    nCore = fiber_spec['nCore']
    if 'NA' in fiber_spec:
        NA = fiber_spec['NA']
        nCladding = np.sqrt(nCore**2 - NA**2)
    else:
        nCladding = fiber_spec['nCladding']
        NA =  np.sqrt(nCore**2 - nCladding**2)
    separator = "="*40
    coreRadius = fiber_spec['coreRadius']
    wavelength = fiber_spec['free_space_wavelength']
    kzmax = nCore*2*np.pi/wavelength
    kzmin = nCladding*2*np.pi/wavelength
    kzspan = kzmax - kzmin
    kzmax = kzmax - 0.001*kzspan
    kzmin = kzmin + 0.001*kzspan
    # split the solution domain in at least 100 parts
    dkz = (kzmax-kzmin)/300
    sol = fiber_spec
    Vnum = 2*np.pi * coreRadius * NA / wavelength
    sol['Vnum'] = Vnum
    numModes = int(Vnum**2/2)
    numModesTE = int(Vnum/2)
    numModesTM = int(Vnum/2)
    numModesHE = int(Vnum*(Vnum-2)/4)
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

    tmfun = tmfungen(λfree=wavelength, 
                    n2=nCladding, 
                    n1=nCore, 
                    a=coreRadius)

    # sol['tmfun'] = tmfun

    tefun = tefungen(λfree=wavelength,  
                    n2=nCladding, 
                    n1=nCore, 
                    a=coreRadius)
    
    # sol['tefun'] = tefun

    print("Calculting TE(0,n) propagation constants ...")
    dkzprime = dkz/numModesTE
    temodes = findallroots(tefun, kzmin, kzmax, dkzprime, dtype=real_dtype, method='brentq', num_sigfigs=6)

    print("Calculting TM(0,n) propagation constants ...")
    tmmodes = findallroots(tmfun, kzmin, kzmax, dkzprime, dtype=real_dtype, method='brentq', num_sigfigs=6)
    kzrange = np.linspace(kzmin, kzmax, 1000, dtype=real_dtype)
    
    if drawPlots:
        tmvals = tmfun(kzrange)
        tevals = tefun(kzrange)
        tmmodes = findallroots(tmfun, kzmin, kzmax, dkz, dtype=real_dtype, method='bisect', num_sigfigs=6, verbose=False)
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

    m = 1
    hemodes = {}
    print("Calculting HE(m,n) propagation constants ...")
    # sol['hefuns'] = {}
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
        hezeros = findallroots(hefun, kzmin, kzmax, dkzprime, dtype=real_dtype, method='secant', num_sigfigs=10, verbose=False)
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
        - 'free_space_wavelenegth' : float
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
    λfree = fiber_sol['free_space_wavelength']
    grid_divider = fiber_sol['grid_divider']
    maxIndex = max(nCore, nCladding, nFree)
    Δs = λfree / maxIndex / grid_divider
    # calculate the coordinates arrays
    numSamples = int(2 * b / Δs)
    xrange = np.linspace(-b, b, numSamples, dtype=real_dtype)
    yrange = np.linspace(-b, b, numSamples, dtype=real_dtype)
    ρrange = np.linspace(0 , np.sqrt(2)*b, numSamples, dtype=real_dtype)
    φrange = np.linspace(-np.pi, np.pi, numSamples, dtype=real_dtype)
    Xg, Yg = np.meshgrid(xrange, yrange)
    ρg     = np.sqrt(Xg**2 + Yg**2)
    φg     = np.arctan2(Yg, Xg)

    crossMask = np.zeros((numSamples, numSamples)).astype(np.bool8)
    crossMask[ρg <= a] = True
    crossMask[ρg > a]  = False

    # #Coords-Calc
    nxy   = np.zeros((numSamples, numSamples))
    nxy[crossMask]  =  nCore
    nxy[~crossMask] =  nCladding

    return a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask, numSamples

def calculate_numerical_basis(fiber_sol):
    '''
    Given a solution for the propagation modes of the fiber, calculate the numerical basis.
    
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
        - 'free_space_wavelenegth' : float
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
        - 'eigenbasis' : 5D array
            The numerical basis. The first dimension is the mode number,
            the second dimension only has two values, 0 and 1, 0 for the
            E  field, and 1 for the H field. The third dimension is used
            for  the  different components of the corresponding field in
            cylindrical  coordinates.  The  first index being the radial
            component,  the  second index being the azimuthal component,
            and  the  third  index  being  the  z component. Finally the
            fourth  and fifth dimensions are arrays that hold the values
            of the corresponding field components.
        - 'eigenbasis_nums': list of tuples
            A  list  of  tuples,  each  tuple has nine values, the first
            value  is  a  string, either 'TE', 'TM', or 'HE', the second
            value is value of m, and the third value is the value of n.m
            The  order  in  this list is so that the n-th element of the
            list  corresponds  to  the  n-th mode in the eigenbasis. The
            fourth  and  fifth values are for the transverse propagation
            constants  the fourth one being the one inside the core, and
            the fifth one being the one outside the core. From the sixth
            to the ninth values are the values for Ae, Ah, Be, and Bh.
    '''
    coord_layout = coordinate_layout(fiber_sol)
    nCore = fiber_sol['nCore']
    nCladding = fiber_sol['nCladding']
    λfree = fiber_sol['free_space_wavelength']
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