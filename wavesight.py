#!/usr/bin/env python3

import numpy as np
from scipy import special
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt
from fieldgen import * 
from convstore import *

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
    drawPlots : bool, optional
        Whether to draw plots of the mode profiles. The default is False.
    verbose : bool, optional
        Whether to print out extra information. The default is False.
    
    Returns
    -------
    sol : dict with the following keys:
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

    sol['tmfun'] = tmfun

    tefun = tefungen(λfree=wavelength, 
                    n2=nCladding, 
                    n1=nCore, 
                    a=coreRadius)
    
    sol['tefun'] = tefun

    print("Calculting TE(0,n) propagation constants ...")
    dkzprime = dkz/numModesTE
    temodes = findallroots(tefun, kzmin, kzmax, dkzprime, method='brentq', num_sigfigs=6)

    print("Calculting TM(0,n) propagation constants ...")
    tmmodes = findallroots(tmfun, kzmin, kzmax, dkzprime, method='brentq', num_sigfigs=6)
    kzrange = np.linspace(kzmin, kzmax, 1000)
    
    if drawPlots:
        tmvals = tmfun(kzrange)
        tevals = tefun(kzrange)
        tmmodes = findallroots(tmfun, kzmin, kzmax, dkz, method='bisect', num_sigfigs=6, verbose=False)
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
    sol['hefuns'] = {}
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
        sol['hefuns'][m] = hefun
        hevals = hefun(kzrange)
        hezeros = findallroots(hefun, kzmin, kzmax, dkzprime, method='secant', num_sigfigs=10, verbose=False)
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
        # if n == 1:
        #     break
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

