#!/usr/bin/env python3

import warnings
import numpy as np
from misc import *
import cmasher as cm
from fields import *
from convstore import * 
import diffkernels as dk
from fieldgenesis import *
from wavegraphics import *
from fungenerators import *
from tqdm.notebook import tqdm
from collections import OrderedDict
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator

real_dtype = np.float64
complex_dtype = np.complex128  

def multisolver(fiber_spec, solve_modes = 'all', drawPlots=False, verbose=False, tm_te_funcs=False):
    '''
    This function solves the propagation constants of a step-index fiber with the given specifications. This assuming that the cladding is so big that it is effectively infinite.

    The propagation constants determine the z-dependence of the fields along the z-direction. The propagation constants are always bounded by what would be the plane wave wavenumbers in the cladding or in the core.

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
            λfree / max(nCore, nCladding, nUpper) / grid_divider
    solve_modes: str
        either 'all' or 'transvserse'
    drawPlots : bool, optional
        Whether to draw plots of the mode profiles. The default is False.
    verbose : bool, optional
        Whether to print out extra information. The default is False.
    tm_te_funcs : bool, optional
        If True then the return dictionary includes the keys tmfun and tefun. The default is False.
    
    Returns
    -------
    sol : dict with all the keys included in fiber_spec plus these others:
        kzmax : float
            2π/λfree * nCladding (no kz larger than this)
        kzmin : float
            2π/λfree * nCore (no kz smaller than this)
        Vnum : float
            The V number of the fiber.
        numModesFromVnum: float
            The number of modes according to the V number.
        totalNumModesTE : int
            The total number of TE modes that were found.
        totalNumModesTM : int
            The total number of TM modes that were found.
        totalNumModesHE : int
            The total number of HE modes that were found, excluding parity.
        totalNumModes : int
            The total number of modes that were found, including parity.
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
        if verbose:
            print("Estimating nCladding from nCore and NA ...")
        NA        = fiber_spec['NA']
        nCladding = np.sqrt(nCore**2 - NA**2)
    else:
        nCladding = fiber_spec['nCladding']
        NA        =  np.sqrt(nCore**2 - nCladding**2)
    separator     = "="*40
    coreRadius    = fiber_spec['coreRadius']
    wavelength    = fiber_spec['λFree']
    kFree         = 2 * np.pi / wavelength
    kzmax         = nCore * 2 * np.pi / wavelength
    kzmin         = nCladding * 2 *np.pi / wavelength
    kzspan        = kzmax - kzmin
    kzmax         = kzmax - 1e-7 * kzspan
    kzmin         = kzmin + 1e-7 * kzspan
    # split the solution domain in at least 300 parts
    dkz           = (kzmax - kzmin) / 1000
    sol           = fiber_spec
    Vnum          = 2 * np.pi * coreRadius * NA / wavelength
    sol['Vnum']   = Vnum
    sol['kFree']  = kFree
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
        dkzprime = dkz/numModesTE/2.
        if verbose:
            print("Calculating TM(0,n) propagation constants ...")
        tmmodes = findallroots(tmfun, kzmin, kzmax, dkzprime, 
                            dtype=real_dtype, method='brentq', num_sigfigs=10, reverse=True)
        if verbose:
            print("Calculating TE(0,n) propagation constants ...")
        temodes = findallroots(tefun, kzmin, kzmax, dkzprime, 
                            dtype=real_dtype, method='brentq', num_sigfigs=10, reverse=True)
        kzrange = np.linspace(kzmin, kzmax, 1000, dtype=real_dtype)
    
    if drawPlots:
        tmvals = tmfun(kzrange)
        tevals = tefun(kzrange)
        tmzerocheck = tmfun(tmmodes)
        tezerocheck = tefun(temodes)

        plt.figure(figsize=(10,3))
        plt.plot(kzrange, tmvals, 'r')
        plt.scatter(tmmodes,tmzerocheck, c='y')
        plt.plot([kzmin, kzmax], [0,0], "w:")
        plt.ylim(-1,1)
        plt.title('TM modes (%d found)' % len(tmmodes))
        plt.show()

        plt.figure(figsize=(10,3))
        plt.plot(kzrange, tevals, 'r')
        plt.scatter(temodes,tezerocheck, c='y')
        plt.plot([kzmin, kzmax], [0,0], "w:")
        plt.ylim(-1,1)
        plt.title('TE modes (%d found)' % len(temodes))
        plt.show()

    hemodes = {}
    if solve_modes == 'all':
        m = 1
        if verbose:
            print("Calculating HE(m,n) propagation constants ...")
        while True:
            approxModes = maxHEmodes - m
            approxModes = max(2, approxModes)
            dkzprime = dkz / approxModes
            if verbose:
                print(f'm={m}',end='\r')
            hefun = hefungen(λfree=wavelength, 
                        m=m, 
                        n2=nCladding, 
                        n1=nCore, 
                        a=coreRadius)
            # sol['hefuns'][m] = hefun
            hevals = hefun(kzrange)
            hezeros = findallroots(hefun, kzmin, kzmax, dkzprime, 
                                dtype=real_dtype, method='brentq', num_sigfigs=10, verbose=False, reverse=True)
            if len(hezeros) == 0:
                break
            hemodes[m] = hezeros
            hezerocheck = hefun(hezeros)
            if drawPlots:
                plt.figure(figsize=(15,3))
                plt.plot(kzrange, hevals, 'r')
                plt.plot([kzmin, kzmax], [0,0], "w:")
                plt.scatter(hezeros, hezerocheck, c='y')
                plt.ylim(-0.01,0.04)
                plt.title('HE(%d, n) roots (%d found)' % (m, len(hezeros)))
                plt.show()
            m = m + 1  

    numCalcModes = (2 * sum(list(map(len,hemodes.values()))), len(temodes), len(tmmodes))
    if verbose:
        print("")
        print(separator)
        print("HE modes = %s\nTE modes = %d\nTM modes = %d\nTOTAL modes = %d\nFROM_Vnum = %d" % (*numCalcModes, sum(numCalcModes), numModes))
        print(separator)
    # put the modes in the solution dictionary
    sol['TEkz'] = {0: temodes}
    sol['TMkz'] = {0: tmmodes}
    sol['HEkz'] = hemodes
    totalModesTE = len(sol['TEkz'][0])
    totalModesTM = len(sol['TMkz'][0])
    totalModesHE = sum(list(map(len, sol['HEkz'].values())))
    sol['totalModesTE'] = totalModesTE
    sol['totalModesTM'] = totalModesTM
    sol['totalModesHE'] = totalModesHE
    sol['totalModes'] = totalModesTE + totalModesTM + 2*totalModesHE
    return sol

def field_dot(E_field, H_field, Δs, mask=None):
    '''
    Parameters
    ----------
    E_field : np.array
        an electric field sampled on a cartesian grid of size Δs

    H_field : np.array
        a magnetic field sampled on a cartesian grid of size Δs

    Δs : float
        the grid spacing

    mask : np.array
        a mask to apply to the fields (same size as E_field[0,:,:] and H_field[0,:,:])
    
    Returns
    -------
    dotp : float
        the dot product of the two given fields
    '''
    if mask is not None:
        E_field[0][~mask] = 0
        E_field[1][~mask] = 0
        H_field[0][~mask] = 0
        H_field[1][~mask] = 0
    sumField = E_field[0] * np.conjugate(H_field[1]) - E_field[1] * np.conjugate(H_field[0])
    dotp = np.sum(sumField)
    dotp = 0.5 * dotp * Δs**2
    return dotp

def boundary_test(Efuncs, Hfuncs, fiber_spec, modeType, tolerance=1e-5):
    '''
    This  function  checks  that a given solution for the fields
    satisfies  the  boundary  conditions  at  the  core/cladding
    interface within tolerance. If the mode is HE the test is on
    the relative difference of all field components being within
    tolerance. If the mode is TM or TE then the test is that all
    the  components  that  should  be zero are exactly zero, and
    that  the relative difference for the non-zero components is
    within tolerance.

    Parameters
    ----------
    Efuncs : tuple
        a  tuple  of  six  functions  which  describe  the three
        components  of  the  H  field  in  the core and cladding
        regions.

    Hfuncs : tuple 
        a  tuple  of  six  functions  which  describe  the three
        components  of  the  H  field  in  the core and cladding
        regions.

    fiber_spec : dict
        a dictionary containing the fiber specifications.

    modeType : str
        must be one of 'HE', 'TE', or 'TM'.

    tolerance :float
        the tolerance for the boundary condition test.

    Returns
    -------
    boundaryTest : bool
        True if test OK, False otherwise.
    '''
    (ECoreρ, ECoreϕ, ECorez, ECladdingρ, ECladdingϕ, ECladdingz) = Efuncs
    (HCoreρ, HCoreϕ, HCorez, HCladdingρ, HCladdingϕ, HCladdingz) = Hfuncs
    nCore = fiber_spec['nCore']
    nCladding = fiber_spec['nCladding']
    a = fiber_spec['coreRadius']
    coreBoundary = OrderedDict()
    claddingBoundary = OrderedDict()

    # the radial component of D is continuous,
    coreBoundary['DCoreρ'] = ECoreρ(a) * nCore**2
    claddingBoundary['DCladdingρ'] = ECladdingρ(a) * nCladding**2

    # the radial component of B is continuous,
    coreBoundary['BCoreρ'] = HCoreρ(a)
    claddingBoundary['BCladdingρ'] = HCladdingρ(a)

    # the azimuthal component of E is continuous,
    coreBoundary['ECoreϕ'] = ECoreϕ(a)
    claddingBoundary['ECladdingϕ'] = ECladdingϕ(a)

    # the longitudinal component of E is continuous,
    coreBoundary['ECorez'] = ECorez(a)
    claddingBoundary['ECladdingz'] = ECladdingz(a)

    # the azimuthal component of H is continuous,
    coreBoundary['HCoreϕ'] = HCoreϕ(a)
    claddingBoundary['HCladdingϕ'] = HCladdingϕ(a)

    # the longitudinal component of H is continuous.
    coreBoundary['HCorez'] = HCorez(a)
    claddingBoundary['HCladdingz'] = HCladdingz(a)
    coreArray = np.array(list(coreBoundary.values()))
    claddingArray = np.array(list(claddingBoundary.values()))
    if modeType == 'HE':
        nonZeroDiff = np.max(np.abs((coreArray / claddingArray)) - 1)
        nonZeroTest = nonZeroDiff < tolerance
        zeroCheck = True
    elif modeType == 'TE':
        coreZeros     = np.zeros(3,dtype=np.complex128)
        claddingZeros = np.zeros(3,dtype=np.complex128)
        # the z-component of E should be zero
        coreZeros[0] = coreBoundary['ECorez']
        claddingZeros[0] = claddingBoundary['ECladdingz']
        # the ρ-component of E should be zero
        coreZeros[1] = coreBoundary['DCoreρ']
        claddingZeros[1] = claddingBoundary['DCladdingρ']
        # the ϕ-component of H should be zero
        coreZeros[2] = coreBoundary['HCoreϕ']
        claddingZeros[2] = claddingBoundary['HCladdingϕ']
        zeroCheck = np.all(coreZeros == np.zeros(3)) and np.all(claddingZeros == np.zeros(3))
        # Now for all the ones that should be non-zero
        coreNonZeros = np.zeros(3,dtype=np.complex128)
        claddingNonZeros = np.zeros(3,dtype=np.complex128)
        coreNonZeros[0] = coreBoundary['ECoreϕ']
        claddingNonZeros[0] = claddingBoundary['ECladdingϕ']
        coreNonZeros[1] = coreBoundary['BCoreρ']
        claddingNonZeros[1] = claddingBoundary['BCladdingρ']
        coreNonZeros[2] = coreBoundary['HCorez']
        claddingNonZeros[2] = claddingBoundary['HCladdingz']
        nonZeroDiff = np.max(np.abs((coreNonZeros)/claddingNonZeros) - 1)
        nonZeroTest = nonZeroDiff < tolerance
    elif modeType == 'TM':
        coreZeros     = np.zeros(3, dtype=np.complex128)
        claddingZeros = np.zeros(3, dtype=np.complex128)
        # the ϕ-component of E should be zero
        coreZeros[0]     = coreBoundary['ECoreϕ']
        claddingZeros[0] = claddingBoundary['ECladdingϕ']
        # the z-component of H should be zero
        coreZeros[1] = coreBoundary['HCorez']
        claddingZeros[1] = claddingBoundary['HCladdingz']
        # the ρ-component of B should be zero
        coreZeros[2] = coreBoundary['BCoreρ']
        claddingZeros[2] = claddingBoundary['BCladdingρ']
        zeroCheck = np.all(coreZeros == np.zeros(3)) and np.all(claddingZeros == np.zeros(3))
        # Now for all the ones that should be non-zero
        coreNonZeros = np.zeros(3,dtype=np.complex128)
        claddingNonZeros = np.zeros(3,dtype=np.complex128)
        coreNonZeros[0] = coreBoundary['DCoreρ']
        claddingNonZeros[0] = claddingBoundary['DCladdingρ']
        coreNonZeros[1] = coreBoundary['ECorez']
        claddingNonZeros[1] = claddingBoundary['ECladdingz']
        coreNonZeros[2] = coreBoundary['HCoreϕ']
        claddingNonZeros[2] = claddingBoundary['HCladdingϕ']
        nonZeroDiff = np.max(np.abs((coreNonZeros)/claddingNonZeros) - 1)
        nonZeroTest = nonZeroDiff < tolerance
    boundaryTest = (nonZeroTest and zeroCheck)
    if not boundaryTest:
        print(claddingBoundary, coreBoundary)
    return (boundaryTest, nonZeroTest, zeroCheck, nonZeroDiff)

def calculate_size_of_grid(fiber_sol):
    '''
    Given a solution for the modes of a multimode fiber, determine
    the half side of the computational domain that would capture
    most of the energy contained in the solved modes.

    Parameters
    ----------
    fiber_sol : dict
        a dictionary containing the solution for the fiber.
    
    Returns
    -------
    b : float
        the width of the computational domain.
    '''
    goal_fraction = 0.99 
    a = fiber_sol['coreRadius']
    nCore = fiber_sol['nCore']
    nCladding = fiber_sol['nCladding']
    nUpper = fiber_sol['nUpper']
    λfree = fiber_sol['λFree']
    kFree = 2*np.pi/λfree
    grid_divider = fiber_sol['grid_divider']
    maxIndex = max(nCore, nCladding, nUpper)
    Δs = λfree / maxIndex / grid_divider
    # calculate the side of the computational domain: START
    allnums = []
    for modetype in ['TE', 'TM', 'HE']:
        allkzs = fiber_sol[modetype + 'kz']
        for m, kzs in allkzs.items():
            for kz in kzs:
                wnums = (modetype, m, kz)
                allnums.append(wnums)
    (modetype, m, kz) = sorted(allnums, key=lambda x: x[-1])[0]
    (Efuncs, Hfuncs) = fieldGenerator(a, kFree, kz, m, nCladding, nCore, modetype)
    ρrange = np.linspace(0, a + 20 * λfree / nCladding, 1000)
    fluxCore = Efuncs[0](ρrange) * np.conjugate(Hfuncs[1](ρrange)) - Efuncs[1](ρrange) * np.conjugate(Hfuncs[0](ρrange))
    fluxCladding = Efuncs[3](ρrange) * np.conjugate(Hfuncs[4](ρrange)) - Efuncs[4](ρrange) * np.conjugate(Hfuncs[3](ρrange))
    flux = fluxCore
    flux[ρrange>a] = fluxCladding[ρrange>a]
    integrand = flux * ρrange
    total = np.sum(integrand)
    insideEnergy =  np.cumsum(integrand)/total
    b = 2*np.interp(goal_fraction, insideEnergy, ρrange) + λfree
    numSigFigsina = sig_figs_in(a)
    b = rounder(b, numSigFigsina)
    return b

def coordinate_layout(fiber_sol):
    '''
    Given a fiber solution, return the coordinate arrays for plotting
    and coordinating the numerical analysis.

    Parameters
    ----------
    fiber_sol : dict  with at least the following keys:
        'coreRadius' : float
            The radius of the core.
        'claddingIndex' : float  
            The refractive index of the cladding.
        'coreIndex' : float
            The refractive index of the core.
        'free_space_wavelength' : float
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
            The coordinate arrays for the ρ-φ directions in the cylindrical system.
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
    nCore = fiber_sol['nCore']
    nCladding = fiber_sol['nCladding']
    nUpper = fiber_sol['nUpper']
    λfree = fiber_sol['λFree']
    kFree = 2*np.pi/λfree
    grid_divider = fiber_sol['grid_divider']
    maxIndex = max(nCore, nCladding, nUpper)
    Δs = λfree / maxIndex / grid_divider
    b = calculate_size_of_grid(fiber_sol)
    # calculate the side of the computational domain: END
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

def calculate_numerical_basis(fiber_sol, verbose=True):
    '''
    Given a solution for the propagation modes of an optical waveguide, calculate a numerical basis.
    
    Parameters
    ----------
    fiber_sol : dict with at least the following keys:
        'coreRadius' : float
            The radius of the core.
        'nCladding' : float
            The refractive index of the cladding.
        'nCore' : float
            The refractive index of the core.
        'free_space_wavelength' : float
            The wavelength of the light in free space.
        'totalModes' : int
            The total number of calculated modes.
        'TEkz' : 1D dict
            A single key equal to m=0, the values are array with
            the TE modes propagation constants.
        'TMkz' : 1D dict
            A single key equal to m=0, the values are array with
            the TM modes propagation constants.
        'HEkz' : 1D dict
            Keys are m values, values are 1D arrays of kz values.
    
    Returns
    -------
    fiber_sol : dict the same dictionary as the input, but with two new keys:
        'coord_layout' : tuple
            The tuple returned by the coordinate_layout function.
        'eigenbasis' : 5D np.array
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
        'eigenbasis_nums' : list of tuples
            A  list  of tuples, each tuple has 7 values, the first value
            is  a  string indicating the type of mode, and can be either
            'TE', 'TM', or 'HE'. The second value is a string indicating
            the  parity  of the mode. The third value is the value of m.
            The  fourth  value  corresponds  to  the  value  of  kz, the
            propagation  constant along the z-direction. The fifth value
            is   the   index  that  the  value  of  kz  has  within  the
            corresponding  array  listing the propagation constants. The
            sixth  value  is  the transverse propagation constant inside
            the  core.  The seventh value is the propagation constant in
            the cladding. (modType, parity, m, kzidx, kz, γ, β)
    '''
    warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
    warnings.filterwarnings('ignore', 'invalid value encountered in multiply')
    warnings.filterwarnings('ignore', 'invalid value encountered in divide')
    coord_layout = coordinate_layout(fiber_sol)
    nCore = fiber_sol['nCore']
    nCladding = fiber_sol['nCladding']
    λfree = fiber_sol['λFree']
    kFree = 2*np.pi/λfree
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
    if verbose:
        iter_fun = tqdm
    else:
        iter_fun = lambda x: x
    for modType in ['TE','TM','HE']:
        if modType == 'HE':
            parities = ['EVEN', 'ODD']
        else:
            parities = ['TETM']
        solkey = modType + 'kz'
        for m, kzs in fiber_sol[solkey].items():
            cosMesh = np.cos(m*φg)
            sinMesh = np.sin(m*φg)
            for kzidx, kz in enumerate(iter_fun(kzs)):
                γ = np.sqrt(nCore**2*4*np.pi**2/λfree**2 - kz**2)
                β = np.sqrt(kz**2 - nCladding**2*4*np.pi**2/λfree**2)
                (Efuncs, Hfuncs) = fieldGenerator(a, kFree, kz, m, nCladding, nCore, modType)
                (ECoreρ, ECoreϕ, ECorez, ECladdingρ, ECladdingϕ, ECladdingz) = Efuncs
                (HCoreρ, HCoreϕ, HCorez, HCladdingρ, HCladdingϕ, HCladdingz) = Hfuncs
                funPairs = (((ECoreρ, ECladdingρ), (HCoreρ, HCladdingρ)),
                            ((ECoreϕ, ECladdingϕ), (HCoreϕ, HCladdingϕ)),
                            ((ECorez, ECladdingz), (HCorez, HCladdingz)))
                this_E = np.zeros((3, numSamples, numSamples), dtype=complex_dtype)
                this_H = np.zeros((3, numSamples, numSamples), dtype=complex_dtype)
                for idx, ((EfunCore, EfunCladding), (HfunCore, HfunCladding)) in enumerate(funPairs):
                    ECorevals             = np.vectorize(EfunCore)(ρrange)
                    ECorevals             = np.interp(ρg, ρrange, ECorevals)
                    ECorevals[~crossMask] = 0
                    ECladdingvals             = np.vectorize(EfunCladding)(ρrange)
                    ECladdingvals             = np.interp(ρg, ρrange, ECladdingvals)
                    ECladdingvals[crossMask]  = 0
                    E_all                  = ECorevals + ECladdingvals
                    HCorevals              = np.vectorize(HfunCore)(ρrange)
                    HCorevals              = np.interp(ρg, ρrange, HCorevals)
                    HCorevals[~crossMask]  = 0
                    HCladdingvals             = np.vectorize(HfunCladding)(ρrange)
                    HCladdingvals             = np.interp(ρg, ρrange, HCladdingvals)
                    HCladdingvals[crossMask]  = 0
                    H_all                  = HCorevals + HCladdingvals
                    E_all[np.isnan(E_all)] = 0.
                    H_all[np.isnan(H_all)] = 0.
                    this_E[idx, :, :] = E_all
                    this_H[idx, :, :] = H_all
                for parity in parities:
                    if parity == 'TETM':
                        for idx in range(3):
                            eigenbasis[counter, 0, idx, :, :] = this_E[idx]
                            eigenbasis[counter, 1, idx, :, :] = this_H[idx]
                    elif parity == 'EVEN':
                        for idx in range(3):
                            ϕPhaseE = {0: cosMesh,  
                                       1: -sinMesh,
                                       2: cosMesh}[idx]
                            ϕPhaseH = {0: -sinMesh,  
                                    1: cosMesh,
                                    2: -sinMesh}[idx]
                            eigenbasis[counter, 0, idx, :, :] = this_E[idx]*ϕPhaseE
                            eigenbasis[counter, 1, idx, :, :] = this_H[idx]*ϕPhaseH
                    elif parity == 'ODD':
                        for idx in range(3):
                            ϕPhaseE = {0: sinMesh,
                                    1: cosMesh, 
                                    2: sinMesh}[idx]
                            ϕPhaseH = {0: cosMesh,
                                    1: sinMesh, 
                                    2: cosMesh}[idx]
                            eigenbasis[counter, 0, idx, :, :] = this_E[idx]*ϕPhaseE
                            eigenbasis[counter, 1, idx, :, :] = this_H[idx]*ϕPhaseH
                    # normalize the field
                    E_field = eigenbasis[counter][0]
                    H_field = eigenbasis[counter][1]
                    norm_sq = field_dot(E_field, H_field, Δs)
                    norm = np.sqrt(norm_sq)
                    eigenbasis[counter][0] = eigenbasis[counter][0] / norm
                    eigenbasis[counter][1] = eigenbasis[counter][1] / norm
                    eigenbasis_nums.append((modType, parity, m, kzidx, kz, γ, β))
                    counter += 1
    fiber_sol['eigenbasis']      = eigenbasis
    fiber_sol['coord_layout']    = coord_layout
    fiber_sol['eigenbasis_nums'] = eigenbasis_nums
    return fiber_sol

def poynting_refractor(Efield, Hfield, nxy, nUpper, verbose=False):
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
    nUpper : float
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
    Sfield = 0.5*np.real(np.cross(Efield, np.conjugate(Hfield), axis=0))

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
    θfield = np.arcsin(nxy/nUpper * np.sin(βfield))

    # #FresnelS-Calc
    if verbose:
        print("Calculating the Fresnel coefficients...")
    
    fresnelS = (2 * nxy * np.cos(βfield) 
                / ( nxy * np.cos(βfield) 
                + np.sqrt(nUpper**2 
                                - nxy**2 * np.sin(βfield)**2)
                    )
                )
    
    # #FresnelP-Calc
    fresnelP = (2 * nUpper * nxy * np.cos(βfield) 
                / (nUpper**2 * np.cos(βfield) 
                +  nxy * np.sqrt(nUpper**2 
                            - nxy**2 * np.sin(βfield)**2)
                    )
                )
    
    # #ζ-Calc
    if verbose:
        print("Calculating the ζ of the local coord system...")
    # calculate the unit vector field perpendicular to the plane of incidence
    # which is basically k X z
    ζfield = np.zeros(kfield.shape, dtype=complex_dtype)
    ζfield[0] = kfield[1]
    ζfield[1] = -kfield[0]
    # normalize it
    ζfield /= np.sqrt(np.sum(np.abs(ζfield)**2, axis=0))

    # in the case of normal incidence, ζ is arbitrary
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
    ESdot = Efield[0] * ζfield[0] + Efield[1] * ζfield[1]
    EincS = np.zeros(ζfield.shape, dtype=complex_dtype)
    EincS[0] = ESdot * ζfield[0]
    EincS[1] = ESdot * ζfield[1]
    # #EincP-Calc
    EincP = Efield - EincS
    del ESdot

    # #HincS-Calc
    if verbose:
        print("Calculating the S and P component of the incident H field...")
    # decompose the field in P and S polarizations
    # first find P-pol and then use the complement to determine S-pol
    # the S-pol can be obtained by the dot product of H with ζ
    HSdot = Hfield[0] * ζfield[0] + Hfield[1] * ζfield[1]
    HincS = np.zeros(ζfield.shape, dtype=complex_dtype)
    HincS[0] = HSdot * ζfield[0]
    HincS[1] = HSdot * ζfield[1]
    # #HincP-Calc
    HincP = Hfield - HincS
    del HSdot

    # #ErefS-Calc
    if verbose:
        print("Calculating the S and P component of the refracted electric field...")
    ErefS = np.zeros(Efield.shape, dtype=complex_dtype)
    ErefS[0] = fresnelS * EincS[0]
    ErefS[1] = fresnelS * EincS[1]
    ErefS[2] = fresnelS * EincS[2]
    ErefS[np.isnan(ErefS)] = 0

    # #HrefS-Calc
    if verbose:
        print("Calculating the S and P component of the refracted H field...")
    HrefS = np.zeros(Hfield.shape, dtype=complex_dtype)
    HrefS[0] = (nUpper / nxy) * fresnelP * HincS[0]
    HrefS[1] = (nUpper / nxy) * fresnelP * HincS[1]
    HrefS[2] = (nUpper / nxy) * fresnelP * HincS[2]
    HrefS[np.isnan(HrefS)] = 0

    # #ErefP-Calc
    ErefP = np.zeros(Efield.shape, dtype=complex_dtype)
    ErefP[0] = fresnelP * EincP[0]
    ErefP[1] = fresnelP * EincP[1]
    ErefP[2] = fresnelP * EincP[2]
    ErefP[np.isnan(ErefP)] = 0

    # #HrefP-Calc
    HrefP = np.zeros(Hfield.shape, dtype=complex_dtype)
    HrefP[0] = (nUpper / nxy) * fresnelS * HincP[0]
    HrefP[1] = (nUpper / nxy) * fresnelS * HincP[1]
    HrefP[2] = (nUpper / nxy) * fresnelS * HincP[2]
    HrefP[np.isnan(HrefP)] = 0

    # #Eref-Calc
    if verbose:
        print("Calculating the total refracted electric field...")
    Eref  = ErefS + ErefP

    # #Href-Calc
    if verbose:
        print("Calculating the total refracted H field...")
    Href  = HrefS + HrefP

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

    return kref, Sfield, Stransverse, βfield, Eref, Href


def from_cyl_cart_to_cart_cart(field):
    '''
    Given  a  field  in  cylindrical  coordinates, convert it to
    cartesian  coordinates.  The  given  field  is assumed to be
    anchored  to a cartesian coordinate system in the sense that
    each  of the indices in its array corresponds to a cartesian
    grid  in the usual sense but the value of the vector at that
    position is given in terms of cylindrical coordinates.

    This  function  assumes  that  the  region  described by the
    cartesian coordinates is a square centered on the axis.

    Parameters
    ----------
    field : np.array
        A  field  in  cylindrical  coordinates  with  shape  (3,
        numSamples,  numSamples) the indices being the ρ, φ, and
        z components respectively.

    Returns
    -------
    ccfield : np.array
        A   field   in  cartesian  coordinates  with  shape  (3,
        numSamples,  numSamples) the indices being the x, y, and
        z components respectively of the given vector field.
    '''
    xrange = np.linspace(-1,1,field.shape[1])
    yrange = np.linspace(-1,1,field.shape[2])
    Xg, Yg = np.meshgrid(xrange, yrange)
    φg     = np.arctan2(Yg, Xg)
    ccfield = np.zeros(field.shape, dtype=field.dtype)
    if field.shape[0] == 3:
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

def from_cart_cart_to_cyl_cart(field):
    '''
    Given  a  field  in  cartesian  coordinates,  convert  it to
    cylindrical  coordinates.  This assumes that the given field
    represents   the   vector  field  in  cartesian  coordinates
    anchored  in  a  centered  cartesian grid. The function then
    returns  a  field that would be sampled in the same centered
    cartesian  grid,  but  the field values are now given in the
    associated cylindrical coordinate system.

    ATTENTION:  This  function assumes that the region described
    by  the  cartesian  coordinates  is a square centered on the
    axis.

    Parameters
    ----------
    field : np.array
        A   field   in  cartesian  coordinates  with  shape  (3,
        numSamples,  numSamples) the indices being the x, y, and
        z components respectively of the given vector field.

    Returns
    -------
    cylfield : np.array
        A  field  in  cylindrical  coordinates  with  shape  (3,
        numSamples,  numSamples) the indices being the ρ, φ, and
        z components respectively.
    '''
    xrange = np.linspace(-1,1,field.shape[1])
    yrange = np.linspace(-1,1,field.shape[2])
    Xg, Yg = np.meshgrid(xrange, yrange)
    φg     = np.arctan2(Yg, Xg)
    cylfield = np.zeros(field.shape, dtype=field.dtype)
    cylfield[2] = field[2]
    # create the cylindrical coordinates of the cartesian unit vector fields
    # first for the x component
    ux = np.zeros((2,field.shape[1],field.shape[2]))
    ux[0] = np.cos(φg)
    ux[1] = -np.sin(φg)
    # now for the y component
    uy = np.zeros((2,field.shape[1],field.shape[2]))
    uy[0] = np.sin(φg)
    uy[1] = np.cos(φg)
    # using these convert the cartesian components to cylindrical
    # adding the ρ components of what comes from the two unit vectors
    # scaled up by the field values in the cartesian basis
    cylfield[0] = field[0]*ux[0] + field[1]*uy[0]
    cylfield[1] = field[0]*ux[1] + field[1]*uy[1]
    return cylfield

def angular_farfield_propagator(field, λFree, nMedium, Zf, Zi, si, sf, Δf = None, options = {}):
    '''This function approximates the farfield of a field given the nearfield
    and the propagation distance.

    This is done by using the angular spectrum method. In which the Fourier
    transform of the nearfield is understood to describe the coefficients that
    approximate the farfield as a superposition of plane waves.

    It assumes that the nearfield is given on a square grid perpendicular to
    the z-axis. The farfield is also given on a square grid perpendicular to
    the z-axis.

    The farfield is approximated by the following formula:
    Efar = (2π 1j / kFree) * ((Zf-Zi)/Rfsq) * np.exp(1j*kFree*Rf) * S2
    with S2 equal to the Fourier transform extrapolated to the position
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
    options : dict (optional) A dictionary of options:
        'return_fourier' : bool
            If True, the function also returns the Fourier transform of the
            nearfield.
        'return_as_dict' : bool
            If False, the function returns the two arrays in the same order
            as the keys in the dictionary.
            If True, the function returns a dictionary with the following
            keys:
            'Efar' : np.array (3, Nf, Nf) The farfield of the field.
            'Efourier' : np.array (3, Ni, Ni) The Fourier transform of the nearfield.
               
    
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
    device_design : dict with at least the following keys:
        coreRadius : float
            the radius of the core in μm
        mlRadius : float
            the radius of the metalens in μm
        Δ : float
            the distance between the end face of the
            fiber and the start of the metalens in μm
        mlPitch : float
            the pitch of the metalens in μm
        emDepth : float
            the  depth of the emitter in the
            crystal  host  in μm, measured from the base of the
            metalens pillars
        emΔxy : float
            the lateral uncertainty (in μm) in
            the position of the emitter
        emΔz : float
            the uncertainty in the axial position
            of the emitter in μm
        mlHeight : float
            the height of the metalens in μm
        λFree  : float
            the  free-space  wavelength of the
            emitter in μm
        nCore : float
            the refractive index of the core
        nHost : float
            the refractive index of the host
        nClad : float
            the refractive index of the cladding
        NA : float
            the numerical aperture of the fiber

    
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
    numSamples : int
        how many evaluation points are avaiable for integration

    Returns
    -------
    BSimpson : np.array
        array of weights for the mixed Simpson's rule.
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

def FFT2D_convolution_integral(xCoords, yCoords, Usamples, kernelFun):
    '''
    FFT2D_convolution_integral  takes  the  sampled values of a function
    U(x,y) on a square domain and computes the convolution integral with
    the  given kernel. The domain of integration is given by xCoords and
    yCoords,   where  it  is  assumed  that  the  elements  of  Usamples
    correspond       to       xCorrds,       yCoords      such      that
    :math:`\\text{Usamples}[i,j]         =         U(\\text{yCoords}[j],
    \\text{xCoords}[i])`.

    Usamples  is  numeric,  whereas  kernelFun  must  be  provided  as a
    bivariate function.

    The convolution has the same domain as the sample values for U.

    .. math:: I(x',y') = \iint_\mathcal{A} U(x,y) \mathcal{K}(x-x', y-y') dxdy

    This   is  done  by  interpreting  the  discretized  integral  as  a
    convolution,  so  that the convolution may be performed with help of
    the fast Fourier transform.

    Parameters
    ----------
    xCoords : np.array
        x-coordinates on the source plane indexed to the given Usamples.

    yCoords : np.array
        y-coordinates on the source plane indexed to the given Usamples.

    Usamples : np.array
        complex  amplitude  of  the  field  in the source plane,
        sampled according to the coordinates provided by xCoords
        and yCoords. Must be a square array.

    kernelFun : func
        a bivariate function of the cartesian x,y coordinates.

    Returns
    -------
    integral : np.array
        resultant  convolution  integral. The top left corner of
        the  array  corresponds  to the lower left corner of the
        observation  plane. The coordinates associated with each
        element  in  the  given array correspond to the original
        xCoords, yCoords.
    '''

    numSamples = len(xCoords)
    assert len(xCoords) == len(yCoords), "Input must be square."
    # side of the integration domain
    Lobs = (xCoords[-1] - xCoords[0])
    Δζ = Lobs / (numSamples-1)
    Δη = Δζ

    # Simpson's rule with 3/8 tail correction
    BSimpson = simpson_weights_1D(numSamples)
    BSimpson = np.matmul(BSimpson.T, BSimpson)

    xCoords = np.linspace(-Lobs/2, Lobs/2, numSamples)
    yCoords = np.linspace(-Lobs/2, Lobs/2, numSamples)
    
    padextra = (numSamples - 1)
    Usamples = Usamples.T
    Usamples = BSimpson * Usamples
    Usamples = np.pad(Usamples, 
            pad_width=((0,padextra), (0,padextra)),
            mode='constant',
            constant_values=0.)

    x0, y0 = xCoords[0], yCoords[0] 
    ζ0, η0 = xCoords[0], yCoords[0]

    # Put together the H array
    Hx1 = np.full((numSamples-1, 2*numSamples-1), x0)
    Hx2 = np.tile(xCoords, (2*numSamples-1,1)).T
    Hx  = np.concatenate((Hx1, Hx2))

    Hζ2 = np.full((numSamples-1, 2*numSamples-1), ζ0)
    Hζ1 = np.tile(xCoords[::-1], (2*numSamples - 1,1)).T
    Hζ  = np.concatenate((Hζ1, Hζ2)).T

    Hxζ = Hx - Hζ.T
    Hy1 = np.full((numSamples-1, 2*numSamples-1), y0)
    Hy2 = np.tile(yCoords, (2*numSamples-1,1)).T
    Hy  = np.concatenate((Hy1, Hy2))

    Hη2 = np.full((numSamples-1, 2*numSamples-1), η0)
    Hη1 = np.tile(yCoords[::-1], (2*numSamples - 1,1)).T
    Hη  = np.concatenate((Hη1, Hη2)).T

    Hyη = (Hy - Hη.T).T

    # Evaluate the kernel
    Hfield = kernelFun(Hxζ, Hyη)

    # compute the Fourier transforms
    FFU = np.fft.fft2(Usamples)
    FFH = np.fft.fft2(Hfield)

    # perform the convolution
    FFUH = FFU * FFH

    # invert the result
    Sfield = np.fft.ifft2(FFUH)

    # get the good parts
    integral = (Δη*Δζ) * Sfield[-numSamples::,-numSamples::]
    integral = integral.T

    return integral

def electric_vectorial_diffraction(zProp, incidentEfield, xCoords, yCoords, λfree, nref):
    '''
    Given  samples  of  the  electric  field  of an electromagnetic wave
    incident  on  the  aperture  plane z=z0, this function estimates the
    electric  field at the observation plane z = z0+zProp.  This is done
    using   the  vectorial  Smythe-Kirchhoff  diffraction  integral.  It
    assumes  that  both  the  space  below  and above the plane z=z0 are
    filled  with  a homogeneous medium with refractive index nref. Since
    the  diffraction integral only requires the transverse components of
    the  electric field, the input can include or not the z-component of
    the field in the source plane.

    This  assumes  that  the  field  is incident from below the aperture
    plane.

    .. math:: 
        \\begin{align}
            \\vec{E}(x,y) &= \\frac{1}{2\\pi} \\nabla \\times \\iint_\\mathcal{A} 
            \\frac{e^{i k R}}{R} \\left(\\hat{z} \\times \\vec{E}(x',y',z=z_0) \\right)
            \\,\\,\\text{dx}\\text{dy} \\\\
            \\vec{R} &= \\vec{r} - \\vec{r}'
        \\end{align}

    Parameters
    ----------
    zProp : float
        the propagation distance along the z-axis.

    incidentEfield : np.array
        either  (3,N,N)  or (2,N,N), the incident electric field
        on  the  aperture  plane. The values given in it so that
        the  first  index references the x, y, or z component of
        the  incident  field.  Also the values are assumed to be
        anchored    to    the    coordinate   system   so   that
        incidentEfield[k,i,j]  gives  the  k-th component of the
        electric field at position xCoords[j], yCoords[i].

    xCoords : np.array (N,)
        coordinate array for the x-axis.

    yCoords : np.array (N,)
        coordinate array for the x-axis.

    Returns
    -------
    diffractedEfield  : np.array (3,N,N)
        the  diffracted field at the observation plane.
    '''
    k = 2*np.pi / λfree * nref
    # get the necessary kernels
    kernel_pairs = dk.electricKernels(zProp, k)
    # create the array to hold the diffracted field
    diffractedEfield = np.zeros((3,)+incidentEfield.shape[-2:], dtype=np.complex128)
    for idx, kernel_pair in enumerate(kernel_pairs):
        if not hasattr(kernel_pair[0],'null'):
            kernel_Ex = kernel_pair[0]
            field = FFT2D_convolution_integral(xCoords, yCoords, incidentEfield[0], kernel_Ex)
            diffractedEfield[idx] += field
        if not hasattr(kernel_pair[1],'null'):
            kernel_Ey = kernel_pair[1]
            field = FFT2D_convolution_integral(xCoords, yCoords, incidentEfield[1], kernel_Ey)
            diffractedEfield[idx] += field
    return diffractedEfield

def magnetic_vectorial_diffraction(zProp, incidentEfield, xCoords, yCoords, λfree, nref):
    '''
    Given  samples  of  the  electric  field  of an electromagnetic wave
    incident  on  the  aperture  plane z=z0, this function estimates the
    magnecic H-field at the observation plane z = z0+zProp. This is done
    using   the  vectorial  Smythe-Kirchhoff  diffraction  integral.  It
    assumes  that  both  the  space  below  and above the plane z=z0 are
    filled  with  a homogeneous medium with refractive index nref. Since
    the  diffraction integral only requires the transverse components of
    the  electric field, the input can include or not the z-component of
    the field in the source plane.

    This  assumes  that  the  field  is incident from below the aperture
    plane.

    .. math:: 
        \\begin{align}
        \\vec{H}(x,y) &= \\frac{1}{2\\pi i \\omega} \\nabla \\left(\\nabla \\times \\iint_\\mathcal{A} 
        \\frac{e^{i k R}}{R} \\left(\\hat{z} \\times \\vec{E}(x',y',z=z_0) \\right)
        \\,\\,\\text{dx}\\text{dy} \\right) \\\\
        \\vec{R} &= \\vec{r} - \\vec{r}'
        \\end{align}

    Parameters
    ----------
    zProp : float
        the propagation distance along the z-axis.

    incidentEfield : np.array
        either  (3,N,N)  or (2,N,N), the incident electric field
        on  the  aperture  plane. The values given in it so that
        the  first  index references the x, y, or z component of
        the  incident  field.  Also the values are assumed to be
        anchored    to    the    coordinate   system   so   that
        incidentEfield[k,i,j]  gives  the  k-th component of the
        electric field at position xCoords[j], yCoords[i].

    xCoords : np.array
        (N,) coordinate array for the x-axis.

    yCoords : np.array (N,)
        coordinate array for the x-axis.

    Returns
    -------
    diffractedHfield : np.array (3,N,N)
        the  diffracted field at the observation plane.
    '''
    kFree = 2*np.pi / λfree
    k = kFree * nref
    ω = kFree
    # get the necessary kernels
    kernel_pairs =  dk.magneticKernels(zProp, k)
    # create the array to hold the diffracted field
    diffractedHfield = np.zeros((3,)+incidentEfield.shape[-2:], dtype=np.complex128)
    for idx, kernel_pair in enumerate(kernel_pairs):
        if not hasattr(kernel_pair[0],'null'):
            kernel_Ex = kernel_pair[0]
            field = FFT2D_convolution_integral(xCoords, yCoords, incidentEfield[0], kernel_Ex)
            diffractedHfield[idx] += field
        if not hasattr(kernel_pair[1],'null'):
            kernel_Ey = kernel_pair[1]
            field = FFT2D_convolution_integral(xCoords, yCoords, incidentEfield[1], kernel_Ey)
            diffractedHfield[idx] += field
    diffractedHfield = -1j / ω * diffractedHfield
    return diffractedHfield

def scalar_diffraction(zProp, incidentField, xCoords, yCoords, λfree, nref):
    '''
    Given  samples  of  the  electric  field  of an electromagnetic wave
    incident  on  the  aperture  plane z=z0, this function estimates the
    electric  field  at the observation plane z = z0+zProp. This is done
    using  the  Rayleigh-Sommerfeld  diffraction integral. This assuming
    that  that  both the space below and above the plane z=z0 are filled
    with  a  homogeneous  medium  with  refractive index nref.

    .. math:: 

        E(x',y') = \\frac{1}{2\\pi} \\iint_\\mathcal{A} \\frac{e^{ikR}}{R} \\frac{z}{R} \\frac{(ikR-1)}{R} E(x',y') \\text{dxdy}

    This  assumes  that  the  field  is incident from below the aperture
    plane.

    Parameters
    ----------
    zProp : float
        the propagation distance along the z-axis.

    incidentEfield : np.array (N,N)
        the incident field on the aperture plane. The values are
        assumed  to be anchored to the coordinate system so that
        incidentEfield[i,j]  gives  the  value  of  the field at
        position xCoords[j], yCoords[i].

    + xCoords : np.array (N,)
        coordinate array for the x-axis.

    + yCoords : np.array  (N,)
        coordinate array for the x-axis.

    Returns
    -------
    diffractedEfield : np.array
        (N,N)  the  diffracted field at the observation plane.
    '''
    k = 2 * np.pi / λfree * nref
    # get the necessary kernel
    kernel_pairs = dk.electricKernels(zProp, k)
    kernel = kernel_pairs[0][0]
    # create the array to hold the diffracte field
    diffractedEfield = FFT2D_convolution_integral(xCoords, yCoords, incidentField, kernel)
    return diffractedEfield

def fresnel_t_θ1θ2(n1, n2, θ1, θ2):
    '''
    This function returns the electric field transmission
    coefficients for S and P polarization.
    Use if both angles are known beforehand.

    ::
    
        ┌───────────────────────────────────────────────┐
        │                       │                 ┌─    │
        │                       │               ┌─┘     │
        │                       │             ┌─┘       │
        │                       ├──┐        ┌─┘         │
        │                       │  └θ2─┐  ┌─┘           │
        │                       │      └┬─┘             │
        │                       │     ┌─┘               │
        │                       │   ┌─┘                 │
        │                       │ ┌─┘             n2    │
        │ ______________________├─┘____________________ │
        │ |||||||||||||||||||||╱│|||||||||||||||||||||| │
        │ ||||||||||||||||||||╱|│|||||||||||||||| n1 || │
        │ |||||||||||||||||||╱||│|||||||||||||||||||||| │
        │ ||||||||||||||||||╱|||│|||||||||||||||||||||| │
        │ |||||||||||||||||╱||||│|||||||||||||||||||||| │
        │ ||||||||||||||||╱|||||│|||||||||||||||||||||| │
        │ |||||||||||||||╱─┐||||│|||||||||||||||||||||| │
        │ ||||||||||||||╱||└θ1┐|│|||||||||||||||||||||| │
        │ |||||||||||||╱||||||└─┤|||||||||||||||||||||| │
        │ ||||||||||||╱|||||||||│|||||||||||||||||||||| │
        └───────────────────────────────────────────────┘

    Parameters
    ----------
    n1 : float or np.array((N,M))
        refractive index of medium 1
    n2 : float or np.array((N,M))
        refractive index of medium 2
    θ1 : float or np.array((N,M))
        angle in medium 1
    θ2 : float or np.array((N,M))
        angle in medium 2
    
    Returns
    -------
    np.array([ts,tp]) : np.array
    '''
    cosθ1 = np.cos(θ1)
    cosθ2 = np.cos(θ2)
    ts    = (2*n1*cosθ1
             / (n1*cosθ1 + n2*cosθ2))
    tp    = (2*n1*cosθ1
             / (n2*cosθ1 + n1*cosθ2))
    return np.array([ts, tp])

def fresnel_ts_θ1θ2(n1, n2, θ1, θ2):
    '''
    This function returns the electric field transmission
    and reflection coefficients for S and P polarization.
    Use if both angles are known beforehand.

    ::
    
        ┌───────────────────────────────────────────────┐
        │                       │                 ┌─    │
        │                       │               ┌─┘     │
        │                       │             ┌─┘       │
        │                       ├──┐        ┌─┘         │
        │                       │  └θ2─┐  ┌─┘           │
        │                       │      └┬─┘             │
        │                       │     ┌─┘               │
        │                       │   ┌─┘                 │
        │                       │ ┌─┘             n2    │
        │ ______________________├─┘____________________ │
        │ |||||||||||||||||||||╱│|||||||||||||||||||||| │
        │ ||||||||||||||||||||╱|│|||||||||||||||| n1 || │
        │ |||||||||||||||||||╱||│|||||||||||||||||||||| │
        │ ||||||||||||||||||╱|||│|||||||||||||||||||||| │
        │ |||||||||||||||||╱||||│|||||||||||||||||||||| │
        │ ||||||||||||||||╱|||||│|||||||||||||||||||||| │
        │ |||||||||||||||╱─┐||||│|||||||||||||||||||||| │
        │ ||||||||||||||╱||└θ1┐|│|||||||||||||||||||||| │
        │ |||||||||||||╱||||||└─┤|||||||||||||||||||||| │
        │ ||||||||||||╱|||||||||│|||||||||||||||||||||| │
        └───────────────────────────────────────────────┘

    Parameters
    ----------
    n1 : float or np.array((N,M))
        refractive index of medium 1
    n2 : float or np.array((N,M))
        refractive index of medium 2
    θ1 : float or np.array((N,M))
        angle in medium 1
    
    Returns
    -------
    np.array([ts,tp]) : np.array
    '''
    sinθ2 = n1 * np.sin(θ1) / n2
    θ2    = np.arcsin(sinθ2)
    ts, tp  = fresnel_t_θ1θ2(n1, n2, θ1, θ2)
    rs    = ts - 1
    rp    = n2/n1*tp-1
    return np.array([ts, tp, rs, rp])

def fresnel_t_θ1(n1, n2, θ1):
    '''
    This function returns the electric field transmission
    coefficients for S and P polarization.
    Use if only one angle is known beforehand.

    ::
    
        ┌───────────────────────────────────────────────┐
        │                       │                 ┌─    │
        │                       │               ┌─┘     │
        │                       │             ┌─┘       │
        │                       ├──┐        ┌─┘         │
        │                       │  └θ2─┐  ┌─┘           │
        │                       │      └┬─┘             │
        │                       │     ┌─┘               │
        │                       │   ┌─┘                 │
        │                       │ ┌─┘             n2    │
        │ ______________________├─┘____________________ │
        │ |||||||||||||||||||||╱│|||||||||||||||||||||| │
        │ ||||||||||||||||||||╱|│|||||||||||||||| n1 || │
        │ |||||||||||||||||||╱||│|||||||||||||||||||||| │
        │ ||||||||||||||||||╱|||│|||||||||||||||||||||| │
        │ |||||||||||||||||╱||||│|||||||||||||||||||||| │
        │ ||||||||||||||||╱|||||│|||||||||||||||||||||| │
        │ |||||||||||||||╱─┐||||│|||||||||||||||||||||| │
        │ ||||||||||||||╱||└θ1┐|│|||||||||||||||||||||| │
        │ |||||||||||||╱||||||└─┤|||||||||||||||||||||| │
        │ ||||||||||||╱|||||||||│|||||||||||||||||||||| │
        └───────────────────────────────────────────────┘

    Parameters
    ----------
    n1 : float or np.array((N,M))
        refractive index of medium 1
    n2 : float or np.array((N,M))
        refractive index of medium 2
    θ1 : float or np.array((N,M))
        angle in medium 1
    θ2 : float or np.array((N,M))
        angle in medium 2
    
    Returns
    -------
    np.array([ts,tp]) : np.array
    '''
    cosθ1 = np.cos(θ1)
    cosθ2 = np.sqrt(1 - ((n1/n2) * np.sin(θ1))**2)
    ts    = (2*n1*cosθ1
             / (n1*cosθ1 + n2*cosθ2))
    tp    = (2*n1*cosθ1
             / (n2*cosθ1 + n1*cosθ2))
    return np.array([ts, tp])

def fresnel_ts_θ1(n1, n2, θ1):
    '''
    This function returns the electric field transmission
    and reflection coefficients for S and P polarization.
    Use if only one angle is known beforehand.

    ::
    
        ┌───────────────────────────────────────────────┐
        │                       │                 ┌─    │
        │                       │               ┌─┘     │
        │                       │             ┌─┘       │
        │                       ├──┐        ┌─┘         │
        │                       │  └θ2─┐  ┌─┘           │
        │                       │      └┬─┘             │
        │                       │     ┌─┘               │
        │                       │   ┌─┘                 │
        │                       │ ┌─┘             n2    │
        │ ______________________├─┘____________________ │
        │ |||||||||||||||||||||╱│|||||||||||||||||||||| │
        │ ||||||||||||||||||||╱|│|||||||||||||||| n1 || │
        │ |||||||||||||||||||╱||│|||||||||||||||||||||| │
        │ ||||||||||||||||||╱|||│|||||||||||||||||||||| │
        │ |||||||||||||||||╱||||│|||||||||||||||||||||| │
        │ ||||||||||||||||╱|||||│|||||||||||||||||||||| │
        │ |||||||||||||||╱─┐||||│|||||||||||||||||||||| │
        │ ||||||||||||||╱||└θ1┐|│|||||||||||||||||||||| │
        │ |||||||||||||╱||||||└─┤|||||||||||||||||||||| │
        │ ||||||||||||╱|||||||||│|||||||||||||||||||||| │
        └───────────────────────────────────────────────┘

    Parameters
    ----------
    n1 : float or np.array((N,M))
        refractive index of medium 1
    n2 : float or np.array((N,M))
        refractive index of medium 2
    θ1 : float or np.array((N,M))
        angle in medium 1
    
    Returns
    -------
    np.array([ts,tp]) : np.array
    '''
    sinθ2 = n1 * np.sin(θ1) / n2
    θ2    = np.arcsin(sinθ2)
    ts, tp  = fresnel_t_θ1θ2(n1, n2, θ1, θ2)
    rs    = ts - 1
    rp    = n2/n1*tp-1
    return np.array([ts, tp, rs, rp])

def field_sampler(funPairs, cross_width, resolution, m, parity,
                  coreRadius, coord_sys = 'cartesian-cylindrical',
                  equiv_currents = False):
    '''
    Given  the  functions that describe the H and E fields in the core and
    cladding, this function can be used to sample either the fields or the
    equivalent  currents  on the centered computational grid of given size
    and resolution.

    Parameters
    ----------
    funPairs : 3-tuple of 2-tuples of 2-tuples of functions
        The  functions  that  describe the field in the core and cladding.
        Having the following structure:
            (
            ((ECoreρ, ECladdingρ), (HCoreρ, HCladdingρ)),
            ((ECoreϕ, ECladdingϕ), (HCoreϕ, HCladdingϕ)),
            ((ECorez, ECladdingz), (HCorez, HCladdingz))
            )
    cross_width : float
        The width of the cross-section of the computational grid.
    resolution : float
        How many pixels per unit length.
    m : int
        The azimuthal mode number.
    parity : str
        The parity of the mode. Can be 'TETM', 'EVEN', or 'ODD'.
    coreRadius : float
        The radius of the core.
    coord_sys : str, optional
        The coordinate system of the field. Can be 'cartesian-cylindrical'
        or  'cartesian-cartesian'. The default is 'cartesian-cylindrical'.
        The  first  word  determines  the coordinate system of the grid in
        which the field is being described; the second word determines the
        coordinate system of the field itself in those points.
    equiv_currents : bool, optional
        Whether  to  return the equivalent electric and magnetic currents.
        The  default  is False. This is assuming that the normal is in the
        z-direction. Furthermore, it assumes that the only necessary sheet
        of  current  is a single plane, whereas in rigor one would need to
        envelop the simulation volume with these currents.
    
    Returns
    -------
    Xg, Yg, E_field, H_field : tuple of ndarrays
        Returned if equiv_currents is False. The first two are the meshgrid
        of  the  computational  grid.
    '''
    cross_height = cross_width
    max_ρ       = np.sqrt(2)*cross_width/2
    numSamples = int(cross_width*resolution)
    numρSamples = int(np.sqrt(2)*numSamples)
    E_field    = np.zeros((3, numSamples, numSamples), dtype=np.complex_)
    H_field   = np.zeros((3, numSamples, numSamples), dtype=np.complex_)
    xrange   = np.linspace(-cross_width/2, cross_width/2, numSamples)
    yrange  = np.linspace(-cross_height/2, cross_height/2, numSamples)
    Xg, Yg = np.meshgrid(xrange, yrange)
    ρg    = np.sqrt(Xg**2 + Yg**2)
    ρrange = np.linspace(0, max_ρ, numρSamples)
    φg    = np.arctan2(Yg, Xg)
    cosMesh = np.cos(m*φg)
    sinMesh = np.sin(m*φg)
    coreMask = (ρg <= coreRadius)
    for idx, ((EfunCore, EfunCladding), (HfunCore, HfunCladding)) in enumerate(funPairs):
        ECorevals                 = np.vectorize(EfunCore)(ρrange)
        ECorevals                 = np.interp(ρg, ρrange, ECorevals)
        ECorevals[~coreMask]      = 0
        ECladdingvals             = np.vectorize(EfunCladding)(ρrange)
        ECladdingvals             = np.interp(ρg, ρrange, ECladdingvals)
        ECladdingvals[coreMask]   = 0
        E_all                     = ECorevals + ECladdingvals
        HCorevals                 = np.vectorize(HfunCore)(ρrange)
        HCorevals                 = np.interp(ρg, ρrange, HCorevals)
        HCorevals[~coreMask]      = 0
        HCladdingvals             = np.vectorize(HfunCladding)(ρrange)
        HCladdingvals             = np.interp(ρg, ρrange, HCladdingvals)
        HCladdingvals[coreMask]   = 0
        H_all                     = HCorevals + HCladdingvals
        E_all[np.isnan(E_all)]    = 0.
        H_all[np.isnan(H_all)]    = 0.
        E_field[idx, :, :] = E_all
        H_field[idx, :, :] = H_all
    if parity == 'TETM':
        pass
    elif parity == 'EVEN':
        for idx in range(3):
            ϕPhaseE = {0: cosMesh,  
                       1: -sinMesh,
                       2: cosMesh}[idx]
            ϕPhaseH = {0: -sinMesh,  
                       1: cosMesh,
                       2: -sinMesh}[idx]
            E_field[idx, :, :] *= ϕPhaseE
            H_field[idx, :, :] *= ϕPhaseH
    elif parity == 'ODD':
        for idx in range(3):
            ϕPhaseE = {0: sinMesh,
                       1: cosMesh, 
                       2: sinMesh}[idx]
            ϕPhaseH = {0: cosMesh,
                       1: sinMesh, 
                       2: cosMesh}[idx]
            E_field[idx, :, :] *= ϕPhaseE
            H_field[idx, :, :] *= ϕPhaseH
    if equiv_currents:
        electric_J = np.zeros((2,) + H_field.shape[1:], dtype=np.complex_)
        magnetic_K = np.zeros((2,) + E_field.shape[1:], dtype=np.complex_)
        electric_J[0] = -H_field[1]
        electric_J[1] = H_field[0]
        magnetic_K[0] = E_field[1]
        magnetic_K[1] = -E_field[0]
        if coord_sys == 'cartesian-cartesian':
            electric_J = from_cyl_cart_to_cart_cart(electric_J)
            magnetic_K = from_cyl_cart_to_cart_cart(magnetic_K)
        return Xg, Yg, electric_J, magnetic_K
    if coord_sys == 'cartesian-cartesian':
        E_field = from_cyl_cart_to_cart_cart(E_field)
        H_field = from_cyl_cart_to_cart_cart(H_field)
    return Xg, Yg, E_field, H_field

def equivCurrents(Efuncs, Hfuncs, coreRadius, m, parity):
    '''
    This  function  returns  the necessary magnetic and electric
    currents  corresponding  to  the  given field functions. The
    currents  are  calculated  in  a  plane  perpedicular to the
    z-axis,  and  are  returned  so  that  they are functions of
    cartesian   x,y  coordinates  and  represent  the  different
    cartesian components of the currents.

    Parameters
    ----------
    Efuncs : 6-tuple of functions
        Having    the   structure   (ECoreρ,   ECoreϕ,   Ecorez,
        ECladdingρ,  ECladdingϕ, Ecladdingz). The z-component is
        not needed, if it is included, then it is neglected. The
        first  three  functions  describing  the  electric field
        inside  of the core and the following three the electric
        field in the cladding.
    Hfuncs : 6-tuple of functions
        Having    the   structure   (HCoreρ,   HCoreϕ,   HCorez,
        HCladdingρ,  HCladdingϕ, HCladdingz). The z-component is
        not needed, if it is included, then it is neglected. The
        first  three  functions describing the H-field inside of
        the  core  and  the  following  three the H-field in the
        cladding.
    coreRadius : float
        The radius of the core.
    m : int
        The azimuthal mode number.
    parity : str
        The parity of the mode. Can be 'TETM', 'EVEN', or 'ODD'.

    Returns
    -------
    JKfuns : tuple of functions
        Provided  as  (Jx, Jy, Kx, Ky). With Jx and Jy being the
        functions  for  the  x-y  components  of  the equivalent
        electric  current, and Kx and Ky being the functions for
        the  x-y  components of the equivalent magnetic current.
        Each  of the functions is a function of vec, which needs
        to be an iterable with *three* elements corresponding to
        the  x,y,z  cartesian components; this is to account for
        the  fact  that  MEEP requires these functions to take a
        single Vector3 argument.
        
    '''
    (ECoreρ, ECoreϕ, _, ECladdingρ, ECladdingϕ, _) = Efuncs
    (HCoreρ, HCoreϕ, _, HCladdingρ, HCladdingϕ, _) = Hfuncs
    if parity == 'TETM':
        def Jx(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return -HCoreϕ(ρ)*np.cos(ϕ) - HCoreρ(ρ)*np.sin(ϕ)
            else:
                return -HCladdingϕ(ρ)*np.cos(ϕ) - HCladdingρ(ρ)*np.sin(ϕ)
        def Jy(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return -HCoreϕ(ρ)*np.sin(ϕ) + HCoreρ(ρ)*np.cos(ϕ)
            else:
                return -HCladdingϕ(ρ)*np.sin(ϕ) + HCladdingρ(ρ)*np.cos(ϕ)
        def Kx(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return ECoreϕ(ρ)*np.cos(ϕ) + ECoreρ(ρ)*np.sin(ϕ)
            else:
                return ECladdingϕ(ρ)*np.cos(ϕ) + ECladdingρ(ρ)*np.sin(ϕ)
        def Ky(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return ECoreϕ(ρ)*np.sin(ϕ) - ECoreρ(ρ)*np.cos(ϕ)
            else:
                return ECladdingϕ(ρ)*np.sin(ϕ) - ECladdingρ(ρ)*np.cos(ϕ)
    elif parity == 'EVEN':
        def Jx(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return -HCoreϕ(ρ)*np.cos(m*ϕ)*np.cos(ϕ) - HCoreρ(ρ)*np.sin(m*φ)*(-1)*np.sin(ϕ)
            else:
                return -HCladdingϕ(ρ)*np.cos(m*φ)*np.cos(ϕ) - HCladdingρ(ρ)*np.sin(m*φ)*(-1)*np.sin(ϕ)
        def Jy(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return -HCoreϕ(ρ)*np.cos(m*ϕ)*np.sin(ϕ) + HCoreρ(ρ)*np.sin(m*φ)*(-1)*np.cos(ϕ)
            else:
                return -HCladdingϕ(ρ)*np.cos(m*φ)*np.sin(ϕ) + HCladdingρ(ρ)*np.sin(m*φ)*(-1)*np.cos(ϕ)
        def Kx(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return ECoreϕ(ρ)*np.sin(m*φ)*(-1)*np.cos(ϕ) + ECoreρ(ρ)*np.cos(m*φ)*np.sin(ϕ)
            else:
                return ECladdingϕ(ρ)*np.sin(m*φ)*(-1)*np.cos(ϕ) + ECladdingρ(ρ)*np.cos(m*φ)*np.sin(ϕ)
        def Ky(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return ECoreϕ(ρ)*np.sin(m*φ)*(-1)*np.sin(ϕ) - ECoreρ(ρ)*np.cos(m*φ)*np.cos(ϕ)
            else:
                return ECladdingϕ(ρ)*np.sin(m*φ)*(-1)*np.sin(ϕ) - ECladdingρ(ρ)*np.cos(m*φ)*np.cos(ϕ)
    elif parity == 'ODD':
        def Jx(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return -HCoreϕ(ρ)*np.sin(m*φ)*np.cos(ϕ) - HCoreρ(ρ)*np.cos(m*φ)*np.sin(ϕ)
            else:
                return -HCladdingϕ(ρ)*np.sin(m*φ)*np.cos(ϕ) - HCladdingρ(ρ)*np.cos(m*φ)*np.sin(ϕ)
        def Jy(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return -HCoreϕ(ρ)*np.sin(m*φ)*np.sin(ϕ) + HCoreρ(ρ)*np.cos(m*φ)*np.cos(ϕ)
            else:
                return -HCladdingϕ(ρ)*np.sin(m*φ)*np.sin(ϕ) + HCladdingρ(ρ)*np.cos(m*φ)*np.cos(ϕ)
        def Kx(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return ECoreϕ(ρ)*np.cos(m*φ)*np.cos(ϕ) + ECoreρ(ρ)*np.sin(m*φ)*np.sin(ϕ)
            else:
                return ECladdingϕ(ρ)*np.cos(m*φ)*np.cos(ϕ) + ECladdingρ(ρ)*np.sin(m*φ)*np.sin(ϕ)
        def Ky(vec):
            x, y, _ = vec
            ϕ = np.arctan2(y,x)
            ρ = np.sqrt(x**2 + y**2)
            if ρ < coreRadius:
                return ECoreϕ(ρ)*np.cos(m*φ)*np.sin(ϕ) - ECoreρ(ρ)*np.sin(m*φ)*np.cos(ϕ)
            else:
                return ECladdingϕ(ρ)*np.cos(m*φ)*np.sin(ϕ) - ECladdingρ(ρ)*np.sin(m*φ)*np.cos(ϕ)
    JKfuns = (Jx, Jy, Kx, Ky)
    return JKfuns

def from_near_to_far_angular(field, xy_span,
                             kMedium, plotFun = np.real,
                             angular_resolution = np.pi/50, make_plots=False,
                             griddata_method = 'nearest'):
    '''
    Parameters
    ----------
    field : np.array (N, N)
        Values, possibly complex, of the given field.
    xy_span : float
        The spatial extent in both the x and y axes that corresponds to the given field array.
    kMedium : float
        The magnitude of the homogeneous-medium wavevector
    plotFun : func
        In case plots are presented, this function is applied to the given array before plotting.
    angular_resolution: float, optional
        To what angular resolution the angular representation of the field is calculated.
    make_plots: bool
        Whether to show plots as the calculation is carried out.
    Returns
    -------
    (angular_rep, theta_range, phi_range, phi_sum, θmax) : tuple
        angular_rep : np.array (M, M)
            The field values in angular coordinates (θ, ϕ).
        theta_range : np.array
            With the values of θ that match the provided angular_rep.
        phi_range : np.array
            With the values of ϕ that match the provided angular_rep.
        phi_sum   : np.array
            angular_rep summed along the ϕ axis
        θmax      : float
            The angle, in degrees, at which phi_sum is max.
    '''
    x_span = xy_span
    y_span = xy_span
    num_theta_samples = int(np.pi / angular_resolution)
    num_phi_samples = num_theta_samples

    x = np.linspace(-x_span/2, x_span/2, field.shape[0])
    y = np.linspace(-y_span/2, y_span/2, field.shape[1])

    fourier_Field = np.fft.fft2(field)
    fourier_Field = np.fft.fftshift(fourier_Field)
    # Infer the sampling rate of the given field
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    # Calculate and shift the corresponding frequencies
    # converted to angular components of wavevector
    freq_x = 2*np.pi*np.fft.fftfreq(fourier_Field.shape[0], dx)
    freq_y = 2*np.pi*np.fft.fftfreq(fourier_Field.shape[1], dy)
    freq_x_shifted = np.fft.fftshift(freq_x)
    freq_y_shifted = np.fft.fftshift(freq_y)

    if make_plots:
        cmap_range = max(np.max(plotFun(fourier_Field)), np.max(-plotFun(fourier_Field)))
        extent = np.array([freq_x_shifted[0],freq_x_shifted[-1],
                        freq_y_shifted[0],freq_y_shifted[-1]])
        extent = extent/kMedium
        fig, ax = plt.subplots()
        ax.imshow(plotFun(fourier_Field),
                cmap=cm.ember,
                vmin=0,
                vmax=cmap_range,
                extent=extent
                )
        ax.add_patch(plt.Circle((0,0),
                    1,
                    fill=False,
                    color='r',
                    linestyle='--',
                    alpha=0.5))
        ax.set_xlabel('kx/k')
        ax.set_ylabel('ky/k')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        plt.show()

    # Convert the Fourier transform to angular variables
    theta_range  = np.linspace(-np.pi/2, np.pi/2, num_theta_samples+1)
    phi_range    = np.linspace(0, np.pi, num_phi_samples+1)
    phi_grid, theta_grid   = np.meshgrid(phi_range, theta_range)
    sx_grid = np.sin(theta_grid) * np.cos(phi_grid)
    sy_grid = np.sin(theta_grid) * np.sin(phi_grid)
    sx_grid_old, sy_grid_old = np.meshgrid(freq_x_shifted/kMedium, freq_y_shifted/kMedium)
    jacobian = np.abs(np.sin(theta_grid))

    angular_rep = griddata((sx_grid_old.ravel(), 
                            sy_grid_old.ravel()), fourier_Field.ravel(), 
                        (sx_grid, sy_grid),
                        method=griddata_method)
    angular_rep = np.abs(angular_rep**2)

    if make_plots:
        extent = np.array([0, 180, -90,  90])
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(angular_rep, cmap=cm.ember, extent=extent)
        ax.set_xlabel('$\phi /deg$')
        ax.set_ylabel(r'$\theta / deg$')
        ax.set_xticks(np.arange(0, 181, 45))
        ax.set_yticks(np.arange(-90, 91, 45))
        plt.show()

    phi_sum = np.sum(angular_rep, axis=1)
    phi_sum /= np.max(phi_sum)
    θmax = np.abs(theta_range[np.argmax(phi_sum)]/np.pi*180)
    if make_plots:
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(theta_range / np.pi * 180, phi_sum)
        ax.set_xlabel('θ/∘')
        plt.show()
    
    return (angular_rep, theta_range, phi_range, phi_sum, θmax) 
