#!/usr/bin/env python3
from scipy import special
import numpy as np

def tmfungen(λfree, n1, n2, a):
    '''
    This function returns the eigenvalue function for TM modes.

    The expressions for the secular equations are generated in the
    accompanying Mathematica notebook wavesight.nb.

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
    def tm(kz):
        return -(n1**2*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.jv(1,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) - (n2**2*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(0,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*special.kn(1,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))/special.kn(0,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))
    return tm

def tefungen(λfree, n1, n2, a):
    '''
    This function returns the eigenvalue function for TE modes.

    The expressions for the secular equations are generated in the
    accompanying Mathematica notebook wavesight.nb.

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
    def te(kz):
        return -(np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.jv(1,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) - (np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(0,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))*special.kn(1,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))/special.kn(0,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))
    return te

def hefungen(λfree, m, n1, n2, a):
    '''
    This function returns the eigenvalue function for HE(n,m) modes.

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
        return -((m**2*(1/(-kz**2 + (4*n1**2*np.pi**2)/λfree**2) + 1/(kz**2 - (4*n2**2*np.pi**2)/λfree**2))*(n1**2/(-kz**2 + (4*n1**2*np.pi**2)/λfree**2) + n2**2/(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))/a**2) + ((special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)))/(2.*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) + (-special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) - special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)))/(2.*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))*((n1**2*(special.jv(-1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))))/(2.*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2)*special.jv(m,a*np.sqrt(-kz**2 + (4*n1**2*np.pi**2)/λfree**2))) + (n2**2*(-special.kn(-1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)) - special.kn(1 + m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))/(2.*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2)*special.kn(m,a*np.sqrt(kz**2 - (4*n2**2*np.pi**2)/λfree**2))))
    return he


