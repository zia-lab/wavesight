import numpy as np
from scipy import special
import warnings
from printech import *

printer("ATTENTION: disabling RuntimeWarning often triggered due to evaluation of square roots with negative arguments")
warnings.filterwarnings("ignore", category=RuntimeWarning)

def TE_ECoregenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_ECoreρ : func
    '''
    def TE_ECoreρ(ρ):
        '''
        Returns the radial component of TE_ECore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_ECoreρ : real
        '''
        return 0
    return TE_ECoreρ

def TE_ECoregenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_ECoreϕ : func
    '''
    def TE_ECoreϕ(ρ):
        '''
        Returns the azimuthal component of TE_ECore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_ECoreϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return -(special.jv(1,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)/special.jv(1,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))
    return TE_ECoreϕ

def TE_ECoregenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_ECorez : func
    '''
    def TE_ECorez(ρ):
        '''
        Returns the transverse component of TE_ECore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_ECorez : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return 0
    return TE_ECorez


def TM_ECoregenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_ECoreρ : func
    '''
    def TM_ECoreρ(ρ):
        '''
        Returns the radial component of TM_ECore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_ECoreρ : real
        '''
        return special.jv(1,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)/special.jv(1,a*np.sqrt(-kz**2 + kFree**2*nCore**2))
    return TM_ECoreρ

def TM_ECoregenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_ECoreϕ : func
    '''
    def TM_ECoreϕ(ρ):
        '''
        Returns the azimuthal component of TM_ECore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_ECoreϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return 0
    return TM_ECoreϕ

def TM_ECoregenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_ECorez : func
    '''
    def TM_ECorez(ρ):
        '''
        Returns the transverse component of TM_ECore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_ECorez : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return (1j*np.sqrt(-kz**2 + kFree**2*nCore**2)*special.jv(0,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ))/(kz*special.jv(1,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))
    return TM_ECorez


def HE_ECoregenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_ECoreρ : func
    '''
    def HE_ECoreρ(ρ):
        '''
        Returns the radial component of HE_ECore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_ECoreρ : real
        '''
        return -0.5*(special.jv(-1 + m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)*(-1 - (2*kFree**2*m*(nCladding**2 - nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))/(a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(kz**2 - kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))) + special.jv(1 + m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)*(1 - (2*kFree**2*m*(nCladding**2 - nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))/(a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(kz**2 - kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))
    return HE_ECoreρ

def HE_ECoregenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_ECoreϕ : func
    '''
    def HE_ECoreϕ(ρ):
        '''
        Returns the azimuthal component of HE_ECore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_ECoreϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return -0.5*(special.jv(-1 + m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)*(-1 - (2*kFree**2*m*(nCladding**2 - nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))/(a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(kz**2 - kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))) - special.jv(1 + m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)*(1 - (2*kFree**2*m*(nCladding**2 - nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))/(a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(kz**2 - kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))
    return HE_ECoreϕ

def HE_ECoregenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_ECorez : func
    '''
    def HE_ECorez(ρ):
        '''
        Returns the transverse component of HE_ECore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_ECorez : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return ((-1j)*np.sqrt(-kz**2 + kFree**2*nCore**2)*special.jv(m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ))/(kz*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))
    return HE_ECorez


def TE_ECladdinggenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_ECladdingρ : func
    '''
    def TE_ECladdingρ(ρ):
        '''
        Returns the radial component of TE_ECladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_ECladdingρ : real
        '''
        return 0
    return TE_ECladdingρ

def TE_ECladdinggenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_ECladdingϕ : func
    '''
    def TE_ECladdingϕ(ρ):
        '''
        Returns the azimuthal component of TE_ECladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_ECladdingϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return -(special.kn(1,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a)/special.kn(1,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return TE_ECladdingϕ

def TE_ECladdinggenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_ECladdingz : func
    '''
    def TE_ECladdingz(ρ):
        '''
        Returns the transverse component of TE_ECladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_ECladdingz : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return 0
    return TE_ECladdingz


def TM_ECladdinggenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_ECladdingρ : func
    '''
    def TM_ECladdingρ(ρ):
        '''
        Returns the radial component of TM_ECladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_ECladdingρ : real
        '''
        return (nCore**2*special.kn(1,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a))/(nCladding**2*special.kn(1,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return TM_ECladdingρ

def TM_ECladdinggenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_ECladdingϕ : func
    '''
    def TM_ECladdingϕ(ρ):
        '''
        Returns the azimuthal component of TM_ECladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_ECladdingϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return 0
    return TM_ECladdingϕ

def TM_ECladdinggenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_ECladdingz : func
    '''
    def TM_ECladdingz(ρ):
        '''
        Returns the transverse component of TM_ECladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_ECladdingz : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return ((-1j)*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*nCore**2*special.kn(0,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a))/(a*kz*nCladding**2*special.kn(1,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return TM_ECladdingz


def HE_ECladdinggenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_ECladdingρ : func
    '''
    def HE_ECladdingρ(ρ):
        '''
        Returns the radial component of HE_ECladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_ECladdingρ : real
        '''
        return -0.5*(a*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a)*(-1 - (2*kFree**2*m*(nCladding**2 - nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))/(a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(kz**2 - kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))) - (1 - (2*kFree**2*m*(nCladding**2 - nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))/(a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(kz**2 - kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))))*special.kn(1 + m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a)))/(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return HE_ECladdingρ

def HE_ECladdinggenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_ECladdingϕ : func
    '''
    def HE_ECladdingϕ(ρ):
        '''
        Returns the azimuthal component of HE_ECladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_ECladdingϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return -0.5*(a*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a)*(-1 - (2*kFree**2*m*(nCladding**2 - nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))/(a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(kz**2 - kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))) + (1 - (2*kFree**2*m*(nCladding**2 - nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))/(a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(kz**2 - kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))))*special.kn(1 + m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a)))/(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return HE_ECladdingϕ

def HE_ECladdinggenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_ECladdingz : func
    '''
    def HE_ECladdingz(ρ):
        '''
        Returns the transverse component of HE_ECladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_ECladdingz : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return ((-1j)*np.sqrt(-kz**2 + kFree**2*nCore**2)*special.kn(m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a))/(kz*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return HE_ECladdingz


def TE_HCoregenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_HCoreρ : func
    '''
    def TE_HCoreρ(ρ):
        '''
        Returns the radial component of TE_HCore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_HCoreρ : real
        '''
        return (kz*special.jv(1,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ))/(kFree*special.jv(1,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))
    return TE_HCoreρ

def TE_HCoregenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_HCoreϕ : func
    '''
    def TE_HCoreϕ(ρ):
        '''
        Returns the azimuthal component of TE_HCore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_HCoreϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return 0
    return TE_HCoreϕ

def TE_HCoregenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_HCorez : func
    '''
    def TE_HCorez(ρ):
        '''
        Returns the transverse component of TE_HCore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_HCorez : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return (1j*np.sqrt(-kz**2 + kFree**2*nCore**2)*special.jv(0,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ))/(kFree*special.jv(1,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))
    return TE_HCorez


def TM_HCoregenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_HCoreρ : func
    '''
    def TM_HCoreρ(ρ):
        '''
        Returns the radial component of TM_HCore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_HCoreρ : real
        '''
        return 0
    return TM_HCoreρ

def TM_HCoregenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_HCoreϕ : func
    '''
    def TM_HCoreϕ(ρ):
        '''
        Returns the azimuthal component of TM_HCore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_HCoreϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return (kFree*nCore**2*special.jv(1,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ))/(kz*special.jv(1,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))
    return TM_HCoreϕ

def TM_HCoregenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_HCorez : func
    '''
    def TM_HCorez(ρ):
        '''
        Returns the transverse component of TM_HCore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_HCorez : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return 0
    return TM_HCorez


def HE_HCoregenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_HCoreρ : func
    '''
    def HE_HCoreρ(ρ):
        '''
        Returns the radial component of HE_HCore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_HCoreρ : real
        '''
        return (kFree*nCore**2*(special.jv(-1 + m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)*(-1 + (-((a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))) + (nCladding**2*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(nCore**2*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(2.*kFree**2*m*(nCladding**2 - nCore**2))) - special.jv(1 + m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)*(1 + (-((a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))) + (nCladding**2*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(nCore**2*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(2.*kFree**2*m*(nCladding**2 - nCore**2)))))/(2.*kz*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))
    return HE_HCoreρ

def HE_HCoregenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_HCoreϕ : func
    '''
    def HE_HCoreϕ(ρ):
        '''
        Returns the azimuthal component of HE_HCore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_HCoreϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return -0.5*(kFree*nCore**2*(special.jv(-1 + m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)*(-1 + (-((a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))) + (nCladding**2*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(nCore**2*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(2.*kFree**2*m*(nCladding**2 - nCore**2))) + special.jv(1 + m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)*(1 + (-((a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))) + (nCladding**2*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(nCore**2*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(2.*kFree**2*m*(nCladding**2 - nCore**2)))))/(kz*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))
    return HE_HCoreϕ

def HE_HCoregenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_HCorez : func
    '''
    def HE_HCorez(ρ):
        '''
        Returns the transverse component of HE_HCore without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_HCorez : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return (2j*kFree*m*(nCladding**2 - nCore**2)*special.jv(m,np.sqrt(-kz**2 + kFree**2*nCore**2)*ρ)*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))/(a*(kz**2 - kFree**2*nCladding**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) - np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*np.sqrt(-kz**2 + kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))
    return HE_HCorez


def TE_HCladdinggenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_HCladdingρ : func
    '''
    def TE_HCladdingρ(ρ):
        '''
        Returns the radial component of TE_HCladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_HCladdingρ : real
        '''
        return (kz*special.kn(1,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a))/(kFree*special.kn(1,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return TE_HCladdingρ

def TE_HCladdinggenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_HCladdingϕ : func
    '''
    def TE_HCladdingϕ(ρ):
        '''
        Returns the azimuthal component of TE_HCladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_HCladdingϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return 0
    return TE_HCladdingϕ

def TE_HCladdinggenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TE_HCladdingz : func
    '''
    def TE_HCladdingz(ρ):
        '''
        Returns the transverse component of TE_HCladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TE_HCladdingz : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return ((-1j)*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*special.kn(0,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a))/(a*kFree*special.kn(1,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return TE_HCladdingz


def TM_HCladdinggenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_HCladdingρ : func
    '''
    def TM_HCladdingρ(ρ):
        '''
        Returns the radial component of TM_HCladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_HCladdingρ : real
        '''
        return 0
    return TM_HCladdingρ

def TM_HCladdinggenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_HCladdingϕ : func
    '''
    def TM_HCladdingϕ(ρ):
        '''
        Returns the azimuthal component of TM_HCladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_HCladdingϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return (kFree*nCore**2*special.kn(1,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a))/(kz*special.kn(1,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return TM_HCladdingϕ

def TM_HCladdinggenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    TM_HCladdingz : func
    '''
    def TM_HCladdingz(ρ):
        '''
        Returns the transverse component of TM_HCladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        TM_HCladdingz : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return 0
    return TM_HCladdingz


def HE_HCladdinggenρ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_HCladdingρ : func
    '''
    def HE_HCladdingρ(ρ):
        '''
        Returns the radial component of HE_HCladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_HCladdingρ : real
        '''
        return (a*kFree*nCore**2*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a)*(-(nCladding**2/nCore**2) + (-((a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))) + (nCladding**2*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(nCore**2*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(2.*kFree**2*m*(nCladding**2 - nCore**2))) + (nCladding**2/nCore**2 + (-((a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))) + (nCladding**2*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(nCore**2*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(2.*kFree**2*m*(nCladding**2 - nCore**2)))*special.kn(1 + m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a)))/(2.*kz*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return HE_HCladdingρ

def HE_HCladdinggenϕ(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_HCladdingϕ : func
    '''
    def HE_HCladdingϕ(ρ):
        '''
        Returns the azimuthal component of HE_HCladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> -sin(mϕ)
          odd  -> cos(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_HCladdingϕ : real

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return -0.5*(a*kFree*nCore**2*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a)*(-(nCladding**2/nCore**2) + (-((a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))) + (nCladding**2*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(nCore**2*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(2.*kFree**2*m*(nCladding**2 - nCore**2))) - (nCladding**2/nCore**2 + (-((a*(kz**2 - kFree**2*nCladding**2)*np.sqrt(-kz**2 + kFree**2*nCore**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))))/special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))) + (nCladding**2*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*(-kz**2 + kFree**2*nCore**2)*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(nCore**2*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))/(2.*kFree**2*m*(nCladding**2 - nCore**2)))*special.kn(1 + m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a)))/(kz*np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))))
    return HE_HCladdingϕ

def HE_HCladdinggenz(a,kFree,kz,m,nCladding,nCore):

    '''
    Parameters
    ----------
    a : float
        radius of the waveguide core.
    kFree : float)
        free space wavenumber.
    kz : float
        longitudinal propagation constant.
    m : int
        mode order, for HE modes m>=1, for TM and TE, m=0.
    nCladding : float
        refractive index of cladding.
    nCore : float
        refractive index of core.

    Reference
    ---------
    Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.

    Returns
    -------
    HE_HCladdingz : func
    '''
    def HE_HCladdingz(ρ):
        '''
        Returns the transverse component of HE_HCladding without the azimuthal dependence.
        The azimuthal dependence depends on the parity of the field:
          even -> cos(mϕ)
          odd  -> sin(mϕ)

        Parameters
        ----------
        ρ : float
            The radial coordinate.

        Returns
        -------
        HE_HCladdingz : complex

        Reference
        ---------
        Snyder, Allan W, and John Love. Optical Waveguide Theory, 1983.
        '''
        return (2j*kFree*m*(nCladding**2 - nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*special.kn(m,(np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*ρ)/a))/(a*(kz**2 - kFree**2*nCladding**2)*(special.jv(-1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)) - special.jv(1 + m,a*np.sqrt(-kz**2 + kFree**2*nCore**2)))*special.kn(m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) - np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))*np.sqrt(-kz**2 + kFree**2*nCore**2)*special.jv(m,a*np.sqrt(-kz**2 + kFree**2*nCore**2))*(special.kn(-1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2))) + special.kn(1 + m,np.sqrt(a**2*(kz**2 - kFree**2*nCladding**2)))))
    return HE_HCladdingz


def fieldGenerator(a,kFree,kz,m,nCladding,nCore,fieldType='HE'):
    if fieldType == 'HE':
        assert m>=1, 'Not an HE mode'
        ECoreρ = HE_ECoregenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECoreϕ = HE_ECoregenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECorez = HE_ECoregenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECladdingρ = HE_ECladdinggenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECladdingϕ = HE_ECladdinggenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECladdingz = HE_ECladdinggenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCoreρ = HE_HCoregenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCoreϕ = HE_HCoregenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCorez = HE_HCoregenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCladdingρ = HE_HCladdinggenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCladdingϕ = HE_HCladdinggenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCladdingz = HE_HCladdinggenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        Efuncs = (ECoreρ, ECoreϕ, ECorez, ECladdingρ, ECladdingϕ, ECladdingz)
        Hfuncs = (HCoreρ, HCoreϕ, HCorez, HCladdingρ, HCladdingϕ, HCladdingz)
    elif fieldType == 'TE':
        assert m==0, 'Not a TE mode'
        ECoreρ = TE_ECoregenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECoreϕ = TE_ECoregenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECorez = TE_ECoregenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECladdingρ = TE_ECladdinggenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECladdingϕ = TE_ECladdinggenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECladdingz = TE_ECladdinggenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCoreρ = TE_HCoregenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCoreϕ = TE_HCoregenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCorez = TE_HCoregenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCladdingρ = TE_HCladdinggenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCladdingϕ = TE_HCladdinggenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCladdingz = TE_HCladdinggenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        Efuncs = (ECoreρ, ECoreϕ, ECorez, ECladdingρ, ECladdingϕ, ECladdingz)
        Hfuncs = (HCoreρ, HCoreϕ, HCorez, HCladdingρ, HCladdingϕ, HCladdingz)
    elif fieldType == 'TM':
        ECoreρ = TM_ECoregenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECoreϕ = TM_ECoregenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECorez = TM_ECoregenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECladdingρ = TM_ECladdinggenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECladdingϕ = TM_ECladdinggenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        ECladdingz = TM_ECladdinggenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCoreρ = TM_HCoregenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCoreϕ = TM_HCoregenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCorez = TM_HCoregenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCladdingρ = TM_HCladdinggenρ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCladdingϕ = TM_HCladdinggenϕ(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        HCladdingz = TM_HCladdinggenz(a=a, kFree=kFree, kz=kz, m=m, nCladding=nCladding, nCore=nCore)
        Efuncs = (ECoreρ, ECoreϕ, ECorez, ECladdingρ, ECladdingϕ, ECladdingz)
        Hfuncs = (HCoreρ, HCoreϕ, HCorez, HCladdingρ, HCladdingϕ, HCladdingz)
    return Efuncs, Hfuncs
