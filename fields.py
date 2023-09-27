#!/usr/bin/env python3

import numpy as np

def electric_dipole(kFree, nref, θdip, ϕdip, η, pmag, ζCoords, ηCoords, fields='EH'):
    '''
    This  function returns the E and H fields produced by
    an electric  dipole  radiator  at frequency :math:`ω = k_\\text{Free} c` in a
    given plane.

    .. math::

        \\begin{align}
            \\vec{E} &= e^{ikr} \\left(\\frac{k^2}{r} (\\hat{r}\\times\\vec{p})\\times \\hat{r} + \\frac{\\left(1-ikr\\right)}{r^3}\\left(3 \\hat{r}(\\hat{r}\\cdot\\vec{p}) - \\vec{p}\\right)\\right)  \\\\
            \\vec{H} &= -Z_n k^2 \\frac{e^{ikr}}{r} \\left(1 + \\frac{i}{kr}\\right) \\left( \\hat{r} \\times \\vec{p} \\right) \\\\
            Z_n &= \\frac{1}{n} \\\\
            \\hat{r} &= \\frac{\\vec{r}}{r} \\\\
            c &= 1 \\\\
            k &= k_{free} n
        \\end{align}

    ::

        ┌───────────────────────────────────────────────────────┐
        │                                                       │
        │                               ηCoords                 │
        │                          ┌─────────────┬─             │
        │                        ┌.┘           ┌─┘              │
        │                   ▲  ┌─┘│          ┌─┘                │
        │                   │┌─┘           ┌─┘  ζCoords         │
        │                  ┌┼┘    │      ┌─┘                    │
        │                 ─┴┼────────────┘                      │
        │                   │     │                             │
        │                   │                                   │
        │                   η                                   │
        │                   │     │                             │
        │                   │                                   │
        │                   │     │  θdip   ┌▶                  │
        │                   │             ┌─┘                   │
        │                   │     │     ┌─┘  │p                 │
        │                   │         ┌─┘                       │
        │                   │     │ ┌─┘      │                  │
        │                   │     ┌─┘                           │
        │           ─ ─ ─ ─ ▼ ─ ──┼ ─ ─ ─ ─ ─│▶  y              │
        │                     ┌ ┘  └ ┐                          │
        │                    ─    │   ─      │                  │
        │                 ┌ ┘          └ ┐                      │
        │                ─       φdip     ─  │                  │
        │           x ┌ ┘                  └ ─                  │
        │            ◀            │                             │
        │                                                       │
        └───────────────────────────────────────────────────────┘

    Parameters
    ----------
    kFree : float
        free-space wavenumber.

    nref : float
        refractive index of the medium.

    θdip, ϕdip : float
        dipole orientation angles.

    η : float
        vertical distance between dipole and plane.

    pmag : float
        dipole moment magnitude.

    ζCoords : np.array (N,)
        x-coordinates of the target plane.

    ηCoords : np.array
        (N,) y-coordinates of the target plane.

    field : str
        'E', 'H', or 'EH' for the field  components to return.

    Returns
    -------
    Efield : np.array (3, N, N)
        electric field at the given
        plane,  with  the  first  dimension indexing the x, y, and z
        components of the field.

    Hfield : np.array (3, N, N)
        H-field at the given plane,
        with the first dimension indexing the x, y, and z components
        of the field.
    
    Reference
    ---------
    Jackson, David. Classical Electrodynamics, 1999, equation 9.18.
    '''
    assert fields in ['E','H','EH','HE'], "fields must be 'E', 'H', 'EH', or 'HE'"
    assert nref >= 1
    k      = kFree * nref
    pdip   = pmag*np.array([np.sin(θdip)*np.cos(ϕdip),
                            np.sin(θdip)*np.sin(ϕdip),
                            np.cos(θdip)]) 
    numSamples = len(ζCoords)
    ζmesh, ηmesh = np.meshgrid(ζCoords, ηCoords)
    rmesh = np.sqrt(ζmesh**2 + ηmesh**2 + η**2)
    rmeshinverse = 1./rmesh
    rhat  = np.array([ζmesh, ηmesh, η*np.ones((numSamples,numSamples))]) * rmeshinverse

    pdipVec = np.zeros((3,numSamples,numSamples), dtype=np.complex128)
    pdipVec[0] = pdip[0]
    pdipVec[1] = pdip[1]
    pdipVec[2] = pdip[2]
    if 'E' in fields:
        ncrossp = np.cross(rhat, pdip, axis=0)
        Evec1  = (k**2 
                * np.cross(ncrossp, rhat, axis=0)
                * rmesh**2)
        Evec2 = ((3 * rhat * np.sum(rhat * pdipVec, axis=0) - pdipVec) 
                * (1. - 1.j * k * rmesh))
        phaser  = np.exp(1j * k * rmesh)
        Efield  = (Evec1 + Evec2) * phaser * rmeshinverse**3
    if 'H' in fields:
        if 'ncrossp' not in locals():
            ncrossp = np.cross(rhat, pdip, axis=0)
        if 'phaser' not in locals():
            phaser  = np.exp(1j * k * rmesh)
        Hfield = 1. / nref * k**2 * ncrossp * phaser * rmeshinverse * (1 + 1j * rmeshinverse / k)
    if ('E' in fields) and ('H' in fields):
        return Efield, Hfield
    elif fields == 'E':
        return Efield
    elif fields == 'H':
        return Hfield

def magnetic_dipole(kFree, nref, θdip, ϕdip, η, mmag, ζCoords, ηCoords, fields='EH'):
    '''
    This  function returns the E and H fields produced by
    a  magnetic  dipole  radiator  at frequency :math:`ω = k_\\text{Free} c` in a
    given plane.

    Assuming this specific form of  :doc:`maxwell` the fields
    for a magnetic dipole radiator are given by.

    .. math::

        \\begin{align}
            \\vec{H} &= e^{ikr} \\left(\\frac{k^2}{r} (\\hat{r}\\times\\vec{m})\\times \\hat{r} + \\frac{\\left(1-ikr\\right)}{r^3}\\left(3 \\hat{r}(\\hat{r}\\cdot\\vec{m}) - \\vec{m}\\right)\\right) \\\\
            \\vec{E} &= -Z_n k^2 \\frac{e^{ikr}}{r} \\left(1 + \\frac{i}{kr}\\right) \\left( \\hat{r} \\times \\vec{m} \\right) \\\\
            Z_n &= \\frac{1}{n} \\\\
            \\hat{r} &= \\frac{\\vec{r}}{r} \\\\
            c &= 1 \\\\
            k &= k_{free} n 
        \\end{align}

    ::

        ┌───────────────────────────────────────────────────────┐
        │                                                       │
        │                               ηCoords                 │
        │                          ┌─────────────┬─             │
        │                        ┌.┘           ┌─┘              │
        │                   ▲  ┌─┘│          ┌─┘                │
        │                   │┌─┘           ┌─┘  ζCoords         │
        │                  ┌┼┘    │      ┌─┘                    │
        │                 ─┴┼────────────┘                      │
        │                   │     │                             │
        │                   │                                   │
        │                   η                                   │
        │                   │     │                             │
        │                   │                                   │
        │                   │     │  θdip   ┌▶                  │
        │                   │             ┌─┘                   │
        │                   │     │     ┌─┘  │p                 │
        │                   │         ┌─┘                       │
        │                   │     │ ┌─┘      │                  │
        │                   │     ┌─┘                           │
        │           ─ ─ ─ ─ ▼ ─ ──┼ ─ ─ ─ ─ ─│▶  y              │
        │                     ┌ ┘  └ ┐                          │
        │                    ─    │   ─      │                  │
        │                 ┌ ┘          └ ┐                      │
        │                ─       φdip     ─  │                  │
        │           x ┌ ┘                  └ ─                  │
        │            ◀            │                             │
        │                                                       │
        └───────────────────────────────────────────────────────┘

    Parameters
    ----------
    kFree : (float)
        free-space wavenumber.

    nref : float
        refractive index of the medium.

    θdip, ϕdip : float
        dipole orientation angles.

    η : float
        vertical distance between dipole and plane.

    mmag : float
        dipole moment magnitude.

    ζCoords : np.array (N,)
        x-coordinates of the target plane.

    ηCoords : np.array (N,)
        y-coordinates of the target plane.

    field : str:
        'E', 'H', or 'EH' for the field components to return.

    Returns
    -------
    
    Efield  : np.array (3, N, N)
        electric field at the given plane,  with  the  first  
        dimension indexing the x, y, and z components of the field.

    Hfield  : np.array (3, N, N)
        H-field at the given plane, with the first dimension indexing
        the x, y, and z components of the field.
    
    Reference
    ---------
    Jackson, David. Classical Electrodynamics, 1999, equation 9.35.
    '''
    assert fields in ['E','H','EH','HE'], "fields must be 'E', 'H', 'EH', or 'HE'"
    assert nref >= 1
    k      = kFree * nref
    # impedance of medium assuming μ=1
    Zn     = 1./nref 
    mdip   = mmag*np.array([np.sin(θdip)*np.cos(ϕdip),
                            np.sin(θdip)*np.sin(ϕdip),
                            np.cos(θdip)]) 
    numSamples = len(ζCoords)
    ζmesh, ηmesh = np.meshgrid(ζCoords, ηCoords)
    rmesh = np.sqrt(ζmesh**2 + ηmesh**2 + η**2)
    rmeshinverse = 1./rmesh
    rhat  = np.array([ζmesh, ηmesh, η*np.ones((numSamples,numSamples))]) * rmeshinverse

    mdipVec = np.zeros((3,numSamples,numSamples), dtype=np.complex128)
    mdipVec[0] = mdip[0]
    mdipVec[1] = mdip[1]
    mdipVec[2] = mdip[2]
    if 'H' in fields:
        ncrossp = np.cross(rhat, mdip, axis=0)
        Hvec1  = (k**2 
                * np.cross(ncrossp, rhat, axis=0)
                * rmesh**2)
        Hvec2 = ((3 * rhat * np.sum(rhat * mdipVec, axis=0) - mdipVec) 
                * (1. - 1.j * k * rmesh))
        phaser  = np.exp(1j * k * rmesh)
        Hfield  = (Hvec1 + Hvec2) * phaser * rmeshinverse**3
    if 'E' in fields:
        if 'ncrossp' not in locals():
            ncrossp = np.cross(rhat, mdip, axis=0)
        if 'phaser' not in locals():
            phaser  = np.exp(1j * k * rmesh)
        Efield = -Zn * k**2 * ncrossp * phaser * rmeshinverse * (1 + 1j * rmeshinverse / k)
    if ('E' in fields) and ('H' in fields):
        return Efield, Hfield
    elif fields == 'E':
        return Efield
    elif fields == 'H':
        return Hfield


def plane_wave(kFree, nref, θk, ϕk, η, Eamp, Eϕ, xCoords, yCoords, fields='EH'):
    '''
    This function returns the E and H fields produced by a plane
    wave  at a given plane. The direction of the polarization of
    the  electric  field  is given by the angle Eϕ, which is the
    counter-clockwise  angle  between  a  unit  vector  that  is
    perpendicular  to  the direction of propagation and lying on
    the  x-y plane. The direction of the propagation is given by
    θk  and  ϕk.  The complex amplitude of the electric field is
    given by Eamp.

    .. image:: ../img/plane_wave_coords.png

    In   the   case   of   normal   incidence  the  unit  vector
    :math:`\hat{u}` is :math:`-\hat{x}`.

    Parameters
    ----------
    kFree : float
        free-space wavenumber.
    nref : float
        refractive index of the medium.
    θk, ϕk : float
        direction of propagation.
    η : float
        fields are returned at the plane z=η.
    Eamp : float
        amplitude of the electric field.
    Eϕ : float
        angle of the electric field polarization.
    xCoords : np.array (N,)
        x-coordinates of the target plane.
    yCoords : np.array (N,)
        y-coordinates of the target plane.
    field : str
        'E', 'H', or 'EH' for the field components to return.
    
    Returns
    -------
    Efield : np.array (3, N, N)
        electric  field  at  the  given  plane,  with  the first
        dimension  indexing  the  x,  y, and z components of the
        field.
    Hfield : np.array (3, N, N)
        H-field  at  the  given  plane, with the first dimension
        indexing the x, y, and z components of the field.
    '''
    μr = 1 # assume magnetic transparency
    k = kFree * nref
    kVec = k * np.array([np.sin(θk) * np.cos(ϕk),
                         np.sin(θk) * np.sin(ϕk),
                         np.cos(θk)])
    # unit vector in direction of propagation
    kDir = kVec / k
    kdirx, kdiry, kdirz = kDir
    # unit vector perpendicular to kDir and in the x-y plane
    if θk == 0:
        northPole = np.array([-1., 00., 0.])
    else:
        northPole = 1/np.sqrt(kdirx**2 + kdiry**2) * np.array([-kdiry, kdirx, 0])
    # complementary unit vector perpendicular to kDir and northPole
    westPole  =  np.cross(kVec, northPole)
    # unit vector in the direction of polarization
    Epol = np.cos(Eϕ) * northPole + np.sin(Eϕ) * westPole
    xMesh, yMesh = np.meshgrid(xCoords, yCoords)
    phase =  np.exp(1j * kVec[0] * xMesh + 1j * kVec[1] * yMesh)
    phase *= np.exp(1j * kVec[2] * η)
    if 'E' in fields:
        Efield = np.zeros((3, len(xCoords), len(yCoords)), dtype=np.complex128)
        Efield[0] = Epol[0] * phase
        Efield[1] = Epol[1] * phase
        Efield[2] = Epol[2] * phase
        Efield *= Eamp
    if 'H' in fields:
        kcrossEpol = np.cross(kVec, Epol)
        Hfield = np.zeros((3, len(xCoords), len(yCoords)), dtype=np.complex128)
        Hfield[0] = kcrossEpol[0] * phase
        Hfield[1] = kcrossEpol[1] * phase
        Hfield[2] = kcrossEpol[2] * phase
        Hfield *= nref * Eamp / (k * np.sqrt(μr))
    if fields == 'E':
        return Efield
    elif fields == 'H':
        return Hfield
    elif fields in ['EH','HE']:
        return Efield, Hfield

