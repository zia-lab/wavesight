import numpy as np
from scipy import special

def Ezgen_1(Ae, m, γ):
    def Ez(ρ):
        return Ae * special.jv(m, ρ * γ)
    return Ez

def Ezgen_2(Be, m, β):
    def Ez(ρ):
        return Be * special.kn(m, ρ * β)
    return Ez

def Hzgen_1(Ah, m, γ):
    def Hz(ρ):
        return Ah * special.jv(m, ρ * γ)
    return Hz

def Hzgen_2(Bh, m, β):
    def Hz(ρ):
        return Bh * special.kn(m, ρ * β)
    return Hz

def Et1genρ(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCladding, nCore):
    def Et1ρ(ρ):
        '''
        Returns the radial component of Et1 sans the global phase factor e(imϕ).
        Parameters
        ----------
        ρ : float
            The radial coordinate.
        Returns
        -------
        Et1ρ : complex
        '''
        return (1j*((2j*Ah*m*np.pi*special.jv(m,γ*ρ))/(λfree*ρ) + (Ae*kz*γ*(special.jv(-1 + m,γ*ρ) - special.jv(1 + m,γ*ρ)))/2.))/γ**2
    return Et1ρ

def Et1genϕ(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCladding, nCore):
    def Et1ϕ(ρ):
        '''
        Returns the azimuthal component of Et1 sans the global phase factor e(imϕ).
        Parameters
        ----------
        ρ : float
            The radial coordinate.
        Returns
        -------
        Et1ϕ : complex
        '''
        return -0.5*((2j*Ah*np.pi*γ*ρ*special.jv(-1 + m,γ*ρ))/λfree + 2*Ae*kz*m*special.jv(m,γ*ρ) - (2j*Ah*np.pi*γ*ρ*special.jv(1 + m,γ*ρ))/λfree)/(γ**2*ρ)
    return Et1ϕ


def Et2genρ(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCladding, nCore):
    def Et2ρ(ρ):
        '''
        Returns the radial component of Et2 sans the global phase factor e(imϕ).
        Parameters
        ----------
        ρ : float
            The radial coordinate.
        Returns
        -------
        Et2ρ : complex
        '''
        return (0.5j*(Be*kz*β*ρ*special.kn(-1 + m,β*ρ) - (4j*Bh*m*np.pi*special.kn(m,β*ρ))/λfree + Be*kz*β*ρ*special.kn(1 + m,β*ρ)))/(β**2*ρ)
    return Et2ρ

def Et2genϕ(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCladding, nCore):
    def Et2ϕ(ρ):
        '''
        Returns the azimuthal component of Et2 sans the global phase factor e(imϕ).
        Parameters
        ----------
        ρ : float
            The radial coordinate.
        Returns
        -------
        Et2ϕ : complex
        '''
        return (-0.5j*((2*Bh*np.pi*β*ρ*special.kn(-1 + m,β*ρ))/λfree + 2j*Be*kz*m*special.kn(m,β*ρ) + (2*Bh*np.pi*β*ρ*special.kn(1 + m,β*ρ))/λfree))/(β**2*ρ)
    return Et2ϕ


def Ht1genρ(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCladding, nCore):
    def Ht1ρ(ρ):
        '''
        Returns the radial component of Ht1 sans the global phase factor e(imϕ).
        Parameters
        ----------
        ρ : float
            The radial coordinate.
        Returns
        -------
        Ht1ρ : complex
        '''
        return (2*Ae*m*nCore**2*np.pi*special.jv(m,γ*ρ))/(γ**2*λfree*ρ) + (0.5j*Ah*kz*(special.jv(-1 + m,γ*ρ) - special.jv(1 + m,γ*ρ)))/γ
    return Ht1ρ

def Ht1genϕ(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCladding, nCore):
    def Ht1ϕ(ρ):
        '''
        Returns the azimuthal component of Ht1 sans the global phase factor e(imϕ).
        Parameters
        ----------
        ρ : float
            The radial coordinate.
        Returns
        -------
        Ht1ϕ : complex
        '''
        return (1j*((1j*Ah*kz*m*special.jv(m,γ*ρ))/ρ + (Ae*nCore**2*np.pi*γ*(special.jv(-1 + m,γ*ρ) - special.jv(1 + m,γ*ρ)))/λfree))/γ**2
    return Ht1ϕ


def Ht2genρ(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCladding, nCore):
    def Ht2ρ(ρ):
        '''
        Returns the radial component of Ht2 sans the global phase factor e(imϕ).
        Parameters
        ----------
        ρ : float
            The radial coordinate.
        Returns
        -------
        Ht2ρ : complex
        '''
        return (0.5j*(Bh*kz*β*ρ*special.kn(-1 + m,β*ρ) + (4j*Be*m*nCladding**2*np.pi*special.kn(m,β*ρ))/λfree + Bh*kz*β*ρ*special.kn(1 + m,β*ρ)))/(β**2*ρ)
    return Ht2ρ

def Ht2genϕ(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, nCladding, nCore):
    def Ht2ϕ(ρ):
        '''
        Returns the azimuthal component of Ht2 sans the global phase factor e(imϕ).
        Parameters
        ----------
        ρ : float
            The radial coordinate.
        Returns
        -------
        Ht2ϕ : complex
        '''
        return (0.5j*((2*Be*nCladding**2*np.pi*β*ρ*special.kn(-1 + m,β*ρ))/λfree - 2j*Bh*kz*m*special.kn(m,β*ρ) + (2*Be*nCladding**2*np.pi*β*ρ*special.kn(1 + m,β*ρ))/λfree))/(β**2*ρ)
    return Ht2ϕ


