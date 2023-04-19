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

def Et1gen_1(Ae, Ah, Be, Bh, m, β, γ, kz, λfree):
    def Et1(x,y):
        return ((np.cos(m*np.arctan2(y,x)) +  1j * np.sin(m*np.arctan2(y,x))) 
                * np.array([(1j*x*((2j*Ah*m*np.pi*special.jv(m,np.sqrt(x**2 + y**2)*γ))/(np.sqrt(x**2 + y**2)*λfree) + (Ae*kz*γ*(special.jv(-1 + m,np.sqrt(x**2 + y**2)*γ) - special.jv(1 + m,np.sqrt(x**2 + y**2)*γ)))/2.))/(np.sqrt(x**2 + y**2)*γ**2) + (y*((2j*Ah*np.pi*np.sqrt(x**2 + y**2)*γ*special.jv(-1 + m,np.sqrt(x**2 + y**2)*γ))/λfree + 2*Ae*kz*m*special.jv(m,np.sqrt(x**2 + y**2)*γ) - (2j*Ah*np.pi*np.sqrt(x**2 + y**2)*γ*special.jv(1 + m,np.sqrt(x**2 + y**2)*γ))/λfree))/(2.*(x**2 + y**2)*γ**2),(1j*y*((2j*Ah*m*np.pi*special.jv(m,np.sqrt(x**2 + y**2)*γ))/(np.sqrt(x**2 + y**2)*λfree) + (Ae*kz*γ*(special.jv(-1 + m,np.sqrt(x**2 + y**2)*γ) - special.jv(1 + m,np.sqrt(x**2 + y**2)*γ)))/2.))/(np.sqrt(x**2 + y**2)*γ**2) - (x*((2j*Ah*np.pi*np.sqrt(x**2 + y**2)*γ*special.jv(-1 + m,np.sqrt(x**2 + y**2)*γ))/λfree + 2*Ae*kz*m*special.jv(m,np.sqrt(x**2 + y**2)*γ) - (2j*Ah*np.pi*np.sqrt(x**2 + y**2)*γ*special.jv(1 + m,np.sqrt(x**2 + y**2)*γ))/λfree))/(2.*(x**2 + y**2)*γ**2)]))
    return Et1


def Et2gen_1(Ae, Ah, Be, Bh, m, β, γ, kz, λfree):
    def Et2(x,y):
        return ((np.cos(m*np.arctan2(y,x)) +  1j * np.sin(m*np.arctan2(y,x))) 
                * np.array([(0.5j*x*(Be*kz*np.sqrt(x**2 + y**2)*β*special.kn(-1 + m,np.sqrt(x**2 + y**2)*β) - (4j*Bh*m*np.pi*special.kn(m,np.sqrt(x**2 + y**2)*β))/λfree + Be*kz*np.sqrt(x**2 + y**2)*β*special.kn(1 + m,np.sqrt(x**2 + y**2)*β)))/((x**2 + y**2)*β**2) + (0.5j*y*((2*Bh*np.pi*np.sqrt(x**2 + y**2)*β*special.kn(-1 + m,np.sqrt(x**2 + y**2)*β))/λfree + 2j*Be*kz*m*special.kn(m,np.sqrt(x**2 + y**2)*β) + (2*Bh*np.pi*np.sqrt(x**2 + y**2)*β*special.kn(1 + m,np.sqrt(x**2 + y**2)*β))/λfree))/((x**2 + y**2)*β**2),(0.5j*y*(Be*kz*np.sqrt(x**2 + y**2)*β*special.kn(-1 + m,np.sqrt(x**2 + y**2)*β) - (4j*Bh*m*np.pi*special.kn(m,np.sqrt(x**2 + y**2)*β))/λfree + Be*kz*np.sqrt(x**2 + y**2)*β*special.kn(1 + m,np.sqrt(x**2 + y**2)*β)))/((x**2 + y**2)*β**2) - (0.5j*x*((2*Bh*np.pi*np.sqrt(x**2 + y**2)*β*special.kn(-1 + m,np.sqrt(x**2 + y**2)*β))/λfree + 2j*Be*kz*m*special.kn(m,np.sqrt(x**2 + y**2)*β) + (2*Bh*np.pi*np.sqrt(x**2 + y**2)*β*special.kn(1 + m,np.sqrt(x**2 + y**2)*β))/λfree))/((x**2 + y**2)*β**2)]))
    return Et2


def Ht1gen_1(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, n1):
    def Ht1(x,y):
        return ((np.cos(m*np.arctan2(y,x)) +  1j * np.sin(m*np.arctan2(y,x))) 
                * np.array([(x*((2*Ae*m*n1**2*np.pi*special.jv(m,np.sqrt(x**2 + y**2)*γ))/(np.sqrt(x**2 + y**2)*γ**2*λfree) + (0.5j*Ah*kz*(special.jv(-1 + m,np.sqrt(x**2 + y**2)*γ) - special.jv(1 + m,np.sqrt(x**2 + y**2)*γ)))/γ))/np.sqrt(x**2 + y**2) - (1j*y*((1j*Ah*kz*m*special.jv(m,np.sqrt(x**2 + y**2)*γ))/np.sqrt(x**2 + y**2) + (Ae*n1**2*np.pi*γ*(special.jv(-1 + m,np.sqrt(x**2 + y**2)*γ) - special.jv(1 + m,np.sqrt(x**2 + y**2)*γ)))/λfree))/(np.sqrt(x**2 + y**2)*γ**2),(y*((2*Ae*m*n1**2*np.pi*special.jv(m,np.sqrt(x**2 + y**2)*γ))/(np.sqrt(x**2 + y**2)*γ**2*λfree) + (0.5j*Ah*kz*(special.jv(-1 + m,np.sqrt(x**2 + y**2)*γ) - special.jv(1 + m,np.sqrt(x**2 + y**2)*γ)))/γ))/np.sqrt(x**2 + y**2) + (1j*x*((1j*Ah*kz*m*special.jv(m,np.sqrt(x**2 + y**2)*γ))/np.sqrt(x**2 + y**2) + (Ae*n1**2*np.pi*γ*(special.jv(-1 + m,np.sqrt(x**2 + y**2)*γ) - special.jv(1 + m,np.sqrt(x**2 + y**2)*γ)))/λfree))/(np.sqrt(x**2 + y**2)*γ**2)]))
    return Ht1


def Ht2gen_1(Ae, Ah, Be, Bh, m, β, γ, kz, λfree, n2):
    def Ht2(x,y):
        return ((np.cos(m*np.arctan2(y,x)) +  1j * np.sin(m*np.arctan2(y,x))) 
                * np.array([(0.5j*x*(Bh*kz*np.sqrt(x**2 + y**2)*β*special.kn(-1 + m,np.sqrt(x**2 + y**2)*β) + (4j*Be*m*n2**2*np.pi*special.kn(m,np.sqrt(x**2 + y**2)*β))/λfree + Bh*kz*np.sqrt(x**2 + y**2)*β*special.kn(1 + m,np.sqrt(x**2 + y**2)*β)))/((x**2 + y**2)*β**2) - (0.5j*y*((2*Be*n2**2*np.pi*np.sqrt(x**2 + y**2)*β*special.kn(-1 + m,np.sqrt(x**2 + y**2)*β))/λfree - 2j*Bh*kz*m*special.kn(m,np.sqrt(x**2 + y**2)*β) + (2*Be*n2**2*np.pi*np.sqrt(x**2 + y**2)*β*special.kn(1 + m,np.sqrt(x**2 + y**2)*β))/λfree))/((x**2 + y**2)*β**2),(0.5j*y*(Bh*kz*np.sqrt(x**2 + y**2)*β*special.kn(-1 + m,np.sqrt(x**2 + y**2)*β) + (4j*Be*m*n2**2*np.pi*special.kn(m,np.sqrt(x**2 + y**2)*β))/λfree + Bh*kz*np.sqrt(x**2 + y**2)*β*special.kn(1 + m,np.sqrt(x**2 + y**2)*β)))/((x**2 + y**2)*β**2) + (0.5j*x*((2*Be*n2**2*np.pi*np.sqrt(x**2 + y**2)*β*special.kn(-1 + m,np.sqrt(x**2 + y**2)*β))/λfree - 2j*Bh*kz*m*special.kn(m,np.sqrt(x**2 + y**2)*β) + (2*Be*n2**2*np.pi*np.sqrt(x**2 + y**2)*β*special.kn(1 + m,np.sqrt(x**2 + y**2)*β))/λfree))/((x**2 + y**2)*β**2)]))
    return Ht2


