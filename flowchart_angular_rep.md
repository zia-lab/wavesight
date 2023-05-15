# Farfield approximation via the angular-spectrum representation

Given a scalar field in a plane $(z=Z_0)$, sampled in a square grid with a spatial resolution $\delta$, calculate the field at a different plane $(z=Z)$ with $k(Z-Z_0) >> 1 $ at the same spatial resolution $\delta$ but in a larger (or smaller) spatial domain. The calculated field should satisfy criteria for conservation of energy. 

This calculation assumes the field to be monochromatic (with angular frequency $\omega$). Even though in here the details are only given for the propagation of a scalar wave, this also applies to vector waves, in that each component should be propagated as if it was a scalar wave.

```

┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                                                                          │
│                                                                          │
│                                                                          │
│                                                                          │
│                                                                          │
│        z=Zf                                                              │
│       ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╳ ─ ─        │
│                                                            ╱             │
│                                                           ╱              │
│                                                          ╱               │
│                                                         ╱                │
│                                                        ╱                 │
│                                                       ╱                  │
│                                                      ╱                   │
│                                                     ╱                    │
│                                                    ╱                     │
│                                                   ╱                      │
│                                                  ╱                       │
│                                                 ╱                        │
│                                                ╱  R                      │
│                                               ╱                          │
│                                              ╱                           │
│                                             ╱                            │
│                                            ╱                             │
│                                           ╱                              │
│                                          ╱                               │
│                                         ╱                                │
│                                        ╱                                 │
│                                       ╱                                  │
│                                      ╱                                   │
│                                     ╱                                    │
│                       ─ ─ ─ ─ ─ ─ ─╱─ ─ ─ ─ ─  z=Zi                      │
│                                                                          │
│                                                                          │
│                                                                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

```

```mermaid
flowchart TB

NearField["Given:
Ei : field at Z=Z_0
ω : angular frequency of field
Δi : sampling resolution of Ei
Δf : sampling resolution of Ef
si : width and height of near field
sf : width and height of far  field
n  : ref index of medium"]

NearField --> CoordLayout["[#CoordinateLayout]
Given si and sf determine:
Ni : ceil(si/Δi)
Nf : ceil(sf/Δf)
xi (np.array (Ni))
yi (np.array (Ni))
xf (np.array (Nf))
yf (np.array (Nf))
Xf (np.array (Nf, Nf))
Yf (np.array (Nf, Nf))
"]

CoordLayout --> Precalc["[#PreCalc]:
Rf (np.array (Nf, Nf)):
XfoRf (np.array (Nf, Nf)): Xf/Rf
YfoRf (np.array (Nf, Nf)): Yf/Rf
S1 (np.array (Nf, Nf)): 2 π i / k₀ * ((Zf - Zi) / r) * eⁱᵏʳ/r"]

CoordLayout & NearField --> FourierTransform["[#FFT]
Calculate:
Ei(kx, ky)
kx (np.array (1,Ni)) 
ky (np.array (1,Ni)) 
Kx (np.array (Ni, Ni))
Ky (np.array (Ni, Ni))
"]

FourierTransform --> AngularSpectrum["[#Ang Spectrum]
Calculate:
S2 (np.array (Nf, Nf)): interpolate Ei over XfoRf and YfoRf
"]

AngularSpectrum & Precalc --> FinalField["[#Assemble]:
Ef (np.array (Nf, Nf)):  S1 * S2
"]

style FinalField fill:darkred
```
