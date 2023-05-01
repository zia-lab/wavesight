# Local-k approximation to refraction

Given a mode in the waveguide whose propagation is suddenly truncated by the end of the waveguide, find the fields right outside the fiber by using the direction of the Poynting vector as a local direction of propagation of the incident fields.

```

┌────────────────────────────────────────────────────────────────────────┐
│ ...................................................................... │
│ ...................................................................... │
│ ...............................┌──────┐............................... │
│ ...............................│E', H'│............................... │
│ ...............................└──────┘....................      ..... │
│ ...........................................................  nr  ..... │
│ ...........................................................      ..... │
│ XXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------XXXXXXXXXXXXXXXXXXXXXXXXXXXX │
│ XXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------XXXXXXXXXXXXXXXX        XXXX │
│ XXXXXXXXXXXXXXXXXXXXXXXXXXXX----┌────┐----XXXXXXXXXXXXXXXX n(x,y) XXXX │
│ XXXXXXXXXXXXXXXXXXXXXXXXXXXX----│E, H│----XXXXXXXXXXXXXXXX        XXXX │
│ XXXXXXXXXXXXXXXXXXXXXXXXXXXX----└────┘----XXXXXXXXXXXXXXXXXXXXXXXXXXXX │
│ XXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------XXXXXXXXXXXXXXXXXXXXXXXXXXXX │
│ XXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------XXXXXXXXXXXXXXXXXXXXXXXXXXXX │
│ XXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------XXXXXXXXXXXXXXXXXXXXXXXXXXXX │
│                                                                        │
│                                   │                                    │
│                                                                        │
│                                   │                                    │
│                                                                        │
│                                   ├┐                      ┌────        │
│                                    └─┐  θ            ┌────┘            │
│                                   │  └─┐        ┌────┘                 │
│                                        └─┐ ┌────┘                      │
│                                   │  ┌───┴─┘                  nr       │
│       ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬────┘─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─      │
│                               ┌─┘ │                        n(x,y)      │
│                             ┌─┘                                        │
│                           ┌┬┘     │                                    │
│                         ┌─┘└─┐                                         │
│                       ┌─┘    └─┐  │                                    │
│                     ┌─┘     β  └─┐                                     │
│                   ┌─┘            └┤                                    │
│                 ┌─┘                                                    │
│                ─┘                 │                                    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

```

```mermaid
flowchart TB

Field["Given:
E, H : fields of a certain mode
k_z  : propagation constant of mode
Δλ   : sampling resolution of arrays
a    : the radius of the waveguide core
b    : side of computational square
nCore  : ref index of core
nCladding : ref index of cladding
nFree : ref index of launching space"]

Field --> Coords["[#Coords-Calc]
Calculate coordinate layout including
a, b, Δs, xrange, yrange, ρrange, 
φrange, Xg, Yg, ρg, φg, nxy, crossMask
"]

Field & Coords  -->  Poynting["[#EXH-Calc]
Poynting Vector Field
<b>P</b>(x,y) = ½ Re(<b>E</b> x <b>H</b>*)"]

Poynting --> localk["[#normIncidentk-Calc]
Normalized incident k <br>k̂ := <b>k</b>/|k| ≈ <b>P</b>/|P|"]

localk --> Incident["[#β-Calc]
Angle of incidence across interface
β(x,y)"]

Incident & Coords -->  Refracted["[#θ-Calc]
Angles of refraction across interface
θ(x,y)"]

Incident & Field --> FresnelS["[#FresnelS-Calc]
Fresnel-S Coefficient
FS(x,y) = E_S'/E_S (S-pol)"]

Incident & Field --> FresnelP["[#FresnelP-Calc]
    Fresnel-P Coefficient<br>FP(x,y) = E_P'/E_P (P-pol)"]

localk --> Sdir["[#ζ-Calc]
    S-Normal vector field
    (Direction normal to plane of incidence)
    <b>ζ(x,y)<b/> = ẑ x k̂
    ζ̂(x,y) = <b>ζ<b/> / |ζ|"]

Sdir --> Spol["[#EincS-Calc]
S-component of E
E_S = <b>E</b> · ζ̂"]

Spol --> Ppol["[#EincP-Calc]
P-component of E
E_P = E - E_S"]

Ppol & FresnelP --> EPref["[#ErefP-Calc]
Refracted E (P-pol)
E_P' = E_P  FP(x,y)"]

Spol & FresnelS --> ESref["[#ErefS-Calc]
Refracted E (S-pol)
E_S' = E_S  FS(x,y)"]

EPref & ESref --> Eref["[#Eref-Calc]
The refracted electric field
E' = E_S + E_P"]

Coords & localk --> localkrefnorm["[#kref-Calc]
Normalized refracted k̂' 
<b>k̂'</b>"]

localkrefnorm & Eref --> Href["[#Href-Calc]
The refracted H-field<br>H' ≈ √(μ'ϵ')/µ' k̂' x E'"]

style Href fill:darkred
style Eref fill:darkred
```
