Maxwell's Equations
========================

This is the following form of Maxwell's equations assumed throughout wavesight.

.. math:: 

    \begin{align}
        \nabla\cdot \vec{D} = 0 \\
        \nabla \times \vec{H} = \frac{\partial}{\partial t}\vec{D} \\
        \nabla \cdot \vec{B} = 0 \\
        \nabla\times{\vec{E}} = - \frac{\partial \vec{B}}{\partial t} \\
        \vec{D} = \epsilon_r \vec{E} \\
        \vec{H} = \frac{1}{\mu_r} \vec{B} 
    \end{align}

In these equations, :math:`\vec{D}` is the electric displacement field, :math:`\vec{H}` is the magnetic field, :math:`\vec{B}` is the magnetic flux density, and :math:`\vec{E}` is the electric field. :math:`\epsilon_r` and :math:`\mu_r` are the relative permittivity and permeability of the medium.

These are Maxwell's equations in SI for a homogeneous linear medium without free charges or current, having set both :math:`\mu_0` and :math:`\epsilon_0` to 1. 

Since here we are only concerned with electromagnetic waves at optical frequencies where matter is mostly magnetically transparent, we can assume :math:`\mu_r = 1`.