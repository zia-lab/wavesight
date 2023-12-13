.. image:: ./img/wavesight_banner.jpg
   :align: center

`wavesight` is a project to study the coupling of electromagnetic waves into optical fibers. It is written in Python and uses the `numpy` and `scipy` libraries for numerical calculations. It uses `MEEP <https://meep.readthedocs.io>`_ as an implementation of FDTD to solve the propagation of waves across metasurfaces, and to launch the guided modes of step index fibers. It implements the analytical solution to step-index optical waveguides and it uses `S^4 <https://web.stanford.edu/group/fan/S4/>`_ as an RCWA implementation useful in the design of metasurfaces. 

In addition to this it also uses the Smythe-Kirchhoff vectorial diffraction integral to propagate electromagnetic fields across homogeneous media that may compose a part of a larger structure. 

This was run at a HPC cluster at Brown University and includes a number of convenience to manage and launch scheduled jobs using `slurm <https://hpc-uit.readthedocs.io/en/latest/jobs/slurm_parameter.html>`_.

From these calculations `wavesight` also provides a set of tools to visualize the results of the simulations. It uses `matplotlib` to plot the results of the simulations, and `ffmpeg` to generate animations of the results. These results are logged to a private Slack channel.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   /usage/across_ml.rst
   /usage/convstore.rst
   /usage/datapipes.rst
   /usage/diffkernels.rst
   /usage/EH4_animate.rst
   /usage/EH4_plotter.rst
   /usage/EH4_to_EH5.rst
   /usage/EH5_assembly.rst
   /usage/fiber_animate.rst
   /usage/fiber_bridge.rst
   /usage/fiber_bundle.rst
   /usage/fiber_platform.rst
   /usage/fiber_plotter.rst
   /usage/fieldgenesis.rst
   /usage/fields.rst
   /usage/fungenerators.rst
   /usage/hail_david.rst
   /usage/housekeeping.rst
   /usage/maxwell.rst
   /usage/meta_designer.rst
   /usage/metal.rst
   /usage/metaorchestra.rst
   /usage/misc.rst
   /usage/ml_plot_in.rst
   /usage/phase_factory.rst
   /usage/printech.rst
   /usage/slacking.rst
   /usage/sniffer.rst
   /usage/standard_candle.rst
   /usage/templates.rst
   /usage/wavegraphics.rst
   /usage/wavesight.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
