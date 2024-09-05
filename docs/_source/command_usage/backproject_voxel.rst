tomodrgn backproject_voxel
===========================

Purpose
--------
Reconstruct a 3-D volume from pre-aligned 2-D tilt-series projections via weighted back projection.


Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline
    tomodrgn \
        backproject_voxel \
        data/10076_classE_32_sim.star \
        --output output/00_backproject/classE_sim.mrc \
        --uninvert-data

    # dose weight
    tomodrgn \
        backproject_voxel \
        data/10076_classE_32_sim.star \
        --output output/00_backproject/classE_sim_doseweight.mrc \
        --uninvert-data \
        --recon-dose-weight

    # manual lowpass filter
    tomodrgn \
        backproject_voxel \
        data/10076_classE_32_sim.star \
        --output output/00_backproject/classE_sim_lowpass60.mrc \
        --uninvert-data \
        --lowpass 60

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.backproject_voxel.add_args
   :prog: backproject_voxel
   :nodescription:
   :noepilog:


Common next steps
------------------

* Backproject a different particle subset (e.g. using ``--ind-ptcls``) to validate structural heterogeneity visualized by tomoDRGN's decoder network
* Use backprojections to create initial models or masks for further particle refinement in STA software including RELION and M