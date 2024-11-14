tomodrgn backproject_voxel
===========================

Purpose
--------
Reconstruct a 3-D volume from pre-aligned 2-D tilt-series projections via weighted back projection.
CTF correction is performed by phase flipping only.


Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs
    tomodrgn \
        backproject_voxel \
        data/10076_classE_32_sim.star \
        --output output/00_backproject/classE_sim.mrc \
        --uninvert-data \
        --recon-dose-weight

    # WarpTools style inputs
    tomodrgn \
        backproject_voxel \
        data/warptools_test_4-tomos_10-ptcls_box-32_angpix-12_optimisation_set.star \
        --output output/backproject/warptools_70S_doseweight.mrc \
        --uninvert-data \
        --flip \   # note: flip is used because the handedness of this toy dataset is inverted
        --recon-dose-weight

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.backproject_voxel.add_args
   :prog: backproject_voxel
   :nodescription:
   :noepilog:


Common next steps
------------------

* Backproject a different particle subset (e.g. using ``--ind-ptcls``) to validate structural heterogeneity in a subset of particles identified by tomoDRGN's decoder network
