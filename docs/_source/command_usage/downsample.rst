tomodrgn downsample
===========================


Purpose
--------
Downsample an image stack or volume by Fourier cropping.
Note that where possible, it is preferred to re-extract particles in the appropriate upstream processing software at the desired box size, rather than downsampling here.


Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs
    tomodrgn \
        downsample \
        data/10076_classE_32_sim.star \
        --source-software cryosrpnt \
        --downsample 16 \
        --batch-size 50 \
        --output output/10076_classE_16_sim.mrcs \
        --write-tiltseries-starfile \
        --lazy

    # WarpTools style inputs
    tomodrgn \
        downsample \
        data/warptools_test_4-tomos_10-ptcls_box-32_angpix-12_optimisation_set.star \
        --downsample 16 \
        --batch-size 50 \
        --output output/warptools_70S_box-16.mrcs \
        --write-tiltseries-starfile \
        --lazy

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.downsample.add_args
   :prog: downsample
   :nodescription:
   :noepilog:

Common next steps
------------------

* Validate that downsampling produced the desired downsampled particles with ``tomodrgn backproject_voxel`` or ``tomodrgn train_nn``
