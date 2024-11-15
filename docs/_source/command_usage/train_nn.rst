tomodrgn train_nn
=================

Purpose
--------
Train a homogeneous tomoDRGN network (i.e. decoder-only) to generate a "consensus" 3-D reconstruction from pre-aligned 2-D tilt-series projections.

Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs
    tomodrgn \
        train_nn \
        data/10076_classE_32_sim.star \
        --source-software cryosrpnt \  # tomoDRGN tries to automatically infer the software used to export particles, but allows this value to be set explicitly
        --outdir output/nn_classE_sim \
        --uninvert-data \
        --num-epochs 40 \
        --l-dose-mask \
        --recon-dose-weight \
        --recon-tilt-weight

    # WarpTools style inputs
    tomodrgn \
        train_nn \
        data/warptools_test_4-tomos_10-ptcls_box-32_angpix-12_optimisation_set.star \
        --outdir output/nn_warptools_70S_dosetiltweightmask \
        --uninvert-data \
        --num-epochs 40 \
        --l-dose-mask \
        --recon-dose-weight \
        --recon-tilt-weight \
        --lazy \  # note: lazy is used because the separate .mrcs file per particle, as used by WarpTools, is well suited to lazy loading
        --num-workers 2 \  # note: num-workers, prefetch-factor, and persistent-workers are best used only if lazy is enabled to avoid excessive memory utilization
        --prefetch-factor 2 \
        --persistent-workers

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.train_nn.add_args
   :prog: train_nn
   :nodescription:
   :noepilog:

Common next steps
------------------

* Assess model convergence with ``tomodrgn convergence_nn``
