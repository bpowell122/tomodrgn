tomodrgn train_vae
===========================

Purpose
--------
Train a heterogeneous tomoDRGN network (i.e. encoder and decoder modules) to learn an embedding of pre-aligned 2-D tilt-series projections to a continuous latent space, and to learn to generate unique 3-D reconstructions consistent with input images given the corresponding latent embedding.


Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs
    tomodrgn \
        train_vae \
        data/10076_both_32_sim.star \
        --source-software cryosrpnt \  # tomoDRGN tries to automatically infer the software used to export particles, but allows this value to be set explicitly
        --outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 \
        --zdim 8 \
        --uninvert-data \
        --num-epochs 40 \
        --l-dose-mask \
        --recon-dose-weight \
        --recon-tilt-weight \
        --batch-size 8

    # WarpTools style inputs
    tomodrgn \
        train_vae \
        data/warptools_test_4-tomos_10-ptcls_box-32_angpix-12_optimisation_set.star \
        --outdir output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8 \
        --zdim 8 \
        --uninvert-data
        --num-epochs 40 \
        --l-dose-mask \
        --recon-dose-weight \
        --recon-tilt-weight \
        --batch-size 8 \
        --lazy \  # note: lazy is used because the separate .mrcs file per particle, as used by WarpTools, is well suited to lazy loading
        --num-workers 2 \  # note: num-workers, prefetch-factor, and persistent-workers are best used only if lazy is enabled to avoid excessive memory utilization
        --prefetch-factor 2 \
        --persistent-workers

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.train_vae.add_args
   :prog: train_vae
   :nodescription:
   :noepilog:

Common next steps
------------------

* Assess model convergence with ``tomodrgn convergence_vae``
* Analyze model at a particular epoch in latent space with ``tomodrgn analyze``
* Analyze model at a particular epoch in volume space with ``tomodrgn analyze_volumes``
* Generate volumes for all particles at a particular epoch with ``tomodrgn eval_vol``
* Embed a (potentially related) dataset of images into the learned latent space with ``tomodrgn eval_images``
* Map back generated volumes (for all particles) to source tomograms to explore spatially contextuallized heterogeneity with ``tomodrgn subtomo2chimerax``
