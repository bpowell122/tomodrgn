tomodrgn eval_images
===========================

Purpose
--------
Embed images to latent space using a pretrained ``train_vae`` model (i.e. evaluating encoder modules only).

Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs
    tomodrgn \
        eval_images data/10076_classE_32_sim.star \
        --source-software cryosrpnt \
        --weights output/vae_classE_sim_zdim8/weights.pkl \
        -c output/vae_classE_sim_zdim8/config.pkl \
        --out-z output/vae_classE_sim_zdim8/eval_images/z_all.pkl

    # WarpTools style inputs
    tomodrgn \
        eval_images \
        data/warptools_test_4-tomos_10-ptcls_box-32_angpix-12_optimisation_set.star \
        --weights output/vae_warptools_70S_zdim8/weights.pkl \
        -c output/vae_warptools_70S_zdim8/config.pkl \
        --out-z output/vae_warptools_70S_zdim8/eval_images/z_all.pkl

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.eval_images.add_args
   :prog: eval_images
   :nodescription:
   :noepilog:

Common next steps
------------------

* Analyze model at a particular epoch in latent space with ``tomodrgn analyze``
* Analyze model at a particular epoch in volume space with ``tomodrgn analyze_volumes``
* Generate volumes for all particles at a particular epoch with ``tomodrgn eval_vol``
* Map back generated volumes (for all particles) to source tomograms to explore spatially contextualized heterogeneity with ``tomodrgn subtomo2chimerax``
