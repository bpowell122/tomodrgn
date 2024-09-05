tomodrgn train_vae
===========================

Purpose
--------
Train a heterogeneous tomoDRGN network (i.e. encoder and decoder modules) to learn an embedding of pre-aligned 2-D tilt-series projections to a continuous latent space, and to learn to generate unique 3-D reconstructions consistent with input images given the corresponding latent embedding.


Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline, 2 class heterogeneity, zdim 2
    tomodrgn \
        train_vae \
        data/10076_both_32_sim.star \
        --outdir output/vae_both_sim_zdim2 \
        --zdim 2 \
        --uninvert-data \
        --seed 42 \
        --log-interval 100 \
        --enc-dim-A 64 \
        --enc-layers-A 2 \
        --out-dim-A 64 \
        --enc-dim-B 32 \
        --enc-layers-B 4 \
        --dec-dim 256 \
        --dec-layers 3 \
        --num-epochs 40

    # 2 class heterogeneity, zdim 8, dose weight, tilt wieght, recon dose mask
    tomodrgn \
        train_vae \
        data/10076_both_32_sim.star \
        --outdir output/vae_both_sim_zdim8_dosetiltweightmask \
        --zdim 8 \
        --uninvert-data \
        --seed 42 \
        --log-interval 100 \
        --enc-dim-A 64 \
        --enc-layers-A 2 \
        --out-dim-A 64 \
        --enc-dim-B 32 \
        --enc-layers-B 4 \
        --dec-dim 256 \
        --dec-layers 3 \
        -n 1 \
        --l-dose-mask \
        --recon-dose-weight \
        --recon-tilt-weight

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
