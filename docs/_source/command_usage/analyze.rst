tomodrgn analyze
===========================


Purpose
--------
Run standard analyses of a ``train_vae`` model: dimensionality reduction and clustering of latent space, and generation of corresponding volumes via the tomoDRGN decoder for further analysis.


Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline, 2 class heterogeneity, zdim 2
    tomodrgn \
        analyze \
        output/vae_both_sim_zdim2 39 \
        --ksample 20

    # 2 class heterogeneity, zdim 8, dose weight, tilt wieght, recon dose mask, batchsize 8
    tomodrgn \
        analyze \
        output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 39 \
        --ksample 20

    # 1 class heterogeneity, zdim 8
    tomodrgn \
        analyze \
        output/vae_classE_sim_zdim8 39 \
        --ksample 20

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.analyze.add_args
   :prog: analyze
   :nodescription:
   :noepilog:

Common next steps
------------------

* Interactively explore correlations between and spatial context of star file parameters, latent embeddings, volume space dimensionality reduction in the generated Jupyter notebooks
* Identify one (or more) sets of particle indices whose particles share a common feature (e.g. in latent space)
* Filter the input star file by particle indices with ``tomodrgn filter_star``
* Generate an array of numeric labels describing a latent space property for each particle to color volumes in tomogram mapbacks with ``tomodrgn subtomo2chimerax``
