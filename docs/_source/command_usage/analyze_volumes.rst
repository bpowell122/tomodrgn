tomodrgn analyze_volumes
===========================


Purpose
--------
Run standard volume-space analyses of a ``train_vae`` model: dimensionality reduction and clustering of a volume ensemble.

Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline
    tomodrgn \
        analyze_volumes \
        --voldir output/vae_both_sim_zdim2/eval_vol_allz \
        --config output/vae_both_sim_zdim2/config.pkl \
        --outdir output/vae_both_sim_zdim2/eval_vol_allz_analyze_volumes_mask_sphere \
        --ksample 20 \
        --mask sphere

    # soft mask (unique per vol)
    tomodrgn \
        analyze_volumes \
        --voldir output/vae_both_sim_zdim2/eval_vol_allz \
        --config output/vae_both_sim_zdim2/config.pkl \
        --outdir output/vae_both_sim_zdim2/eval_vol_allz_analyze_volumes_mask_soft \
        --ksample 20 \
        --mask soft

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.analyze_volumes.add_args
   :prog: analyze_volumes
   :nodescription:
   :noepilog:

Common next steps
------------------

* Interactively explore correlations between and spatial context of star file parameters, latent embeddings, volume space dimensionality reduction in the ``tomodrgn analyze`` Jupyter notebooks
* Identify one (or more) sets of particle indices whose particles share a common feature (e.g. in volume space)
* Filter the input star file by particle indices with ``tomodrgn filter_star``
* Generate an array of numeric labels describing a volume space property for each particle to color volumes in tomogram mapbacks with ``tomodrgn subtomo2chimerax``
