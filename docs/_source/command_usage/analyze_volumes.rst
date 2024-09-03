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
# TODO
