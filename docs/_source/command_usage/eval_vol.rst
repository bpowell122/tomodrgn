tomodrgn eval_vol
===========================

Purpose
--------
Generate volumes from corresponding latent embeddings using a pretrained ``train_vae`` model (i.e. evaluating decoder module only).


Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline, 2 class heterogeneity, zdim 2 -> eval batch size 1
    tomodrgn \
        eval_vol \
        --weights output/vae_both_sim_zdim2/weights.pkl \
        -c output/vae_both_sim_zdim2/config.pkl \
        -o output/vae_both_sim_zdim2/eval_vol_allz \
        --zfile output/vae_both_sim_zdim2/z.train.pkl

    # baseline, 2 class heterogeneity, zdim 2 -> eval batch size 32
    tomodrgn \
        eval_vol \
        --weights output/vae_both_sim_zdim2/weights.pkl \
        -c output/vae_both_sim_zdim2/config.pkl \
        -o output/vae_both_sim_zdim2/eval_vol_allz \
        --zfile output/vae_both_sim_zdim2/z.train.pkl \
        -b 32

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.eval_vol.add_args
   :prog: eval_vol
   :nodescription:
   :noepilog:

Common next steps
------------------
# TODO
