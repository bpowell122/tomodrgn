tomodrgn eval_vol
===========================

Purpose
--------
Generate volumes from corresponding latent embeddings using a pretrained ``train_vae`` model (i.e. evaluating decoder module only).


Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs
    tomodrgn \
        eval_vol \
        --weights output/vae_both_sim_zdim2/weights.pkl \
        -c output/vae_both_sim_zdim2/config.pkl \
        -o output/vae_both_sim_zdim2/eval_vol_allz \
        --zfile output/vae_both_sim_zdim2/z.train.pkl \
        -b 32

    # WarpTools style inputs
    tomodrgn \
        eval_vol \
        --weights output/vae_warptools_70S_zdim2/weights.pkl \
        -c output/vae_warptools_70S_zdim2/config.pkl \
        -o output/vae_warptools_70S_zdim2/eval_vol_allz \
        --zfile output/vae_warptools_70S_zdim2/z.train.pkl \
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

* Analyze a volume ensemble using dimensionality reduction with ``tomodrgn analyze_volumes``
* Use external tools such as `MAVEn <https://github.com/lkinman/MAVEn>`_ or  `SIREn <https://github.com/lkinman/SIREn>`_ to systematically quantify structural heterogeneity either with or without an atomic model to guide analysis
