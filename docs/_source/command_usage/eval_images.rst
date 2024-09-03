tomodrgn eval_images
===========================

Purpose
--------
Embed images to latent space using a pretrained ``train_vae`` model (i.e. evaluating encoder modules only).

Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline
    tomodrgn \
        eval_images \
        data/10076_classE_32_sim.star \
        --weights output/vae_classE_sim_zdim8/weights.pkl \
        -c output/vae_classE_sim_zdim8/config.pkl \
        --out-z output/vae_classE_sim_zdim8/eval_images/z_all.pkl

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.eval_images.add_args
   :prog: eval_images
   :nodescription:
   :noepilog:

Common next steps
------------------
# TODO
