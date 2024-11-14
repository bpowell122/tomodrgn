tomodrgn convergence_vae
===========================

Purpose
--------
Evaluate convergence of a ``train_vae`` model by stability of latent space and reconstructed volumes.

Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs
    tomodrgn \
        convergence_vae \
        output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 \
        --final-maxima 2 \
        --ground-truth data/10076_class*_32.mrc

    # WarpTools style inputs
    tomodrgn \
        convergence_vae \
        output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8 \
        --final-maxima 2 \
        --ground-truth data/warptools_test_box-32_angpix-12_reconstruct.mrc

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.convergence_vae.add_args
   :prog: convergence_vae
   :nodescription:
   :noepilog:



Common next steps
------------------

* Extend model training with ``tomodrgn train_vae [...] --load latest`` if not yet converged
* Analyze model at a particular epoch in latent space with ``tomodrgn analyze``
* Analyze model at a particular epoch in volume space with ``tomodrgn analyze_volumes``
