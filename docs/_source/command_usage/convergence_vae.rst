tomodrgn convergence_vae
===========================

Purpose
--------
Evaluate convergence of a ``train_vae`` model by stability of latent space and reconstructed volumes.

Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline, 2 class heterogeneity, zdim 2
    tomodrgn \
        convergence_vae \
        output/vae_both_sim_zdim2 \
        latest \
        --random-seed 42 \
        --final-maxima 2 \
        --ground-truth data/10076_class*_32.mrc

    # 2 class heterogeneity, zdim 8, dose weight, tilt wieght, recon dose mask, batchsize 8
    tomodrgn \
        convergence_vae \
        output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 \
        latest \
        --random-seed 42 \
        --final-maxima 2 \
        --ground-truth data/10076_class*_32.mrc

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
