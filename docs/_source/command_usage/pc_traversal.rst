tomodrgn pc_traversal
===========================

Purpose
--------
Generate volumes along specified principal components of latent space.

Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline
    tomodrgn \
        pc_traversal \
        output/vae_both_sim_zdim2/z.train.pkl \
        -o output/vae_both_sim_zdim2/pc_traversal

    # percentile spacing
    tomodrgn \
        pc_traversal \
        output/vae_both_sim_zdim2/z.train.pkl \
        -o output/vae_both_sim_zdim2/pc_traversal \
        --use-percentile-spacing

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.pc_traversal.add_args
   :prog: pc_traversal
   :nodescription:
   :noepilog:

Common next steps
------------------

* Validate the inferred latent space PC traversal by isolating indices of particles proximal to each trajectory path, and performing homogeneous reconstructions with ``tomodrgn backproject_voxel``
