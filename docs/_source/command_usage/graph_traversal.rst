tomodrgn graph_traversal
===========================

Purpose
--------
Identify particle indices forming the most efficient path through latent space, while connecting specified starting and ending points by way of specified anchor points.

Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs
    tomodrgn \
        graph_traversal \
        output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl \
        --anchors 5 10 15 20 \
        -o output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/graph_traversal \
        --max-neighbors 20 \
        --avg-neighbors 20

    # WarpTools style inputs
    tomodrgn \
        graph_traversal \
        output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl \
        --anchors 5 10 15 20 \
        -o output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8/graph_traversal \
        --max-neighbors 20 \
        --avg-neighbors 20

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.graph_traversal.add_args
   :prog: graph_traversal
   :nodescription:
   :noepilog:

Common next steps
------------------

* Validate the inferred latent space graph traversal by isolating indices of particles proximal to each neighbor point or anchor point along the path, and performing homogeneous reconstructions with ``tomodrgn backproject_voxel`` or external STA software
