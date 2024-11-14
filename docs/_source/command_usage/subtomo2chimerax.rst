tomodrgn subtomo2chimerax
===========================

Purpose
--------
Generate ChimeraX command script to visualize selected particles in the spatial context of the source tomogram.
Particles may be visualized as tomoDRGN ``train_vae``-generated unique volumes, a single "consensus" volume, or spherical markers.

Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs -- mapback particles as spherical markers
    tomodrgn \
        subtomo2chimerax \
        data/10076_both_32_sim_vols.star \
        --mode markers \
        --outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_markers \
        --tomoname both.tomostar \
        --star-angpix-override 10 \
        --coloring-labels data/ptcl_labels_D0_E1.pkl

    # Warp v1 style inputs -- mapback particles as consensus volume
    tomodrgn \
        subtomo2chimerax \
        data/10076_both_32_sim_vols.star \
        --mode volume \
        --outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volume \
        --tomoname both.tomostar \
        --star-angpix-override 10 \
        --vol-path output/backproject/classE_sim_doseweight.mrc \
        --vol-render-level 0.7 \
        --coloring-labels data/ptcl_labels_D0_E1.pkl

    # Warp v1 style inputs -- mapback particles as tomoDRGN-generated unique volumes
    tomodrgn \
        subtomo2chimerax \
        data/10076_both_32_sim_vols.star \
        --mode volumes \
        --outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volumes_cmaptab10 \
        --tomoname both.tomostar \
        --star-angpix-override 10 \
        --weights output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/weights.39.pkl \
        --config output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/config.pkl \
        --zfile output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl \
        --vol-render-level 0.7 \
        --coloring-labels data/ptcl_labels_D0_E1.pkl \
        --colormap tab10

    # WarpTools style inputs -- mapback particles as tomoDRGN-generated unique volumes
    tomodrgn \
        subtomo2chimerax \
        data/warptools_test_4-tomos_10-ptcls_box-32_angpix-12_optimisation_set.star \
        --mode volumes \
        --outdir output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8/mapback_volumes_cmaptab10 \
        --tomoname Unt_076.tomostar \
        --weights output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8/weights.39.pkl \
        --config output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8/config.pkl \
        --zfile output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl \
        --vol-render-level 0.7 \
        --coloring-labels output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8/analyze.39/kmeans20/labels.pkl \
        --colormap tab20

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.subtomo2chimerax.add_args
   :prog: subtomo2chimerax
   :nodescription:
   :noepilog:

Common next steps
------------------

* Quantify spatially contextualized heterogeneity trends using custom scripts or external software, perhaps involving tomogram segmentations
* Make cool images and/or movies in ChimeraX