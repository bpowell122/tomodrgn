tomodrgn subtomo2chimerax
===========================

Purpose
--------
Generate ChimeraX command script to visualize selected particles in the spatial context of the source tomogram. Particles may be visualized as tomoDRGN ``train_vae``-generated unique volumes, a single "consensus" volume, or spherical markers.

Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # mode markers
    tomodrgn \
        subtomo2chimerax \
        data/10076_both_32_sim_vols.star \
        --mode markers \
        --outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_markers \
        --tomoname both.tomostar \
        --star-angpix-override 10 \
        --coloring-labels data/ptcl_labels_D0_E1.pkl

    # mode volume
    tomodrgn \
        subtomo2chimerax \
        data/10076_both_32_sim_vols.star \
        -mode volume \
        --outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volume \
        --tomoname both.tomostar \
        --star-angpix-override 10 \
        --vol-path output/00_backproject/classE_sim_doseweight.mrc \
        --vol-render-level 0.7 \
        --coloring-labels data/ptcl_labels_D0_E1.pkl

    # mode volumes
    tomodrgn \
        subtomo2chimerax \
        data/10076_both_32_sim_vols.star \
        --mode volumes \
        --outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volumes \
        --tomoname both.tomostar \
        --star-angpix-override 10 \
        --weights output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/weights.39.pkl \
        --config output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/config.pkl \
        --zfile output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl \
        --vol-render-level 0.7 \
        --coloring-labels data/ptcl_labels_D0_E1.pkl

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.subtomo2chimerax.add_args
   :prog: subtomo2chimerax
   :nodescription:
   :noepilog:

Common next steps
------------------
# TODO
