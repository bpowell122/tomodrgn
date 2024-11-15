tomodrgn filter_star
===========================

Purpose
--------
Filter a star file by selected particle indices or by selected class labels.

Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs -- image series star file filtered by particle indices
    tomodrgn \
        filter_star \
        data/10076_both_32_sim.star \
        --starfile-type imageseries \
        --tomo-id-col _rlnImageName \
        --ind data/ind_ptcl_first10last10.pkl \
        -o output/10076_both_32_sim_filtered.star

    # Warp v1 style inputs -- image series star file filtered by class labels
    tomodrgn \
        filter_star \
        data/10076_both_32_sim.star \
        --starfile-type imageseries \
        --tomo-id-col _rlnImageName \
        --labels data/labels_D-0_E-1.pkl \
        --labels-sel 0 \
        -o output/10076_both_32_sim_filtered_by_labels.star

    # Warp v1 style inputs -- volume series star file filtered by class labels
    tomodrgn \
        filter_star \
        data/10076_both_32_sim_vols.star \
        --starfile-type volumeseries \
        --tomo-id-col _rlnImageName \
        --labels data/labels_D-0_E-1.pkl \
        --labels-sel 0 1 \
        -o output/10076_both_32_sim_vols_filtered_by_labels.star

    # WarpTools style inputs -- filtered by class labels
    tomodrgn \
        filter_star \
        data/warptools_test_4-tomos_10-ptcls_box-32_angpix-12_optimisation_set.star \
        --starfile-type optimisation_set \
        --tomo-id-col _rlnTomoName \
        --labels output/vae_warptools_70S_zdim8_dosetiltweightmask_batchsize8/analyze.39/kmeans20/labels.pkl \
        --labels-sel 0 1 2 3 4 \
        -o output/warptools_70S_filtered_by_labels_optimisation_set.star

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.filter_star.add_args
   :prog: filter_star
   :nodescription:
   :noepilog:

Common next steps
------------------

* Validate that the filtered particle subset is structurally homogeneous for the tomoDRGN-identified feature with ``tomodrgn backproject_voxel``
* Export this particle subset to external STA software
* Train a new model on this subset of particles with ``tomodrgn train_vae`` to explore residual heterogeneity
