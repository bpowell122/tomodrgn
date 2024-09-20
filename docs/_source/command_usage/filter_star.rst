tomodrgn filter_star
===========================

Purpose
--------
Filter a star file ("image series" or "volume series") by selected particle indices or by selected class labels.

Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # image series star file filtered by particle indices
    tomodrgn \
        filter_star data/10076_both_32_sim.star \
        --starfile-type imageseries \
        --tomo-id-col _rlnImageName \
        --ind data/ind_ptcl_first10last10.pkl \
        -o output/10076_both_32_sim_filtered.star

    # image series star file filtered by class labels
    tomodrgn \
        filter_star data/10076_both_32_sim.star \
        --starfile-type imageseries \
        --tomo-id-col _rlnImageName \
        --labels data/labels_D-0_E-1.pkl \
        --labels-sel 0 \
        -o output/10076_both_32_sim_filtered_by_labels.star

    # volume series star file filtered by particle indices
    tomodrgn \
        filter_star \
        data/10076_both_32_sim_vols.star \
        --starfile-type volumeseries \
        --tomo-id-col _rlnImageName \
        --ind data/ind_ptcl_first10last10.pkl \
        -o output/10076_both_32_sim_vols_filtered.star

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.filter_star.add_args
   :prog: filter_star
   :nodescription:
   :noepilog:

Common next steps
------------------

* Validate that the filtered particle subset is structurall homogeneous for the tomoDRGN-identified feature with ``tomodrgn backproject_voxel``
* Export this particle subset to external STA software including RELION and M
* Train a new model on this subset of particles with ``tomodrgn train_vae`` to explore residual heterogeneity
