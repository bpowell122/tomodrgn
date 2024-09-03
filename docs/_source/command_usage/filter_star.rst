tomodrgn filter_star
===========================

Purpose
--------
Filter a star file ("image series" or "volume series") by selected particle indices.

Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline TiltSeriesStarfile
    tomodrgn \
        filter_star data/10076_both_32_sim.star \
        --starfile-type imageseries \
        --tomo-id-col _rlnImageName \
        --ind data/ind_ptcl_first10last10.pkl \
        -o output/10076_both_32_sim_filtered.star

    # baseline volumeseries star file
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
# TODO
