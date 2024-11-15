tomodrgn cleanup
===========================


Purpose
--------
Clean an analyzed ``train_vae`` output directory of various types of outputs.
Any model outputs that have been analyzed (i.e., those for which an ``analyze.EPOCH`` or ``convergence.EPOCH`` directory exist) will not be removed.

Sample usage
------------
The examples below are adapted from ``tomodrgn/testing/commandtest*.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # Warp v1 style inputs -- dry run
    tomodrgn \
        cleanup \
        output/vae_classE_sim_zdim8 \
        --weights \
        --zfiles \
        --volumes \
        --test

    # Warp v1 style inputs -- actually remove files
    tomodrgn \
        cleanup \
        output/vae_classE_sim_zdim8 \
        --weights \
        --zfiles \
        --volumes \
        --test

    # WarpTools style inputs -- dry run
    tomodrgn \
        cleanup output/vae_warptools_70S_zdim8 \
        --weights \
        --zfiles \
        --volumes \
        --test

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.cleanup.add_args
   :prog: cleanup
   :nodescription:
   :noepilog:

Common next steps
------------------

* Share the resulting cleaned directory with collaborators, or upload to Zenodo (or another data sharing service) as a component of data availability for publications
