tomodrgn cleanup
===========================


Purpose
--------
Clean an analyzed ``train_vae`` output directory of various types of outputs.

Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline
    tomodrgn \
        cleanup \
        output/vae_both_sim_zdim2_copy_cleaned \
        --weights \
        --zfiles \
        --volumes \
        --test

    # actually remove files
    tomodrgn \
        cleanup \
        output/vae_both_sim_zdim2_copy_cleaned \
        --weights \
        --zfiles \
        --volumes

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.cleanup.add_args
   :prog: cleanup
   :nodescription:
   :noepilog:

Common next steps
------------------
# TODO
