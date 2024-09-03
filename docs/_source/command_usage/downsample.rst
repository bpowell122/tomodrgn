tomodrgn downsample
===========================


Purpose
--------
Downsample an image stack or volume by Fourier cropping.


Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline
    tomodrgn \
        downsample \
        data/10076_classE_32_sim.star \
        --downsample 16 \
        --batch-size 50 \
        --output output/10076_classE_16_sim.mrcs \
        -write-tiltseries-starfile

    # lazy loading
    tomodrgn \
        downsample \
        data/10076_classE_32_sim.star \
        --downsample 16 \
        --batch-size 50 \
        --output output/10076_classE_16_sim.mrcs \
        --write-tiltseries-starfile \
        --lazy

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.downsample.add_args
   :prog: downsample
   :nodescription:
   :noepilog:

Common next steps
------------------
# TODO
