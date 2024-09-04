tomodrgn train_nn
=================

Purpose
--------
Train a homogeneous tomoDRGN network (i.e. decoder-only) to generate a "consensus" 3-D reconstruction from pre-aligned 2-D tilt-series projections.

Sample usage
------------
The examples below are taken from ``tomodrgn/testing/commandtest.py``, and rely on other outputs from ``commandtest.py`` to execute successfully.

.. code-block:: bash

    # baseline
    tomodrgn \
        train_nn data/10076_classE_32_sim.star \
        --outdir output/nn_classE_sim \
        --uninvert-data \
        --seed 42 \
        --layers 3 \
        --dim 256 \
        --num-epochs 40

    # lazy
    tomodrgn \
        train_nn \
        data/10076_classE_32_sim.star \
        --outdir output/nn_classE_sim_lazy \
        -uninvert-data \
        --seed 42 \
        --layers 3 \
        --dim 256 \
        --num-epochs 1 \
        --lazy

Arguments
---------

.. argparse::
   :ref: tomodrgn.commands.train_nn.add_args
   :prog: train_nn
   :nodescription:
   :noepilog:

Common next steps
------------------

* Assess model convergence with ``tomodrgn convergence_nn``
