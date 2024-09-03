Learn structural heterogeneity
===============================

Purpose
--------

Learning structural heterogeneity from a dataset is the core purpose of tomoDRGN.
Once you have confirmed that your particles have been extracted correctly, training a single tomoDRGN network should enable parsing of significant heterogeneity.
Additional, iterative rounds of model training on progressively-smaller particle subsets are possible for deeper analyses.
As with the homogeneous reconstructions, all commands on this page benefit from access to a machine with a GPU available for computation, and sufficient RAM to hold all particles in memory.

Train heterogeneous tomoDRGN model
-----------------------------------

We begin by training a tomoDRGN model on all particles, as done with the homogeneous reconstructions.
The full list of command line arguments can be found here [TODO ref to train_vae documentation].
On our testing machine, this command took XXXXX hours.

.. code-block:: python

    tomodrgn train_vae \
        particleseries.star \
        -o 03_heterogeneity-1_train_vae \
        --datadir .../path/to/particleseries \
        --recon-dose-weight \
        --recon-tilt-weight \
        --l-dose-mask \
        --enc-layers-A 3 \
        --enc-dim-A 256 \
        --out-dim-A 128 \
        --enc-layers-B 3 \
        --enc-dim-B 256 \
        --zdim 128 \
        --dec-layers 3 \
        --dec-dim 512 \
        -n 50

This command produces several outputs in the ``03_heterogeneity-1_train_vae`` directory:

* ``config.pkl``: this is a configuration file containing all parameters (specified and defaults) to work with this model again via downstream commands such as ``eval_vol``

* ``run.log``: this contains (most of) the output produced by the program. We are currently reworking logging, and recommend instead redirecting STDOUT to a log file as the more complete log history (``run.log`` is a subset of STDOUT)

* ``weights.*.pkl``

* ``reconstruct.*.mrc``


Interpreting outputs
---------------------
tomodrgn analyze (latent space)

* summary of heterogeneity,
* latent heterogeneity (l-PCA, l-UMAP)

tomodrgn analyze_volumes (volume space)

* reconstructed heterogeneity
* volume heterogeneity (v-PCA-UMAP, k?? volumes, pc traversals



Assessing model convergence and overfitting
--------------------------------------------
convergence_vae interpretation and options to continue training
overfitting signs and options to use earlier epochs



Systematic, model-guided assessment of heterogeneity: MAVEn
-------------------------------------------------------------



Systematic, model-free inspection of heterogeneity: SIREn and MAVEn
---------------------------------------------------------------------

