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
If you set ``--uninvert-data`` on the previous page's homogeneous reconstruction, set it again here.
The full list of command line arguments can be found :doc:`here <../command_usage/train_vae>`.

.. code-block:: python

    tomodrgn train_vae \
        imageseries.star \
        -o 03_heterogeneity-1_train_vae \
        --datadir .../path/to/particleseries/images \
        --recon-dose-weight \
        --recon-tilt-weight \
        --l-dose-mask \
        --num-epochs 50

This command produces several outputs in the ``03_heterogeneity-1_train_vae`` directory:

* ``config.pkl``: this is a configuration file containing all parameters (specified and defaults) to work with this model again via downstream tomoDRGN commands.
* ``run.log``: this contains (most of) the output produced by the program. We are currently reworking logging, and recommend instead redirecting STDOUT to a log file as the more complete log history (``run.log`` is a subset of STDOUT).
* ``weights.*.pkl``: these files, generated once per epoch according to ``train_vae ...  --checkpoint`` interval, contain the learned model weights at the named epoch. These weight files are used to resume model training from a checkpoint, or to load a model for downstream latent/volume space analysis.
* ``z.*.pkl``: these files, generated at the same interval as ``weights.*.pkl``, contain the learned latent embeddings for each particle as a numpy array of shape (``num_particles, latent_dimensionality``)


Interpreting outputs
---------------------
Unlike the homogeneous reconstruction steps previously, here the outputs do not include files that can be immediately viewed (e.g., plots of latent space, volumes that can be viewed in ChimeraX, etc.).
Instead, the trained model encoding your dataset's structural heterogeneity can be analyzed in several different ways.
See the next page on analyzing learned structural heterogeneity to learn about various tools that can be used to gain insight into the structural landscape of your dataset.


Assessing model convergence and overfitting
--------------------------------------------
The tool ``tomodrgn convergence_vae`` can be used to learn about the model's training dynamics, and to gain a sense of whether training has converged.
The full list of command line arguments can be found :doc:`here <../command_usage/convergence_vae>`.

.. code-block:: python

    tomodrgn convergence_vae \
        03_heterogeneity-1_train_vae

The outputs of this command include the following:

* plots

  - the model's loss (reconstructed image error, KL divergence from the standard normal distributed latent, and total loss) at each epoch
  - PCA-projected latent space at every few epochs
  - UMAP-projected latent space at every few epochs
  - the mean relative change in latent embeddings for each particle in successive epochs
  - a sampling of well-supported regions of latent space (i.e., highly populated neighborhoods of latent space) and the correlations between volumes generated from these neighborhoods every few epochs as evaluated by volumetric correlation coefficient and by FSC

* a number of supporting files involved in generating these plots, if you desire to take a closer look at the model's training dynamics

Generally, we expect a well-converged model to exhibit (relative) stabilization in most of these plots, though not all plots may fully stabilize.
We suggest that further downstream analysis should be performed on the model at the epoch at which stabilization appears to begin, so as to minimize the potential for overfitting artifacts.

Overfitting is periodically observed, and is generally characterized by

* a loss curve that descends significantly with no visible improvement in map quality, map heterogeneity, or latent space distribution
* maps that exhibit streaks along a particular axis (similar to preferred orientation issues), or exhibit increasing amounts of high frequency noise

In the event that your model is not yet converged, you can resume training with the same ``train_vae`` command as above, appended with ``--load latest`` to resume training from the most recent model checkpoint (a ``weights.*.pkl`` file).


Common pitfalls
----------------

* model is overfit / underfit: see the section above
* NaN or otherwise non-decreasing reconstruction loss
  - Reconstruction (aka MSE aka generative) loss should generally decrease from about 1 - 1.5 (at the start of training) to 0.8 - 0.9 (at the end of training). Other loss curve behavior (increasing loss, NaN loss, wildly oscillating loss) is usually a result of the learning rate being too high leading to unstable training. The learning rate is set to a low default value, but thereafter increases as the square root of the batch size, and may also need tuning for different datasets. We recommend that you increase the ``--batch-size`` until your GPU is saturated (i.e. you get a "CUDA out of memory error", typically batch sizes 1-16 depending on particle box size), then decrease the learning rate ``--lr`` until the reconstruction loss behaves as described above.
* the latent space looks like a homogeneous blob:
  - the model may have nonetheless learned structural heterogeneity! We frequently observe that conformational heterogeneity (and even small-scale compositional heterogeneity) results in relatively continuous latent spaces. See the next page for analyzing structural heterogeneity.
  - if no structural heterogeneity has been learned, then try training a new model with a decreased beta value (``--beta``) to decrease the impact of KL regularization, and/or an increased latent dimensionality (``--zdim``) to provide a larger latent space.
