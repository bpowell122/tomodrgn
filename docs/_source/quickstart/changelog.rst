Changelog
=========

* v0.2.1

  - New features

    * Added ``--dose-override`` option when loading images to manually specify dose in e-/A2/image
    * Added ``--starfile-source`` option to ``filter_star.py`` to allow filtering RELION3.1-format star files from RELION
    * Added ``--write-tiltseries-starfile`` option to ``downsample.py`` to write RELION3.0-format star file to downsampled images
    * Added multiple options for pooling tilt images between ``train_vae.py`` encoder A and encoder B per particle
    * Exposed pytorch dataloader options to user in ``train_vae.py``, ``train_nn.py``, ``eval_images.py``
    * ``convergence_nn.py`` saves FSCs calculated per epoch to pkl file
    * Added ``--batch-size`` to ``eval_vol.py`` calling parallelized ``models.FTPositionalDecoder.gen_volume_batch`` for accelerated volume generation
    * Added ``--vol-path`` to ``subtomo2chimerax.py`` allowing use of consensus reconstruction instead of unique tomodrgn volumes
    * Added new ``quicktest.py`` and ``unittest.py`` interface
    * Added ``load`` method using ``config.pkl`` file to easily create ``dataset.TiltSeriesMRCData`` object

  - Changes

    * Faster dataset loading with improved file I/O
    * Dataset preprocessing refactored to numpy arrays
    * Decreased memory utilization in ``dataset.TiltSeriesMRCData`` preprocessing
    * Decreased memory utilization in ``starfile.TiltSeriesStarfile`` initialization
    * Decreased memory utilization by gaussian positional encoding
    * Changed default positional encoding to ``gaussian``
    * Changed default latent dimensionality to ``--zdim 128``
    * ``train_vae.py`` checks for NaN/inf during checkpoint evaluation
    * ``utils.calc_fsc`` allows input volumes to be either paths to .mrc files or direct numpy arrays
    * New plots in ``convergence_vae.py`` including all loss types, all-to-all volumetric correlation coefficients among tomodrgn (and optionally ground truth) volumes
    * Changed default ``subtomo2chimerax.py`` color map to ChimeraX color scheme
    * Added requirement for pytorch>=1.11

  - Bugfixes

    * Fixed syntax of internal calls to ``eval_vol.py``
    * Fixed dataset index filtering of pre-filtered datasets
    * Added assert for convergence scripts that requisite input files exist
    * Fixed bug where ``config.pkl`` containing CUDA tensors could not be loaded on cpu
    * Fixed bug where precalculated ``norm`` would be ignored when loading dataset from ``config.pkl`` with ``--lazy``

* v0.2.0

  - Features

    * added support for datasets with variable tilts per tomogram (tilt counts/schemes/etc)
    * new scripts ``subtomo2chimerax`` and ``filter_star``
    * additional features and checks in ``tomoDRGN_viz+filt.ipynb``
    * ``eval_images`` supports ``z.pkl`` output; ``eval_vol`` supports ``z.pkl`` input
    * new tests ``unittest.sh`` and ``quicktest.sh``
    * validated compatibility with python 3.7 - 3.10 and pytorch 1.8 - 1.12

  - Changes

    * refactored the majority of codebase for explicit internal tomoDRGN compatibility and some performance improvements
    * updated tomoDRGN install requirements and changed to ``pip``-based installation
    * ``--amp`` is now enabled by default and can be disabled with ``--no-amp``
    * ``--do-dose-weighting``, ``--do-tilt-weighting``, ``--dose-mask`` have been renamed to ``--recon-dose-weight``, ``--recon-tilt-weight``, ``--l-dose-mask``, respectively
    * ``tomodrgn downsample`` now benefits from pytorch-based GPU acceleration
    * updated training hyperparameter defaults
    * (many) various bugfixes

* v0.1.0
    - initial tomoDRGN release