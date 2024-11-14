Isolate particle subsets
=========================

Purpose
--------
Once structural heterogeneity has been analyzed, we typically want to isolate homogeneous subpopulations of particles.
Subsequent 3-D reconstructions with ``tomodrgn backproject_voxel`` or in traditional STA tools can validate the tomoDRGN-observed structural state, while subsequent 3-D refinements can improve the reconstruction resolution both globally (particularly if using tomoDRGN to filter junk particles) and locally (in the local region of the annotated heterogeneity).

Filtering a star file
----------------------
TomoDRGN requires as input an "image series" star file, in which each particle is described by multiple related rows which each describe a single image of that particle.
However, many STA software require metadata in a "volume series" star file format, in which each particle is described by a single row (more analogous to SPA).
Therefore, we recommend that both types of star file are generated when extracting particles (as necessary for your STA software), and note that tomoDRGN supports filtering both types of star files by the same particle indices.

.. tab-set::

   .. tab-item:: Filtering with indices.pkl

        Particle filtration is most often performed with an ``indices.pkl`` file which describes which particles should be retained for further analysis.
        The ``indices.pkl`` file contains a numpy array of shape ``(<= num_particles)`` of nonredundant integer-valued elements.
        Each element represents a particle to be retained, in the order and indexing of the input star file used for tomoDRGN analysis.

        The ``indices.pkl`` file can be generated in several ways:

        * ``tomoDRGN_viz+filt_legacy.ipynb`` jupyter notebook. This notebook contains extensive functionality to generate indices on the basis of desired clusters from latent space clustering, latent space outliers, interactive particle selection, and several additional approaches.
        * Outputs from MAVEn and SIREn can be used to identify which particles exhibit a structural feature of interest, perhaps defined as a specific occupancy pattern above a particular threshold.
        * A "quick and dirty" method based on selecting k-means classes of interest through a few lines of python is illustrated below, to skip much of the other functionality in the jupyter notebook

        .. code-block:: python

            import numpy as np
            from tomodrgn import utils
            kmeans_labels = utils.load_pkl('analyze.49/kmeans100/labels.pkl')  # or use the volume space k-means labels at e.g. analyze.49/all_vols_analysis/kmeans100/voxel_kmeans100_labels.pkl
            kmeans_labels_to_keep = np.array([0, 4, 5, 6, 50, 51, 52])  # define your own labels here
            indices_to_keep = np.array([i for i, label in enumerate(kmeans_labels) if label in kmeans_labels_to_keep])
            utils.save_pkl(indices_to_keep, f'analyze.49/kmeans100/indices_kept_{len(indices_to_keep)}-particles.pkl')

        Once the indices file is generated, we can filter the image series and/or volume series star files with ``tomodrgn filter_star``.
        The full list of command line arguments can be found :doc:`here <../command_usage/filter_star>`.

        .. code-block:: python

            tomodrgn filter_star \
                imageseries.star \
                --starfile-type imageseries \
                --tomo-id-col _rlnImageName \  # useful when filtering the output star file for a particular tomogram's particles
                --ind path/to/indices.pkl \
                -o path/to/imageseries_filtered.star

            tomodrgn filter_star \
                volumeseries.star \
                --starfile-type volumeseries \
                --tomo-id-col _rlnImageName \
                --ind path/to/indices.pkl \
                -o path/to/imageseries_filtered.star

   .. tab-item:: Filtering with labels.pkl

        Particle filtration can also be performed with with a ``labels.pkl`` file which describes a single class label for each particle, coupled with a space-separated list of which class labels should be retained after filtering.
        A frequently used source of the ``labels.pkl`` file is ``tomodrgn analyze``, using the kmeans labels.

        Once the set of desired class labels is known, we can filter the image series and/or volume series star files with ``tomodrgn filter_star``.
        The full list of command line arguments can be found :doc:`here <../command_usage/filter_star>`.

        .. code-block:: python

            tomodrgn filter_star \
                path/to/imageseries.star \
                --starfile-type imageseries \
                --tomo-id-col _rlnImageName \
                --labels 03_heterogeneity-1_train_vae/analyze.49/kmeans100/labels.pkl \
                --labels-sel 0 1 2 3 \
                -o path/to/imageseries_filtered.star

Interpreting outputs
---------------------
The outputs are regular plain text .star files and may be viewed in any text editor or other package of choice.

Common pitfalls
----------------
* If an ``indices.pkl`` is generated and then immediately used to train a new tomoDRGN model, the outputs of that model will not have indexing that can be directly aligned to the input star file (because a subset of particles from the input star file are being used for training / latent space embedding / etc.).

  - This is handled in the background for you if generating particle indices using the jupyter notebook approach
  - If using a different approach, you will have to re-index your indices as follows:

    .. code-block:: python

        import numpy as np
        from tomodrgn import utils

        # generate your indices just as before
        kmeans_labels = utils.load_pkl('analyze.49/kmeans100/labels.pkl')  # or use the volume space k-means labels at e.g. analyze.49/all_vols_analysis/kmeans100/voxel_kmeans100_labels.pkl
        kmeans_labels_to_keep = np.array([0, 4, 5, 6, 50, 51, 52])  # define your own labels here
        indices_to_keep = np.array([i for i, label in enumerate(kmeans_labels) if label in kmeans_labels_to_keep])

        # NEW: re-index your selected indices to correspond to the unfiltered star file indexing
        prefiltered_indices = utils.load_pkl('path/to/indices/used/for/training.pkl')
        indices_to_keep = np.asarray([ind for i, ind in enumerate(prefiltered_indices) if i in indices_to_keep])

        # save the indices.pkl file just as before
        utils.save_pkl(indices_to_keep, f'analyze.49/kmeans100/indices_kept_{len(indices_to_keep)}-particles.pkl')

  - To avoid this problem, we recommend filtering the star file prior to to training a new tomoDRGN model, and not using the ``--ind-ptcls`` option of ``tomodrgn train_vae``

* Mapping particle indices derived from an imageseries star file (with tomoDRGN) to a volumeseries star file relies on the two star files describing the same set of particles in the same order. Usually particle extraction software adheres to this. However, the least error-prone way to ensure your star files are "in alignment" is to extract both image series and volume series subtomograms at the same time, rather than extracting volume series subtomograms only when needed for e.g. the current star file filtering.
