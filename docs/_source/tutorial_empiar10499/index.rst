Tutorial: EMPIAR-10499 70S ribosomes
====================================

Here we present tutorials for processing heterogeneous ribosome data from cryo-ET benchmark dataset EMPIAR-10499, as described in our tomoDRGN manuscript.
This tutorial will cover the following stages of processing:

1. upstream processing and obtaining input data for tomoDRGN
2. validating that particles and metadata were extracted correctly: ``tomodrgn backproject_voxel``, or ``tomodrgn train_nn`` with ``tomodrgn convergence_nn``
3. learning structural heterogeneity within the dataset: ``tomodrgn train_vae``, ``tomodrgn convergence_vae``, and ``tomodrgn analyze``
4. isolating particle subsets of interest: ``tomodrgn filter_star``
5. visualizing structural heterogeneity patterns in the tomogram's spatial context: ``tomodrgn subtomo2chimerax``
6. Iterating particle refinement by reprocessing in RELION (v3) or M

With these steps as building blocks, many additional types of analyses are possible.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: U-M cryoET workshop 2024

    um_cryoet_workshop_2024


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: EMPIAR-10499 ribosomes

    upstream_preprocessing
    validate_homogeneous_reconstruction
    learn_structural_heterogeneity
    isolate_particle_subsets
    visualize_spatially_contextualized_heterogeneity
    external_iterative_processing