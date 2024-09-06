Tutorial: EMPIAR-10499 70S ribosomes
====================================

Here we present tutorials for processing heterogeneous ribosome data from cryo-ET benchmark dataset EMPIAR-10499, as described in our tomoDRGN manuscript.
This tutorial will cover the following stages of processing:

#. upstream processing and obtaining input data for tomoDRGN
#. validating that particles and metadata were extracted correctly: ``tomodrgn backproject_voxel``, or ``tomodrgn train_nn`` with ``tomodrgn convergence_nn``
#. learning structural heterogeneity within the dataset: ``tomodrgn train_vae`` with ``tomodrgn convergence_vae``
#. analyzing structural heterogeneity within the dataset: ``tomodrgn analyze``, ``tomodrgn eval_vol`` and ``tomodrgn analyze_volumes``, and external tools including `SIREn <https://github.com/lkinman/SIREn>`_ and `MAVEn <https://github.com/lkinman/MAVEn>`_
#. visualizing structural heterogeneity patterns in the tomogram's spatial context: tomoDRGN's interactive visualization jupyter notebook and ``tomodrgn subtomo2chimerax``
#. isolating particle subsets of interest: tomoDRGN's interactive filtering jupyter notebook and ``tomodrgn filter_star``
#. taking homogeneous particle subsets back into external STA tools for further refinement

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
    analyze_structural_heterogeneity
    visualize_spatially_contextualized_heterogeneity
    isolate_particle_subsets
    external_iterative_processing