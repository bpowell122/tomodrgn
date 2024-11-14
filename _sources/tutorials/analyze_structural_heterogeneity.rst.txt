Analyze structural heterogeneity
=================================

Standard latent space analysis
-------------------------------
TomoDRGN's learned latent space is populated by a unique latent embedding for each particle.
This distribution of latent embeddings is represented as an array of shape ``(num_particles, latent_dimensionality)`` and saved as ``z.*.pkl`` at each epoch.
Several standard analyses of this array of embeddings are implemented in the ``tomodrgn analyze`` command.

The full list of command line arguments can be found :doc:`here <../command_usage/analyze>`.

.. code-block:: python

    tomodrgn analyze \
        03_heterogeneity-1_train_vae \
        --epoch 49 \
        --ksample 100

Interpreting outputs
^^^^^^^^^^^^^^^^^^^^^

* latent space dimensionality reduction via PCA and UMAP

  - plots of PCA explained variance and PCA-projected latent embeddings are saved to ``analyze.49/``
  - plots of UMAP-projected latent embeddings are saved to ``analyze.49/``, the array of UMAP-projected embeddings is saved to ``analyze.49/umap.pkl``
  - the plot axes are labeled as ``l-PCA`` and ``l-UMAP`` to emphasize their *latent space* origin

* latent space interpolation (and subsequent volume generation) along specified principal components

  - latent embeddings sampled at the 5th to 95th percentile in decile steps along each principal component are saved to ``analyze.49/pcN``
  - corresponding volumes generated at each sampled latent embedding are saved in the same directory
  - plots of the PCA-projected latent embeddings with the sampled latent coordinates annotated are saved in the same directory

* latent space k-means clustering (and subsequent volume generation at each cluster's centroid)

  - latent embeddings sampled at the centroid of each k-means class, and the indices of these representative centroid particles, are saved to ``analyze.49/kmeansN`` as ``centers.txt`` and ``centers_ind.txt``, respectively
  - corresponding volumes generated at each class centroid latent embedding are saved in the same directory
  - the class label of each particle (i.e., which k-means class was each particle assigned to) is saved to ``analyze.49/kmeansN/labels.pkl`` as an array of shape ``(num_particles, 1)``
  - the distribution of class labels per tomogram (i.e., how populated is each class in each tomogram) is saved to ``analyze.49/kmeansN/tomogram_label_distribution.png``
  - images of the first few particles in each class are saved to ``analyze.49/kmeansN/particle_images_kmeanslabelN.png``
  - plots of the PCA-projected and UMAP-projected latent embeddings with the sampled centroid latent coordinates annotated are saved in the same directory

* plots describing the correlation of the input star file's numerical columns with the latent UMAP dimensionality reduction

  - a plot is saved for each numerical column to ``analyze.49/controls/*.png``
  - as we (generally) do not expect structural heterogeneity to correlate with parameters such as particle pose or CTF parameters, these plots allow one to test this assumption

* generation of interactive jupyter notebooks to explore further potential parameter correlations, also aiding selection of distinct particle subsets

  - these notebooks can be opened in Jupyter Notebook.

    * One way to launch Jupyter Notebook is to run ``jupyter notebook`` at terminal. Note that port forwarding will be required if you are running your notebooks on a remote machine (e.g. HPC cluster).
    * If you are new to Jupyter Notebook, there are tons of online resources: check out `this tutorial <https://www.dataquest.io/blog/jupyter-notebook-tutorial/>`_ or `this cheat sheet <https://www.edureka.co/blog/wp-content/uploads/2018/10/Jupyter_Notebook_CheatSheet_Edureka.pdf>`_

  - the ``tomoDRGN_viz+filt_legacy.ipynb`` notebook contains functionality to interactively recreate many of the analyses described above, perhaps changing the number of k-means classes, axis limits, and so on. It also contains functionality used to select particle subsets, as will be discussed later.
  - the ``tomoDRGN_interactive_viz.ipynb`` notebook contains streamlined functionality to jump straight into interactively exploring potential correlations and clustering among parameters associated with each particle.
  - jupyter notebooks are designed to be very interactive -- these templates are useful to us in our analyses, but you can easily add new python code to try new types of analyses as appropriate for your dataset and your structural investigation!

Standard volume space analysis
-------------------------------
The latent space is typically correlated well with structural heterogeneity.
However, it can also be instructive to directly explore structural heterogeneity in "volume space".
We perform this by generating a large ensemble of unique volumes, then performing all of the analyses described above for latent space analysis directly in the volume space array.

We first need a large ensemble of unique volumes.
This ensemble can be generated indirectly, e.g. by using the kmeans100 volumes generated by ``tomodrgn analyze`` above.
However, it is also possible and reasonably performant to generate larger volume ensembles (potentially up to a unique volume for every particle in the dataset, as demonstrated below) directly with ``tomodrgn eval_vol``.
When generating this many volumes, we strongly recommend generating downsampled volumes, typically around box size 32px - 64px.
This minimizes the time required to generate the volumes, time required to analyze the volumes, and disk space required to store all outputs.
As a reminder, each volume will use approximately :math:`\frac{4*(boxsize)^{3}}{1024^{2}}` MiB of disk space; a useful reference point is that 1 box64 volume is 1 MiB.

The full list of command line arguments can be found :doc:`here <../command_usage/eval_vol>`.

.. code-block:: python

    tomodrgn eval_vol \
        --weights 03_heterogeneity-1_train_vae/weights.49.pkl \
        -c 03_heterogeneity-1_train_vae/config.pkl \
        -o 03_heterogeneity-1_train_vae/all_vols \
        --zfile 03_heterogeneity-1_train_vae/z.49.train.pkl \
        --downsample 64 \
        -b 32

Once an ensemble of volumes has been generated through some means, we can run the volume space analogue to each of the analyses described above for latent space.
This includes volume space PCA and UMAP dimensionality reduction and interpolation, volume space k-means clustering, generation of volumes along principal components and k-means centroids, and numerical attribute correlation with controls.

The full list of command line arguments can be found :doc:`here <../command_usage/analyze_volumes>`.

.. code-block:: python

    tomodrgn analyze_volumes \
        --voldir 03_heterogeneity-1_train_vae/all_vols \
        --config 03_heterogeneity-1_train_vae/config.pkl \
        --outdir 03_heterogeneity-1_train_vae/all_vols_analysis \
        --ksample 100 \
        --mask soft

Interpreting outputs
^^^^^^^^^^^^^^^^^^^^^
See the section above for interpreting outputs of ``tomodrgn analyze`` latent space analysis; an analogous set of outputs are generated here.
The exception is the addition of two new files containing the dimensionality-reduced array of volumes: ``all_vols_analysis/voxel_pc.pkl` and ``all_vols_analysis/voxel_pc_umap.pkl``


Systematic, model-guided assessment of heterogeneity: MAVEn
-------------------------------------------------------------
In some cases, the structural heterogeneity present in the dataset can be well parameterized as an atomic model exhibiting compositional heterogeneity.
We can perform a quantitative analysis of the learned structural heterogeneity guided by this atomic model using the tool `MAVEn <https://github.com/lkinman/MAVEn>`_.
MAVEn is designed to quantify the relative occupancy of many real space masks across the reconstructed volume (quantified as the amount of density), then cluster the resulting array of ``(num_volumes, num_masks)`` to identify structural classes sharing certain structural features, and to identify structural blocks of atoms (masks) that exhibit correlated occupancy.

MAVEn requires as inputs:

#. an ensemble of volumes (for example, generated by ``tomodrgn analyze`` or ``tomodrgn eval_vol`` above)
#. a PDB file from which to generate masks to quantify occupancy of distinct real space regions (obtained elsewhere, perhaps from the PDB or from model building and refinement into your consensus reconstruction)

The MAVEn pipeline is documented in more detail at the link above.


Systematic, model-free inspection of heterogeneity: SIREn
----------------------------------------------------------
In many cases, we may not have a suitable atomic model with which to quantitate structural heterogeneity patterns.
The tool `SIREn <https://github.com/lkinman/SIREn>`_ has been developed to perform (atomic) model-free analysis of an ensemble of volumes.
SIREn exploits statistically significant pairwise voxel correlations among the volume ensemble to infer what regions of the structure are likely to form distinct structural blocks, including both compositional and conformational structural heterogeneity.

SIREn requires as input:

#. an ensemble of volumes (for example, generated by ``tomodrgn analyze`` or ``tomodrgn eval_vol`` above)

The SIREn pipeline is documented in more detail at the link above.
