Visualize spatially-contextualized heterogeneity
==================================================

Purpose
--------
While many interesting structural or biological insights can be gleaned from examining individual volumes or volume ensembles in isolation, in cryo-ET we have knowledge of where each particle originates in the source tomogram.
With this knowledge, we can map particles (and by extension, their structural heterogeneity) back to the underlying tomogram / cellular context, which opens significant further avenues of investigation and hypothesis generation.

Visualizing particle mapbacks
------------------------------
TomoDRGN provides two ways to view particles mapped back to tomogram positions: an interactive 3-D plot built into Jupyter notebook, and ChimeraX command scripts.

Interactive 3-D plot
^^^^^^^^^^^^^^^^^^^^^
The ``tomoDRGN_interactive_viz.ipynb`` Jupyter notebook contains a widget to generate an interactive 3-D scatter plot of all particles.
This can be a useful way to rapidly explore spatially contextualized heterogeneity in a relatively lightweight application.
Open the notebook, run the first few cells to import dependencies, enable widgets, and load data, and proceed to the section titled "View particle distributions in tomogram context".

#. The first cell here asserts that all required columns to map particles to tomogram coordinates are present in the loaded data.
#. The second cell requires user input to define the path to where all generated tomograms are stored on the filesystem, and to define how each tomogram name is mapped to a unique value in the star file so that the correct particle subset can be plotted.
#. Running the third cell will produce the interactive plotting widget.

.. figure:: ../assets/interactive_scatter3d_mapback.png
    :alt: tomoDRGN interactive particle viewer running in a Jupyter notebook
    :scale: 100%

    The Jupyter notebook interactive viewer displaying particles and a tomogram from a toy dataset.

The widget provides functionality broadly grouped into

* tomogram and particle loading (top row)
* tomogram rendering options (top left)
* particle rendering options (top-middle left)
* particle subset annotation options (bottom-middle left)
* log output (bottom left)

Hovering over a particle reports its index in the input star file used for training; if "particle color by" is set then the value of that attribute for that particle is also reported.

ChimeraX command scripts
^^^^^^^^^^^^^^^^^^^^^^^^^
TomoDRGN also provides functionality to view spatially contextuallized heterogeneity in ChimeraX through the command ``tomodrgn subtomo2chimerax``.
The subtomo2chimerax command has been adapted and modified with permission from ``relionsubtomo2ChimeraX.py``, written by `Huy Bui at McGill University <https://doi.org/10.5281/zenodo.6820119>`_.
Subtomo2chimerax generates outputs in three possible rendering ``--mode``'s: placing spherical markers at each particle position, placing a homogeneous consensus structure at each particle position, or generating a unique tomoDRGN volume for each particle and placing the corresponding volume at its particle's source location.
In all cases, it requires a volume series star file describing the same set of particles in the same order that tomoDRGN was trained on (in order to correctly map heterogeneity to spatial location).

The full list of command line arguments can be found :doc:`here <../command_usage/subtomo2chimerax>`.

.. code-block:: python

    tomodrgn subtomo2chimerax \
        path/to/volumeseries.star \
        --mode volumes \
        --outdir 03_heterogeneity-1_train_vae/analyze.49/tomogram_mapbacks \
        --weights 03_heterogeneity-1_train_vae/weights.49.pkl \
        --config 03_heterogeneity-1_train_vae/config.pkl \
        --zfile 03_heterogeneity-1_train_vae/z.49.train.pkl \
        --coloring-labels 03_heterogeneity-1_train_vae/analyze.49/kmeans100/labels.pkl
        --downsample 64

By default, ``tomodrgn subtomo2chimerax`` will create a separate output folder for each tomogram identified in the star file.
One such tomogram's mapback is shown below, where each particle is colored by its latent kmeans class.

.. figure:: ../assets/empiar10499_00255_ribosomes.png
    :alt: example subtomo2chimerax tomogram mapback using EMPIAR dataset EMPIAR-10499
    :scale: 100%

    Heterogeneous ribosome volumes learned and generated with tomoDRGN mapped back to the source tomogram of a Mycoplasma pneumoniae cell from EMPIAR-10499


Interpreting outputs
---------------------
* For each tomogram, ``tomodrgn subtomo2chimerax`` will produce a folder containing a few outputs:

  - the unique volume generated for each particle, named following the ordering of the input star file
  - ``rgba_labels.txt``: a mapping of unique labels found in ``--coloring-labels``, to the corresponding RGBA specification, to which models in ChimeraX share this unique label
  - ``mapback.cxc``: the ChimeraX command file that should be opened in an empty ChimeraX session to create the scene of particle volumes at source tomogram coordinates. Note that ``mapback.cxc`` uses relative paths for ChimeraX to find and load the correct volumes.


Common pitfalls
----------------
* The image series star file used for model training, and the volume series star file used for deriving particle 3-D coordinates, must describe the same set of particles in the same order. See the discussion on the next page for further information.
* The shape of the array specified by ``--coloring-labels`` must be the ``(num_particles)`` and must match the particle indexing specified by the input star file. The file ``analyze.49/kmeans100/labels.pkl`` is a good example that meets these requirements.
* A useful way to check that your volume series star file and coloring labels are being parsed correctly is to use an array of every particle's X-coordinate in the tomogram as the ``--coloring-labels`` input, and using a continuous colormap such as ``--colormap viridis``.

