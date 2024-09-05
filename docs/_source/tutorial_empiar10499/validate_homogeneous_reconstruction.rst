Validate particle extraction
=============================

Purpose
--------

Once you have extracted (or downloaded) a subtomogram particleseries, it's a good idea to validate that the extraction worked correctly.
A quick way to confirm successful particle extraction is to generate a homogeneous reconstruction of the extracted particles, either via ``tomodrgn backproject_voxel`` or ``tomodrgn train_nn``.
Both approaches to homogeneous reconstructions benefit from access to a machine with a GPU available for computation, and sufficient RAM to hold all particles in memory. This is particularly important for ``tomodrgn train_nn``.


Performing homogeneous reconstruction
--------------------------------------

.. tab-set::

    .. tab-item:: tomodrgn backproject_voxel

        In this example, we backproject all particles referenced in ``particleseries.star``.
        The full list of command line arguments can be found here [TODO ref to backproject_voxel documentation].
        On our testing machine, this command took XXXXX minutes.

        .. code-block:: console

            mkdir 01_backproject

        .. code-block:: python

            tomodrgn backproject_voxel \
                particleseries.star \
                -o 01_validate_backproject/backproject_weighted.mrc \
                --datadir .../path/to/particleseries \
                --recon-dose-weight \
                --recon-tilt-weight

        This command produces several outputs in the ``01_validate_backproject`` directory:

        * ``backproject_weighted.mrc``: this is the unfiltered backprojected reconstruction

        * ``backproject_weighted_half*.mrc``: these two maps are the unfiltered backprojected half-map reconstructions from randomly selected halves of the dataset

        * ``backproject_weighted_fsc.png``: this is the FSC between the two half-maps, calculated with an automatically generated soft mask

        * ``backproject_weighted_filt.mrc``: this is the backprojected reconstruction, lowpass filtered to the resolution at which half-maps FSC drops below 0.143

    .. tab-item:: tomodrgn train_nn

        In this example, we train a homogeneous (decoder-only) tomoDRGN network to reconstruct the particles referenced in ``particleseries.star``.
        The full list of command line arguments can be found here [TODO ref to train_nn documentation]
        On our testing machine, this command took XXXXX minutes.

        .. code-block:: python

            tomodrgn train_nn \
                particleseries.star \
                -o 02_validate_train-nn \
                --datadir .../path/to/particleseries \
                --recon-dose-weight \
                --recon-tilt-weight \
                -- l-dose-mask \
                -- layers 3 \
                -- dim 512 \
                -- n 20

        This command produces several outputs in the ``02_validate_train-nn`` directory:

        * ``config.pkl``

        * ``run.log``

        * ``weights.*.pkl``

        * ``reconstruct.*.mrc``


Interpreting outputs
---------------------

Open your ``backproject_weighted_filt.mrc`` or ``reconstruct.19.mrc``, as appropriate, in ChimeraX or a similar 3D volume viewer.
If all went well, you should see a reconstruction that looks like your desired structure:
# TODO add picture


Common pitfalls
----------------

If your reconstruction does not look like an interpretable structure similar to that produced by upstream processing, here are a few things to check:

- volume looks hollow: try adding ``--uninvert-data`` to your ``backproject_voxel`` or ``train_nn`` command to fix the data sign convention for your particles (light-on-dark vs dark-on-light)

- volume looks like a featureless ball: the rotations specified in your star file may be inaccurate. Check these poses by reconstructing these particles with ``relion_reconstruct --i particleseries.star --o reconstruct_relion.mrc --ctf``.

- volume looks like a cube of noise:

    1. try setting a stronger lowpass filter if using ``tomodrgn backproject_voxel``, or training for fewer epochs with more particles if using ``tomodrgn train_nn``

    2. confirm that your particle coordinates were supplied for extraction with the correct pixel size and align with your desired particles (e.g. with Cube). Check these poses by reconstructing these particles with ``relion_reconstruct --i particleseries.star --o reconstruct_relion.mrc --ctf``.

