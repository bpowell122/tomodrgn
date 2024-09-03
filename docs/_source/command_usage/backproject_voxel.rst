tomodrgn backproject_voxel
===========================

Purpose
--------
Reconstruct a 3-D volume from pre-aligned 2-D tilt-series projections via weighted back projection.


Sample usage
------------

Example 1: minimal usage
    * Reconstruction is lowpass filtered to FSC=0.143 half-map resolution

    .. code-block:: python

        tomodrgn backproject_voxel \
            imageseries.star \
            -o backproject.mrc

Example 2: more complex usage
    * Particles are located in a different directory than specified in ``imageseries.star``, so ``--datadir`` is specified as a relative path.
    * Images are weighted by dose and tilt during backprojection
    * Only a subset of particles are backprojected, as specifed with ``--ind``
    * Reconstruction is lowpass filtered to FSC=0.143 half-map resolution

    .. code-block:: python

        tomodrgn backproject_voxel \
            imageseries.star \
            --datadir ../../particleseries/ \
            --recon-dose-weight \
            --recon-tilt-weight \
            --ind ind_desired.pkl \
            -o backproject.mrc

Arguments
---------

.. option:: particles

    Path to input "image series" subtomogram star file. First positional argument.

.. option:: -o OUTPUT

    Path to output .mrc 3-D reconstruction.

.. option:: --uninvert-data

    Do not invert data sign.

.. option:: --datadir DATADIR

    Path prefix to directory containing particle stack if location on disk differs from path referenced in particles star file

.. option:: --ind IND.PKL

    Path to indices of selected particles to use for backprojection. Indices are 0-indexed, stored in a 1-dimensional numpy array, and saved to disk as a binary pickle file. Indices are applied to sequentially unique elements of the ``_rlnGroupName`` in the particles star file.

.. option:: --first FIRST

    Number of particles to use for backprojection. Default is to use all particles in particles star file.

.. option:: --recon-tilt-weight

    Weight images in reciprocal space by ``cosine(tilt_angle)``, as parsed from particles star file.

.. option:: --recon-dose-weight

    Weight images in reciprocal space per tilt per spatial frequency by dose dependent amplitude attenuation.

.. option:: --lowpass LOWPASS

    Resolution (in angstroms) to lowpass filter the reconstructed volume. Default is the lowest resolution where reconstructed half-maps decrease below FSC=0.143 correlation.

.. option:: --flip

    Flip the handedness of the output volume.


Common next steps
------------------
* Backproject a different particle subset to validate structural heterogeneity visualized by tomoDRGN's decoder network
* Use backprojections to create initial models or masks for further particle refinement in RELION or M