Upstream processing
====================

# TODO update to reflect new upstream STA and extraction supported software

tomoDRGN inputs and nomenclature
---------------------------------
Several input data types from upstream processing are required to take full advantage of tomoDRGN's heterogeneity analysis, validation, and iterative processing potential.
However, only some files are required for minimal functionality.

Required
**********

#. "Image series" subtomograms

   * Real-space 2-D projection images, typically extracted from motion-corrected micrographs
   * Each particle should be sampled by multiple images, collected at a range of different stage tilts
   * Warp and M offer this option via "Subtomogram Extraction > Image series"

#. Star file metadata for "image series" subtomograms

   * RELION v3 star file (due to Warp/M compatibility)
   * each tilt image is referenced by one row, detailing pose (translation and rotation), CTF (defocus, pixel size, etc.), path on disk, and shared particle identity

Recommended
************

3. "Volume series" subtomograms
4. Star file metadata for "volume series" subtomograms


Obtaining particles
--------------------
.. tab-set::

    .. tab-item:: Warp / RELIONv3 / M

        The most well-developed upstream processing pipeline to generate required inputs features Warp (for tilt movie alignment, CTF fitting, subtomogram extraction), RELION (subtomogram alignment and averaging), and optionally M (multi-particle refinement, subtomogram re-extraction).
        [TODO] sample particle extraction dialog box in Warp and in M

    .. tab-item:: Download from EMPIAR

        The ribosomes analyzed from EMPIAR-10499 in our manuscript [REF] have been deposited to EMPIAR via accession ID EMPIAR-XXXXX.
        Six datasets are deposited at that accession ID, containing the same set of particles re-extracted at different box and pixel sizes:
        For the sake of simplicity, we will only be working with the following datasets, which require about XXX GB storage capacity:

        * 22,291 ribosomes extracted with box size 96 px and pixel size 3.71 Å/px as "image series"
        * 22,291 ribosomes extracted with box size XX px and pixel size XXXX Å/px as "volume series"
        * star files containing metadata for each dataset

        .. tab-set::

            .. tab-item:: Globus

                1. Create a Globus Connect Personal account, or sign into an existing Globus installation if available.

                2. In the Globus web transfer interface, navigate to the File Transfer tab.

                3. On the left side of the window, select the repository as "XXXXX". Below this, enter the following path to EMPIAR-XXXXX: ``/path/to/empiar/XXXXX``

                4. Select "XXXXXX dataset", "XXXXX dataset", and "XXXX starfiles".

                5. On the right side of the file transfer window, select the path on your local machine where you would like to store these files.

                6. Click the ">" button to initiate the transfer from EMPIAR to your local system. Transfer progress can be monitored in the "Activity" tab.

            .. tab-item:: rsync

                .. code-block:: console

                    rsync -avz XXXX/path/to/empiar/dataset1 \
                        /.../path/on/local/machine

                .. code-block:: console

                    rsync -avz XXXX/path/to/empiar/dataset2 \
                        /.../path/on/local/machine

                .. code-block:: console

                    rsync -avz XXXX/path/to/empiar/starfiles \
                        /.../path/on/local/machine


Alternative particle sources
-----------------------------

RELION v4
***********

May be possible with TeamTomo Warp - RELION v4 - M suite of tools that should enable conversion from RELION v4 STA to Warp/M for subtomogram extraction as imageseries.
https://gist.github.com/biochem-fan/c21b4701cc633201c5c99582b4ca16b3
However, we have not tested this yet.

CisTEM
*******

CisTEM can extract images from tilt series, just as required for tomoDRGN.
Support for cisTEM star files in tomoDRGN is nearly complete.
This approach should also enable emClarity-processed subtomograms to be extracted in cisTEM through existing emClarity - cisTEM pipelines.

Others
********

Cryo-ET is an exciting, rapidly developing field with many distinct software tools.
We do not yet support import from other software packages (e.g. STOPGAP, nextPyP, PyTOM, ...).
If your workflow uses other software packages, and this metadata cannot be converted into one of the supported subtomogram extraction pipelines described above, please reach out to us with your use case.
