Background
===========


Motivation
-----------

Designed to learn continuous, multidimensional structural heterogeneity from tilt series data...


Training requirements and limitations
-------------------------------------

TomoDRGN model training requires as inputs:

1. a ``.mrcs`` stack containing the 2D pixel array of each particle extracted from each stage tilt micrograph, and
2. a corresponding RELION3.0-format ``.star`` file describing pose and CTF parameters for each particle's tilt image as derived from traditional subtomogram averaging.

At the moment, the only tested workflow to generate these inputs is Warp+IMOD tomogram generation, STA using RELION 3.1 (and optionally M), followed by subtomogram extraction as image series in Warp (or M) (making use of the excellent [dynamo2warp](https://github.com/alisterburt/dynamo2m) tools from Alister Burt). We are actively seeking to expand the range of validated workflows; please let us know if you are exploring a novel approach.


Additional details about required inputs for train_vae
-------------------------------------------------------

TomoDRGN requires a number of specific metadata values to be supplied in the star file (a sample star file is provided at ``tomodrgn/testing/data/10076_both_32_sim.star``). Users testing upstream workflows that do not conclude with 2D particle extraction in Warp/M should ensure their RELION 3.0 format star files contain the following headers:

* ``_rlnGroupName``: used to identify multiple tilt images associated with the same particle, with format ``tomogramID_particleID`` as used in Warp (e.g. ``001_000004`` identifies particle 5 from tomogram 2)
* ``_rlnCtfScalefactor``: used to calculate stage tilt of each image, equal to ``cosine(stage_tilt_radians)`` as defined in Warp
* ``_rlnCtfBfactor``: used to calculate cumulative dose imparted to each image, equal to ``dose_e-_per_A2 * -4`` as defined in Warp
* ``_rlnImageName``: used to load image data
* ``'_rlnDetectorPixelSize','_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration', '_rlnAmplitudeContrast', '_rlnPhaseShift'``: used to calculate the CTF and optimal dose
* ``_rlnAngleRot, _rlnAngleTilt, _rlnAnglePsi``: used to calculate pose as defined in RELION
* ``_rlnOriginX, _rlnOriginY``: used to calculate pose as defined in RELION (optional, Warp re-centers upon particle extraction and therefore does not have these headers by default)
* tomoDRGN ignores but maintains all other headers, notably allowing interactive post-training analysis to probe per-particle correlations with user-defined metadata