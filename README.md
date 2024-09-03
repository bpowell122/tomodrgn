# TomoDRGN: a cryoDRGN fork for heterogeneity analysis of particles from cryo-electron tomography

<p style="text-align: center;">
    <img src="docs/_source/assets/empiar10499_00255_ribosomes.png" alt="Unique per-particle ribosome volumes, calculated from a tomoDRGN variational autoencoder trained on EMPIAR-10499 _Mycoplasma pneumonaie_ ribosomes, mapped back to the tomographic cellular environment"/> 
</p>

[CryoDRGN](https://github.com/zhonge/cryodrgn) has proven a powerful deep learning method for heterogeneity analysis in single particle cryo-EM. In particular, the method models a continuous distribution over 3D structures by using a Variational Auto-Encoder (VAE) based architecture to generate a reconstruction voxel-by-voxel once given a fixed coordinate from a continuous learned  latent space.

TomoDRGN extends the cryoDRGN framework to cryo-ET by learning heterogeneity from datasets in which each particle is sampled by multiple projection images at different stage tilt angles. For cryo-ET samples imaging particles _in situ_, tomoDRGN therefore enables continuous heterogeneity analysis at a single particle level within the native cellular environment. This new type of input necessitates modification of the cryoDRGN architecture, enables tomography-specific processing opportunities (e.g. dose weighting for loss weighting and efficient voxel subset evaluation during training), and benefits from tomography-specific interactive visualizations.

## Installation / dependencies:
Setting up a new tomoDRGN environment:

    # Create conda environment
    conda create --name tomodrgn "python>=3.10"
    conda activate tomodrgn

    # Clone source code and install
    git clone https://github.com/bpowell122/tomodrgn.git
    cd tomodrgn
    python -m pip install .

Potential errors during installation

    # on my Ubuntu 24.04 machine, I had to install the following in order to build fastcluster dependency during install
    sudo apt install make
    sudo apt install build-essential
    sudo apt install cmake

    # tomoDRGN requires pytorch>=2.3, but pytorch does not distribute prebuilt pip packages for x86 Macs starting with pytorch 2.3 (https://github.com/pytorch/pytorch/issues/114602)
    # therefore pytorch must be built from source for x86 Macs (https://github.com/pytorch/pytorch#from-source)
    pip install mkl-static mkl-include
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    conda install cmake ninja
    pip install -r requirements.txt
    python3 setup.py develop

Optional: verify code+dependency functionality on your system
    
    cd tomodrgn/testing

    # ~1 minute
    # tests train_vae and analyze
    python ./quicktest.py

    # ~50 minutes on Macbook Pro, ~10 minutes on Ubuntu workstation with 4060Ti
    # tests all commands with multiple options (except jupyter notebooks)
    # a useful reference for commonly used command syntax
    python ./commandtest.py

Optional: build documentation

    # documentation is built with sphinx in the tomodrgn environment
    python -m pip install .[docs]
    cd docs
    make clean && make html
    # documentation is accessible at ./docs/_build/html/index.html and can be viewed in a web browser


## Training requirements and limitations
TomoDRGN model training requires as inputs:
1. a .mrcs stack containing the 2D pixel array of each particle extracted from each stage tilt micrograph, and
2. a corresponding RELION3.0-format .star file describing pose and CTF parameters for each particle's tilt image as derived from traditional subtomogram averaging. 

At the moment, the only tested workflow to generate these inputs is Warp+IMOD tomogram generation, STA using RELION 3.1 (and optionally M), followed by subtomogram extraction as image series in Warp (or M) (making use of the excellent [dynamo2warp](https://github.com/alisterburt/dynamo2m) tools from Alister Burt). We are actively seeking to expand the range of validated workflows; please let us know if you are exploring a novel approach.


## Additional details about required inputs for train_vae
TomoDRGN requires a number of specific metadata values to be supplied in the star file (a sample star file is provided at `tomodrgn/testing/data/10076_both_32_sim.star`). Users testing upstream workflows that do not conclude with 2D particle extraction in Warp/M should ensure their RELION 3.0 format star files contain the following headers:
* `_rlnGroupName`: used to identify multiple tilt images associated with the same particle, with format `tomogramID_particleID` as used in Warp (e.g. `001_000004` identifies particle 5 from tomogram 2)
* `_rlnCtfScalefactor`: used to calculate stage tilt of each image, equal to `cosine(stage_tilt_radians)` as defined in Warp
* `_rlnCtfBfactor`: used to calculate cumulative dose imparted to each image, equal to `dose_e-_per_A2 * -4` as defined in Warp
* `_rlnImageName`: used to load image data
* `'_rlnDetectorPixelSize','_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration', '_rlnAmplitudeContrast', '_rlnPhaseShift'`: used to calculate the CTF and optimal dose
* `_rlnAngleRot, _rlnAngleTilt, _rlnAnglePsi`: used to calculate pose as defined in RELION
* `_rlnOriginX, _rlnOriginY`: used to calculate pose as defined in RELION (optional, Warp re-centers upon particle extraction and therefore does not have these headers by default)
* tomoDRGN ignores but maintains all other headers, notably allowing interactive post-training analysis to probe per-particle correlations with user-defined metadata


## Changelog
* v0.2.1
  * New features
    * Added `--dose-override` option when loading images to manually specify dose in e-/A2/image
    * Added `--starfile-source` option to `filter_star.py` to allow filtering RELION3.1-format star files from RELION
    * Added `--write-tiltseries-starfile` option to `downsample.py` to write RELION3.0-format star file to downsampled images
    * Added multiple options for pooling tilt images between `train_vae.py` encoder A and encoder B per particle
    * Exposed pytorch dataloader options to user in `train_vae.py`, `train_nn.py`, `eval_images.py`
    * `convergence_nn.py` saves FSCs calculated per epoch to pkl file
    * Added `--batch-size` to `eval_vol.py` calling parallelized `models.FTPositionalDecoder.gen_volume_batch` for accelerated volume generation
    * Added `--vol-path` to `subtomo2chimerax.py` allowing use of consensus reconstruction instead of unique tomodrgn volumes
    * Added new `quicktest.py` and `unittest.py` interface
    * Added `load` method using `config.pkl` file to easily create `dataset.TiltSeriesMRCData` object 
  * Changes
    * Faster dataset loading with improved file I/O
    * Dataset preprocessing refactored to numpy arrays
    * Decreased memory utilization in `dataset.TiltSeriesMRCData` preprocessing
    * Decreased memory utilization in `starfile.TiltSeriesStarfile` initialization
    * Decreased memory utilization by gaussian positional encoding 
    * Changed default positional encoding to `gaussian`
    * Changed default latent dimensionality to `--zdim 128`
    * `train_vae.py` checks for NaN/inf during checkpoint evaluation
    * `utils.calc_fsc` allows input volumes to be either paths to .mrc files or direct numpy arrays
    * New plots in `convergence_vae.py` including all loss types, all-to-all volumetric correlation coefficients among tomodrgn (and optionally ground truth) volumes
    * Changed default `subtomo2chimerax.py` color map to ChimeraX color scheme
    * Added requirement for pytorch>=1.11
  * Bugfixes
    * Fixed syntax of internal calls to `eval_vol.py` 
    * Fixed dataset index filtering of pre-filtered datasets
    * Added assert for convergence scripts that requisite input files exist
    * Fixed bug where `config.pkl` containing CUDA tensors could not be loaded on cpu
    * Fixed bug where precalculated `norm` would be ignored when loading dataset from `config.pkl` with `--lazy`

* v0.2.0
  * Features
    * added support for datasets with variable tilts per tomogram (tilt counts/schemes/etc)
    * new scripts `subtomo2chimerax` and `filter_star`
    * additional features and checks in `tomoDRGN_viz+filt.ipynb`
    * `eval_images` supports `z.pkl` output; `eval_vol` supports `z.pkl` input
    * new tests `unittest.sh` and `quicktest.sh`
    * validated compatibility with python 3.7 - 3.10 and pytorch 1.8 - 1.12
  * Changes
    * refactored the majority of codebase for explicit internal tomoDRGN compatibility and some performance improvements
    * updated tomoDRGN install requirements and changed to `pip`-based installation 
    * `--amp` is now enabled by default and can be disabled with `--no-amp`
    * `--do-dose-weighting`, `--do-tilt-weighting`, `--dose-mask` have been renamed to `--recon-dose-weight`, `--recon-tilt-weight`, `--l-dose-mask`, respectively
    * `tomodrgn downsample` now benefits from pytorch-based GPU acceleration
    * updated training hyperparameter defaults
    * (many) various bugfixes

* v0.1.0
  * initial tomoDRGN release


## Relevant literature
1. Zhong, E., Bepler, T., Berger, B. & Davis, J. CryoDRGN: Reconstruction of Heterogeneous cryo-EM Structures Using Neural Networks. Nature Methods, [doi:10.1038/s41592-020-01049-4](https://doi.org/10.1038/s41592-020-01049-4) (2021)
2. Kinman, L., Powell, B., Zhong, E., Berger, B. & Davis, J. Uncovering structural ensembles from single particle cryo-EM data using cryoDRGN. bioRxiv, [doi:10.1101/2022.08.09.503342](https://doi.org/10.1101/2022.08.09.503342) (2022)
3. Sun, J., Kinman, L., Jahagirdar, D., Ortega, J., Davis. J. KsgA facilitates ribosomal small subunit maturation by proofreading a key structural lesion. bioRxiv, [doi:10.1101/2022.07.13.499473](https://doi.org/10.1101/2022.07.13.499473) (2022)


## Contact
Please reach out with bug reports, feature requests, etc to bmp[at]mit[dot]edu.