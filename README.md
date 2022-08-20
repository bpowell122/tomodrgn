# TomoDRGN: a cryoDRGN fork for heterogeneity analysis of particles from cryo-electron tomography

<p style="text-align: center;">
    <img src="assets/empiar10499_00255_ribosomes.png" alt="Unique per-particle ribosome volumes, calculated from a tomoDRGN variational autoencoder trained on EMPIAR-10499 _Mycoplasma pneumonaie_ ribosomes, mapped back to the tomographic cellular environment" width="80%"/> 
</p>

[CryoDRGN](https://github.com/zhonge/cryodrgn) has proven a powerful deep learning method for heterogeneity analysis in single particle cryo-EM. In particular, the method models a continuous distribution over 3D structures by using a Variational Auto-Encoder (VAE) based architecture to generate a reconstruction voxel-by-voxel once given a fixed coordinate from a continuous learned  latent space.

TomoDRGN extends the cryoDRGN framework to cryo-ET by learning heterogeneity from datasets in which each particle is sampled by multiple projection images at different stage tilt angles. For cryo-ET samples imaging particles _in situ_, tomoDRGN therefore enables continuous heterogeneity analysis at a single particle level within the native cellular environment. This new type of input necessitates modification of the cryoDRGN architecture, enables tomography-specific processing opportunities (e.g. dose weighting for loss weighting and efficient voxel subset evaluation during training), and benefits from tomography-specific interactive visualizations.

TomoDRGN aims to parallel familiar cryoDRGN usage syntax and integrate directly with pre-existing cryoDRGN analysis and visualization tools. 


## Installation / dependencies:
TomoDRGN shares essentially all of cryoDRGN's dependencies, with the addition of a few new ones such as ipyvolume (for interactive 3D visualization of results). Therefore, users may choose to update existing cryoDRGN environments to run tomoDRGN as well (replacing the "create conda environment" section below with `conda activate cryodrgn`), or instead to set up a dedicated tomoDRGN environment:

    # Create conda environment
    conda create --name tomodrgn python=3.7
    conda activate tomodrgn

    # Install dependencies
    conda install pytorch cudatoolkit=10.1 -c pytorch # Replace cudatoolkit version if needed
    conda install pandas
    
    # Install dependencies for latent space visualization 
    conda install seaborn scikit-learn 
    conda install umap-learn jupyterlab ipywidgets cufflinks-py ipyvolume "nodejs>=15.12.0" -c conda-forge
    jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
    jupyter labextension install jupyterlab-plotly --no-build
    jupyter labextension install plotlywidget --no-build
    jupyter lab build

    # Clone source code and install
    git clone https://github.com/bpowell122/tomodrgn.git
    cd tomodrgn
    python setup.py install


## Example usage
Below are minimalist examples of various common tomoDRGN usage syntax. By design, usage syntax parallels cryoDRGN where possible. To see all options available for a given `command` with corresponding defaults (e.g. `train_vae`), run: `tomodrgn train_vae --help`
0. Initialize the conda environment: `conda activate tomodrgn`
1. Train a decoder-only network to learn a homogeneous structure: `tomodrgn train_nn particles.star --outdir 01_nn_particles --batch-size 1 --num-epochs 30`
2. Assess convergence of a decoder-only network relative to an external volume: `tomodrgn convergence_nn 01_nn_particles sta_reference_volume.mrc`
3. Train a VAE to simultaneously learn a latent heterogeneity landscape and a volume-generative decoder: `tomodrgn train_vae particles.star --zdim 8 --outdir 02_vae_z8_box256 --batch-size 1 --num-epochs 30`
4. Assess convergence of a VAE model after 30 epochs training using internal / self-consistency heuristics: `tomodrgn convergence_vae 02_vae_z8_box256 29`
5. Run standard analysis of a VAE model after 30 epochs training (PCA and UMAP of latent space, volume generation along PC's, _k_-means sampling of latent space and generation of corresponding volumes): `tomodrgn analyze 02_vae_z8_box256 29 --Apix 3.5`
6. Interactive analysis of a VAE model: run `jupyter notebook` to open `tomoDRGN_viz+filt_template.ipynb` placed in `02_vae_z8_box256/analyze.29` by `tomodrgn analyze`
7. Filter a star file by particle indices isolated by interactive analysis of a VAE: `tomodrgn filter_star particles.star --ind ind.pkl --outstar particles_filt.star`

## Training requirements and limitations
TomoDRGN requires as inputs:
1. a .mrcs stack containing the 2D pixel array of each particle extracted from each stage tilt micrograph, and
2. a corresponding .star file describing pose and CTF parameters for each particle as derived from traditional subtomogram averaging. 

At the moment, the only tested workflow to generate these inputs is Warp+IMOD tomogram generation, STA using RELION 3.1, followed by 2D particle extraction in Warp (making use of the excellent [dynamo2warp](https://github.com/alisterburt/dynamo2m) tools from Alister Burt). We are actively seeking to expand the range of validated workflows; please let us know if you have success with a novel approach.

## Additional details about required inputs
TomoDRGN requires a number of specific metadata values to be supplied in the star file (a sample star file is provided in the `tomodrgn/testing/data` folder). Users testing upstream workflows that do not conclude with 2D particle extraction in Warp should ensure their RELION 3.0 format star files contain the following headers:
* `_rlnGroupName`: used to identify multiple tilt images associated with the same particle, equal to a unique value per particle
* `_rlnCtfScalefactor`: used to calculate stage tilt of each image, equal to `cosine(stage_tilt_radians)` as defined in Warp
* `_rlnCtfBfactor`: used to calculate cumulative dose imparted to each image, equal to `dose_e-_per_A2 * -4` as defined in Warp
* `_rlnImageName`: used to load image data
* `'_rlnDetectorPixelSize','_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration', '_rlnAmplitudeContrast', '_rlnPhaseShift'`: used to calculate the CTF and optimal dose
* `_rlnAngleRot, _rlnAngleTilt, _rlnAnglePsi`: used to calculate pose as defined in RELION
* `_rlnOriginX, _rlnOriginY`: used to calculate pose as defined in RELION (optional, Warp re-centers upon particle extraction and therefore does not have these headers by default)
* tomoDRGN ignores but maintains all other headers, notably allowing interactive post-training analysis to probe per-particle correlations with user-defined metadata


## Contact
Please reach out with bug reports, feature requests, etc to bmp[at]mit[dot]edu.