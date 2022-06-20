# TomoDRGN: a cryoDRGN fork for heterogeneity analysis of particles from cryo-ET

[CryoDRGN](https://github.com/zhonge/cryodrgn) is a neural network based algorithm for heterogeneous cryo-EM reconstruction. In particular, the method models a *continuous* distribution over 3D structures by using a neural network based representation for the volume.

TomoDRGN extends on the cryoDRGN framework to cryo-ET by handling data in which each particle's heterogeneity is sampled by multiple projection images at different stage tilt angles. For samples containing particles _in situ_, tomoDRGN therefore enables continuous heterogeneity analysis at a single particle level within the native cellular context. This new data format necessitates modification of the cryoDRGN architecture, and enables tomography-specific processing opportunities (eg dose weighting for loss weighting and efficient voxel subset evaluation during training).

TomoDRGN aims to parallel familiar cryoDRGN usage syntax and integrate directly with pre-existing cryoDRGN analysis and visualization tools. 


## Installation/dependencies:

TomoDRGN shares essentially all of cryoDRGN's dependencies, with the addition of a few new ones such as ipyvolume. Therefore users may choose to update their existing cryoDRGN environments to run tomoDRGN as well (skipping the "create conda environment" section below), or to set up a dedicated tomoDRGN environment.

To install tomoDRGN, git clone the source code and install the following dependencies with anaconda, replacing the cudatoolkit version as necessary:

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


## Requirements and Limitations
TomoDRGN requires as inputs a .mrcs stack containing the 2D pixel array of each particle extracted from at each stage tilt micrograph, and a corresponding .star file describing pose and CTF parameters for each particle as derived from traditional subtomogram averaging. At the moment, the only tested workflow to generate these inputs is Warp+IMOD tomogram processing, STA using RELION 3.1, followed by particle extraction in Warp (making use of the excellent [dynamo2warp](https://github.com/alisterburt/dynamo2m) tools from Alister Burt). 

Additionally, the current encoder architecture requires that all particles have images from the same set of stage tilts. This means a dataset containing 41 tilts for tilt series A, and 40 tilts after discarding 1 tilt for tilt series B, will crash during training. The current recommended workaround is to limit all datasets to the minimal set of common tilt angles during final particle export. A proper fix is in the works for the next tomoDRGN release. 


## Contact

Please reach out with bug reports, feature requests, etc to bmp[at]mit[dot]edu.