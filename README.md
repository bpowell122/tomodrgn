# TomoDRGN: a cryoDRGN fork for heterogeneity analysis of particles from cryo-electron tomography

<p style="text-align: center;">
    <img src="assets/empiar10499_00255_ribosomes.png" alt="Unique per-particle ribosome volumes, calculated from a tomoDRGN variational autoencoder trained on EMPIAR-10499 _Mycoplasma pneumonaie_ ribosomes, mapped back to the tomographic cellular environment"/> 
</p>

[CryoDRGN](https://github.com/zhonge/cryodrgn) has proven a powerful deep learning method for heterogeneity analysis in single particle cryo-EM. In particular, the method models a continuous distribution over 3D structures by using a Variational Auto-Encoder (VAE) based architecture to generate a reconstruction voxel-by-voxel once given a fixed coordinate from a continuous learned  latent space.

TomoDRGN extends the cryoDRGN framework to cryo-ET by learning heterogeneity from datasets in which each particle is sampled by multiple projection images at different stage tilt angles. For cryo-ET samples imaging particles _in situ_, tomoDRGN therefore enables continuous heterogeneity analysis at a single particle level within the native cellular environment. This new type of input necessitates modification of the cryoDRGN architecture, enables tomography-specific processing opportunities (e.g. dose weighting for loss weighting and efficient voxel subset evaluation during training), and benefits from tomography-specific interactive visualizations.

## Installation / dependencies:
TomoDRGN shares essentially all of cryoDRGN's dependencies, with the addition of a few new ones such as ipyvolume (for interactive 3D visualization of results). Therefore, users may choose to update existing cryoDRGN environments to run tomoDRGN as well, or instead to set up a dedicated tomoDRGN environment. However, we advise users to set up a separate tomoDRGN environment if possible.

Setting up a dedicated tomoDRGN environment (recommended):

    # Create conda environment
    conda create --name tomodrgn "python>=3.7"
    conda activate tomodrgn
      
    # Install dependencies
    conda install "pytorch-gpu>=1.8.0" "cudatoolkit>=11.0" -c pytorch  # TODO this should really be pytorch-gpu>=1.11.0 (inference mode, mem check, etc)
    conda install "pandas>=1.3.5" "seaborn>=0.11.2" "scikit-learn>=1.0.2"
    conda install "umap-learn>=0.5.3" "cufflinks-py>=0.17.3" "ipyvolume>=0.5" "healpy>=1.16.1"  "ipywidgets<8.0" "typing_extensions>=3.7.4" "pythreejs<2.4.0" "notebook>5.3" -c conda-forge
    pip install ipyvolume==0.6.0a10

    # Clone source code and install
    git clone https://github.com/bpowell122/tomodrgn.git
    cd tomodrgn
    pip install .

<details>
   <summary> Additional setup details </summary>

Adding tomoDRGN to an existing cryoDRGN environment (possible but not recommended):

    # Activate cryodrgn conda environment
    conda activate cryodrgn

    # Install new dependencies
    conda install ipywidgets healpy -c conda-forge
    pip install ipyvolume>=0.6.0a10
    jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly plotlywidget --no-build
    jupyter lab build

    # Clone source code and install
    git clone https://github.com/bpowell122/tomodrgn.git
    cd tomodrgn
    pip install .

Note: the majority of these dependency pins are in place to help conda solve the dependency matrix in reasonable runtime using known workable versions, but are not explicitly required. The *truly* required pins are:
* cudatoolkit>=11.0 : nvidia 30-series GPUs with higher CUDA capability (eg. sm_86) require this cudatoolkit version
* typing_extensions>=3.7.4 : older versions cause an error when initializing some tomodrgn scripts
* ipyvolume>=0.5 : tomodrgn's analysis notebook uses some features introduced in 0.5
* ipywidgets<8.0 : ipyvolume is not compatible with ipywidgets>=8.0 as of 9/1/22, see [here](https://github.com/widgetti/ipyvolume/pull/411)
* pythreejs<2.4.0 : 2.4 causes ipyvolume interactive widgets to not render and jupyter notebook to freeze
* ipyvolume==0.6.0a10 : tomodrgn's analysis notebook uses some features introduced in 0.6.0a10, which is itself not currently available on conda-forge

</details>

Optional: verify code+dependency functionality on your system
    
    cd tomodrgn/testing
    bash ./quicktest.sh  # ~1 minute, tests train_vae and analyze
    bash ./unittest.sh  # ~20 minutes, tests all commands with multiple options (except jupyter notebooks)

## Example usage
Below are minimal examples of various common tomoDRGN commands. By design, syntax parallels cryoDRGN's syntax where possible. All commands require initialization of the conda environment: `conda activate tomodrgn`

1. Train a decoder-only network to learn a homogeneous structure: `tomodrgn train_nn particles_imageseries.star --outdir 01_nn_particles --num-epochs 30`
    <details> 
        <summary> Click to see all options: <code> tomodrgn train_nn --help </code> </summary>
   
        usage: tomodrgn train_nn [-h] -o OUTDIR [--load WEIGHTS.PKL] [--checkpoint CHECKPOINT] [--log-interval LOG_INTERVAL] [-v]
                                 [--seed SEED] [--ind IND] [--uninvert-data] [--no-window] [--window-r WINDOW_R] [--datadir DATADIR]
                                 [--lazy] [--Apix APIX] [--recon-tilt-weight] [--recon-dose-weight] [--dose-override DOSE_OVERRIDE]
                                 [--l-dose-mask] [--sample-ntilts SAMPLE_NTILTS] [-n NUM_EPOCHS] [-b BATCH_SIZE] [--wd WD] [--lr LR]
                                 [--norm NORM NORM] [--no-amp] [--multigpu] [--layers LAYERS] [--dim DIM] [--l-extent L_EXTENT]
                                 [--pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}] [--pe-dim PE_DIM]
                                 [--activation {relu,leaky_relu}] [--feat-sigma FEAT_SIGMA]
                                 particles
        
        Train a NN to model a 3D density map given 2D images from a tilt series with consensus pose assignments
        
        positional arguments:
          particles             Input particles_imageseries.star
        
        options:
          -h, --help            show this help message and exit
          -o OUTDIR, --outdir OUTDIR
                                Output directory to save model (default: None)
          --load WEIGHTS.PKL    Initialize training from a checkpoint (default: None)
          --checkpoint CHECKPOINT
                                Checkpointing interval in N_EPOCHS (default: 1)
          --log-interval LOG_INTERVAL
                                Logging interval in N_PTCLS (default: 200)
          -v, --verbose         Increases verbosity (default: False)
          --seed SEED           Random seed (default: 40184)
        
        Dataset loading:
          --ind IND             Filter particle stack by these indices (default: None)
          --uninvert-data       Do not invert data sign (default: True)
          --no-window           Turn off real space windowing of dataset (default: True)
          --window-r WINDOW_R   Windowing radius (default: 0.85)
          --datadir DATADIR     Path prefix to particle stack if loading relative paths from a .star or .cs file (default: None)
          --lazy                Lazy loading if full dataset is too large to fit in memory (default: False)
          --Apix APIX           Override A/px from input starfile; useful if starfile does not have _rlnDetectorPixelSize col (default:
                                1.0)
          --recon-tilt-weight   Weight reconstruction loss by cosine(tilt_angle) (default: False)
          --recon-dose-weight   Weight reconstruction loss per tilt per pixel by dose dependent amplitude attenuation (default: False)
          --dose-override DOSE_OVERRIDE
                                Manually specify dose in e- / A2 / tilt (default: None)
          --l-dose-mask         Do not train on frequencies exposed to > 2.5x critical dose. Training lattice is intersection of this with
                                --l-extent (default: False)
          --sample-ntilts SAMPLE_NTILTS
                                Number of tilts to sample from each particle per epoch. Default: min(ntilts) from dataset (default: None)
        
        Training parameters:
          -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                                Number of training epochs (default: 20)
          -b BATCH_SIZE, --batch-size BATCH_SIZE
                                Minibatch size (default: 1)
          --wd WD               Weight decay in Adam optimizer (default: 0)
          --lr LR               Learning rate in Adam optimizer (scale linearly with --batch-size) (default: 0.0002)
          --norm NORM NORM      Data normalization as shift, 1/scale (default: mean, std of dataset) (default: None)
          --no-amp              Disable use of mixed-precision training (default: False)
          --multigpu            Parallelize training across all detected GPUs. Specify GPUs i,j via `export CUDA_VISIBLE_DEVICES=i,j`
                                before tomodrgn train_vae (default: False)
        
        Network Architecture:
          --layers LAYERS       Number of hidden layers (default: 3)
          --dim DIM             Number of nodes in hidden layers (default: 256)
          --l-extent L_EXTENT   Coordinate lattice size (if not using positional encoding) (default: 0.5)
          --pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}
                                Type of positional encoding (default: geom_lowf)
          --pe-dim PE_DIM       Num sinusoid features in positional encoding (default: D/2) (default: None)
          --activation {relu,leaky_relu}
                                Activation (default: relu)
          --feat-sigma FEAT_SIGMA
                                Scale for random Gaussian features (default: 0.5)
    </details>

2. Assess convergence of a decoder-only network relative to an external volume: `tomodrgn convergence_nn 01_nn_particles sta_reference_volume.mrc`
   <details> 
        <summary> Click to see all options: <code> tomodrgn convergence_nn --help </code> </summary>

        usage: tomodrgn convergence_nn [-h] [--max-epoch MAX_EPOCH] [--include-dc] [--fsc-mask {none,sphere,tight,soft}]
                                       training_directory reference_volume
        
        Assess convergence of a decoder-only network relative to an external volume by FSC
        
        positional arguments:
          training_directory    train_nn directory containing reconstruct.N.mrc
          reference_volume      volume against which to calculate FSC
        
        options:
          -h, --help            show this help message and exit
          --max-epoch MAX_EPOCH
                                Maximum epoch for which to calculate FSCs (default: None)
          --include-dc          Include FSC calculation for DC component, default False because DC component default excluded during
                                training (default: False)
          --fsc-mask {none,sphere,tight,soft}
                                Type of mask applied to volumes before calculating FSC (default: soft)
    </details>

3. Train a VAE to simultaneously learn a latent heterogeneity landscape and a volume-generative decoder: `tomodrgn train_vae particles_imageseries.star --zdim 8 --outdir 02_vae_z8 --num-epochs 30`
   <details> 
        <summary> Click to see all options: <code> tomodrgn train_vae --help </code> </summary>
   
        usage: tomodrgn train_vae [-h] -o OUTDIR --zdim ZDIM [--load WEIGHTS.PKL] [--checkpoint CHECKPOINT] [--log-interval LOG_INTERVAL]
                                  [-v] [--seed SEED] [--ind PKL] [--uninvert-data] [--no-window] [--window-r WINDOW_R] [--datadir DATADIR]
                                  [--lazy] [--Apix APIX] [--recon-tilt-weight] [--recon-dose-weight] [--dose-override DOSE_OVERRIDE]
                                  [--l-dose-mask] [--sample-ntilts SAMPLE_NTILTS] [-n NUM_EPOCHS] [-b BATCH_SIZE] [--wd WD] [--lr LR]
                                  [--beta BETA] [--beta-control BETA_CONTROL] [--norm NORM NORM] [--no-amp] [--multigpu]
                                  [--enc-layers-A QLAYERSA] [--enc-dim-A QDIMA] [--out-dim-A OUT_DIM_A] [--enc-layers-B QLAYERSB]
                                  [--enc-dim-B QDIMB] [--enc-mask ENC_MASK] [--dec-layers PLAYERS] [--dec-dim PDIM] [--l-extent L_EXTENT]
                                  [--pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}]
                                  [--feat-sigma FEAT_SIGMA] [--pe-dim PE_DIM] [--activation {relu,leaky_relu}]
                                  particles
        
        Train a VAE for heterogeneous reconstruction with known pose for tomography data
        
        positional arguments:
          particles             Input particles (.mrcs, .star, .cs, or .txt)
        
        options:
          -h, --help            show this help message and exit
          -o OUTDIR, --outdir OUTDIR
                                Output directory to save model (default: None)
          --zdim ZDIM           Dimension of latent variable (default: None)
          --load WEIGHTS.PKL    Initialize training from a checkpoint (default: None)
          --checkpoint CHECKPOINT
                                Checkpointing interval in N_EPOCHS (default: 1)
          --log-interval LOG_INTERVAL
                                Logging interval in N_PTCLS (default: 200)
          -v, --verbose         Increases verbosity (default: False)
          --seed SEED           Random seed (default: 38057)
        
        Dataset loading:
          --ind PKL             Filter particle stack by these indices (default: None)
          --uninvert-data       Do not invert data sign (default: True)
          --no-window           Turn off real space windowing of dataset (default: True)
          --window-r WINDOW_R   Windowing radius (default: 0.85)
          --datadir DATADIR     Path prefix to particle stack if loading relative paths from a .star or .cs file (default: None)
          --lazy                Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD) (default: False)
          --Apix APIX           Override A/px from input starfile; useful if starfile does not have _rlnDetectorPixelSize col (default:
                                1.0)
          --recon-tilt-weight   Weight reconstruction loss by cosine(tilt_angle) (default: False)
          --recon-dose-weight   Weight reconstruction loss per tilt per pixel by dose dependent amplitude attenuation (default: False)
          --dose-override DOSE_OVERRIDE
                                Manually specify dose in e- / A2 / tilt (default: None)
          --l-dose-mask         Do not train on frequencies exposed to > 2.5x critical dose. Training lattice is intersection of this with
                                --l-extent (default: False)
          --sample-ntilts SAMPLE_NTILTS
                                Number of tilts to sample from each particle per epoch. Default: min(ntilts) from dataset (default: None)
        
        Training parameters:
          -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                                Number of training epochs (default: 20)
          -b BATCH_SIZE, --batch-size BATCH_SIZE
                                Minibatch size (default: 1)
          --wd WD               Weight decay in Adam optimizer (default: 0)
          --lr LR               Learning rate in Adam optimizer (default: 0.0002)
          --beta BETA           Choice of beta schedule or a constant for KLD weight (default: None)
          --beta-control BETA_CONTROL
                                KL-Controlled VAE gamma. Beta is KL target (default: None)
          --norm NORM NORM      Data normalization as shift, 1/scale (default: 0, std of dataset) (default: None)
          --no-amp              Disable use of mixed-precision training (default: False)
          --multigpu            Parallelize training across all detected GPUs (default: False)
        
        Encoder Network:
          --enc-layers-A QLAYERSA
                                Number of hidden layers for each tilt (default: 3)
          --enc-dim-A QDIMA     Number of nodes in hidden layers for each tilt (default: 256)
          --out-dim-A OUT_DIM_A
                                Number of nodes in output layer of encA == ntilts * number of nodes input to encB (default: 128)
          --enc-layers-B QLAYERSB
                                Number of hidden layers encoding merged tilts (default: 1)
          --enc-dim-B QDIMB     Number of nodes in hidden layers encoding merged tilts (default: 256)
          --enc-mask ENC_MASK   Circular mask of image for encoder (default: D/2; -1 for no mask) (default: None)
        
        Decoder Network:
          --dec-layers PLAYERS  Number of hidden layers (default: 3)
          --dec-dim PDIM        Number of nodes in hidden layers (default: 256)
          --l-extent L_EXTENT   Coordinate lattice size (if not using positional encoding) (default: 0.5)
          --pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}
                                Type of positional encoding (default: geom_lowf)
          --feat-sigma FEAT_SIGMA
                                Scale for random Gaussian features (default: 0.5)
          --pe-dim PE_DIM       Num features in positional encoding (default: None)
          --activation {relu,leaky_relu}
                                Activation (default: relu)
   </details>

4. Assess convergence of a VAE model after 30 epochs training using internal / self-consistency heuristics: `tomodrgn convergence_vae 02_vae_z8 29`
   <details> 
        <summary> Click to see all options: <code> tomodrgn convergence_vae --help </code> </summary>

       usage: tomodrgn convergence_vae [-h] [-o OUTDIR] [--epoch-interval EPOCH_INTERVAL] [--force-umap-cpu] [--subset SUBSET]
                                       [--random-seed RANDOM_SEED] [--random-state RANDOM_STATE] [--n-epochs-umap N_EPOCHS_UMAP]
                                       [--skip-umap] [--n-bins N_BINS] [--smooth SMOOTH] [--smooth-width SMOOTH_WIDTH]
                                       [--pruned-maxima PRUNED_MAXIMA] [--radius RADIUS] [--final-maxima FINAL_MAXIMA] [--Apix APIX]
                                       [--flip] [-d DOWNSAMPLE] [--cuda CUDA] [--skip-volgen] [--max-threads MAX_THREADS]
                                       [--thresh THRESH] [--dilate DILATE] [--dist DIST]
                                       workdir epoch
       
       Assess convergence and training dynamics of a heterogeneous VAE network
       
       positional arguments:
         workdir               Directory with tomoDRGN results
         epoch                 Latest epoch number N to analyze convergence (0-based indexing, corresponding to z.N.pkl, weights.N.pkl
       
       options:
         -h, --help            show this help message and exit
         -o OUTDIR, --outdir OUTDIR
                               Output directory for convergence analysis results (default: [workdir]/convergence.[epoch]) (default: None)
         --epoch-interval EPOCH_INTERVAL
                               Interval of epochs between calculating most convergence heuristics (default: 5)
       
       UMAP  calculation arguments:
         --force-umap-cpu      Override default UMAP GPU-bound implementation via cuML to use umap-learn library instead (default: False)
         --subset SUBSET       Max number of particles to be used for UMAP calculations. 'None' means use all ptcls (default: 50000)
         --random-seed RANDOM_SEED
                               Manually specify the seed used for selection of subset particles (default: None)
         --random-state RANDOM_STATE
                               Random state seed used by UMAP for reproducibility at slight cost of performance (default 42, None means
                               slightly faster but non-reproducible) (default: 42)
         --n-epochs-umap N_EPOCHS_UMAP
                               Number of epochs to train the UMAP embedding via cuML for a given z.pkl, as described in the cuml.UMAP
                               documentation (default: 25000)
         --skip-umap           Skip UMAP embedding. Requires that UMAP be precomputed for downstream calcs. Useful for tweaking volume
                               generation settings. (default: False)
       
       Sketching UMAP via local maxima arguments:
         --n-bins N_BINS       the number of bins along UMAP1 and UMAP2 (default: 30)
         --smooth SMOOTH       smooth the 2D histogram before identifying local maxima (default: True)
         --smooth-width SMOOTH_WIDTH
                               width of gaussian kernel for smoothing 2D histogram expressed as multiple of one bin's width (default:
                               1.0)
         --pruned-maxima PRUNED_MAXIMA
                               prune poorly-separated maxima until this many maxima remain (default: 12)
         --radius RADIUS       distance at which two maxima are considered poorly-separated and are candidates for pruning (euclidean
                               distance in bin-space) (default: 5.0)
         --final-maxima FINAL_MAXIMA
                               select this many local maxima, sorted by highest bin count after pruning, for which to generate volumes
                               (default: 10)
       
       Volume generation arguments:
         --Apix APIX           A/pix of output volume (default: 1.0)
         --flip                Flip handedness of output volume (default: False)
         -d DOWNSAMPLE, --downsample DOWNSAMPLE
                               Downsample volumes to this box size (pixels). Recommended for boxes > 250-300px (default: None)
         --cuda CUDA           Specify cuda device for volume generation (default: None)
         --skip-volgen         Skip volume generation. Requires that volumes already exist for downstream CC + FSC calcs (default: False)
       
       Mask generation arguments:
         --max-threads MAX_THREADS
                               Max number of threads used to parallelize mask generation (default: 8)
         --thresh THRESH       Float, isosurface at which to threshold mask; default None uses 50th percentile (default: None)
         --dilate DILATE       Number of voxels to dilate thresholded isosurface outwards from mask boundary (default: 3)
         --dist DIST           Number of voxels over which to apply a soft cosine falling edge from dilated mask boundary (default: 10)
    </details>

5. Run standard analysis of a VAE model after 30 epochs training (PCA and UMAP of latent space, volume generation along PC's, _k_-means sampling of latent space and generation of corresponding volumes): `tomodrgn analyze 02_vae_z8 29 --Apix 3.5`
    <details> 
        <summary> Click to see all options: <code> tomodrgn analyze --help </code> </summary>

        usage: tomodrgn analyze [-h] [--device DEVICE] [-o OUTDIR] [--skip-vol] [--skip-umap] [--Apix APIX] [--flip] [-d DOWNSAMPLE]
                                [--pc PC] [--pc-ondata] [--ksample KSAMPLE]
                                workdir epoch
        
        Visualize latent space and generate volumes
        
        positional arguments:
          workdir               Directory with tomoDRGN results
          epoch                 Epoch number N to analyze (0-based indexing, corresponding to z.N.pkl, weights.N.pkl)
        
        options:
          -h, --help            show this help message and exit
          --device DEVICE       Optionally specify CUDA device (default: None)
          -o OUTDIR, --outdir OUTDIR
                                Output directory for analysis results (default: [workdir]/analyze.[epoch]) (default: None)
          --skip-vol            Skip generation of volumes (default: False)
          --skip-umap           Skip running UMAP (default: False)
        
        Extra arguments for volume generation:
          --Apix APIX           Pixel size to add to .mrc header (default: 1 A/pix)
          --flip                Flip handedness of output volumes (default: False)
          -d DOWNSAMPLE, --downsample DOWNSAMPLE
                                Downsample volumes to this box size (pixels) (default: None)
          --pc PC               Number of principal component traversals to generate (default: 2)
          --pc-ondata           Find closest on-data latent point to each PC percentile (default: False)
          --ksample KSAMPLE     Number of kmeans samples to generate (default: 20)
    </details>

6. Interactive analysis of a VAE model: run `jupyter notebook` to open `02_vae_z8/analyze.29/tomoDRGN_viz+filt_template.ipynb` generated by `tomodrgn analyze`. Note: certain widget features used in the notebook are only available in jupyter notebook, not jupyter lab. 
    <details> 
        <summary> Click to see two ways to run jupyter notebook </summary>
    To run a local instance of Jupyter Notebook (e.g. you are viewing a monitor directly connected to a computer with direct access to the filesystem containing 02_vae_z8_box256 and the tomodrgn conda environment):
   
         jupyter notebook
    To run a remote instance of Jupyter Notebook (e.g. you are viewing a monitor NOT connected to a computer with direct access to the filesystem containing 02_vae_z8_box256 and the tomodrgn conda environment, perhaps having run tomodrgn on a remote cluster):
         
         # In one terminal window, set up port forwarding to your local machine
         ssh -t -t username@cluster-head-node -L 8888:localhost:8888 ssh active-worker-node -L 8888:localhost:8888

         # In a second terminal window connected to your remote system, launch the notebook
         jupyter notebook --no-browser --port 8888
   </details>

7. Filter a star file (imageseries or volumeseries) by particle indices identified and written by `tomoDRGN_viz+filt_template.ipynb`: `tomodrgn filter_star particles.star --ind ind.pkl --outstar particles_filt.star` 
    <details> 
        <summary> Click to see all options: <code> tomodrgn filter_star --help </code> </summary>
   
        usage: tomodrgn filter_star [-h] [--tomo-id-col TOMO_ID_COL] [--ptcl-id-col PTCL_ID_COL] [--ind IND] [--ind-type {particle,image}]
                                    [--tomogram TOMOGRAM] [--action {keep,drop}] -o O
                                    input
        
        Filter a .star file generated by Warp subtomogram export
        
        positional arguments:
          input                 Input .star file
        
        options:
          -h, --help            show this help message and exit
          --tomo-id-col TOMO_ID_COL
                                Name of column in input starfile with unique values per tomogram (default: _rlnMicrographName)
          --ptcl-id-col PTCL_ID_COL
                                Name of column in input starfile with unique values per particle, if `index` then each row is treated as a
                                unique particle (default: _rlnGroupName)
          --ind IND             selected indices array (.pkl) (default: None)
          --ind-type {particle,image}
                                use indices to filter by particle or by individual image (default: particle)
          --tomogram TOMOGRAM   optionally select by individual tomogram name (if `all` then writes individual star files per tomogram
                                (default: None)
          --action {keep,drop}  keep or remove particles associated with ind.pkl (default: keep)
          -o O                  Output .star file (default: None)
    </details>

8. Generate 3-D volumes from (per-particle) unique positions in latent space: `tomodrgn eval_vol 02_vae_z8/weights.pkl --config 02_vae_z8/config.pkl -o 02_vae_z8/tomogram_vols --zfile 02_vae_z8/z.pkl --downsample 64`. This command can benefit from per-tomomgram `z.pkl` filtering within `tomoDRGN_viz+filt_template.ipynb` to create volumes for all particles associated with a particular tomogram.
    <details> 
        <summary> Click to see all options: <code> tomodrgn eval_vol --help </code> </summary>
   
        usage: tomodrgn eval_vol [-h] -c CONFIG -o O [--prefix PREFIX] [--no-amp] [-v] [-z [Z ...]] [--z-start [Z_START ...]]
                                 [--z-end [Z_END ...]] [-n N] [--zfile ZFILE] [--Apix APIX] [--flip] [--invert] [-d DOWNSAMPLE]
                                 weights
        
        Evaluate the decoder at specified values of z
        
        positional arguments:
          weights               Model weights.pkl from train_vae
        
        options:
          -h, --help            show this help message and exit
          -c CONFIG, --config CONFIG
                                config.pkl file from train_vae (default: None)
          -o O                  Output .mrc or directory (default: None)
          --prefix PREFIX       Prefix when writing out multiple .mrc files (default: vol_)
          --no-amp              Disable use of mixed-precision training (default: False)
          -v, --verbose         Increases verbosity (default: False)
        
        Specify z values:
          -z [Z ...]            Specify one z-value (default: None)
          --z-start [Z_START ...]
                                Specify a starting z-value (default: None)
          --z-end [Z_END ...]   Specify an ending z-value (default: None)
          -n N                  Number of structures between [z_start, z_end] (default: 10)
          --zfile ZFILE         Text/.pkl file with z-values to evaluate (default: None)
        
        Volume arguments:
          --Apix APIX           Pixel size to add to output .mrc header (default: 1)
          --flip                Flip handedness of output volume (default: False)
          --invert              Invert contrast of output volume (default: False)
          -d DOWNSAMPLE, --downsample DOWNSAMPLE
                                Downsample volumes to this box size (pixels) (default: None)

    </details>

9. Write a ChimeraX script to place tomoDRGN-generated individual particle volumes in the spatial context of a chosen tomogram with optional color mapping: `tomodrgn subtomo2chimerax particles_volumeseries.star -o mytomogram_tomodrgnvols.cxs --tomoname mytomogram.tomostar --vols-dir 02_vae_z8/tomogram_vols --coloring-labels  02_vae_z8/analyze.29/kmeans20/labels.pkl`
    <details> 
        <summary> Click to see all options: <code> tomodrgn subtomo2chimerax --help </code> </summary>

        usage: tomodrgn subtomo2chimerax [-h] -o OUTFILE [--tomoname TOMONAME] [--tomo-id-col TOMO_ID_COL]
                                         [--star-apix-override STAR_APIX_OVERRIDE] --vols-dir VOLS_DIR
                                         [--vols-apix-override VOLS_APIX_OVERRIDE] [--ind IND] [--vols-render-level VOLS_RENDER_LEVEL]
                                         [--coloring-labels COLORING_LABELS] [--matplotlib-colormap MATPLOTLIB_COLORMAP]
                                         starfile
        
        Create a .cxs script for ChimeraX to arrange uniquely generated tomoDRGN volumes into tomogram with optional label-based color
        mapping Adapted from relionsubtomo2ChimeraX.py, written by Huy Bui, McGill University, doi: https://doi.org/10.5281/zenodo.6820119
        
        positional arguments:
          starfile              Input particle_volumeseries starfile from Warp subtomogram export
        
        options:
          -h, --help            show this help message and exit
          -o OUTFILE, --outfile OUTFILE
                                Output .cxc script to be opened in ChimeraX (default: None)
          --tomoname TOMONAME   Name of tomogram in input starfile for which to display tomoDRGN vols in ChimeraX (default: None)
          --tomo-id-col TOMO_ID_COL
                                Name of column in input starfile to filter by --tomoname (default: _rlnMicrographName)
          --star-apix-override STAR_APIX_OVERRIDE
                                Override pixel size of input particle_volumeseries starfile (A/px) (default: None)
          --vols-dir VOLS_DIR   Path to tomoDRGN reconstructed volumes (default: None)
          --vols-apix-override VOLS_APIX_OVERRIDE
                                Override pixel size of tomoDRGN-reconstructed particle volumes (A/px) (default: None)
          --ind IND             Ind.pkl used in training run that produced volumes in vols-dir (if applicable) (default: None)
        
        ChimeraX rendering options:
          --vols-render-level VOLS_RENDER_LEVEL
                                Isosurface level to render all tomoDRGN reconstructed volumes in ChimeraX (default: 0.7)
          --coloring-labels COLORING_LABELS
                                labels.pkl file used to assign colors to each volume (typically kmeans labels.pkl (default: None)
          --matplotlib-colormap MATPLOTLIB_COLORMAP
                                Name of colormap to sample labels, from https://matplotlib.org/stable/tutorials/colors/colormaps.html
                                (default: tab20)
    </details>

10. Generate latent embeddings for new images using a pretrained model: `tomodrgn eval_images particles_imageseries_new.star 02_vae_z8/weights.pkl -c 02_vae_z8/config.pkl --out-z 02_vae_z8/eval_images/z.pkl`
    <details> 
        <summary> Click to see all options: <code> tomodrgn eval_images --help </code> </summary>

        usage: tomodrgn eval_images [-h] -c CONFIG --out-z PKL [--log-interval LOG_INTERVAL] [-b BATCH_SIZE] [-v] [--lazy]
                                    particles weights
        
        Evaluate z for a stack of images
        
        positional arguments:
          particles             Input particles (.mrcs, .star, .cs, or .txt)
          weights               Model weights
        
        options:
          -h, --help            show this help message and exit
          -c CONFIG, --config CONFIG
                                config.pkl file from train_vae (default: None)
          --out-z PKL           Output pickle for z (default: None)
          --log-interval LOG_INTERVAL
                                Logging interval in N_IMGS (default: 1000)
          -b BATCH_SIZE, --batch-size BATCH_SIZE
                                Minibatch size (default: 64)
          -v, --verbose         Increases verbosity (default: False)
        
        Dataset loading:
          --lazy                Lazy loading if full dataset is too large to fit in memory (default: False)
    </details>

11. Find particle indices and latent embeddings most directly connecting chosen "anchor" particles in latent space: `tomodrgn graph_traversal 02_vae_z8/z.pkl --anchors 137 10 20 -o 02_vae_z8/graph_traversal/path.txt --out-z 02_vae_z8/graph_traversal/z.path.txt`
    <details> 
        <summary> Click to see all options: <code> tomodrgn graph_traversal --help </code> </summary>

        usage: tomodrgn graph_traversal [-h] --anchors ANCHORS [ANCHORS ...] [--max-neighbors MAX_NEIGHBORS]
                                        [--avg-neighbors AVG_NEIGHBORS] [--batch-size BATCH_SIZE] [--max-images MAX_IMAGES] -o O --out-z
                                        OUT_Z
                                        data
        
        Find shortest path along nearest neighbor graph
        
        positional arguments:
          data                  Input z.pkl embeddings
        
        options:
          -h, --help            show this help message and exit
          --anchors ANCHORS [ANCHORS ...]
                                Index of anchor points (default: None)
          --max-neighbors MAX_NEIGHBORS
          --avg-neighbors AVG_NEIGHBORS
          --batch-size BATCH_SIZE
          --max-images MAX_IMAGES
          -o O                  Output .txt or .pkl file for path indices (default: None)
          --out-z OUT_Z         Output .txt or .pkl file for path z-values (default: None)
    </details>


## Training requirements and limitations
TomoDRGN model training requires as inputs:
1. a .mrcs stack containing the 2D pixel array of each particle extracted from each stage tilt micrograph, and
2. a corresponding .star file describing pose and CTF parameters for each particle's tilt image as derived from traditional subtomogram averaging. 

At the moment, the only tested workflow to generate these inputs is Warp+IMOD tomogram generation, STA using RELION 3.1, followed by subtomogram extraction as image series in Warp (making use of the excellent [dynamo2warp](https://github.com/alisterburt/dynamo2m) tools from Alister Burt). We are actively seeking to expand the range of validated workflows; please let us know if you have success with a novel approach.


## Additional details about required inputs for train_vae
TomoDRGN requires a number of specific metadata values to be supplied in the star file (a sample star file is provided at `tomodrgn/testing/data/10076_both_32_sim.star`). Users testing upstream workflows that do not conclude with 2D particle extraction in Warp should ensure their RELION 3.0 format star files contain the following headers:
* `_rlnGroupName`: used to identify multiple tilt images associated with the same particle, with format `tomogramID_particleID` as used in Warp (e.g. `001_000004` identifies particle 5 from tomogram 2)
* `_rlnCtfScalefactor`: used to calculate stage tilt of each image, equal to `cosine(stage_tilt_radians)` as defined in Warp
* `_rlnCtfBfactor`: used to calculate cumulative dose imparted to each image, equal to `dose_e-_per_A2 * -4` as defined in Warp
* `_rlnImageName`: used to load image data
* `'_rlnDetectorPixelSize','_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration', '_rlnAmplitudeContrast', '_rlnPhaseShift'`: used to calculate the CTF and optimal dose
* `_rlnAngleRot, _rlnAngleTilt, _rlnAnglePsi`: used to calculate pose as defined in RELION
* `_rlnOriginX, _rlnOriginY`: used to calculate pose as defined in RELION (optional, Warp re-centers upon particle extraction and therefore does not have these headers by default)
* tomoDRGN ignores but maintains all other headers, notably allowing interactive post-training analysis to probe per-particle correlations with user-defined metadata


## Changelog
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
1. TomoDRGN (bioRxiv) 2022
2. Zhong, E., Bepler, T., Berger, B. & Davis, J. CryoDRGN: Reconstruction of Heterogeneous cryo-EM Structures Using Neural Networks. Nature Methods, [doi:10.1038/s41592-020-01049-4](https://doi.org/10.1038/s41592-020-01049-4) (2021)
3. Kinman, L., Powell, B., Zhong, E., Berger, B. & Davis, J. Uncovering structural ensembles from single particle cryo-EM data using cryoDRGN. bioRxiv, [doi:10.1101/2022.08.09.503342](https://doi.org/10.1101/2022.08.09.503342) (2022)
4. Sun, J., Kinman, L., Jahagirdar, D., Ortega, J., Davis. J. KsgA facilitates ribosomal small subunit maturation by proofreading a key structural lesion. bioRxiv, [doi:10.1101/2022.07.13.499473](https://doi.org/10.1101/2022.07.13.499473) (2022)


## Contact
Please reach out with bug reports, feature requests, etc to bmp[at]mit[dot]edu.