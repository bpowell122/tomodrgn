# TomoDRGN: a cryoDRGN fork for heterogeneity analysis of particles from cryo-electron tomography

<p style="text-align: center;">
    <img src="assets/empiar10499_00255_ribosomes.png" alt="Unique per-particle ribosome volumes, calculated from a tomoDRGN variational autoencoder trained on EMPIAR-10499 _Mycoplasma pneumonaie_ ribosomes, mapped back to the tomographic cellular environment"/> 
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
Below are minimal examples of various common tomoDRGN commands. By design, syntax parallels cryoDRGN's syntax where possible. All commands require initialization of the conda environment: `conda activate tomodrgn`

1. Train a decoder-only network to learn a homogeneous structure: `tomodrgn train_nn particles_imageseries.star --outdir 01_nn_particles --batch-size 1 --num-epochs 30`
    <details> 
        <summary> Click to see all options: <code> tomodrgn train_nn --help </code> </summary>
   
        usage: tomodrgn train_nn [-h] -o OUTDIR [--load WEIGHTS.PKL]
                         [--checkpoint CHECKPOINT]
                         [--log-interval LOG_INTERVAL] [-v] [--seed SEED]
                         [--ind IND] [--uninvert-data] [--no-window]
                         [--window-r WINDOW_R] [--datadir DATADIR] [--lazy]
                         [--Apix APIX] [--recon-tilt-weight]
                         [--recon-dose-weight] [--dose-override DOSE_OVERRIDE]
                         [--l-dose-mask] [--sample-ntilts SAMPLE_NTILTS]
                         [-n NUM_EPOCHS] [-b BATCH_SIZE] [--wd WD] [--lr LR]
                         [--norm NORM NORM] [--no-amp] [--multigpu]
                         [--layers LAYERS] [--dim DIM] [--l-extent L_EXTENT]
                         [--pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}]
                         [--pe-dim PE_DIM] [--activation {relu,leaky_relu}]
                         [--feat-sigma FEAT_SIGMA]
                         particles

       Train a NN to model a 3D density map given 2D images from a tilt series with
       consensus pose assignments

       positional arguments:
         particles             Input particles (.mrcs, .star, .cs, or .txt)
    
       optional arguments:
         -h, --help            show this help message and exit
         -o OUTDIR, --outdir OUTDIR
                               Output directory to save model (default: None)
         --load WEIGHTS.PKL    Initialize training from a checkpoint (default: None)
         --checkpoint CHECKPOINT
                               Checkpointing interval in N_EPOCHS (default: 1)
         --log-interval LOG_INTERVAL
                               Logging interval in N_PTCLS (default: 200)
         -v, --verbose         Increases verbosity (default: False)
         --seed SEED           Random seed (default: 1605)
    
       Dataset loading:
         --ind IND             Filter particle stack by these indices (default: None)
         --uninvert-data       Do not invert data sign (default: True)
         --no-window           Turn off real space windowing of dataset (default:
                               True)
         --window-r WINDOW_R   Windowing radius (default: 0.85)
         --datadir DATADIR     Path prefix to particle stack if loading relative
                               paths from a .star or .cs file (default: None)
         --lazy                Lazy loading if full dataset is too large to fit in
                               memory (default: False)
         --Apix APIX           Override A/px from input starfile; useful if starfile
                               does not have _rlnDetectorPixelSize col (default: 1.0)
         --recon-tilt-weight   Weight reconstruction loss by cosine(tilt_angle)
                               (default: False)
         --recon-dose-weight   Weight reconstruction loss per tilt per pixel by dose
                               dependent amplitude attenuation (default: False)
         --dose-override DOSE_OVERRIDE
                               Manually specify dose in e- / A2 / tilt (default:
                               None)
         --l-dose-mask         Do not train on frequencies exposed to > 2.5x critical
                               dose. Training lattice is intersection of this with
                               --l-extent (default: False)
         --sample-ntilts SAMPLE_NTILTS
                               Number of tilts to sample from each particle per
                               epoch. Default: min(ntilts) from dataset (default:
                               None)
    
       Training parameters:
         -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                               Number of training epochs (default: 20)
         -b BATCH_SIZE, --batch-size BATCH_SIZE
                               Minibatch size (default: 8)
         --wd WD               Weight decay in Adam optimizer (default: 0)
         --lr LR               Learning rate in Adam optimizer (default: 0.0001)
         --norm NORM NORM      Data normalization as shift, 1/scale (default: mean,
                               std of dataset) (default: None)
         --no-amp              Disable use of mixed-precision training (default:
                               False)
         --multigpu            Parallelize training across all detected GPUs. Specify
                               GPUs i,j via `export CUDA_VISIBLE_DEVICES=i,j` before
                               tomodrgn train_vae (default: False)
    
       Network Architecture:
         --layers LAYERS       Number of hidden layers (default: 3)
         --dim DIM             Number of nodes in hidden layers (default: 256)
         --l-extent L_EXTENT   Coordinate lattice size (if not using positional
                               encoding) (default: 0.5)
         --pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}
                               Type of positional encoding (default: geom_lowf)
         --pe-dim PE_DIM       Num sinusoid features in positional encoding (default:
                               D/2) (default: None)
         --activation {relu,leaky_relu}
                               Activation (default: relu)
         --feat-sigma FEAT_SIGMA
                               Scale for random Gaussian features (default: 0.5)
    </details>

2. Assess convergence of a decoder-only network relative to an external volume: `tomodrgn convergence_nn 01_nn_particles sta_reference_volume.mrc`
   <details> 
        <summary> Click to see all options: <code> tomodrgn convergence_nn --help </code> </summary>

        usage: tomodrgn convergence_nn [-h] [--max-epoch MAX_EPOCH] [--include-dc]
                               [--fsc-mask {None,sphere,tight,soft}]
                               training_directory reference_volume
        Assess convergence of a decoder-only network relative to an external volume by FSC
   
        positional arguments:
          training_directory    train_nn directory containing reconstruct.N.mrc
          reference_volume      volume against which to calculate FSC
   
        optional arguments:
          -h, --help            show this help message and exit
          --max-epoch MAX_EPOCH
                                Maximum epoch for which to calculate FSCs (default:
                                None)
          --include-dc          Include FSC calculation for DC component, default
                                False because DC component default excluded during
                                training (default: False)
          --fsc-mask {None,sphere,tight,soft}
                                Type of mask applied to volumes before calculating FSC
                                (default: soft)
    </details>

3. Train a VAE to simultaneously learn a latent heterogeneity landscape and a volume-generative decoder: `tomodrgn train_vae particles_imageseries.star --zdim 8 --outdir 02_vae_z8_box256 --batch-size 1 --num-epochs 30`
   <details> 
        <summary> Click to see all options: <code> tomodrgn train_vae --help </code> </summary>
   
        usage: tomodrgn train_vae [-h] -o OUTDIR --zdim ZDIM [--load WEIGHTS.PKL]
                             [--checkpoint CHECKPOINT]
                             [--log-interval LOG_INTERVAL] [-v] [--seed SEED]
                             [--ind PKL] [--uninvert-data] [--no-window]
                             [--window-r WINDOW_R] [--datadir DATADIR] [--lazy]
                             [--Apix APIX] [--recon-tilt-weight]
                             [--recon-dose-weight]
                             [--dose-override DOSE_OVERRIDE] [--l-dose-mask]
                             [--sample-ntilts SAMPLE_NTILTS] [-n NUM_EPOCHS]
                             [-b BATCH_SIZE] [--wd WD] [--lr LR] [--beta BETA]
                             [--beta-control BETA_CONTROL] [--norm NORM NORM]
                             [--no-amp] [--multigpu] [--enc-layers-A QLAYERSA]
                             [--enc-dim-A QDIMA] [--out-dim-A OUT_DIM_A]
                             [--enc-layers-B QLAYERSB] [--enc-dim-B QDIMB]
                             [--enc-mask ENC_MASK] [--dec-layers PLAYERS]
                             [--dec-dim PDIM] [--l-extent L_EXTENT]
                             [--pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}]
                             [--feat-sigma FEAT_SIGMA] [--pe-dim PE_DIM]
                             [--activation {relu,leaky_relu}]
                             particles
   
        Train a VAE for heterogeneous reconstruction with known pose for tomography data
        
        positional arguments:
        particles             Input particles (.mrcs, .star, .cs, or .txt)
        
        optional arguments:
          -h, --help            show this help message and exit
          -o OUTDIR, --outdir OUTDIR Output directory to save model (default: None)
          --zdim ZDIM           Dimension of latent variable (default: None)
          --load WEIGHTS.PKL    Initialize training from a checkpoint (default: None)
          --checkpoint CHECKPOINT
                              Checkpointing interval in N_EPOCHS (default: 1)
          --log-interval LOG_INTERVAL
                              Logging interval in N_PTCLS (default: 200)
          -v, --verbose         Increaes verbosity (default: False)
          --seed SEED           Random seed (default: 50236)
   
        Dataset loading:
          --ind PKL             Filter particle stack by these indices (default: None)
          --uninvert-data       Do not invert data sign (default: True)
          --no-window           Turn off real space windowing of dataset (default:
                                True)
          --window-r WINDOW_R   Windowing radius (default: 0.85)
          --datadir DATADIR     Path prefix to particle stack if loading relative
                                paths from a .star or .cs file (default: None)
          --lazy                Lazy loading if full dataset is too large to fit in
                                memory (Should copy dataset to SSD) (default: False)
          --Apix APIX           Override A/px from input starfile; useful if starfile
                                does not have _rlnDetectorPixelSize col (default: 1.0)
          --recon-tilt-weight   Weight reconstruction loss by cosine(tilt_angle)
                                (default: False)
          --recon-dose-weight   Weight reconstruction loss per tilt per pixel by dose
                                dependent amplitude attenuation (default: False)
          --dose-override DOSE_OVERRIDE
                                Manually specify dose in e- / A2 / tilt (default:
                                None)
          --l-dose-mask         Do not train on frequencies exposed to > 2.5x critical
                                dose. Training lattice is intersection of this with
                                --l-extent (default: False)
          --sample-ntilts SAMPLE_NTILTS
                                Number of tilts to sample from each particle per
                                epoch. Default: min(ntilts) from dataset (default:
                                None)
        
        Training parameters:
          -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                                Number of training epochs (default: 20)
          -b BATCH_SIZE, --batch-size BATCH_SIZE
                                Minibatch size (default: 8)
          --wd WD               Weight decay in Adam optimizer (default: 0)
          --lr LR               Learning rate in Adam optimizer (default: 0.0001)
          --beta BETA           Choice of beta schedule or a constant for KLD weight
                                (default: None)
          --beta-control BETA_CONTROL
                                KL-Controlled VAE gamma. Beta is KL target (default:
                                None)
          --norm NORM NORM      Data normalization as shift, 1/scale (default: 0, std
                                of dataset) (default: None)
          --no-amp              Disable use of mixed-precision training (default:
                                False)
          --multigpu            Parallelize training across all detected GPUs
                                (default: False)
        
        Encoder Network:
          --enc-layers-A QLAYERSA
                                Number of hidden layers for each tilt (default: 3)
          --enc-dim-A QDIMA     Number of nodes in hidden layers for each tilt
                                (default: 256)
          --out-dim-A OUT_DIM_A
                                Number of nodes in output layer of encA == ntilts *
                                number of nodes input to encB (default: 128)
          --enc-layers-B QLAYERSB
                                Number of hidden layers encoding merged tilts
                                (default: 1)
          --enc-dim-B QDIMB     Number of nodes in hidden layers encoding merged tilts
                                (default: 256)
          --enc-mask ENC_MASK   Circular mask of image for encoder (default: D/2; -1
                                for no mask) (default: None)
        
        Decoder Network:
          --dec-layers PLAYERS  Number of hidden layers (default: 3)
          --dec-dim PDIM        Number of nodes in hidden layers (default: 256)
          --l-extent L_EXTENT   Coordinate lattice size (if not using positional
                                encoding) (default: 0.5)
          --pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}
                                Type of positional encoding (default: geom_lowf)
          --feat-sigma FEAT_SIGMA
                                Scale for random Gaussian features (default: 0.5)
          --pe-dim PE_DIM       Num features in positional encoding (default: None)
          --activation {relu,leaky_relu}
                                Activation (default: relu)
   </details>

4. Assess convergence of a VAE model after 30 epochs training using internal / self-consistency heuristics: `tomodrgn convergence_vae 02_vae_z8_box256 29`
   <details> 
        <summary> Click to see all options: <code> tomodrgn convergence_vae --help </code> </summary>

        usage: tomodrgn convergence_vae [-h] [-o OUTDIR]
                                [--epoch-interval EPOCH_INTERVAL]
                                [--force-umap-cpu] [--subset SUBSET]
                                [--random-seed RANDOM_SEED]
                                [--random-state RANDOM_STATE]
                                [--n-epochs-umap N_EPOCHS_UMAP] [--skip-umap]
                                [--n-bins N_BINS] [--smooth SMOOTH]
                                [--smooth-width SMOOTH_WIDTH]
                                [--pruned-maxima PRUNED_MAXIMA]
                                [--radius RADIUS]
                                [--final-maxima FINAL_MAXIMA] [--Apix APIX]
                                [--flip] [--invert] [-d DOWNSAMPLE]
                                [--cuda CUDA] [--skip-volgen]
                                [--max-threads MAX_THREADS] [--thresh THRESH]
                                [--dilate DILATE] [--dist DIST]
                                workdir epoch

        Visualize convergence and training dynamics
        
        positional arguments:
          workdir               Directory with tomoDRGN results
          epoch                 Latest epoch number N to analyze convergence (0-based
                                indexing, corresponding to z.N.pkl, weights.N.pkl
        
        optional arguments:
          -h, --help            show this help message and exit
          -o OUTDIR, --outdir OUTDIR
                                Output directory for convergence analysis results
                                (default: [workdir]/convergence.[epoch]) (default:
                                None)
          --epoch-interval EPOCH_INTERVAL
                                Interval of epochs between calculating most
                                convergence heuristics (default: 5)
        
        UMAP  calculation arguments:
          --force-umap-cpu      Override default UMAP GPU-bound implementation via
                                cuML to use umap-learn library instead (default:
                                False)
          --subset SUBSET       Max number of particles to be used for UMAP
                                calculations. 'None' means use all ptcls (default:
                                50000)
          --random-seed RANDOM_SEED
                                Manually specify the seed used for selection of subset
                                particles (default: None)
          --random-state RANDOM_STATE
                                Random state seed used by UMAP for reproducibility at
                                slight cost of performance (default 42, None means
                                slightly faster but non-reproducible) (default: 42)
          --n-epochs-umap N_EPOCHS_UMAP
                                Number of epochs to train the UMAP embedding via cuML
                                for a given z.pkl, as described in the cuml.UMAP
                                documentation (default: 25000)
          --skip-umap           Skip UMAP embedding. Requires that UMAP be precomputed
                                for downstream calcs. Useful for tweaking volume
                                generation settings. (default: False)
        
        Sketching UMAP via local maxima arguments:
          --n-bins N_BINS       the number of bins along UMAP1 and UMAP2 (default: 30)
          --smooth SMOOTH       smooth the 2D histogram before identifying local
                                maxima (default: True)
          --smooth-width SMOOTH_WIDTH
                                width of gaussian kernel for smoothing 2D histogram
                                expressed as multiple of one bin's width (default:
                                1.0)
          --pruned-maxima PRUNED_MAXIMA
                                prune poorly-separated maxima until this many maxima
                                remain (default: 12)
          --radius RADIUS       distance at which two maxima are considered poorly-
                                separated and are candidates for pruning (euclidean
                                distance in bin-space) (default: 5.0)
          --final-maxima FINAL_MAXIMA
                                select this many local maxima, sorted by highest bin
                                count after pruning, for which to generate volumes
                                (default: 10)
        
        Volume generation arguments:
          --Apix APIX           A/pix of output volume (default: 1.0)
          --flip                Flip handedness of output volume (default: False)
          --invert              Invert contrast of output volume (default: False)
          -d DOWNSAMPLE, --downsample DOWNSAMPLE
                                Downsample volumes to this box size (pixels).
                                Recommended for boxes > 250-300px (default: None)
          --cuda CUDA           Specify cuda device for volume generation (default:
                                None)
          --skip-volgen         Skip volume generation. Requires that volumes already
                                exist for downstream CC + FSC calcs (default: False)
        
        Mask generation arguments:
          --max-threads MAX_THREADS
                                Max number of threads used to parallelize mask
                                generation (default: 8)
          --thresh THRESH       Float, isosurface at which to threshold mask; default
                                None uses 50th percentile (default: None)
          --dilate DILATE       Number of voxels to dilate thresholded isosurface
                                outwards from mask boundary (default: 3)
          --dist DIST           Number of voxels over which to apply a soft cosine
                                falling edge from dilated mask boundary (default: 10)
    </details>

5. Run standard analysis of a VAE model after 30 epochs training (PCA and UMAP of latent space, volume generation along PC's, _k_-means sampling of latent space and generation of corresponding volumes): `tomodrgn analyze 02_vae_z8_box256 29 --Apix 3.5`
   <details> 
        <summary> Click to see all options: <code> tomodrgn analyze --help </code> </summary>

        usage: tomodrgn analyze [-h] [--device DEVICE] [-o OUTDIR] [--skip-vol]
                                [--skip-umap] [--Apix APIX] [--flip] [--invert]
                                [-d DOWNSAMPLE] [--pc PC] [--pc-ondata]
                                [--ksample KSAMPLE]
                                workdir epoch
        
        Visualize latent space and generate volumes
        
        positional arguments:
          workdir               Directory with tomoDRGN results
          epoch                 Epoch number N to analyze (0-based indexing,
                                corresponding to z.N.pkl, weights.N.pkl)
        
        optional arguments:
          -h, --help            show this help message and exit
          --device DEVICE       Optionally specify CUDA device (default: None)
          -o OUTDIR, --outdir OUTDIR
                                Output directory for analysis results (default:
                                [workdir]/analyze.[epoch]) (default: None)
          --skip-vol            Skip generation of volumes (default: False)
          --skip-umap           Skip running UMAP (default: False)
        
        Extra arguments for volume generation:
          --Apix APIX           Pixel size to add to .mrc header (default: 1 A/pix)
          --flip                Flip handedness of output volumes (default: False)
          --invert              Invert contrast of output volumes (default: False)
          -d DOWNSAMPLE, --downsample DOWNSAMPLE
                                Downsample volumes to this box size (pixels) (default:
                                None)
          --pc PC               Number of principal component traversals to generate
                                (default: 2)
          --pc-ondata           Find closest on-data latent point to each PC
                                percentile (default: False)
          --ksample KSAMPLE     Number of kmeans samples to generate (default: 20)
   </details>

6. Interactive analysis of a VAE model: run `jupyter notebook` to open `tomoDRGN_viz+filt_template.ipynb` placed in `02_vae_z8_box256/analyze.29` by `tomodrgn analyze`.
    <details> 
        <summary> Click to see two ways to run jupyter notebook </summary>
    To run a local instance of Jupyter Notebook (e.g. you are viewing a monitor directly connected to a computer with direct access to the filesystem containing 02_vae_z8_box256 and the tomodrgn conda environment):
   
         jupyter notebook 02_vae_z8_box256/analyze.29/tomoDRGN_viz+filt_template.ipynb
    To run a remote instance of Jupyter Notebook (e.g. you are viewing a monitor NOT connected to a computer with direct access to the filesystem containing 02_vae_z8_box256 and the tomodrgn conda environment, perhaps having run tomodrgn on a remote cluster):
         
         # In one terminal window, set up port forwarding to your local machine
         ssh -t -t username@cluster-head-node -L 8888:localhost:8888 ssh active-worker-node -L 8888:localhost:8888

         # In a second terminal window connected to your remote system, launch the notebook
         jupyter notebook 02_vae_z8_box256/analyze.29/tomoDRGN_viz+filt_template.ipynb --no-browser --port 8888
   </details>

7. Generate 3-D volumes from (per-particle) unique positions in latent space: `tomodrgn eval_vol 02_vae_z8_box256/weights.pkl --config 02_vae_z8_box256/config.pkl -o 02_vae_z8_box256/tomogram_vols --zfile 02_vae_z8_box256/z.pkl --downsample 64`. This command can benefit from per-tomomgram `z.pkl` filtering within `tomoDRGN_viz+filt_template.ipynb` to create volumes for all particles associated with a particular tomogram.

8. Write a ChimeraX script to place per-particle volumes in the spatial context of a chosen tomogram with optional color mapping: `tomodrgn subtomo2chimerax particles_volumeseries.star -o mytomogram_tomodrgnvols.cxs --tomoname mytomogram.tomostar --vols-dir 02_vae_z8_box256/tomogram_vols --coloring-labels  02_vae_z8_box256/analyze.29/kmeans20/labels.pkl`

9. Filter a star file by particle indices identified and written by `tomoDRGN_viz+filt_template.ipynb`: `tomodrgn filter_star particles.star --ind ind.pkl --outstar particles_filt.star`
    <details> 
        <summary> Click to see all options: <code> tomodrgn filter_star --help </code> </summary>

        usage: filter_star.py [-h]
                              [--input-type {warp_particleseries,warp_volumeseries,m_volumeseries}]
                              [--ind IND] [--ind-type {particle,image,tilt}]
                              [--tomogram TOMOGRAM] [--action {keep,drop}] -o O
                              input
        
        Filter a .star file generated by Warp subtomogram export
        
        positional arguments:
          input                 Input .star file
        
        optional arguments:
          -h, --help            show this help message and exit
          --input-type {warp_particleseries,warp_volumeseries,m_volumeseries}
                                input data .star source (subtomos as images vs as
                                volumes
          --ind IND             optionally select by indices array (.pkl)
          --ind-type {particle,image,tilt}
                                use indices to filter by particle, by individual
                                image, or by tilt index
          --tomogram TOMOGRAM   optionally select by individual tomogram name (`all`
                                means write individual star files per tomogram
          --action {keep,drop}  keep or remove particles associated with ind/tomogram
                                selection
          -o O                  Output .star file
    </details>

## Training requirements and limitations
TomoDRGN model training requires as inputs:
1. a .mrcs stack containing the 2D pixel array of each particle extracted from each stage tilt micrograph, and
2. a corresponding .star file describing pose and CTF parameters for each particle's tilt image as derived from traditional subtomogram averaging. 

At the moment, the only tested workflow to generate these inputs is Warp+IMOD tomogram generation, STA using RELION 3.1, followed by subtomogram extraction as image series in Warp (making use of the excellent [dynamo2warp](https://github.com/alisterburt/dynamo2m) tools from Alister Burt). We are actively seeking to expand the range of validated workflows; please let us know if you have success with a novel approach.

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


## Relevant literature
1. TomoDRGN (bioRxiv) 2022
2. Zhong, E., Bepler, T., Berger, B. & Davis, J. CryoDRGN: Reconstruction of Heterogeneous cryo-EM Structures Using Neural Networks. Nature Methods, [doi:10.1038/s41592-020-01049-4](https://doi.org/10.1038/s41592-020-01049-4) (2020)
3. Kinman, L., Powell, B., Zhong, E., Berger, B. & Davis, J. Uncovering structural ensembles from single particle cryo-EM data using cryoDRGN. bioRxiv, [doi:10.1101/2022.08.09.503342](https://doi.org/10.1101/2022.08.09.503342) (2022)
4. Sun, J., Kinman, L., Jahagirdar, D., Ortega, J., Davis. J. KsgA facilitates ribosomal small subunit maturation by proofreading a key structural lesion. bioRxiv, [doi:10.1101/2022.07.13.499473](https://doi.org/10.1101/2022.07.13.499473) (2022)


## Contact
Please reach out with bug reports, feature requests, etc to bmp[at]mit[dot]edu.