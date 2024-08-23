"""
Assess convergence and training dynamics of a heterogeneous VAE network
"""

import argparse
import glob
import os
import sys
from datetime import datetime as dt

import numpy as np

from tomodrgn import analysis, utils, convergence

flog = utils.flog
log = utils.log


def add_args(_parser):
    _parser.add_argument('workdir', type=os.path.abspath, help='Directory with tomoDRGN results')
    _parser.add_argument('epoch', type=str, help='Latest epoch number N to analyze convergence (0-based indexing, corresponding to z.N.pkl, weights.N.pkl), "latest" for last detected epoch')
    _parser.add_argument('-o', '--outdir', type=os.path.abspath, help='Output directory for convergence analysis results (default: [workdir]/convergence.[epoch])')
    _parser.add_argument('--epoch-interval', type=int, default=5, help='Interval of epochs between calculating most convergence heuristics')

    group = _parser.add_argument_group('UMAP  calculation arguments')
    group.add_argument('--subset', default=50000, type=int, help='Max number of particles to be used for UMAP calculations. \'None\' means use all ptcls')
    group.add_argument('--random-seed', type=int, default=np.random.randint(0, 100000), help='Manually specify the seed used for selection of subset particles and other numpy operations')
    group.add_argument('--random-state', type=int, default=42,
                       help='Random state seed used by UMAP for reproducibility at slight cost of performance (default 42, None means slightly faster but non-reproducible)')
    group.add_argument('--skip-umap', action='store_true', help='Skip UMAP embedding. Requires that UMAP be precomputed for downstream calcs. Useful for tweaking volume generation settings.')

    group = _parser.add_argument_group('Sketching UMAP via local maxima arguments')
    group.add_argument('--n-bins', type=int, default=30, help='the number of bins along UMAP1 and UMAP2')
    group.add_argument('--smooth', type=bool, default=True, help='smooth the 2D histogram before identifying local maxima')
    group.add_argument('--smooth-width', type=float, default=1.0, help='width of gaussian kernel for smoothing 2D histogram expressed as multiple of one bin\'s width')
    group.add_argument('--pruned-maxima', type=int, default=12, help='prune poorly-separated maxima until this many maxima remain')
    group.add_argument('--radius', type=float, default=5.0, help='distance at which two maxima are considered poorly-separated and are candidates for pruning (euclidean distance in bin-space)')
    group.add_argument('--final-maxima', type=int, default=10, help='select this many local maxima, sorted by highest bin count after pruning, for which to generate volumes')

    group = _parser.add_argument_group('Volume generation arguments')
    group.add_argument('--downsample', type=int, help='Downsample volumes to this box size (pixels)')
    group.add_argument('--lowpass', type=float, default=None, help='Lowpass filter to this resolution in Å')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volumes')
    group.add_argument('--invert', action='store_true', help='Invert contrast of output volumes')
    group.add_argument('--cuda', type=int, default=None, help='Specify cuda device for volume generation')
    group.add_argument('--skip-volgen', action='store_true', help='Skip volume generation. Requires that volumes already exist for downstream CC + FSC calculations')
    group.add_argument('--ground-truth', type=os.path.abspath, nargs='+', default=None, help='Relative path containing wildcards to ground_truth_vols*.mrc for map-map CC calcs')

    group = _parser.add_argument_group('Mask generation arguments')
    group.add_argument('--mask', type='str', choices=['none', 'sphere', 'tight', 'soft'], help='Type of mask to generate for each generated volume when calculating volume-based metrics.')
    group.add_argument('--thresh', type=float, default=None, help='Isosurface percentile at which to threshold volume; default is to use 99th percentile.')
    group.add_argument('--dilate', type=int, default=None, help='Number of voxels to dilate thresholded isosurface outwards from mask boundary; default is to use 1/30th of box size (px).')
    group.add_argument('--dist', type=int, default=None, help='Number of voxels over which to apply a soft cosine falling edge from dilated mask boundary; default is to use 1/30th of box size (px)')

    return _parser


def get_latest(workdir: str) -> int:
    # assumes args.num_epochs > latest checkpoint
    log('Detecting latest checkpoint...')
    files = glob.glob(f'{workdir}/z.*.train.pkl')
    epochs = [int(file.split('.')[-2]) for file in files]
    epoch = max(epochs)
    return epoch


def main(args):
    t1 = dt.now()

    # Configure paths
    workdir = args.workdir
    config = f'{workdir}/config.pkl'
    runlog = f'{workdir}/run.log'
    assert glob.glob(f'{args.workdir}/weights*.pkl'), f'No weights.*.pkl files detected in {args.workdir}; exiting...'
    assert glob.glob(f'{args.workdir}/z.*.train.pkl'), f'No z.*.train.pkl files detected in {args.workdir}; exiting...'

    # get the array of epochs at which to calculate convergence metrics
    final_epoch = get_latest(args.workdir) if args.epoch == 'latest' else int(args.epoch)
    epochs = np.arange(4, final_epoch + 1, args.epoch_interval)
    if epochs[-1] != final_epoch:
        epochs = np.append(epochs, final_epoch)
    log(f'Will analyze epochs: {epochs}')

    # assert all required files are locatable
    for e in range(final_epoch):
        assert os.path.isfile(workdir + f'/z.{e}.train.pkl'), f'Could not find training file {workdir}/z.{e}.train.pkl'
    for e in epochs:
        assert os.path.isfile(workdir + f'/weights.{e}.pkl'), f'Could not find training file {workdir}/weights.{e}.pkl'
    assert os.path.isfile(config), f'Could not find training file {config}'
    assert os.path.isfile(runlog), f'Could not find training file {runlog}'

    # Configure output paths
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = f'{workdir}/convergence.{final_epoch}'
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir + '/plots', exist_ok=True)
    os.makedirs(outdir + '/pcs', exist_ok=True)
    os.makedirs(outdir + '/umaps', exist_ok=True)
    os.makedirs(outdir + '/repr_particles', exist_ok=True)
    logfile = f'{outdir}/convergence.log'
    flog(args, logfile)
    if len(epochs) < 3:
        flog('WARNING: Too few epochs have been selected for some analyses. Try decreasing --epoch-interval to a shorter interval, or analyzing a later epoch.', logfile)
    if len(epochs) < 2:
        flog('WARNING: Too few epochs have been selected for any analyses. Try decreasing --epoch-interval to a shorter interval, or analyzing a later epoch.', logfile)
        sys.exit()
    flog(f'Saving all results to {outdir}', logfile)

    # Get total number of particles, latent space dimensionality
    nptcls, zdim = utils.load_pkl(workdir + f'/z.{final_epoch}.train.pkl').shape

    # Convergence 0: total loss
    flog('Plotting total loss curve', logfile)
    convergence.plot_loss(runlog=runlog,
                          outdir=outdir)

    # Convergence 1: PCA latent
    flog('Calculating and plotting PCA at each epoch', logfile)
    convergence.plot_latent_pca(workdir=workdir, outdir=outdir, epochs=epochs)

    # Convergence 2: UMAP latent embeddings
    if args.skip_umap:
        flog('Skipping UMAP calculation', logfile)
    else:
        flog(f'Calculating and plotting UMAP embeddings at each epoch', logfile)
        convergence.plot_latent_umap(workdir=workdir,
                                     outdir=outdir,
                                     epochs=epochs,
                                     nptcls=nptcls,
                                     subset=args.subset,
                                     random_seed=args.random_seed,
                                     random_state=args.random_state)

    # Convergence 3: latent encoding shifts
    flog(f'Calculating and plotting latent embedding vector shifts between all epochs', logfile)
    convergence.encoder_latent_shifts(workdir=workdir,
                                      outdir=outdir,
                                      final_epoch=final_epoch)

    # Convergence 4: correlation of generated volumes
    flog(f'Sketching epoch {final_epoch}\'s latent space to find representative and well-supported latent encodings', logfile)
    binned_ptcls_mask, labels = convergence.sketch_via_umap_local_maxima(outdir=outdir,
                                                                         sketch_epoch=final_epoch,
                                                                         n_bins=args.n_bins,
                                                                         smooth=args.smooth,
                                                                         smooth_width=args.smooth_width,
                                                                         pruned_maxima=args.pruned_maxima,
                                                                         radius=args.radius,
                                                                         final_maxima=args.final_maxima)

    flog('Locating representative latent encodings in earlier epochs', logfile)
    convergence.follow_candidate_particles(workdir=workdir,
                                           outdir=outdir,
                                           epochs=epochs,
                                           binned_ptcls_mask=binned_ptcls_mask,
                                           labels=labels)

    if args.skip_volgen:
        flog('Skipping volume generation ...', logfile)
    else:
        flog(f'Generating volumes at representative latent encodings in earlier epochs', logfile)
        for epoch in epochs:
            vg = analysis.VolumeGenerator(weights_path=f'{workdir}/weights.{epoch}.pkl',
                                          config_path=f'{workdir}/config.pkl',
                                          downsample=args.downsample,
                                          lowpass=args.lowpass,
                                          flip=args.flip,
                                          invert=args.invert,
                                          cuda=args.device, )
            z_values = np.asarray(np.loadtxt(f'{outdir}/repr_particles/latent_representative.{epoch}.txt'), dtype=np.float32)
            vg.gen_volumes(z_values=z_values,
                           outdir=f'{outdir}/vols.{epoch}')

    flog(f'Calculating successive epoch pairwise map-map CCs at representative latent encodings', logfile)
    convergence.calc_ccs_pairwise_epochs(outdir=outdir,
                                         epochs=epochs,
                                         labels=labels,
                                         mask=args.mask,
                                         thresh=args.thresh,
                                         dilate=args.dilate,
                                         dist=args.dist, )

    flog(f'Calculating successive epoch pairwise map-map FSCs at representative latent encodings', logfile)
    convergence.calc_fscs_pairwise_epochs(outdir=outdir,
                                          epochs=epochs,
                                          labels=labels,
                                          mask=args.mask,
                                          thresh=args.thresh,
                                          dilate=args.dilate,
                                          dist=args.dist, )

    flog(f'Calculating intra-epoch all-to-all map-map CCs at representative latent encodings', logfile)
    convergence.calc_ccs_alltoall_intraepoch(outdir=outdir,
                                             epochs=epochs,
                                             labels=labels,
                                             mask=args.mask,
                                             thresh=args.thresh,
                                             dilate=args.dilate,
                                             dist=args.dist, )

    if args.ground_truth is not None:
        flog(f'Calculating ground truth map-map CCs', logfile)
        flog(f'Using ground truth maps {args.ground_truth}', logfile)
        convergence.calc_ccs_alltogroundtruth(outdir=outdir,
                                              epochs=epochs,
                                              labels=labels,
                                              ground_truth_paths=args.ground_truth,
                                              mask=args.mask,
                                              thresh=args.thresh,
                                              dilate=args.dilate,
                                              dist=args.dist)

    flog(f'Finished in {dt.now() - t1}', logfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
