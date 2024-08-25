"""
Evaluate a trained TiltSeriesHetOnlyVAE model to embed particle images in the learned latent space
"""
import argparse
import os
import pickle
import pprint
from datetime import datetime as dt

import torch

from tomodrgn import utils, dataset
from tomodrgn.commands.train_vae import encoder_inference
from tomodrgn.models import TiltSeriesHetOnlyVAE
from tomodrgn.starfile import TiltSeriesStarfile

log = utils.log


def add_args(_parser):
    _parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, or .txt)')
    _parser.add_argument('-w', '--weights', help='Model weights')
    _parser.add_argument('-c', '--config', required=True, help='config.pkl file from train_vae')
    _parser.add_argument('--out-z', metavar='PKL', type=os.path.abspath, required=True, help='Output pickle for z')
    _parser.add_argument('--log-interval', type=int, default=1000, help='Logging interval in N_IMGS')
    _parser.add_argument('-b', '--batch-size', type=int, default=64, help='Minibatch size')
    _parser.add_argument('-v', '--verbose', action='store_true', help='Increases verbosity')
    _parser.add_argument('--no-amp', action='store_true', help='Disable use of automatic mixed precision')

    group = _parser.add_argument_group('Override configuration values -- star file')
    group.add_argument('--source-software', type=str, choices=('auto', 'warp_v1', 'nextpyp', 'relion_v5', 'warp_v2'), default='auto',
                       help='Manually set the software used to extract particles. Default is to auto-detect.')
    group.add_argument('--ind-ptcls', type=os.path.abspath, metavar='PKL', help='Filter starfile by particles (unique rlnGroupName values) using np array pkl as indices')
    group.add_argument('--ind-imgs', type=os.path.abspath, help='Filter starfile by particle images (star file rows) using np array pkl as indices')
    group.add_argument('--sort-ptcl-imgs', choices=('unsorted', 'dose_ascending', 'random'), default='unsorted', help='Sort the star file images on a per-particle basis by the specified criteria')
    group.add_argument('--use-first-ntilts', type=int, default=-1, help='Keep the first `use_first_ntilts` images of each particle in the sorted star file.'
                                                                        'Default -1 means to use all. Will drop particles with fewer than this many tilt images.')
    group.add_argument('--use-first-nptcls', type=int, default=-1, help='Keep the first `use_first_nptcls` particles in the sorted star file. Default -1 means to use all.')

    group = _parser.add_argument_group('Override configuration values -- data handling')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')

    group = _parser.add_argument_group('Dataloader arguments')
    group.add_argument('--num-workers', type=int, default=0, help='Number of workers to use when batching particles for training. Has moderate impact on epoch time')
    group.add_argument('--prefetch-factor', type=int, default=2, help='Number of particles to prefetch per worker for training. Has moderate impact on epoch time')
    group.add_argument('--persistent-workers', action='store_true', help='Whether to persist workers after dataset has been fully consumed. Has minimal impact on run time')
    group.add_argument('--pin-memory', action='store_true', help='Whether to use pinned memory for dataloader. Has large impact on epoch time. Recommended.')

    return _parser


def main(args):
    t1 = dt.now()
    log(args)

    # make output directory
    if not os.path.exists(os.path.dirname(args.out_z)):
        os.makedirs(os.path.dirname(args.out_z))

    # set the device
    device = utils.get_default_device()
    if device == torch.device('cpu'):
        args.no_amp = True
        log('Warning: pytorch AMP does not support non-CUDA (e.g. cpu) devices. Automatically disabling AMP and continuing')
        # https://github.com/pytorch/pytorch/issues/55374
    torch.set_grad_enabled(False)
    use_amp = not args.no_amp

    # update the configuration with user-specified parameters
    log('Updating config with new parameters passed to this command ...')
    cfg = utils.load_pkl(args.config)
    if args.particles is not None:
        cfg['starfile_args']['sourcefile'] = args.particles
    if args.source_software is not None:
        cfg['starfile_args']['source_software'] = args.source_software
    if args.ind_ptcls is not None:
        cfg['starfile_args']['ind_ptcls'] = args.ind_ptcls
    if args.ind_imgs is not None:
        cfg['starfile_args']['ind_imgs'] = args.ind_imgs
    if args.sort_ptcl_imgs is not None:
        cfg['starfile_args']['sort_ptcl_imgs'] = args.sort_ptcl_imgs
    if args.use_first_ntilts is not None:
        cfg['starfile_args']['use_first_ntilts'] = args.use_first_ntilts
    if args.use_first_nptcls is not None:
        cfg['starfile_args']['use_first_nptcls'] = args.use_first_nptcls
    if args.datadir is not None:
        cfg['dataset_args']['datadir'] = args.datadir
    if args.lazy:
        cfg['training_args']['lazy'] = args.lazy
    if not args.invert_data:
        cfg['dataset_args']['invert_data'] = args.invert_data

    # load star file
    ptcls_star = TiltSeriesStarfile(args.particles,
                                    source_software=args.source_software)

    # filter star file
    ptcls_star.filter(ind_imgs=args.ind_imgs,
                      ind_ptcls=args.ind_ptcls,
                      sort_ptcl_imgs=args.sort_ptcl_imgs,
                      use_first_ntilts=args.use_first_ntilts,
                      use_first_nptcls=args.use_first_nptcls)

    # save filtered star file for future convenience (aligning latent embeddings with particles, re-extracting particles, mapbacks, etc.)
    outstar = f'{os.path.dirname(args.out_z)}/{os.path.splitext(os.path.basename(args.particles))[0]}_tomodrgn_preprocessed.star'
    ptcls_star.sourcefile_filtered = outstar
    ptcls_star.write(outstar)

    # update the config file to point to the newly filtered star file for dataset loading
    cfg['starfile_args']['sourcefile_filtered'] = outstar

    # embed all images associated with each particle
    cfg['dataset_args']['star_random_subset'] = -1

    log('Using configuration:')
    pprint.pprint(cfg)

    # load the particles
    log('Loading the dataset using specified config ...')
    data = dataset.TiltSeriesMRCData.load(cfg)

    # instantiate model and lattice
    log('Loading the model using specified config ...')
    model, lattice = TiltSeriesHetOnlyVAE.load(config=cfg,
                                               weights=args.weights,
                                               device=device)

    # check that ntilts of new input images is compatible with ntilts of trained model encoder b
    if model.encoder.pooling_function in ['concatenate', 'set_encoder']:
        assert model.encoder.ntilts <= data.ntilts_range[0], \
            f'The loaded model requires a minimum of {model.encoder.ntilts} tilt images per particle, but some input particles contain as few as {data.ntilts_range[0]} tilt images'

    # evaluation loop
    log('Embedding particle images in latent space ...')
    z_mu, z_logvar = encoder_inference(model=model,
                                       lat=lattice,
                                       data=data,
                                       use_amp=use_amp,
                                       batchsize=args.batch_size)

    log(f'Saving latent embeddings to {args.out_z}')
    with open(args.out_z, 'wb') as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)

    log(f'Finished in {dt.now() - t1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
