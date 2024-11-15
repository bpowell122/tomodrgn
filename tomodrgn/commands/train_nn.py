"""
Train a FTPositionalDecoder neural network to reconstruct a 3D density map given 2D images from a tilt series with known pose and CTF parameters
"""
import argparse
import os
import sys
from datetime import datetime as dt
from typing import get_args

import numpy as np
import torch
import torch.nn
import torch.amp
import torch.utils.data

from tomodrgn import utils, ctf, config
from tomodrgn.models import VolumeGenerator, mlp_ascii
from tomodrgn.dataset import load_sta_dataset
from tomodrgn.lattice import Lattice
from tomodrgn.models import FTPositionalDecoder, DataParallelPassthrough
from tomodrgn.starfile import load_sta_starfile, KNOWN_STAR_SOURCES
from tomodrgn.commands.train_vae import preprocess_batch

log = utils.log
vlog = utils.vlog


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        # this script is called from tomodrgn.__main__ entry point, in which case a parser is already created
        pass

    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, or .txt)')

    group = parser.add_argument_group('Core arguments')
    group.add_argument('--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    group.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    group.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    group.add_argument('--log-interval', type=int, default=200, help='Logging interval in N_PTCLS (default: %(default)s)')
    group.add_argument('--verbose', action='store_true', help='Increases verbosity')
    group.add_argument('--seed', type=int, default=np.random.randint(0, 100000), help='Random seed')
    group.add_argument('--plot-format', type=str, choices=['png', 'svgz'], default='png', help='File format with which to save plots')

    group = parser.add_argument_group('Particle starfile loading and filtering')
    group.add_argument('--source-software', type=str, choices=get_args(KNOWN_STAR_SOURCES), default='auto',
                       help='Manually set the software used to extract particles. Default is to auto-detect.')
    group.add_argument('--ind-ptcls', type=os.path.abspath, metavar='PKL', help='Filter starfile by particles (unique rlnGroupName values) using np array pkl as indices')
    group.add_argument('--ind-imgs', type=os.path.abspath, help='Filter starfile by particle images (star file rows) using np array pkl as indices')
    group.add_argument('--sort-ptcl-imgs', choices=('unsorted', 'dose_ascending', 'random'), default='unsorted', help='Sort the star file images on a per-particle basis by the specified criteria')
    group.add_argument('--use-first-ntilts', type=int, default=-1, help='Keep the first `use_first_ntilts` images of each particle in the sorted star file.'
                                                                        'Default -1 means to use all. Will drop particles with fewer than this many tilt images.')
    group.add_argument('--use-first-nptcls', type=int, default=-1, help='Keep the first `use_first_nptcls` particles in the sorted star file. Default -1 means to use all.')

    group = parser.add_argument_group('Dataset loading and preprocessing')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.8, help='Real space inner windowing radius to begin cosine falloff')
    group.add_argument('--window-r-outer', type=float, default=.9, help='Real space outer windowing radius to end cosine falloff')
    group.add_argument('--datadir', type=os.path.abspath, default=None, help='Path prefix to particle stack if loading relative paths from a .star file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD)')
    group.add_argument('--sequential-tilt-sampling', action='store_true', help='Supply particle images of one particle to encoder in filtered starfile order')

    group = parser.add_argument_group('Weighting and masking')
    group.add_argument('--recon-tilt-weight', action='store_true', help='Weight reconstruction loss by cosine(tilt_angle)')
    group.add_argument('--recon-dose-weight', action='store_true', help='Weight reconstruction loss per tilt per pixel by dose dependent amplitude attenuation')
    group.add_argument('--l-dose-mask', action='store_true', help='Do not train on frequencies exposed to > 2.5x critical dose. Training lattice is intersection of this with --l-extent')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs')
    group.add_argument('-b', '--batch-size', type=int, default=1, help='Minibatch size')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer')
    group.add_argument('--lr', type=float, default=0.0001, help='Learning rate in Adam optimizer for batch size 1. Is automatically further scaled as square-root of batch size.')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: mean, std of dataset)')
    group.add_argument('--no-amp', action='store_true', help='Disable use of mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs. Specify GPUs i,j via `export CUDA_VISIBLE_DEVICES=i,j` before tomodrgn train_vae')

    group = parser.add_argument_group('Network Architecture')
    group.add_argument('--layers', type=int, default=3, help='Number of hidden layers')
    group.add_argument('--dim', type=int, default=512, help='Number of nodes in hidden layers')
    group.add_argument('--l-extent', type=float, default=0.5, help='Coordinate lattice size (if not using positional encoding)')
    group.add_argument('--pe-type', choices=('geom_ft', 'geom_full', 'geom_lowf', 'geom_nohighf', 'linear_lowf', 'gaussian', 'none'), default='gaussian', help='Type of positional encoding')
    group.add_argument('--pe-dim', type=int, help='Num sinusoid features in positional encoding (default: D/2)')
    group.add_argument('--activation', choices=('relu', 'leaky_relu'), default='relu', help='Activation')
    group.add_argument('--feat-sigma', type=float, default=0.5, help="Scale for random Gaussian features")

    group = parser.add_argument_group('Dataloader arguments')
    group.add_argument('--num-workers', type=int, default=0, help='Number of workers to use when batching particles for training. Has moderate impact on epoch time')
    group.add_argument('--prefetch-factor', type=int, default=None, help='Number of particles to prefetch per worker for training. Has moderate impact on epoch time')
    group.add_argument('--persistent-workers', action='store_true', help='Whether to persist workers after dataset has been fully consumed. Has minimal impact on run time')
    group.add_argument('--pin-memory', action='store_true', help='Whether to use pinned memory for dataloader. Has large impact on epoch time. Recommended.')

    return parser


def save_checkpoint(model: FTPositionalDecoder | DataParallelPassthrough,
                    lat: Lattice,
                    optim: torch.optim.Optimizer,
                    epoch: int,
                    norm: tuple[float, float],
                    config_path: str,
                    amp: bool,
                    out_mrc: str,
                    out_weights: str) -> None:
    """
    Evaluate the model with the currently loaded weights; save the resulting consensus volume and model.

    :param model: FTPositionalDecoder model to be evaluated
    :param lat: lattice object specifying coordinates to evaluate
    :param optim: optimizer object, to be saved into model pkl
    :param epoch: current epoch number, 0-indexed, saved into model pkl
    :param norm: mean and standard deviation of dataset by which input dataset was normalized, to be unscaled following volume generation
    :param config_path: path to trained model `config.pkl`
    :param amp: Enable or disable use of mixed-precision model inference
    :param out_mrc: name of the output mrc file to save consensus reconstruction
    :param out_weights: name of the output pkl file to save model state_dict, optimizer state_dict
    :return: None
    """
    # instantiate the volume generator
    vg = VolumeGenerator(config=config_path,
                         weights_path=None,
                         model=model,
                         lat=lat,
                         amp=amp)

    # generate volumes
    vg.generate_volumes(z=None,
                        out_dir=os.path.dirname(out_mrc),
                        out_name=os.path.splitext(os.path.basename(out_mrc))[0],
                        downsample=None,
                        lowpass=None,
                        flip=False,
                        invert=False,
                        batch_size=1)

    # save the model
    torch.save({
        'norm': norm,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, out_weights)


def train_batch(model: FTPositionalDecoder | DataParallelPassthrough,
                scaler: torch.amp.GradScaler,
                optim: torch.optim.Optimizer,
                lat: Lattice,
                batch_images: torch.Tensor,
                batch_rots: torch.Tensor,
                batch_trans: torch.Tensor,
                batch_ctf_params: torch.Tensor,
                batch_recon_error_weights: torch.Tensor,
                batch_hartley_2d_mask: torch.Tensor,
                image_ctf_premultiplied: bool,
                image_dose_weighted: bool,
                use_amp: bool = False) -> float:
    """
    Train a FTPositionalDecoder model on a batch of tilt series particle images.

    :param model: FTPositionalDecoder object to be trained
    :param scaler: GradScaler object to be used for scaling loss involving fp16 tensors to avoid over/underflow
    :param optim: torch.optim.Optimizer object to be used for optimizing the model
    :param lat: Hartley-transform lattice of points for voxel grid operations
    :param batch_images: Batch of images to be used for training, shape (batchsize, ntilts, boxsize_ht, boxsize_ht)
    :param batch_rots: Batch of 3-D rotation matrices corresponding to `batch_images` known poses, shape (batchsize, ntilts, 3, 3)
    :param batch_trans: Batch of 2-D translation matrices corresponding to `batch_images` known poses, shape (batchsize, ntilts, 2).
            May be `torch.zeros((batchsize))` instead to indicate no translations should be applied to the input images.
    :param batch_ctf_params: Batch of CTF parameters corresponding to `batch_images` known CTF parameters, shape (batchsize, ntilts, 9).
            May be `torch.zeros((batchsize))` instead to indicate no CTF corruption should be applied to the reconstructed slice.
    :param batch_recon_error_weights: Batch of 2-D weights to be applied to the per-spatial-frequency error between the reconstructed slice and the input image.
            Calculated from critical dose exposure curves and electron beam vs sample tilt geometry.
            May be `torch.zeros((batchsize))` instead to indicate no weighting should be applied to the reconstructed slice error.
    :param batch_hartley_2d_mask: Batch of 2-D masks to be applied per-spatial-frequency.
            Calculated as the intersection of critical dose exposure curves and a Nyquist-limited circular mask in reciprocal space, including masking the DC component.
    :param image_ctf_premultiplied: Whether images were multiplied by their CTF during particle extraction.
    :param image_dose_weighted: Whether images were multiplied by their exposure-dependent frequency weighting during particle extraction.
    :param use_amp: If true, use Automatic Mixed Precision to reduce memory consumption and accelerate code execution via `autocast` and `GradScaler`
    :return: generative loss between reconstructed slices and input images
    """
    # prepare to train a new batch
    optim.zero_grad()
    model.train()

    # autocast auto-enabled and set to correct device
    with torch.amp.autocast(device_type=lat.device.type, enabled=use_amp):
        # center images via translation and calculate CTF
        batch_images_preprocessed, batch_ctf_weights = preprocess_batch(lat=lat,
                                                                        batch_images=batch_images,
                                                                        batch_trans=batch_trans,
                                                                        batch_ctf_params=batch_ctf_params,
                                                                        image_ctf_premultiplied=True)  # use image_ctf_premultiplied True to disable phase flipping of input images
        # decode the lattice coordinate positions given the encoder-generated embeddings
        batch_images_recon = decode_batch(model=model,
                                          lat=lat,
                                          batch_rots=batch_rots,
                                          batch_hartley_2d_mask=batch_hartley_2d_mask)
        # calculate the model loss
        gen_loss = loss_function(batch_images=batch_images_preprocessed,
                                 batch_images_recon=batch_images_recon,
                                 batch_ctf_weights=batch_ctf_weights,
                                 batch_recon_error_weights=batch_recon_error_weights,
                                 batch_hartley_2d_mask=batch_hartley_2d_mask,
                                 image_ctf_premultiplied=image_ctf_premultiplied,
                                 image_dose_weighted=image_dose_weighted)

    # backpropogate the scaled loss and optimize model weights
    scaler.scale(gen_loss).backward()
    scaler.step(optim)
    scaler.update()

    return gen_loss.item()


def decode_batch(*,
                 model: FTPositionalDecoder | DataParallelPassthrough,
                 lat: Lattice,
                 batch_rots: torch.Tensor,
                 batch_hartley_2d_mask: torch.Tensor) -> torch.Tensor:
    """
    Decode a batch of particles represented by multiple images from per-particle latent embeddings and corresponding lattice positions to evaluate

    :param model: FTPositionalDecoder object to be trained
    :param lat: Hartley-transform lattice of points for voxel grid operations
    :param batch_rots: Batch of 3-D rotation matrices corresponding to `batch_images` known poses, shape (batchsize, ntilts, 3, 3)
    :param batch_hartley_2d_mask: Batch of 2-D masks to be applied per-spatial-frequency, shape (batchsize, ntilts, boxsize_ht**2)
            Calculated as the intersection of critical dose exposure curves and a Nyquist-limited circular mask in reciprocal space, including masking the DC component.
    :return: Reconstructed central slices of Fourier space volumes corresponding to each particle in the batch, shape (batchsize * ntilts * boxsize_ht**2 [`batch_hartley_2d_mask`]).
            Note that the returned array is completely flattened (including along batch dimension) due to potential for uneven/ragged reconstructed image tensors per-particle after masking
    """

    # prepare lattice at rotated fourier components
    input_coords = lat.coords @ batch_rots  # shape batchsize x ntilts x boxsize_ht**2 x 3 from [boxsize_ht**2 x 3] @ [batchsize x ntilts x 3 x 3]

    # filter by dec_mask to skip decoding coordinates that have low contribution to SNR
    # decode each particle one at a time due to variable # pixels in batch_hartley_2d_mask per particle producing ragged tensor
    # this will eventually be replaceable with pytorch NestedTensor, but as of torch v2.4 this still does not work at backprop (and does not support many common operations we need prior to that)
    batch_images_recon = torch.cat([model(coords=coords_ptcl[mask_ptcl].unsqueeze(0), z=None).squeeze(0)
                                    for coords_ptcl, mask_ptcl in zip(input_coords, batch_hartley_2d_mask, strict=True)], dim=0)

    return batch_images_recon


def loss_function(*,
                  batch_images: torch.Tensor,
                  batch_images_recon: torch.Tensor,
                  batch_ctf_weights: torch.Tensor,
                  batch_recon_error_weights: torch.Tensor,
                  batch_hartley_2d_mask: torch.Tensor,
                  image_ctf_premultiplied: bool,
                  image_dose_weighted: bool) -> torch.Tensor:
    """
    Calculate generative loss between reconstructed and input images

    :param batch_images: Batch of images to be used for training, shape (batchsize, ntilts, boxsize_ht**2)
    :param batch_images_recon: Reconstructed central slices of Fourier space volumes corresponding to each particle in the batch, shape (batchsize * ntilts * boxsize_ht**2 [`batch_hartley_2d_mask`])
    :param batch_ctf_weights: CTF evaluated at each spatial frequency corresponding to input images, shape (batchsize, ntilts, boxsize_ht**2) or None if no CTF should be applied
    :param batch_recon_error_weights: Batch of 2-D weights to be applied to the per-spatial-frequency error between each reconstructed slice and input image, shape (batchsize, ntilts, boxsize_ht**2).
            Calculated from critical dose exposure curves and electron beam vs sample tilt geometry.
    :param batch_hartley_2d_mask: Batch of 2-D masks to be applied per-spatial-frequency.
            Calculated as the intersection of critical dose exposure curves and a Nyquist-limited circular mask in reciprocal space, including masking the DC component.
    :param image_ctf_premultiplied: Whether images were multiplied by their CTF during particle extraction.
    :param image_dose_weighted: Whether images were multiplied by their exposure-dependent frequency weighting during particle extraction.    :param beta: scaling factor to apply to KLD during loss calculation.
    :return: generative loss between reconstructed slices and input images
    """
    # locally disable autocast for numerical stability in calculating losses, particularly with batch size > 1 or equivalent large denominator for kld or large num pixels for mse
    with torch.amp.autocast(device_type=batch_images.device.type, enabled=False):
        # upcast tensors for numerical stability in loss if amp was enabled upstream
        batch_images = batch_images.to(torch.float32)
        batch_images_recon = batch_images_recon.to(torch.float32)
        batch_ctf_weights = batch_ctf_weights.to(torch.float32) if batch_ctf_weights is not None else None
        batch_recon_error_weights = batch_recon_error_weights.to(torch.float32)

        # reconstruction error
        batch_images = batch_images[batch_hartley_2d_mask].view(-1)
        batch_recon_error_weights = batch_recon_error_weights[batch_hartley_2d_mask].view(-1)
        if batch_ctf_weights is not None:
            batch_ctf_weights = batch_ctf_weights[batch_hartley_2d_mask].view(-1)
            batch_images_recon = batch_images_recon * batch_ctf_weights  # apply CTF to reconstructed image
            if not image_ctf_premultiplied:
                # undo phase flipping in place from preprocess_batch, this was only applied if images are not ctf premultiplied
                batch_images = batch_images * batch_ctf_weights.sign()
            else:
                # reconstructed image needs to correspond to input ctf_premultiplied images, i.e. images that are doubly convolved with the CTF
                batch_images_recon = batch_images_recon * batch_ctf_weights  # apply CTF to reconstructed image again
        if image_dose_weighted:
            # images may have been extracted with dose weights pre-applied
            batch_images_recon = batch_images_recon * batch_recon_error_weights
        gen_loss = torch.mean(batch_recon_error_weights * ((batch_images_recon - batch_images) ** 2))

    return gen_loss


def main(args):
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    logfile = f'{args.outdir}/run.log'

    def flog(msg):  # HACK: switch to logging module
        return utils.flog(msg, logfile)

    if args.load == 'latest':
        latest_epoch = utils.get_latest_epoch(args.outdir)
        args.load = f'{args.outdir}/weights.{latest_epoch}.pkl'
    flog(' '.join(sys.argv))
    flog(args)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set the device
    device = utils.get_default_device()
    if device == torch.device('cpu'):
        args.no_amp = True
        flog('Warning: pytorch AMP does not support non-CUDA (e.g. cpu) devices. Automatically disabling AMP and continuing')
        # https://github.com/pytorch/pytorch/issues/55374

    # load star file
    ptcls_star = load_sta_starfile(star_path=args.particles,
                                   source_software=args.source_software)
    ptcls_star.plot_particle_uid_ntilt_distribution(outpath=f'{args.outdir}/{os.path.basename(ptcls_star.sourcefile)}_particle_uid_ntilt_distribution.{args.plot_format}')

    # filter star file
    ptcls_star.filter(ind_imgs=args.ind_imgs,
                      ind_ptcls=args.ind_ptcls,
                      sort_ptcl_imgs=args.sort_ptcl_imgs,
                      use_first_ntilts=args.use_first_ntilts,
                      use_first_nptcls=args.use_first_nptcls)

    # save filtered star file for future convenience (aligning latent embeddings with particles, re-extracting particles, mapbacks, etc.)
    outstar = f'{args.outdir}/{os.path.splitext(os.path.basename(ptcls_star.sourcefile))[0]}_tomodrgn_preprocessed.star'
    ptcls_star.sourcefile_filtered = outstar
    ptcls_star.write(outstar)

    # load the particles
    flog(f'Loading dataset from {args.particles}')
    datadir = args.datadir if args.datadir is not None else os.path.dirname(ptcls_star.sourcefile)
    data = load_sta_dataset(ptcls_star=ptcls_star,
                             datadir=datadir,
                             lazy=args.lazy,
                             norm=args.norm,
                             invert_data=args.invert_data,
                             window=args.window,
                             window_r=args.window_r,
                             window_r_outer=args.window_r_outer,
                             recon_dose_weight=args.recon_dose_weight,
                             recon_tilt_weight=args.recon_tilt_weight,
                             l_dose_mask=args.l_dose_mask,
                             constant_mintilt_sampling=False,
                             sequential_tilt_sampling=args.sequential_tilt_sampling)
    if args.batch_size > 1 and (data.ntilts_range[0] != data.ntilts_range[1]):
        # particles have variable numbers of tilt images; collate_fn will fail when processing multiple particles as a ragged tensor
        args.batch_size = 1
    boxsize_ht = data.boxsize_ht
    nptcls = data.nptcls
    image_ctf_premultiplied = data.star.image_ctf_premultiplied
    image_dose_weighted = data.star.image_dose_weighted
    ctf.print_ctf_params(data.ctf_params[0])

    # instantiate lattice
    lat = Lattice(boxsize_ht, extent=args.l_extent, device=device)

    # instantiate model
    activation = {"relu": torch.nn.ReLU, "leaky_relu": torch.nn.LeakyReLU}[args.activation]
    model = FTPositionalDecoder(boxsize_ht=boxsize_ht,
                                in_dim=3,
                                hidden_layers=args.layers,
                                hidden_dim=args.dim,
                                activation=activation,
                                pe_type=args.pe_type,
                                pe_dim=args.pe_dim,
                                feat_sigma=args.feat_sigma)
    for line in mlp_ascii(input_dim=model.in_dim,
                          hidden_dims=[model.hidden_dim for _ in range(model.hidden_layers)],
                          output_dim=2):
        print(line)

    # save configuration
    out_config = f'{args.outdir}/config.pkl'
    config.save_config(args=args,
                       star=ptcls_star,
                       data=data,
                       lat=lat,
                       model=model,
                       out_config=out_config)

    # optimizer
    args.lr = args.lr * (args.batch_size ** 0.5)
    if not args.sequential_tilt_sampling:
        log('Scaling learning rate larger by 2 due to using random tilt sampling')
        args.lr = args.lr * 2
    log(f'Final learning rate after scaling by square root of batch size: {args.lr}')
    optim = torch.optim.AdamW(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.wd,
                              eps=1e-4)  # https://github.com/pytorch/pytorch/issues/40497#issuecomment-1084807134

    # Mixed precision training with AMP
    use_amp = not args.no_amp
    flog(f'AMP acceleration enabled (autocast + gradscaler) : {use_amp}')
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    if use_amp:
        if not args.batch_size % 8 == 0:
            flog('Warning: recommended to have batch size divisible by 8 for AMP training')
        if not (boxsize_ht - 1) % 8 == 0:
            flog('Warning: recommended to have image size divisible by 8 for AMP training')
        assert args.dim % 8 == 0, "Decoder hidden layer dimension must be divisible by 8 for AMP training"

    # load weights if restarting from checkpoint
    if args.load:
        flog('Loading checkpoint from {}'.format(args.load))
        checkpoint = torch.load(args.load, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        assert start_epoch < args.num_epochs
        try:
            scaler.load_state_dict(checkpoint['scaler'])
        except KeyError:
            flog('No GradScaler instance found in specified checkpoint; creating new GradScaler')
    else:
        start_epoch = 0
        model.to(device)
    epoch = start_epoch

    # parallelize
    if args.multigpu and torch.cuda.device_count() > 1:
        log(f'Using {torch.cuda.device_count()} GPUs!')
        args.batch_size *= torch.cuda.device_count()
        log(f'Increasing batch size to {args.batch_size}')
        model = DataParallelPassthrough(model)
    elif args.multigpu:
        log(f'WARNING: --multigpu selected, but {torch.cuda.device_count()} GPUs detected')

    # train
    flog('Done all preprocessing; starting training now!')
    data_generator = torch.utils.data.DataLoader(dataset=data,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 prefetch_factor=args.prefetch_factor,
                                                 persistent_workers=args.persistent_workers,
                                                 pin_memory=args.pin_memory)
    for epoch in range(start_epoch, args.num_epochs):
        t2 = dt.now()
        loss_accum = 0
        batch_it = 0
        for batch_images, batch_rots, batch_trans, batch_ctf_params, batch_recon_error_weights, batch_hartley_2d_mask, batch_indices in data_generator:
            # impression counting
            batch_it += len(batch_images)

            # transfer to GPU
            batch_images = batch_images.to(device)
            batch_rots = batch_rots.to(device)
            batch_trans = batch_trans.to(device)
            batch_ctf_params = batch_ctf_params.to(device)
            batch_recon_error_weights = batch_recon_error_weights.to(device)
            batch_hartley_2d_mask = batch_hartley_2d_mask.to(device)

            # training minibatch
            loss_batch = train_batch(model=model,
                                     scaler=scaler,
                                     optim=optim,
                                     lat=lat,
                                     batch_images=batch_images,
                                     batch_rots=batch_rots,
                                     batch_trans=batch_trans,
                                     batch_ctf_params=batch_ctf_params,
                                     batch_recon_error_weights=batch_recon_error_weights,
                                     batch_hartley_2d_mask=batch_hartley_2d_mask,
                                     image_ctf_premultiplied=image_ctf_premultiplied,
                                     image_dose_weighted=image_dose_weighted,
                                     use_amp=use_amp)
            loss_accum += loss_batch * len(batch_images)
            if batch_it % args.log_interval == 0:
                flog(f'# [Train Epoch: {epoch + 1}/{args.num_epochs}] [{batch_it}/{nptcls} particles]  loss={loss_batch:.6f}')

        flog(f'# =====> Epoch: {epoch + 1} Average loss = {loss_accum / batch_it:.6}; Finished in {dt.now() - t2}')
        if args.checkpoint and epoch % args.checkpoint == 0:
            flog(f'Memory usage: {utils.check_memory_usage(device=device)}')
            save_checkpoint(model=model,
                            lat=lat,
                            optim=optim,
                            epoch=epoch,
                            norm=data.norm,
                            config_path=out_config,
                            out_mrc=f'{args.outdir}/reconstruct.{epoch}.mrc',
                            out_weights=f'{args.outdir}/weights.{epoch}.pkl',
                            amp=use_amp)

    # save model weights and evaluate the model on 3D lattice
    save_checkpoint(model=model,
                    lat=lat,
                    optim=optim,
                    epoch=epoch,
                    norm=data.norm,
                    config_path=out_config,
                    out_mrc=f'{args.outdir}/reconstruct.mrc',
                    out_weights=f'{args.outdir}/weights.pkl',
                    amp=use_amp)

    td = dt.now() - t1
    flog(f'Finished in {td} ({td / (args.num_epochs - start_epoch)} per epoch)')


if __name__ == '__main__':
    main(add_args().parse_args())
