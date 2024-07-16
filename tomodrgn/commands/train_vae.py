'''
Train a VAE for heterogeneous reconstruction with known pose for tomography data
'''
import numpy as np
import sys, os
import argparse
import pickle
from datetime import datetime as dt
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tomodrgn
from tomodrgn import utils, dataset, ctf, starfile
from tomodrgn.dataset import TiltSeriesMRCData
from tomodrgn.models import TiltSeriesHetOnlyVAE, DataParallelPassthrough
from tomodrgn.lattice import Lattice
from tomodrgn.beta_schedule import get_beta_schedule

log = utils.log
vlog = utils.vlog


def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    parser.add_argument('--zdim', type=int, required=True, default=128, help='Dimension of latent variable')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=200, help='Logging interval in N_PTCLS (default: %(default)s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increases verbosity')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 100000), help='Random seed')
    parser.add_argument('--pose', type=os.path.abspath, help='Optionally override star file poses with cryodrgn-format pose.pkl')
    parser.add_argument('--ctf', type=os.path.abspath, help='Optionally override star file CTF with cryodrgn-format ctf.pkl')

    group = parser.add_argument_group('Particle starfile loading and train/test split')
    group.add_argument('--ind', type=os.path.abspath, metavar='PKL', help='Filter starfile by particles (unique rlnGroupName values) using np array pkl')
    group.add_argument('--first-ntilts', type=int, default=None, help='Derive new train/test split filtering each particle to first n tilts before train/test split')
    group.add_argument('--fraction-train', type=float, default=1., help='Derive new train/test split with this fraction of each particles images assigned to train')
    group.add_argument('--ind-img-train', type=os.path.abspath, help='Filter starfile by images (rows) to train model using pre-existing np array pkl')
    group.add_argument('--ind-img-test', type=os.path.abspath, help='Filter starfile by images (rows) to test model using pre-existing np array pkl')
    group.add_argument('--sequential-tilt-sampling', action='store_true', help='Supply particle images of one particle to encoder in starfile order')

    group = parser.add_argument_group('Dataset loading and preprocessing')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.8, help='Real space inner windowing radius for cosine falloff to radius 1')
    group.add_argument('--window-r-outer', type=float, default=.9, help='Real space outer windowing radius for cosine falloff to radius 1')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD)')
    group.add_argument('--Apix', type=float, default=1.0, help='Override A/px from input starfile; useful if starfile does not have _rlnDetectorPixelSize col')

    group.add_argument_group('Weighting and masking')
    group.add_argument('--recon-tilt-weight', action='store_true', help='Weight reconstruction loss by cosine(tilt_angle)')
    group.add_argument('--recon-dose-weight', action='store_true', help='Weight reconstruction loss per tilt per pixel by dose dependent amplitude attenuation')
    group.add_argument('--l-dose-mask', action='store_true', help='Do not train on frequencies exposed to > 2.5x critical dose. Training lattice is intersection of this with --l-extent')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs')
    group.add_argument('-b', '--batch-size', type=int, default=1, help='Minibatch size')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer')
    group.add_argument('--lr', type=float, default=0.0002, help='Learning rate in Adam optimizer. Should co-scale linearly with batch size.')
    group.add_argument('--beta', default=None, help='Choice of beta schedule or a constant for KLD weight')
    group.add_argument('--beta-control', type=float, help='KL-Controlled VAE gamma. Beta is KL target')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: 0, std of dataset)')
    group.add_argument('--no-amp', action='store_true', help='Disable use of mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')

    group = parser.add_argument_group('Encoder Network')
    group.add_argument('--enc-layers-A', dest='qlayersA', type=int, default=3, help='Number of hidden layers for each tilt')
    group.add_argument('--enc-dim-A', dest='qdimA', type=int, default=256, help='Number of nodes in hidden layers for each tilt')
    group.add_argument('--out-dim-A', type=int, default=128, help='Number of nodes in output layer of encA == ntilts * number of nodes input to encB')
    group.add_argument('--enc-layers-B', dest='qlayersB', type=int, default=3, help='Number of hidden layers encoding merged tilts')
    group.add_argument('--enc-dim-B', dest='qdimB', type=int, default=512, help='Number of nodes in hidden layers encoding merged tilts')
    group.add_argument('--enc-mask', type=int, help='Circular mask of image for encoder (default: D/2; -1 for no mask)')
    group.add_argument('--pooling-function', type=str, choices=('concatenate', 'max', 'mean', 'median', 'set_encoder'), default='concatenate',
                       help='Function used to pool features along ntilts dimension after encA')
    group.add_argument('--num-seeds', type=int, default=1, help='number of seeds for PMA')
    group.add_argument('--num-heads', type=int, default=4, help='number of heads for multi head attention blocks')
    group.add_argument('--layer-norm', action='store_true', help='whether to apply layer normalization in the set transformer block')

    group = parser.add_argument_group('Decoder Network')
    group.add_argument('--dec-layers', dest='players', type=int, default=3, help='Number of hidden layers')
    group.add_argument('--dec-dim', dest='pdim', type=int, default=256, help='Number of nodes in hidden layers')
    group.add_argument('--l-extent', type=float, default=0.5, help='Coordinate lattice size (if not using positional encoding) (default: %(default)s)')
    group.add_argument('--pe-type', choices=('geom_ft', 'geom_full', 'geom_lowf', 'geom_nohighf', 'linear_lowf', 'gaussian', 'none'), default='gaussian', help='Type of positional encoding')
    group.add_argument('--feat-sigma', type=float, default=0.5, help="Scale for random Gaussian features")
    group.add_argument('--pe-dim', type=int, help='Num features in positional encoding')
    group.add_argument('--activation', choices=('relu', 'leaky_relu'), default='relu', help='Activation')

    group = parser.add_argument_group('Dataloader arguments')
    group.add_argument('--num-workers', type=int, default=0, help='Number of workers to use when batching particles for training. Has moderate impact on epoch time')
    group.add_argument('--prefetch-factor', type=int, default=2, help='Number of particles to prefetch per worker for training. Has moderate impact on epoch time')
    group.add_argument('--persistent-workers', action='store_true', help='Whether to persist workers after dataset has been fully consumed. Has minimal impact on run time')
    group.add_argument('--pin-memory', action='store_true', help='Whether to use pinned memory for dataloader. Has large impact on epoch time. Recommended.')
    return parser


def get_latest(args: argparse.Namespace,
               flog) -> argparse.Namespace:
    """
    Detect the latest completed epoch of model training and update args to load that epoch's model weights.
    :param args: argparse namespace from parse_args. Requires `num_epochs`, `outdir`, and `load` parameters
    :param flog: file to log output to disk
    :return: updated argparse namespace
    """
    # assumes args.num_epochs > latest checkpoint
    flog('Detecting latest checkpoint...')
    weights = [f'{args.outdir}/weights.{i}.pkl' for i in range(args.num_epochs)]
    weights = [f for f in weights if os.path.exists(f)]
    args.load = weights[-1]
    flog(f'Loading {args.load}')
    return args


def save_config(args: argparse.Namespace,
                data: TiltSeriesMRCData,
                lattice: Lattice,
                model: TiltSeriesHetOnlyVAE | DataParallelPassthrough,
                out_config: str) -> None:
    """
    Save input arguments and precalculated data, lattice, and model metadata.
    :param args: argparse namespace from parse_args
    :param data: TiltSeriesMRCData object containing input data
    :param lattice: Lattice object created using input data
    :param model: TiltSeriesHetOnlyVAE object created using input data (no weights included)
    :param out_config: name of output `.pkl` file in whcih to save configuration
    :return: None
    """
    dataset_args = dict(particles=args.particles,
                        norm=data.norm,
                        ntilts=data.ntilts_training,
                        invert_data=args.invert_data,
                        ind=args.ind,
                        ind_img=np.hstack(data.ptcls_to_imgs_ind),
                        window=args.window,
                        window_r=args.window_r,
                        window_r_outer=args.window_r_outer,
                        datadir=args.datadir,
                        sequential_tilt_sampling=args.sequential_tilt_sampling,
                        pose=args.pose,
                        ctf=args.ctf,
                        starfile_source=args.starfile_source)
    lattice_args = dict(D=lattice.D,
                        extent=lattice.extent,
                        ignore_DC=lattice.ignore_DC)
    model_args = dict(in_dim=model.in_dim.item(),  # .item() converts tensor to int for unpickling on non-gpu systems
                      qlayersA=args.qlayersA,
                      qdimA=args.qdimA,
                      qlayersB=args.qlayersB,
                      qdimB=args.qdimB,
                      out_dimA=args.out_dim_A,
                      players=args.players,
                      pdim=args.pdim,
                      zdim=args.zdim,
                      enc_mask=args.enc_mask,
                      pe_type=args.pe_type,
                      feat_sigma=args.feat_sigma,
                      pe_dim=args.pe_dim,
                      domain='fourier',
                      activation=args.activation,
                      l_dose_mask=args.l_dose_mask,
                      pooling_function=args.pooling_function,
                      num_seeds=args.num_seeds,
                      num_heads=args.num_heads,
                      layer_norm=args.layer_norm)
    training_args = dict(n=args.num_epochs,
                         B=args.batch_size,
                         wd=args.wd,
                         lr=args.lr,
                         beta=args.beta,
                         beta_control=args.beta_control,
                         amp=not args.no_amp,
                         multigpu=args.multigpu,
                         lazy=args.lazy,
                         recon_dose_weight=args.recon_dose_weight,
                         recon_tilt_weight=args.recon_tilt_weight,
                         verbose=args.verbose,
                         log_interval=args.log_interval,
                         checkpoint=args.checkpoint,
                         outdir=args.outdir)
    config = dict(dataset_args=dataset_args,
                  lattice_args=lattice_args,
                  model_args=model_args,
                  training_args=training_args)
    config['seed'] = args.seed
    with open(out_config, 'wb') as f:
        pickle.dump(config, f)
        meta = dict(time=dt.now(),
                    cmd=sys.argv,
                    version=tomodrgn.__version__)
        pickle.dump(meta, f)


def train_batch(*,
                model: TiltSeriesHetOnlyVAE | DataParallelPassthrough,
                scaler: torch.GradScaler,
                optim: torch.optim.Optimizer,
                lattice: Lattice,
                batch_images: torch.tensor,
                batch_rots: torch.tensor,
                batch_trans: torch.tensor,
                batch_ctf_params: torch.tensor,
                batch_recon_error_weights: torch.tensor,
                batch_hartley_2d_mask: torch.tensor,
                beta: float,
                beta_control: float | None = None,
                use_amp: bool = False) -> np.ndarray:
    """
    Train a TiltSeriesHetOnlyVAE model on a batch of tilt series particle images.
    :param model: TiltSeriesHetOnlyVAE object to be trained
    :param scaler: torch.GradScaler object to be used for scaling loss involving fp16 tensors to avoid over/underflow
    :param optim: torch.optim.Optimizer object to be used for optimizing the model
    :param lattice: Hartley-transform lattice of points for voxel grid operations
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
    :param beta: scaling factor to apply to KLD during loss calculation.
    :param beta_control: KL-Controlled VAE gamma. Beta is KL target.
    :param use_amp: If true, use Automatic Mixed Precision to reduce memory consumption and accelerate code execution via `torch.autocast` and `torch.scaler`
    :return: numpy array of losses: total loss, generative loss between reconstructed slices and input images, and kld loss between latent embeddings and standard normal
    """
    # prepare to train a new batch
    optim.zero_grad()
    model.train()

    # get key dimension sizes
    batchsize, ntilts, boxsize_ht, boxsize_ht = batch_images.shape

    # autocast auto-enabled and set to correct device
    with torch.autocast(device_type=batch_images.device.type, enabled=use_amp):
        # center images via translation and phase flip for partial CTF correction
        batch_images_preprocessed, ctf_weights = preprocess_batch(lattice=lattice,
                                                                  batch_images=batch_images,
                                                                  batch_trans=batch_trans,
                                                                  batch_ctf_params=batch_ctf_params)
        # encode the translated and CTF-phase-flipped images
        z_mu, z_logvar, z = encode_batch(model=model,
                                         batch_images=batch_images_preprocessed)
        # decode the lattice coordinate positions given the encoder-generated embeddings
        y_reconstructed = decode_batch(model=model,
                                       lattice=lattice,
                                       batch_rots=batch_rots,
                                       batch_hartley_2d_mask=batch_hartley_2d_mask,
                                       z=z)
        # calculate the model loss
        loss, gen_loss, kld_loss = loss_function(z_mu,
                                                 z_logvar,
                                                 batch_images_preprocessed,
                                                 y_reconstructed,
                                                 ctf_weights,
                                                 batch_recon_error_weights,
                                                 batch_hartley_2d_mask,
                                                 beta,
                                                 beta_control)

    # backpropogate the scaled loss and optimize model weights
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    return torch.tensor((loss, gen_loss, kld_loss)).detach().cpu().numpy()


def preprocess_batch(*,
                     lattice: Lattice,
                     batch_images: torch.tensor,
                     batch_trans: torch.tensor,
                     batch_ctf_params: torch.tensor) -> tuple[torch.tensor, Union[torch.tensor, None]]:
    """
    Center images via translation and phase flip for partial CTF correction, as needed
    :param lattice: Hartley-transform lattice of points for voxel grid operations
    :param batch_images: Batch of images to be used for training, shape (batchsize, ntilts, boxsize_ht, boxsize_ht)
    :param batch_trans: Batch of 2-D translation matrices corresponding to `batch_images` known poses, shape (batchsize, ntilts, 2).
            May be `torch.zeros((batchsize))` instead to indicate no translations should be applied to the input images.
    :param batch_ctf_params: Batch of CTF parameters corresponding to `batch_images` known CTF parameters, shape (batchsize, ntilts, 9).
            May be `torch.zeros((batchsize))` instead to indicate no CTF corruption should be applied to the reconstructed slice.
    :return batch_images: translationally-centered and phase-flipped batch of images to be used for training, shape (batchsize, ntilts, boxsize_ht, boxsize_ht)
    :return batch_ctf_weights: CTF evaluated at each spatial frequency corresponding to input images, shape (batchsize, ntilts, boxsize_ht, boxsize_ht)
    """

    # get key dimension sizes
    batchsize, ntilts, boxsize_ht, boxsize_ht = batch_images.shape

    # translate the image
    if not torch.all(batch_trans == 0):
        batch_images = lattice.translate_ht(batch_images.view(batchsize * ntilts, -1), batch_trans.view(batchsize * ntilts, 1, 2))

    batch_images = batch_images.view(batchsize, ntilts, boxsize_ht, boxsize_ht)

    # phase flip the input CTF-corrupted image and calculate CTF weights to apply later
    if not torch.all(batch_ctf_params == 0):
        freqs = lattice.freqs2d.unsqueeze(0).expand(batchsize * ntilts, *lattice.freqs2d.shape).clone()  # one lattice copy per image in batch
        freqs /= batch_ctf_params[:, :, 1].view(batchsize * ntilts, 1, 1)  # convert units from 1/px to 1/Angstrom
        batch_ctf_weights = ctf.compute_ctf(freqs, *torch.split(batch_ctf_params.view(batchsize * ntilts, -1)[:, 2:], 1, 1))
        batch_ctf_weights = batch_ctf_weights.view(batchsize, ntilts, boxsize_ht, boxsize_ht)
        batch_images *= batch_ctf_weights.sign()  # phase flip by CTF to be all positive amplitudes
    else:
        batch_ctf_weights = None

    return y, ctf_weights

    return batch_images, batch_ctf_weights


def encode_batch(*,
                 model: TiltSeriesHetOnlyVAE,
                 batch_images: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
                 model: TiltSeriesHetOnlyVAE | DataParallelPassthrough,
    """
    Encode a batch of particles represented by multiple images to per-particle latent embeddings
    :param model: TiltSeriesHetOnlyVAE object to be trained
    :param batch_images: Batch of images to be used for training, shape (batchsize, ntilts, boxsize_ht, boxsize_ht)
    :return z_mu: Direct output of encoder module parameterizing the mean of the latent embedding for each particle, shape (batchsize, zdim)
    :return z_logvar: Direct output of encoder module parameterizing the log variance of the latent embedding for each particle, shape (batchsize, zdim)
    :return z: Resampling of the latent embedding for each particle parameterized as a gaussian with mean z_mu and variance z_logvar, shape (batchsize, zdim)
    """

    # get key dimension sizes
    batchsize, ntilts, _, _ = batch_images.shape

    # encode
    z_mu, z_logvar = model.encode(batch_images, batchsize, ntilts)  # ouput is B x zdim, i.e. one value per ptcl (not per img)
    z = model.reparameterize(z_mu, z_logvar)

    return z_mu, z_logvar, z


def decode_batch(*,
                 model: TiltSeriesHetOnlyVAE | DataParallelPassthrough,
                 lattice: Lattice,
                 batch_rots: torch.Tensor,
                 batch_hartley_2d_mask: torch.Tensor,
                 z: torch.Tensor) -> torch.Tensor:
    """
    Decode a batch of particles represented by multiple images from per-particle latent embeddings and corresponding lattice positions to evaluate
    :param model: TiltSeriesHetOnlyVAE object to be trained
    :param lattice: Hartley-transform lattice of points for voxel grid operations
    :param batch_rots: Batch of 3-D rotation matrices corresponding to `batch_images` known poses, shape (batchsize, ntilts, 3, 3)
    :param batch_hartley_2d_mask: Batch of 2-D masks to be applied per-spatial-frequency.
            Calculated as the intersection of critical dose exposure curves and a Nyquist-limited circular mask in reciprocal space, including masking the DC component.
    :param z: Resampled latent embedding for each particle
    :return: Reconstructed central slices of Fourier space volumes corresponding to each particle in the batch, shape (batchsize * ntilts * boxsize_ht**2 [`batch_hartley_2d_mask`])
    """

    # get key dimension sizes
    boxsize_ht = lattice.D
    batchsize, ntilts, _, _ = batch_rots.shape

    # TODO reimplement decoding more cleanly and efficiently with NestedTensors when autograd available: https://github.com/pytorch/nestedtensor
    # prepare lattice at rotated fourier components
    input_coords = lattice.coords @ batch_rots  # shape batchsize x ntilts x boxsize_ht**2 x 3 from [boxsize_ht**2 x 3] @ [batchsize x ntilts x 3 x 3]

    # concatenate with z
    z = z.view(z.shape[0], *([1] * (input_coords.ndimension() - 2)), z.shape[-1])  # shape batchsize x 1 x 1 x zdim
    z_percoord = z.expand(*input_coords.shape[:-1], z.shape[-1])  # shape batchsize x ntilts x boxsize_ht**2 x zdim
    input_coords_z = torch.cat((input_coords, z_percoord), dim=-1)  # shape batchsize x ntilts x boxsize_ht**2 x 3+zdim

    # filter by dec_mask
    input_coords_z_masked = input_coords_z[batch_hartley_2d_mask.view(batchsize, ntilts, boxsize_ht * boxsize_ht), :].unsqueeze(0)
    pseudo_batchsize = utils.first_n_factors(input_coords_z_masked.shape[-2], lower_bound=8)[0]
    input_coords_z_masked = input_coords_z_masked.view(pseudo_batchsize, -1, input_coords_z_masked.shape[-1])

    # pass to decoder
    batch_images_recon = model(input_coords_z_masked).view(1, -1)
    return batch_images_recon


def _unparallelize(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def loss_function(z_mu, z_logvar, y, y_recon, ctf_weights, cumulative_weights, dec_mask, beta, beta_control=None):
    # reconstruction error
    y = y[dec_mask].view(1, -1)
    if ctf_weights is not None:
        y_recon *= ctf_weights[dec_mask].view(1, -1)  # apply CTF to reconstructed image
        y *= ctf_weights[dec_mask].sign().view(1, -1)  # undo phase flipping from preprocess_batch
    gen_loss = torch.mean((cumulative_weights[dec_mask].view(1, -1) * ((y_recon - y) ** 2)))

    # latent loss
    kld = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0)
    if beta_control is None:
        kld_loss = beta * kld / dec_mask.sum().float()
    else:
        kld_loss = beta_control * (beta - kld) ** 2 / dec_mask.sum().float()

    # total loss
    loss = gen_loss + kld_loss
    return loss, gen_loss, kld_loss


def eval_z(model, lattice, data, args, device, use_amp=False):
    model.eval()
    assert not model.training
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=use_amp):
            z_mu_all = torch.zeros((data.nptcls, model.zdim), device=device, dtype=torch.half if use_amp else torch.float)
            z_logvar_all = torch.zeros((data.nptcls, model.zdim), device=device, dtype=torch.half if use_amp else torch.float)
            data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                        persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
            for batch_images, _, batch_trans, batch_ctf_params, _, _, batch_indices in data_generator:
                B, ntilts, D, D = batch_images.shape

                # transfer to GPU
                batch_images = batch_images.to(device)
                batch_ctf_params = batch_ctf_params.to(device)
                batch_trans = batch_trans.to(device)

                # use run_batch to reduce redundant code errors
                batch_images, ctf_weights = preprocess_batch(lattice=lattice,
                                                             batch_images=batch_images,
                                                             batch_trans=batch_trans,
                                                             batch_ctf_params=batch_ctf_params)
                z_mu, z_logvar, z = encode_batch(model=model,
                                                 batch_images=batch_images)

                z_mu_all[batch_indices] = z_mu
                z_logvar_all[batch_indices] = z_logvar
            z_mu_all = z_mu_all.cpu().numpy()
            z_logvar_all = z_logvar_all.cpu().numpy()

            if np.any(z_mu_all == np.nan) or np.any(z_mu_all == np.inf):
                nan_count = np.sum(np.isnan(z_mu_all))
                inf_count = np.sum(np.isinf(z_mu_all))
                sys.exit(f'Latent evaluation at end of epoch failed: z.pkl would contain {nan_count} NaN and {inf_count} Inf')

            return z_mu_all, z_logvar_all


def save_checkpoint(model, optim, epoch, z_mu_train, z_logvar_train, z_mu_test, z_logvar_test, out_weights, out_z_train, out_z_test, scaler):
    '''
    Save model weights and latent encoding z
    '''
    # save model weights
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None
    }, out_weights)
    # save z
    with open(out_z_train, 'wb') as f:
        pickle.dump(z_mu_train.astype(np.float32), f)
        pickle.dump(z_logvar_train.astype(np.float32), f)
    if z_mu_test is not None:
        with open(out_z_test, 'wb') as f:
            pickle.dump(z_mu_test.astype(np.float32), f)
            pickle.dump(z_logvar_test.astype(np.float32), f)


def convergence_latent(z_mu_train, z_logvar_train, z_mu_test, z_logvar_test, workdir, epoch, significance_threshold=0.05):
    '''
    Calculates the fraction of particles whose latent embeddings overlap between train and test
    with significant p value relative to reparamaterization epsilon

    Currently evaluating as two-sample z-test (train and test) with sample size 1 (per particle)
    Multiplying results along latent dim (treated as indendependent) and updating p_value according to bonferroni correction
        (Or could generalize to mahalanobis distance which allows non-diagonal multivariate gaussians)

    https://www.statology.org/p-value-from-z-score-python/
    https://www.probabilitycourse.com/chapter4/4_2_3_normal.php
    https://www.analyticsvidhya.com/blog/2020/06/statistics-analytics-hypothesis-testing-z-test-t-test/#Deciding_Between_Z-Test_and_T-Test
    http://homework.uoregon.edu/pub/class/es202/ztest.html

    :param epoch:
    :param workdir:
    :param z_mu_train: np array (n_particles, zdim) of predicted latent means for each particle's train images
    :param z_logvar_train: np array (n_particles, zdim) of predicted latent log variance for each particle's train images
    :param z_mu_test: np array (n_particles, zdim) of predicted latent means for each particle's test images
    :param z_logvar_test: np array (n_particles, zdim) of predicted latent log variance for each particle's test images
    :param significance_threshold: float, significance level aka p value for single axis
    :return: fraction_particles_different: float, fraction of particles significantly different in latent embedding between train and test
    '''

    # TODO What is the right null/alternate hypothesis to test?
    # run the two-sample z-test
    z_scores = (z_mu_train - z_mu_test) / (np.sqrt(np.exp(z_logvar_train) ** 2 + np.exp(z_logvar_test) ** 2))

    # convert z-scores to p-values given two-tailed z-score
    p_values = norm.sf(abs(z_scores)) * 2

    # multiply p values along zdim axis given independent latent dimensions aka independent tests
    p_values = np.prod(p_values, axis=1)

    # apply significace test with bonferroni-corrected p-value
    bonferroni = z_mu_train.shape[1]
    significance_threshold = significance_threshold / bonferroni

    # save data and log summary value
    utils.save_pkl(p_values, f'{workdir}/convergence_latent_pvalues.{epoch}.pkl')
    log(f'Convergence epoch {epoch}: fraction of particles with significantly similar latent: {np.sum(p_values < significance_threshold) / p_values.shape[0]}')


def convergence_volumes_generate(z_train, z_test, epoch, workdir, weights_path, config_path, volume_count=100):
    '''
    Select indices and generate corresponding test/train volumes for convergence metrics
    :param z_train: numpy array of shape (n_particles, zdim) of latent values deriving from train particle images
    :param z_test: numpy array of shape (n_particles) zdim) of latent values deriving from test particle images
    :param epoch:
    :param workdir: str, absolute path to model workdir to store generated volumes
    :param weights_path:
    :param config_path:
    :param volume_count: int, number of volumes to generate for each of train/test
    '''
    # sample volume_count particles
    n_particles = z_train.shape[0]
    ind_sel = np.sort(np.random.choice(n_particles, size=volume_count))

    # generate the train and test volumes for corresponding particles
    from tomodrgn.commands.analyze import VolumeGenerator
    vg = VolumeGenerator(weights_path, config_path, vol_args=dict())
    os.mkdir(f'{workdir}/scratch.{epoch}.train')
    os.mkdir(f'{workdir}/scratch.{epoch}.test')
    vg.gen_volumes(f'{workdir}/scratch.{epoch}.train', z_train[ind_sel])
    vg.gen_volumes(f'{workdir}/scratch.{epoch}.test', z_test[ind_sel])

    # save data
    utils.save_pkl(ind_sel, f'{workdir}/convergence_volumes_sel.{epoch}.pkl')


def convergence_volumes_testtrain_correlation(workdir, epoch, volume_count=100):
    '''
    Calculate the correlation between volumes generated from test and train latent embeddings of the same particles in the same epoch
    :param workdir: str, absolute path to model workdir
    :param epoch: int, current epoch being evaluated
    :param volume_count: int, number of volumes to generate for each of train/test
    '''
    # calculate pairwise FSC and return resolution of FSC=0.5
    resolutions_point5 = np.zeros(volume_count)
    fscs = []
    for i in range(volume_count):
        vol_train = f'{workdir}/scratch.{epoch}.train/vol_{i:03d}.mrc'
        vol_test = f'{workdir}/scratch.{epoch}.test/vol_{i:03d}.mrc'
        x, fsc = utils.calc_fsc(vol_train, vol_test, mask='soft')
        resolutions_point5[i] = x[-1] if np.all(fsc >= 0.5) else x[np.argmax(fsc < 0.5)]
        fscs.append(fsc)

    # save data and log summary value
    utils.save_pkl(np.asarray(fscs), f'{workdir}/convergence_volumes_testtrain_correlation.{epoch}.pkl')
    log(f'Convergence epoch {epoch}: 90th percentile of test/train map-map FSC 0.5 resolutions (units: 1/px): {np.percentile(resolutions_point5, 90)}')


def convergence_volumes_testtest_scale(workdir, epoch, volume_count = 100):
    '''
    Calculate the scale of heterogeneity among all volumes generated from test latent embeddings of the same particles in the same epoch
    :param workdir: str, absolute path to model workdir
    :param epoch: int, current epoch being evaluated
    :param volume_count: int, number of volumes to generate for each of train/test
    '''
    # initialize empty pairwise distance matrix for CCs and corresponding upper triangle (non-redundant) pairwise indices
    pairwise_ccs = np.zeros((volume_count, volume_count))
    ind_triu = np.triu_indices(n=volume_count, k=1, m=volume_count)
    row_ind, col_ind = ind_triu

    # iterate through zipped indices, loading volumes and calculating CC
    from tomodrgn.mrc import parse_mrc
    from tomodrgn.commands.convergence_vae import calc_cc
    for i, j in zip(row_ind, col_ind):
        vol_train, vol_train_header = parse_mrc(f'{workdir}/scratch.{epoch}.train/vol_{i:03d}.mrc')
        vol_test, vol_test_header = parse_mrc(f'{workdir}/scratch.{epoch}.test/vol_{j:03d}.mrc')
        pairwise_ccs[i,j] = calc_cc(vol_train, vol_test)

    # calculate pairwise volume distances as 1 - CC
    pairwise_cc_dist = np.ones_like(pairwise_ccs) - pairwise_ccs

    # save data and log summary value
    utils.save_pkl(pairwise_cc_dist, f'{workdir}/convergence_volumes_testtest_scale.{epoch}.pkl')
    log(f'Convergence epoch {epoch}: sum of complement-CC distances for all pairwise test/test volumes: {np.sum(pairwise_cc_dist[ind_triu])}')


def main(args):
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    LOG = f'{args.outdir}/run.log'

    def flog(msg):  # HACK: switch to logging module
        return utils.flog(msg, LOG)

    if args.load == 'latest':
        args = get_latest(args, flog)
    flog(' '.join(sys.argv))
    flog(args)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## set the device
    device = utils.get_default_device()
    if device == torch.device('cpu'):
        args.no_amp = True
        flog('Warning: pytorch AMP does not support non-CUDA (e.g. cpu) devices. Automatically disabling AMP and continuing')
        # https://github.com/pytorch/pytorch/issues/55374

    # load star file
    ptcls_star = starfile.TiltSeriesStarfile(args.particles)

    # load the particle indices
    ptcls_to_imgs_ind = ptcls_star.get_ptcl_img_indices()
    if args.ind is not None:
        flog(f'Reading supplied particle indices {args.ind}')
        ind = pickle.load(open(args.ind, 'rb'))
        assert len(ptcls_to_imgs_ind) >= len(ind), 'More particles specified than particles found in star file'
        assert len(ptcls_to_imgs_ind) >= max(ind), 'Specified particles exceed the number of particles found in star file'
    else:
        ind = None

    # prepare particle image train/test split
    if args.ind_img_train:
        flog(f'Reading supplied train image indices {args.ind_img_train}')
        inds_train = utils.load_pkl(args.ind_img_train)
        assert np.max(np.hstack(ptcls_to_imgs_ind)) >= len(inds_train), 'More particle images specified for train split than particle images found in star file'
        assert np.max(np.hstack(ptcls_to_imgs_ind)) >= max(inds_train), 'Specified particle images for train split exceed then number of particle images found in star file'
        if args.ind_img_test:
            flog(f'Reading supplied test image indices {args.ind_img_test}')
            inds_test = utils.load_pkl(args.ind_img_test)
            assert np.max(np.hstack(ptcls_to_imgs_ind)) >= (len(inds_train) + len(inds_test)), 'More particle images specified for train and test splits than particle images found in star file'
            assert np.max(np.hstack(ptcls_to_imgs_ind)) >= max(inds_test), 'Specified particle images for test split exceed then number of particle images found in star file'
        else:
            flog('No test image indices supplied, so none will be used')
            inds_test = np.array([])
    else:
        flog(f'Creating new train/test split indices with fraction_train={args.fraction_train} and first_ntilts={args.first_ntilts}')
        inds_train, inds_test = ptcls_star.make_test_train_split(fraction_train=args.fraction_train,
                                                                 first_ntilts=args.first_ntilts,
                                                                 show_summary_stats=True)
        utils.save_pkl(inds_train, f'{args.outdir}/inds_imgs_train.pkl')
        utils.save_pkl(inds_test, f'{args.outdir}/inds_imgs_test.pkl')

    # load the particles + poses + ctf from input starfile
    flog(f'Loading dataset from {args.particles}')
    data_train = dataset.TiltSeriesMRCData(ptcls_star,
                                           norm=args.norm,
                                           invert_data=args.invert_data,
                                           ind_ptcl=ind,
                                           ind_img=inds_train,
                                           window=args.window,
                                           datadir=args.datadir,
                                           window_r=args.window_r,
                                           window_r_outer=args.window_r_outer,
                                           recon_dose_weight=args.recon_dose_weight,
                                           recon_tilt_weight=args.recon_tilt_weight,
                                           l_dose_mask=args.l_dose_mask,
                                           lazy=args.lazy,
                                           use_all_images=True,
                                           sequential_tilt_sampling=args.sequential_tilt_sampling)
    if len(inds_test) > 0:
        data_test = dataset.TiltSeriesMRCData(ptcls_star,
                                              norm=data_train.norm,  # share normalization with train images
                                              invert_data=args.invert_data,
                                              ind_ptcl=ind,
                                              ind_img=inds_test,
                                              window=args.window,
                                              datadir=args.datadir,
                                              window_r=args.window_r,
                                              window_r_outer=args.window_r_outer,
                                              recon_dose_weight=args.recon_dose_weight,
                                              recon_tilt_weight=args.recon_tilt_weight,
                                              l_dose_mask=args.l_dose_mask,
                                              lazy=args.lazy,
                                              use_all_images=True,
                                              sequential_tilt_sampling=args.sequential_tilt_sampling)
    else:
        data_test = None
    boxsize_ht = data_train.boxsize_ht
    nptcls = data_train.nptcls

    # load pose and CTF from pkl if supplied
    if args.pose or args.ctf:
        def load_pose_pkl(pose_path, ind_img=None):
            rots, trans = utils.load_pkl(pose_path)
            rots = np.asarray(rots, dtype=np.float32)
            trans = np.asarray(trans, dtype=np.float32)
            if ind_img is not None:
                rots = rots[ind_img]
                if trans is not None:
                    trans = trans[ind_img]
            return rots, trans

        def load_ctf_pkl(ctf_path, ind_img=None):
            ctf_params = utils.load_pkl(ctf_path)
            ctf_params = np.asarray(ctf_params, dtype=np.float32)
            if ind_img is not None:
                ctf_params = ctf_params[ind_img]
            return ctf_params

        if args.pose is not None:
            flog(f'Updating dataset to use poses from {args.pose}')
            rots_train, trans_train = load_pose_pkl(args.pose, inds_train)
            assert rots_train.shape == (data_train.nimgs, 3, 3)
            data_train.rot = rots_train
            if trans_train is not None:
                assert trans_train.shape == (data_train.nimgs, 2)
                data_train.trans = trans_train
            if len(inds_test) > 0:
                rots_test, trans_test = load_pose_pkl(args.pose, inds_test)
                assert rots_test.shape == (data_test.nimgs, 3, 3)
                data_test.rot = rots_test
                if trans_test is not None:
                    assert trans_test.shape == (data_test.nimgs, 2)
                    data_test.trans = trans_test

        if args.ctf is not None:
            flog(f'Updating dataset to use CTF parameters from {args.ctf}')
            ctf_params_train = load_ctf_pkl(args.ctf, inds_train)
            assert ctf_params_train.shape == (data_train.nimgs, 9)
            data_train.ctf_params = ctf_params_train
            if len(inds_test) > 0:
                ctf_params_test = load_ctf_pkl(args.ctf, inds_test)
                assert ctf_params_test.shape == (data_test.nimgs, 9)
                data_test.ctf_params = ctf_params_test

    # instantiate lattice
    lattice = Lattice(boxsize_ht, extent=args.l_extent, device=device)

    # determine which pixels to encode (equivalently applicable to all particles)
    if args.enc_mask is None:
        args.enc_mask = boxsize_ht // 2
    if args.enc_mask > 0:
        # encode pixels within defined circular radius in fourier space
        assert args.enc_mask <= boxsize_ht // 2
        enc_mask = lattice.get_circular_mask(args.enc_mask)
        in_dim = enc_mask.sum()
    elif args.enc_mask == -1:
        # encode all pixels in fourier space
        enc_mask = None
        in_dim = lattice.D ** 2
    else:
        raise RuntimeError("Invalid argument for encoder mask radius {}".format(args.enc_mask))
    flog(f'Pixels encoded per tilt (+ enc-mask):  {in_dim.sum()}')

    # determine the number of tilts to randomly sample per particle during training and testing
    if args.pooling_function in ['max', 'mean', 'median']:
        '''
        if batchsize == 1:
            can have any number of tilts per particle (recognize via ntilts_training==None?)
        if batchsize > 1:
            if number of tilts differs between particles:
                may cause errors during batch collation before starting batch
                may be able to get by with nestedtensor or alternate clever dtype 
                or zero padding "fake" images and subselecting on GPU as appropriate
                otherwise require use_first_ntilts to truncate?
            else:   
                can have any number of tilts per particle
        '''
        if args.batch_size == 1 and not args.multigpu:
            # can sample all images specified by inds_train and inds_test for each particle in each batch
            data_train.ntilts_training = None
            data_test.ntilts_training = None
            flog(f'Will sample {data_train.ntilts_training} tilts per particle for train split, to pass to encoder, due to model requirements of --pooling-function {args.pooling_function}')
            if len(inds_test) > 0:
                flog(f'Will sample {data_test.ntilts_training} tilts per particle for test split, to pass to encoder, due to model requirements of --pooling-function {args.pooling_function}')
        else:
            # TODO implement ntilt_training calculation for set-style encoder with batchsize > 1
            raise NotImplementedError
    elif args.pooling_function in ['concatenate', 'set_encoder']:
        # requires same number of tilts for both test and train for every particle
        # therefore add further subset train/test to sample same number of tilts from each for each particle
        if len(inds_test) > 0:
            n_tilts_for_model = min(data_train.ntilts_range[0], data_test.ntilts_range[0])
            data_train.ntilts_training = n_tilts_for_model
            data_test.ntilts_training = n_tilts_for_model
            flog(f'Will sample {n_tilts_for_model} tilts per particle for model input, for each of test and train, due to model requirements of --pooling-function {args.pooling_function}')
        else:
            n_tilts_for_model = data_train.ntilts_range[0]
            data_train.ntilts_training = n_tilts_for_model
            flog(f'Will sample {n_tilts_for_model} tilts per particle for model input, for train split, due to model requirements of --pooling-function {args.pooling_function}')

    # instantiate model
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    flog(f'Pooling function prior to encoder B: {args.pooling_function}')
    model = TiltSeriesHetOnlyVAE(lattice, args.qlayersA, args.qdimA, args.out_dim_A, data_train.ntilts_training,
                                 args.qlayersB, args.qdimB, args.players, args.pdim, in_dim, args.zdim,
                                 enc_mask=enc_mask, enc_type=args.pe_type, enc_dim=args.pe_dim,
                                 domain='fourier', activation=activation, l_dose_mask=args.l_dose_mask,
                                 feat_sigma=args.feat_sigma, pooling_function=args.pooling_function,
                                 num_seeds=args.num_seeds, num_heads=args.num_heads, layer_norm=args.layer_norm)
    # model.to(device)
    flog(model)
    flog('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    flog('{} parameters in encoder'.format(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)))
    flog('{} parameters in decoder'.format(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)))

    # save configuration
    out_config = f'{args.outdir}/config.pkl'
    save_config(args, data_train, lattice, model, out_config)

    # set beta schedule
    if args.beta is None:
        args.beta = 1. / args.zdim
    beta_schedule = get_beta_schedule(args.beta, n_iterations=args.num_epochs * nptcls + args.batch_size)

    # instantiate optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-4)  # https://github.com/pytorch/pytorch/issues/40497#issuecomment-1084807134

    # Mixed precision training with AMP
    use_amp = not args.no_amp
    flog(f'AMP acceleration enabled (autocast + gradscaler) : {use_amp}')
    scaler = torch.GradScaler(device_type=device.type, enabled=use_amp)
    if use_amp:
        if not args.batch_size % 8 == 0: flog('Warning: recommended to have batch size divisible by 8 for AMP training')
        if not (boxsize_ht - 1) % 8 == 0: flog('Warning: recommended to have image size divisible by 8 for AMP training')
        if in_dim % 8 != 0: flog('Warning: recommended to have masked image dimensions divisible by 8 for AMP training')
        assert args.qdimA % 8 == 0, "Encoder hidden layer dimension must be divisible by 8 for AMP training"
        assert args.qdimB % 8 == 0, "Encoder hidden layer dimension must be divisible by 8 for AMP training"
        if args.zdim % 8 != 0: flog('Warning: recommended to have z dimension divisible by 8 for AMP training')
        assert args.pdim % 8 == 0, "Decoder hidden layer dimension must be divisible by 8 for AMP training"

    # restart from checkpoint
    if args.load:
        flog('Loading checkpoint from {}'.format(args.load))
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        assert start_epoch < args.num_epochs
        try:
            scaler.load_state_dict(checkpoint['scaler'])
        except:
            flog('No GradScaler instance found in specified checkpoint; creating new GradScaler')
    else:
        start_epoch = 0
        model.to(device)

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
    data_train_generator = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                      persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
    for epoch in range(start_epoch, args.num_epochs):
        t2 = dt.now()
        losses_accum = np.zeros((3), dtype=np.float32)
        batch_it = 0

        for batch_images, batch_rots, batch_trans, batch_ctf_params, batch_recon_error_weights, batch_hartley_2d_mask, batch_indices in data_train_generator:
            # impression counting
            batch_it += len(batch_indices)  # total number of ptcls seen
            global_it = nptcls * epoch + batch_it
            beta = beta_schedule(global_it)

            # transfer to GPU
            batch_images = batch_images.to(device)
            batch_rots = batch_rots.to(device)
            batch_trans = batch_trans.to(device)
            batch_ctf_params = batch_ctf_params.to(device)
            batch_recon_error_weights = batch_recon_error_weights.to(device)
            batch_hartley_2d_mask = batch_hartley_2d_mask.to(device)

            # training minibatch
            losses_batch = train_batch(model=model,
                                       scaler=scaler,
                                       optim=optim,
                                       lattice=lattice,
                                       batch_images=batch_images,
                                       batch_rots=batch_rots,
                                       batch_trans=batch_trans,
                                       batch_ctf_params=batch_ctf_params,
                                       batch_recon_error_weights=batch_recon_error_weights,
                                       batch_hartley_2d_mask=batch_hartley_2d_mask,
                                       beta=beta,
                                       beta_control=args.beta_control,
                                       use_amp=use_amp)

            # logging
            if batch_it % args.log_interval == 0:
                log(f'# [Train Epoch: {epoch + 1}/{args.num_epochs}] [{batch_it}/{nptcls} subtomos] gen loss={losses_batch[1]:.6f}, kld={losses_batch[2]:.6f}, beta={beta:.6f}, loss={losses_batch[0]:.6f}')
            losses_accum += losses_batch * len(batch_images)
        flog(
            f'# =====> Epoch: {epoch + 1} Average gen loss = {losses_accum[1] / batch_it:.6f}, KLD = {losses_accum[2] / batch_it:.6f}, total loss = {losses_accum[0] / batch_it:.6f}; Finished in {dt.now() - t2}')
        if args.checkpoint and (epoch + 1) % args.checkpoint == 0:
            if device.type != 'cpu':
                flog(f'GPU memory usage: {utils.check_memory_usage()}')
            out_weights = f'{args.outdir}/weights.{epoch}.pkl'
            out_z_train = f'{args.outdir}/z.{epoch}.train.pkl'
            z_mu_train, z_logvar_train = eval_z(model, lattice, data_train, args, device, use_amp=use_amp)
            if data_test:
                out_z_test = f'{args.outdir}/z.{epoch}.test.pkl'
                z_mu_test, z_logvar_test = eval_z(model, lattice, data_test, args, device, use_amp=use_amp)
            else:
                z_mu_test, z_logvar_test, out_z_test = None, None, None
            save_checkpoint(model, optim, epoch, z_mu_train, z_logvar_train, z_mu_test, z_logvar_test, out_weights, out_z_train, out_z_test, scaler)
            if data_test:
                flog('Calculating convergence metrics using test/train split...')
                convergence_latent(z_mu_train, z_logvar_train, z_mu_test, z_logvar_test, args.outdir, epoch)
                convergence_volumes_generate(z_mu_train, z_mu_test, epoch, args.outdir, f'{args.outdir}/weights.{epoch}.pkl', f'{args.outdir}/config.pkl', volume_count=100)
                convergence_volumes_testtrain_correlation(args.outdir, epoch, volume_count=100)
                convergence_volumes_testtest_scale(args.outdir, epoch, volume_count=100)

    # save model weights, latent encoding, and evaluate the model on 3D lattice
    out_weights = f'{args.outdir}/weights.pkl'
    out_z_train = f'{args.outdir}/z.train.pkl'
    z_mu_train, z_logvar_train = eval_z(model, lattice, data_train, args, device, use_amp=use_amp)
    if data_test:
        out_z_test = f'{args.outdir}/z.test.pkl'
        z_mu_test, z_logvar_test = eval_z(model, lattice, data_test, args, device, use_amp=use_amp)
    else:
        z_mu_test, z_logvar_test, out_z_test = None, None, None
    save_checkpoint(model, optim, epoch, z_mu_train, z_logvar_train, z_mu_test, z_logvar_test, out_weights, out_z_train, out_z_test, scaler)

    td = dt.now() - t1
    flog(f'Finished in {td} ({td / (args.num_epochs - start_epoch)} per epoch)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)
