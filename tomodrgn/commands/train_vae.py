"""
Train a VAE for heterogeneous reconstruction with known pose for tomography data
"""
import numpy as np
import sys
import os
import argparse
import pickle
from datetime import datetime as dt
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from tomodrgn import utils, ctf, config
from tomodrgn.starfile import TiltSeriesStarfile
from tomodrgn.dataset import TiltSeriesMRCData
from tomodrgn.models import TiltSeriesHetOnlyVAE, DataParallelPassthrough, print_tiltserieshetonlyvae_ascii
from tomodrgn.lattice import Lattice
from tomodrgn.beta_schedule import get_beta_schedule
from tomodrgn.analysis import VolumeGenerator

log = utils.log
vlog = utils.vlog


def add_args(_parser):
    _parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, or .txt)')
    _parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    _parser.add_argument('--zdim', type=int, required=True, default=128, help='Dimension of latent variable')
    _parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    _parser.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    _parser.add_argument('--log-interval', type=int, default=200, help='Logging interval in N_PTCLS (default: %(default)s)')
    _parser.add_argument('-v', '--verbose', action='store_true', help='Increases verbosity')
    _parser.add_argument('--seed', type=int, default=np.random.randint(0, 100000), help='Random seed')

    group = _parser.add_argument_group('Particle starfile loading and filtering')
    group.add_argument('--source-software', type=str, choices=('auto', 'warp_v1', 'nextpyp', 'relion_v5', 'warp_v2'), default='auto',
                       help='Manually set the software used to extract particles. Default is to auto-detect.')
    group.add_argument('--ind-ptcls', type=os.path.abspath, metavar='PKL', help='Filter starfile by particles (unique rlnGroupName values) using np array pkl as indices')
    group.add_argument('--ind-imgs', type=os.path.abspath, help='Filter starfile by particle images (star file rows) using np array pkl as indices')
    group.add_argument('--sort-ptcl-imgs', choices=('unsorted', 'dose_ascending', 'random'), default='unsorted', help='Sort the star file images on a per-particle basis by the specified criteria')
    group.add_argument('--use-first-ntilts', type=int, default=-1, help='Keep the first `use_first_ntilts` images of each particle in the sorted star file.'
                                                                        'Default -1 means to use all. Will drop particles with fewer than this many tilt images.')
    group.add_argument('--use-first-nptcls', type=int, default=-1, help='Keep the first `use_first_nptcls` particles in the sorted star file. Default -1 means to use all.')

    group = _parser.add_argument_group('Particle starfile train/test split')
    group.add_argument('--fraction-train', type=float, default=1., help='Derive new train/test split with this fraction of each particles images assigned to train')
    group.add_argument('--show-summary-stats', type=bool, default=True, help='Log distribution statistics of particle sampling for test/train splits')

    group = _parser.add_argument_group('Dataset loading and preprocessing')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.8, help='Real space inner windowing radius for cosine falloff to radius 1')
    group.add_argument('--window-r-outer', type=float, default=.9, help='Real space outer windowing radius for cosine falloff to radius 1')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD)')
    group.add_argument('--sequential-tilt-sampling', action='store_true', help='Supply particle images of one particle to encoder in filtered starfile order')

    group = _parser.add_argument_group('Weighting and masking')
    group.add_argument('--recon-tilt-weight', action='store_true', help='Weight reconstruction loss by cosine(tilt_angle)')
    group.add_argument('--recon-dose-weight', action='store_true', help='Weight reconstruction loss per tilt per pixel by dose dependent amplitude attenuation')
    group.add_argument('--l-dose-mask', action='store_true', help='Do not train on frequencies exposed to > 2.5x critical dose. Training lattice is intersection of this with --l-extent')

    group = _parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs')
    group.add_argument('-b', '--batch-size', type=int, default=1, help='Minibatch size')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer')
    group.add_argument('--lr', type=float, default=0.0002, help='Learning rate in Adam optimizer. Should co-scale linearly with batch size.')
    group.add_argument('--beta', default=None, help='Choice of beta schedule or a constant for KLD weight')
    group.add_argument('--beta-control', type=float, help='KL-Controlled VAE gamma. Beta is KL target')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: 0, std of dataset)')
    group.add_argument('--no-amp', action='store_true', help='Disable use of mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')

    group = _parser.add_argument_group('Encoder Network')
    group.add_argument('--enc-layers-A', dest='qlayersA', type=int, default=3, help='Number of hidden layers for each tilt')
    group.add_argument('--enc-dim-A', dest='qdimA', type=int, default=256, help='Number of nodes in hidden layers for each tilt')
    group.add_argument('--out-dim-A', type=int, default=128, help='Number of nodes in output layer of encA == ntilts * number of nodes input to encB')
    group.add_argument('--enc-layers-B', dest='qlayersB', type=int, default=3, help='Number of hidden layers encoding merged tilts')
    group.add_argument('--enc-dim-B', dest='qdimB', type=int, default=512, help='Number of nodes in hidden layers encoding merged tilts')
    group.add_argument('--enc-mask', type=int, help='Diameter of circular mask of image for encoder in pixels (default: boxsize+1 to use up to Nyquist; -1 for no mask)')
    group.add_argument('--pooling-function', type=str, choices=('concatenate', 'max', 'mean', 'median', 'set_encoder'), default='concatenate',
                       help='Function used to pool features along ntilts dimension after encA')
    group.add_argument('--num-seeds', type=int, default=1, help='number of seeds for PMA')
    group.add_argument('--num-heads', type=int, default=4, help='number of heads for multi head attention blocks')
    group.add_argument('--layer-norm', action='store_true', help='whether to apply layer normalization in the set transformer block')

    group = _parser.add_argument_group('Decoder Network')
    group.add_argument('--dec-layers', dest='players', type=int, default=3, help='Number of hidden layers')
    group.add_argument('--dec-dim', dest='pdim', type=int, default=256, help='Number of nodes in hidden layers')
    group.add_argument('--l-extent', type=float, default=0.5, help='Coordinate lattice size (if not using positional encoding) (default: %(default)s)')
    group.add_argument('--pe-type', choices=('geom_ft', 'geom_full', 'geom_lowf', 'geom_nohighf', 'linear_lowf', 'gaussian', 'none'), default='gaussian', help='Type of positional encoding')
    group.add_argument('--feat-sigma', type=float, default=0.5, help="Scale for random Gaussian features")
    group.add_argument('--pe-dim', type=int, help='Num features in positional encoding')
    group.add_argument('--activation', choices=('relu', 'leaky_relu'), default='relu', help='Activation')

    group = _parser.add_argument_group('Dataloader arguments')
    group.add_argument('--num-workers', type=int, default=0, help='Number of workers to use when batching particles for training. Has moderate impact on epoch time')
    group.add_argument('--prefetch-factor', type=int, default=None, help='Number of particles to prefetch per worker for training. Has moderate impact on epoch time')
    group.add_argument('--persistent-workers', action='store_true', help='Whether to persist workers after dataset has been fully consumed. Has minimal impact on run time')
    group.add_argument('--pin-memory', action='store_true', help='Whether to use pinned memory for dataloader. Has large impact on epoch time. Recommended.')
    return _parser


def train_batch(*,
                model: TiltSeriesHetOnlyVAE | DataParallelPassthrough,
                scaler: GradScaler,
                optim: torch.optim.Optimizer,
                lat: Lattice,
                batch_images: torch.Tensor,
                batch_rots: torch.Tensor,
                batch_trans: torch.Tensor,
                batch_ctf_params: torch.Tensor,
                batch_recon_error_weights: torch.Tensor,
                batch_hartley_2d_mask: torch.Tensor,
                beta: float,
                beta_control: float | None = None,
                use_amp: bool = False) -> np.ndarray:
    """
    Train a TiltSeriesHetOnlyVAE model on a batch of tilt series particle images.
    :param model: TiltSeriesHetOnlyVAE object to be trained
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
    :param beta: scaling factor to apply to KLD during loss calculation.
    :param beta_control: KL-Controlled VAE gamma. Beta is KL target.
    :param use_amp: If true, use Automatic Mixed Precision to reduce memory consumption and accelerate code execution via `autocast` and `GradScaler`
    :return: numpy array of losses: total loss, generative loss between reconstructed slices and input images, and kld loss between latent embeddings and standard normal
    """
    # prepare to train a new batch
    optim.zero_grad()
    model.train()

    # autocast auto-enabled and set to correct device
    with autocast(enabled=use_amp):
        # center images via translation and phase flip for partial CTF correction
        batch_images_preprocessed, batch_ctf_weights = preprocess_batch(lat=lat,
                                                                        batch_images=batch_images,
                                                                        batch_trans=batch_trans,
                                                                        batch_ctf_params=batch_ctf_params)
        # encode the translated and CTF-phase-flipped images
        z_mu, z_logvar, z = encode_batch(model=model,
                                         batch_images=batch_images_preprocessed)
        # decode the lattice coordinate positions given the encoder-generated embeddings
        batch_images_recon = decode_batch(model=model,
                                          lat=lat,
                                          batch_rots=batch_rots,
                                          batch_hartley_2d_mask=batch_hartley_2d_mask,
                                          z=z)
        # calculate the model loss
        loss, gen_loss, kld_loss = loss_function(z_mu=z_mu,
                                                 z_logvar=z_logvar,
                                                 batch_images=batch_images_preprocessed,
                                                 batch_images_recon=batch_images_recon,
                                                 batch_ctf_weights=batch_ctf_weights,
                                                 batch_recon_error_weights=batch_recon_error_weights,
                                                 batch_hartley_2d_mask=batch_hartley_2d_mask,
                                                 beta=beta,
                                                 beta_control=beta_control)

    # backpropogate the scaled loss and optimize model weights
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    return torch.tensor((loss, gen_loss, kld_loss)).detach().cpu().numpy()


def preprocess_batch(*,
                     lat: Lattice,
                     batch_images: torch.Tensor,
                     batch_trans: torch.Tensor,
                     batch_ctf_params: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """
    Center images via translation and phase flip for partial CTF correction, as needed
    :param lat: Hartley-transform lattice of points for voxel grid operations
    :param batch_images: Batch of images to be used for training, shape (batchsize, ntilts, boxsize_ht, boxsize_ht)
    :param batch_trans: Batch of 2-D translation matrices corresponding to `batch_images` known poses, shape (batchsize, ntilts, 2).
            May be `torch.zeros((batchsize))` instead to indicate no translations should be applied to the input images.
    :param batch_ctf_params: Batch of CTF parameters corresponding to `batch_images` known CTF parameters, shape (batchsize, ntilts, 9).
            May be `torch.zeros((batchsize))` instead to indicate no CTF corruption should be applied to the reconstructed slice.
    :return batch_images: translationally-centered and phase-flipped batch of images to be used for training, shape (batchsize, ntilts, boxsize_ht**2)
    :return batch_ctf_weights: CTF evaluated at each spatial frequency corresponding to input images, shape (batchsize, ntilts, boxsize_ht**2) or None if no CTF should be applied
    """

    # get key dimension sizes
    batchsize, ntilts, boxsize_ht, boxsize_ht = batch_images.shape

    # translate the image
    if not torch.all(batch_trans == torch.zeros(*batch_trans.shape, device=batch_trans.device)):
        batch_images = lat.translate_ht(batch_images.view(batchsize * ntilts, -1), batch_trans.view(batchsize * ntilts, 1, 2))

    # restore separation of batch dim and tilt dim (merged into batch dim during translation)
    batch_images = batch_images.view(batchsize, ntilts, boxsize_ht * boxsize_ht)

    # phase flip the input CTF-corrupted image and calculate CTF weights to apply later
    if not torch.all(batch_ctf_params == torch.zeros(*batch_ctf_params.shape, device=batch_ctf_params.device)):
        batch_ctf_weights = ctf.compute_ctf(lat, *torch.split(batch_ctf_params[:, :, 1:], 1, 2))
        batch_images = batch_images * batch_ctf_weights.sign()  # phase flip by CTF to be all positive amplitudes
    else:
        batch_ctf_weights = None

    # TODO weight y (inplace) by cumulative weights (unmasked) ? done in commit df7aa8

    return batch_images, batch_ctf_weights


def encode_batch(*,
                 model: TiltSeriesHetOnlyVAE | DataParallelPassthrough,
                 batch_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode a batch of particles represented by multiple images to per-particle latent embeddings
    :param model: TiltSeriesHetOnlyVAE object to be trained
    :param batch_images: Batch of images to be used for training, shape (batchsize, ntilts, boxsize_ht*2)
    :return: z_mu: Direct output of encoder module parameterizing the mean of the latent embedding for each particle, shape (batchsize, zdim)
    :return: z_logvar: Direct output of encoder module parameterizing the log variance of the latent embedding for each particle, shape (batchsize, zdim)
    :return: z: Resampling of the latent embedding for each particle parameterized as a gaussian with mean `z_mu` and variance `z_logvar`, shape (batchsize, zdim)
    """
    z_mu, z_logvar = model.encode(batch_images)  # ouput is B x zdim, i.e. one value per ptcl (not per img)
    z = model.encoder.reparameterize(z_mu, z_logvar)

    return z_mu, z_logvar, z


def decode_batch(*,
                 model: TiltSeriesHetOnlyVAE | DataParallelPassthrough,
                 lat: Lattice,
                 batch_rots: torch.Tensor,
                 batch_hartley_2d_mask: torch.Tensor,
                 z: torch.Tensor) -> torch.Tensor:
    """
    Decode a batch of particles represented by multiple images from per-particle latent embeddings and corresponding lattice positions to evaluate
    :param model: TiltSeriesHetOnlyVAE object to be trained
    :param lat: Hartley-transform lattice of points for voxel grid operations
    :param batch_rots: Batch of 3-D rotation matrices corresponding to `batch_images` known poses, shape (batchsize, ntilts, 3, 3)
    :param batch_hartley_2d_mask: Batch of 2-D masks to be applied per-spatial-frequency, shape (batchsize, ntilts, boxsize_ht**2)
            Calculated as the intersection of critical dose exposure curves and a Nyquist-limited circular mask in reciprocal space, including masking the DC component.
    :param z: Resampled latent embedding for each particle, shape (batchsize, zdim)
    :return: Reconstructed central slices of Fourier space volumes corresponding to each particle in the batch, shape (batchsize * ntilts * boxsize_ht**2 [`batch_hartley_2d_mask`]).
            Note that the returned array is completely flattened (including along batch dimension) due to potential for uneven/ragged reconstructed image tensors per-particle after masking
    """

    # prepare lattice at rotated fourier components
    input_coords = lat.coords @ batch_rots  # shape batchsize x ntilts x boxsize_ht**2 x 3 from [boxsize_ht**2 x 3] @ [batchsize x ntilts x 3 x 3]

    # filter by dec_mask to skip decoding coordinates that have low contribution to SNR
    # decode each particle one at a time due to variable # pixels in batch_hartley_2d_mask per particle producing ragged tensor
    # this will eventually be replaceable with pytorch NestedTensor, but as of torch v2.4 this still does not work at backprop (and does not support many common operations we need prior to that)
    batch_images_recon = torch.cat([model.decode(coords=coords_ptcl[mask_ptcl].unsqueeze(0), z=z_ptcl.unsqueeze(0)).squeeze(0)
                                    for coords_ptcl, mask_ptcl, z_ptcl in zip(input_coords, batch_hartley_2d_mask, z, strict=True)], dim=0)

    return batch_images_recon


def loss_function(*,
                  z_mu: torch.Tensor,
                  z_logvar: torch.Tensor,
                  batch_images: torch.Tensor,
                  batch_images_recon: torch.Tensor,
                  batch_ctf_weights: torch.Tensor,
                  batch_recon_error_weights: torch.Tensor,
                  batch_hartley_2d_mask: torch.Tensor,
                  beta: float,
                  beta_control: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate generative loss between reconstructed and input images, and beta-weighted KLD between latent embeddings and standard normal
    :param z_mu: Direct output of encoder module parameterizing the mean of the latent embedding for each particle, shape (batchsize, zdim)
    :param z_logvar: Direct output of encoder module parameterizing the log variance of the latent embedding for each particle, shape (batchsize, zdim)
    :param batch_images: Batch of images to be used for training, shape (batchsize, ntilts, boxsize_ht**2)
    :param batch_images_recon: Reconstructed central slices of Fourier space volumes corresponding to each particle in the batch, shape (batchsize * ntilts * boxsize_ht**2 [`batch_hartley_2d_mask`])
    :param batch_ctf_weights: CTF evaluated at each spatial frequency corresponding to input images, shape (batchsize, ntilts, boxsize_ht**2) or None if no CTF should be applied
    :param batch_recon_error_weights: Batch of 2-D weights to be applied to the per-spatial-frequency error between each reconstructed slice and input image, shape (batchsize, ntilts, boxsize_ht**2).
            Calculated from critical dose exposure curves and electron beam vs sample tilt geometry.
    :param batch_hartley_2d_mask: Batch of 2-D masks to be applied per-spatial-frequency.
            Calculated as the intersection of critical dose exposure curves and a Nyquist-limited circular mask in reciprocal space, including masking the DC component.
    :param beta: scaling factor to apply to KLD during loss calculation.
    :param beta_control: KL-Controlled VAE gamma. Beta is KL target.
    :return: total summed loss, generative loss between reconstructed slices and input images, and beta-weighted kld loss between latent embeddings and standard normal
    """
    # reconstruction error
    batch_images = batch_images[batch_hartley_2d_mask].view(-1)
    batch_recon_error_weights = batch_recon_error_weights[batch_hartley_2d_mask].view(-1)
    if batch_ctf_weights is not None:
        batch_ctf_weights = batch_ctf_weights[batch_hartley_2d_mask].view(-1)
        batch_images_recon = batch_images_recon * batch_ctf_weights  # apply CTF to reconstructed image
        batch_images = batch_images * batch_ctf_weights.sign()  # undo phase flipping in place from preprocess_batch
    gen_loss = torch.mean(batch_recon_error_weights * ((batch_images_recon - batch_images) ** 2))

    # latent loss
    kld = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0)
    if beta_control is None:
        kld_loss = beta * kld / torch.sum(batch_hartley_2d_mask, dtype=kld.dtype)
    else:
        kld_loss = beta_control * (beta - kld) ** 2 / torch.sum(batch_hartley_2d_mask, dtype=kld.dtype)

    # total loss
    loss = gen_loss + kld_loss
    return loss, gen_loss, kld_loss


def encoder_inference(*,
                      model: TiltSeriesHetOnlyVAE | DataParallelPassthrough,
                      lat: Lattice,
                      data: TiltSeriesMRCData,
                      use_amp: bool = False,
                      batchsize: int = 1,
                      **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the encoder module using the specified data as input to be embedded in latent space.
    :param model: TiltSeriesHetOnlyVAE object to be used for encoder module inference. Informs device on which to run inference.
    :param lat: Hartley-transform lattice of points for voxel grid operations
    :param data: TiltSeriesMRCData object for accessing tilt images with known CTF and pose parameters, to be embedded in latent space
    :param use_amp: If true, use Automatic Mixed Precision to reduce memory consumption and accelerate code execution via `torch.autocast`
    :param batchsize: batch size used in DataLoader for model inference
    :param kwargs: All other key word arguments are passed to `torch.DataLoader`
    :return: Direct output of encoder module parameterizing the mean of the latent embedding for each particle, shape (batchsize, zdim).
            Direct output of encoder module parameterizing the log variance of the latent embedding for each particle, shape (batchsize, zdim)
    """

    # prepare to run model inference
    model.eval()
    assert not model.training
    with torch.inference_mode():

        # autocast auto-enabled and set to correct device
        with autocast(enabled=use_amp):

            # pre-allocate tensors to store outputs
            z_mu_all = torch.zeros((data.nptcls, model.zdim),
                                   device=lat.device,
                                   dtype=torch.half if use_amp else torch.float)
            z_logvar_all = torch.zeros_like(z_mu_all)

            # create the dataloader iterator with shuffle False to preserve index alignment between input dataset and output latent
            data_generator = DataLoader(data,
                                        batch_size=batchsize,
                                        shuffle=False,
                                        **kwargs)

            for batch_images, _, batch_trans, batch_ctf_params, _, _, batch_indices in data_generator:
                # transfer to GPU
                batch_images = batch_images.to(lat.device)
                batch_ctf_params = batch_ctf_params.to(lat.device)
                batch_trans = batch_trans.to(lat.device)

                # encode the translated and CTF-phase-flipped images
                batch_images, ctf_weights = preprocess_batch(lat=lat,
                                                             batch_images=batch_images,
                                                             batch_trans=batch_trans,
                                                             batch_ctf_params=batch_ctf_params)

                # decode the lattice coordinate positions given the encoder-generated embeddings
                z_mu, z_logvar, z = encode_batch(model=model,
                                                 batch_images=batch_images)

                # store latent embeddings in master array
                z_mu_all[batch_indices] = z_mu
                z_logvar_all[batch_indices] = z_logvar

            # when dataset is fully exhausted, move latent embeddings to cpu-based numpy arrays
            z_mu_all = z_mu_all.cpu().numpy()
            z_logvar_all = z_logvar_all.cpu().numpy()

            # sanity check that no latent embeddings are NaN or inf, which could happen with AMP enabled
            if np.any(z_mu_all == np.nan) or np.any(z_mu_all == np.inf):
                nan_count = np.sum(np.isnan(z_mu_all))
                inf_count = np.sum(np.isinf(z_mu_all))
                sys.exit(f'Latent evaluation at end of epoch failed: z.pkl would contain {nan_count} NaN and {inf_count} Inf')

            return z_mu_all, z_logvar_all


def save_checkpoint(*,
                    model: TiltSeriesHetOnlyVAE | DataParallelPassthrough,
                    scaler: GradScaler,
                    optim: torch.optim.Optimizer,
                    epoch: int,
                    z_mu_train: np.ndarray,
                    z_logvar_train: np.ndarray,
                    out_weights: str,
                    out_z_train: str,
                    z_mu_test: np.ndarray = None,
                    z_logvar_test: np.ndarray = None,
                    out_z_test: str = None) -> None:
    """
    Save model weights and latent encoding z
    :param model: TiltSeriesHetOnlyVAE object used for model training and evaluation
    :param scaler: GradScaler object used for scaling loss involving fp16 tensors to avoid over/underflow
    :param optim: torch.optim.Optimizer object used for optimizing the model
    :param epoch: epoch count at which checkpoint is being saved
    :param z_mu_train: array of latent embedding means for the dataset train split
    :param z_logvar_train: array of latent embedding log variances for the dataset train split
    :param out_weights: name of output file to save model, optimizer, and scaler state dicts
    :param out_z_train: name of output file to save latent embeddings for dataset train split
    :param z_mu_test: array of latent embedding log variances for the dataset test split
    :param z_logvar_test: array of latent embedding log variances for the dataset test split
    :param out_z_test: name of output file to save latent embeddings for dataset test split
    :return: None
    """

    # save model weights
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None
    }, out_weights)

    # save latent embeddings from dataset train split
    with open(out_z_train, 'wb') as f:
        pickle.dump(z_mu_train.astype(np.float32), f)
        pickle.dump(z_logvar_train.astype(np.float32), f)

    # save latent embeddings from dataset test split
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
    """
    Select indices and generate corresponding test/train volumes for convergence metrics
    :param z_train: numpy array of shape (n_particles, zdim) of latent values deriving from train particle images
    :param z_test: numpy array of shape (n_particles) zdim) of latent values deriving from test particle images
    :param epoch:
    :param workdir: str, absolute path to model workdir to store generated volumes
    :param weights_path:
    :param config_path:
    :param volume_count: int, number of volumes to generate for each of train/test
    """
    # sample volume_count particles
    n_particles = z_train.shape[0]
    ind_sel = np.sort(np.random.choice(n_particles, size=volume_count))

    # generate the train and test volumes for corresponding particles
    vg = VolumeGenerator(weights_path=weights_path, config_path=config_path)
    os.mkdir(f'{workdir}/scratch.{epoch}.train')
    os.mkdir(f'{workdir}/scratch.{epoch}.test')
    vg.gen_volumes(z_values=z_train[ind_sel], outdir=f'{workdir}/scratch.{epoch}.train', )
    vg.gen_volumes(z_values=z_test[ind_sel], outdir=f'{workdir}/scratch.{epoch}.test', )

    # save data
    utils.save_pkl(ind_sel, f'{workdir}/convergence_volumes_sel.{epoch}.pkl')


def convergence_volumes_testtrain_correlation(workdir, epoch, volume_count=100):
    """
    Calculate the correlation between volumes generated from test and train latent embeddings of the same particles in the same epoch
    :param workdir: str, absolute path to model workdir
    :param epoch: int, current epoch being evaluated
    :param volume_count: int, number of volumes to generate for each of train/test
    """
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


def convergence_volumes_testtest_scale(workdir, epoch, volume_count=100):
    """
    Calculate the scale of heterogeneity among all volumes generated from test latent embeddings of the same particles in the same epoch
    :param workdir: str, absolute path to model workdir
    :param epoch: int, current epoch being evaluated
    :param volume_count: int, number of volumes to generate for each of train/test
    """
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
        pairwise_ccs[i, j] = calc_cc(vol_train, vol_test)

    # calculate pairwise volume distances as 1 - CC
    pairwise_cc_dist = np.ones_like(pairwise_ccs) - pairwise_ccs

    # save data and log summary value
    utils.save_pkl(pairwise_cc_dist, f'{workdir}/convergence_volumes_testtest_scale.{epoch}.pkl')
    log(f'Convergence epoch {epoch}: sum of complement-CC distances for all pairwise test/test volumes: {np.sum(pairwise_cc_dist[ind_triu])}')


def main(args):
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    logfile = f'{args.outdir}/run.log'

    def flog(msg):  # HACK: switch to logging module
        return utils.flog(msg, logfile)

    if args.load == 'latest':
        args = utils.get_latest(args=args)
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
    ptcls_star = TiltSeriesStarfile(args.particles,
                                    source_software=args.source_software)
    ptcls_star.plot_particle_uid_ntilt_distribution(outpath=f'{args.outdir}/{os.path.basename(ptcls_star.sourcefile)}_particle_uid_ntilt_distribution.png')

    # filter star file
    ptcls_star.filter(ind_imgs=args.ind_imgs,
                      ind_ptcls=args.ind_ptcls,
                      sort_ptcl_imgs=args.sort_ptcl_imgs,
                      use_first_ntilts=args.use_first_ntilts,
                      use_first_nptcls=args.use_first_nptcls)

    # split ptcls_star by into half sets per-particle for independent backprojection
    ptcls_star.make_test_train_split(fraction_split1=args.fraction_train,
                                     show_summary_stats=True)

    # save filtered star file for future convenience (aligning latent embeddings with particles, re-extracting particles, mapbacks, etc.)
    outstar = f'{args.outdir}/{os.path.splitext(os.path.basename(ptcls_star.sourcefile))[0]}_tomodrgn_preprocessed.star'
    ptcls_star.sourcefile_filtered = outstar
    ptcls_star.write(outstar)

    # load the particles + poses + ctf from input starfile
    flog(f'Loading dataset from {args.particles}')
    data_train = TiltSeriesMRCData(ptcls_star=ptcls_star,
                                   star_random_subset=1,
                                   datadir=args.datadir,
                                   lazy=args.lazy,
                                   norm=args.norm,
                                   invert_data=args.invert_data,
                                   window=args.window,
                                   window_r=args.window_r,
                                   window_r_outer=args.window_r_outer,
                                   recon_dose_weight=args.recon_dose_weight,
                                   recon_tilt_weight=args.recon_tilt_weight,
                                   l_dose_mask=args.l_dose_mask,
                                   constant_mintilt_sampling=True,
                                   sequential_tilt_sampling=args.sequential_tilt_sampling)
    if args.fraction_train < 1:
        data_test = TiltSeriesMRCData(ptcls_star,
                                      star_random_subset=2,
                                      datadir=args.datadir,
                                      lazy=args.lazy,
                                      norm=data_train.norm,
                                      invert_data=args.invert_data,
                                      window=args.window,
                                      window_r=args.window_r,
                                      window_r_outer=args.window_r_outer,
                                      recon_dose_weight=args.recon_dose_weight,
                                      recon_tilt_weight=args.recon_tilt_weight,
                                      l_dose_mask=args.l_dose_mask,
                                      constant_mintilt_sampling=True,
                                      sequential_tilt_sampling=args.sequential_tilt_sampling)
    else:
        data_test = None
    boxsize_ht = data_train.boxsize_ht
    nptcls = data_train.nptcls

    # instantiate lattice
    lat = Lattice(boxsize_ht, extent=args.l_extent, device=device)

    # determine which pixels to encode (equivalently applicable to all particles)
    if args.enc_mask is None:
        args.enc_mask = boxsize_ht
    if args.enc_mask > 0:
        # encode pixels within defined circular radius in fourier space
        assert args.enc_mask <= boxsize_ht
        enc_mask = lat.get_circular_mask(args.enc_mask)
        in_dim = enc_mask.sum()
    elif args.enc_mask == -1:
        # encode all pixels in fourier space
        enc_mask = None
        in_dim = lat.boxsize ** 2
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
            flog(f'Will sample {data_train.ntilts_training} tilts per particle for train split, to pass to encoder, due to model requirements of --pooling-function {args.pooling_function}')
            if args.fraction_train < 1:
                data_test.ntilts_training = None
                flog(f'Will sample {data_test.ntilts_training} tilts per particle for test split, to pass to encoder, due to model requirements of --pooling-function {args.pooling_function}')
        else:
            # TODO implement ntilt_training calculation for set-style encoder with batchsize > 1
            raise NotImplementedError
    elif args.pooling_function in ['concatenate', 'set_encoder']:
        # requires same number of tilts for both test and train for every particle
        # therefore add further subset train/test to sample same number of tilts from each for each particle
        if args.fraction_train < 1:
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
    model = TiltSeriesHetOnlyVAE(in_dim=in_dim,
                                 hidden_layers_a=args.qlayersA,
                                 hidden_dim_a=args.qdimA,
                                 out_dim_a=args.out_dim_A,
                                 ntilts=data_train.ntilts_training,
                                 hidden_layers_b=args.qlayersB,
                                 hidden_dim_b=args.qdimB,
                                 zdim=args.zdim,
                                 hidden_layers_decoder=args.players,
                                 hidden_dim_decoder=args.pdim,
                                 lat=lat,
                                 activation=activation,
                                 enc_mask=enc_mask,
                                 pooling_function=args.pooling_function,
                                 feat_sigma=args.feat_sigma,
                                 num_seeds=args.num_seeds,
                                 num_heads=args.num_heads,
                                 layer_norm=args.layer_norm,
                                 pe_type=args.pe_type,
                                 pe_dim=args.pe_dim, )
    print_tiltserieshetonlyvae_ascii(model)
    # model.print_model_info()

    # save configuration
    out_config = f'{args.outdir}/config.pkl'
    config.save_config(args=args,
                       star=ptcls_star,
                       data=data_train,
                       lat=lat,
                       model=model,
                       out_config=out_config)

    # set beta schedule
    if args.beta is None:
        args.beta = 1. / args.zdim
    beta_schedule = get_beta_schedule(args.beta, n_iterations=args.num_epochs * nptcls + args.batch_size)

    # instantiate optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-4)  # https://github.com/pytorch/pytorch/issues/40497#issuecomment-1084807134

    # Mixed precision training with AMP
    use_amp = not args.no_amp
    flog(f'AMP acceleration enabled (autocast + gradscaler) : {use_amp}')
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        if not args.batch_size % 8 == 0:
            flog('Warning: recommended to have batch size divisible by 8 for AMP training')
        if not (boxsize_ht - 1) % 8 == 0:
            flog('Warning: recommended to have image size divisible by 8 for AMP training')
        if in_dim % 8 != 0:
            flog('Warning: recommended to have masked image dimensions divisible by 8 for AMP training')
        assert args.qdimA % 8 == 0, "Encoder hidden layer dimension must be divisible by 8 for AMP training"
        assert args.qdimB % 8 == 0, "Encoder hidden layer dimension must be divisible by 8 for AMP training"
        if args.zdim % 8 != 0:
            flog('Warning: recommended to have z dimension divisible by 8 for AMP training')
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
    data_train_generator = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                      persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
    for epoch in range(start_epoch, args.num_epochs):
        t2 = dt.now()
        losses_accum = np.zeros(3, dtype=np.float32)
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
                                       lat=lat,
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
                log(f'# [Train Epoch: {epoch + 1}/{args.num_epochs}] [{batch_it}/{nptcls} subtomos] '
                    f'gen loss={losses_batch[1]:.6f}, '
                    f'kld={losses_batch[2]:.6f}, beta={beta:.6f}, '
                    f'loss={losses_batch[0]:.6f}')
            losses_accum += losses_batch * len(batch_images)
        flog(
            f'# =====> Epoch: {epoch + 1} '
            f'Average gen loss = {losses_accum[1] / batch_it:.6f}, '
            f'KLD = {losses_accum[2] / batch_it:.6f}, '
            f'total loss = {losses_accum[0] / batch_it:.6f}; '
            f'Finished in {dt.now() - t2}')
        if args.checkpoint and (epoch + 1) % args.checkpoint == 0:
            if device.type != 'cpu':
                flog(f'GPU memory usage: {utils.check_memory_usage()}')
            out_weights = f'{args.outdir}/weights.{epoch}.pkl'
            out_z_train = f'{args.outdir}/z.{epoch}.train.pkl'
            z_mu_train, z_logvar_train = encoder_inference(model=model,
                                                           lat=lat,
                                                           data=data_train,
                                                           use_amp=use_amp,
                                                           batchsize=args.batch_size)
            if data_test:
                out_z_test = f'{args.outdir}/z.{epoch}.test.pkl'
                z_mu_test, z_logvar_test = encoder_inference(model=model,
                                                             lat=lat,
                                                             data=data_test,
                                                             use_amp=use_amp,
                                                             batchsize=args.batch_size)
            else:
                z_mu_test, z_logvar_test, out_z_test = None, None, None
            save_checkpoint(model=model,
                            scaler=scaler,
                            optim=optim,
                            epoch=epoch,
                            z_mu_train=z_mu_train,
                            z_logvar_train=z_logvar_train,
                            out_weights=out_weights,
                            out_z_train=out_z_train,
                            z_mu_test=z_mu_test,
                            z_logvar_test=z_logvar_test,
                            out_z_test=out_z_test)
            if data_test:
                flog('Calculating convergence metrics using test/train split...')
                convergence_latent(z_mu_train, z_logvar_train, z_mu_test, z_logvar_test, args.outdir, epoch)
                convergence_volumes_generate(z_mu_train, z_mu_test, epoch, args.outdir, f'{args.outdir}/weights.{epoch}.pkl', f'{args.outdir}/config.pkl', volume_count=100)
                convergence_volumes_testtrain_correlation(args.outdir, epoch, volume_count=100)
                convergence_volumes_testtest_scale(args.outdir, epoch, volume_count=100)

    # save model weights, latent encoding, and evaluate the model on 3D lattice
    out_weights = f'{args.outdir}/weights.pkl'
    out_z_train = f'{args.outdir}/z.train.pkl'
    z_mu_train, z_logvar_train = encoder_inference(model=model,
                                                   lat=lat,
                                                   data=data_train,
                                                   use_amp=use_amp,
                                                   batchsize=args.batch_size)
    if data_test:
        out_z_test = f'{args.outdir}/z.test.pkl'
        z_mu_test, z_logvar_test = encoder_inference(model=model,
                                                     lat=lat,
                                                     data=data_test,
                                                     use_amp=use_amp,
                                                     batchsize=args.batch_size)
    else:
        z_mu_test, z_logvar_test, out_z_test = None, None, None
    save_checkpoint(model=model,
                    scaler=scaler,
                    optim=optim,
                    epoch=epoch,
                    z_mu_train=z_mu_train,
                    z_logvar_train=z_logvar_train,
                    out_weights=out_weights,
                    out_z_train=out_z_train,
                    z_mu_test=z_mu_test,
                    z_logvar_test=z_logvar_test,
                    out_z_test=out_z_test)
    td = dt.now() - t1
    flog(f'Finished in {td} ({td / (args.num_epochs - start_epoch)} per epoch)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
