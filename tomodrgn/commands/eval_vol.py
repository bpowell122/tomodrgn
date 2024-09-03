"""
Evaluate a trained FTPositionalDecoder model, optionally conditioned on values of latent embedding z.
"""
import numpy as np
import os
import argparse
from datetime import datetime as dt
import pprint
import itertools
import multiprocessing as mp
import multiprocessing.pool

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast

from tomodrgn import mrc, utils, fft
from tomodrgn.models import TiltSeriesHetOnlyVAE, FTPositionalDecoder

log = utils.log
vlog = utils.vlog


def add_args(_parser):
    _parser.add_argument('-w', '--weights', help='Model weights from train_vae')
    _parser.add_argument('-c', '--config', required=True, help='config.pkl file from train_vae')
    _parser.add_argument('-o', type=os.path.abspath, required=True, help='Output .mrc or directory')
    _parser.add_argument('--prefix', default='vol_', help='Prefix when writing out multiple .mrc files')
    _parser.add_argument('--no-amp', action='store_true', help='Disable use of mixed-precision training')
    _parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size to parallelize volume generation (32-64 works well for box64 volumes)')
    _parser.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs. Specify GPUs i,j via `export CUDA_VISIBLE_DEVICES=i,j`')
    _parser.add_argument('-v', '--verbose', action='store_true', help='Increases verbosity')

    group = _parser.add_argument_group('Specify z values')
    group.add_argument('-z', type=np.float32, nargs='*', help='Specify one z-value')
    group.add_argument('--z-start', type=np.float32, nargs='*', help='Specify a starting z-value')
    group.add_argument('--z-end', type=np.float32, nargs='*', help='Specify an ending z-value')
    group.add_argument('-n', type=int, default=10, help='Number of structures between [z_start, z_end]')
    group.add_argument('--zfile', type=os.path.abspath, help='Text/.pkl file with z-values to evaluate')

    group = _parser.add_argument_group('Volume arguments')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volume')
    group.add_argument('--invert', action='store_true', help='Invert contrast of output volume')
    group.add_argument('--downsample', type=int, help='Downsample volumes to this box size (pixels)')
    group.add_argument('--lowpass', type=float, default=None, help='Lowpass filter to this resolution in Å. Requires settings --Apix.')

    return _parser


def check_z_inputs(args):
    if args.z_start:
        assert args.z_end, "Must provide --z-end with argument --z-start"


class ZDataset(Dataset):
    """
    Dataset to allow easy pytorch batch-parallelized passing of latent embeddings to the decoder, particularly when using nn.DataParallel.
    """

    def __init__(self,
                 z: np.ndarray):
        """
        Initialize the (latent embedding) Dataset.

        :param z: array of latent embeddings to decode, shape (nptcls, zdim)
        """
        self.z = z

    def __len__(self) -> int:
        return self.z.shape[0]

    def __getitem__(self,
                    index: int) -> np.ndarray:
        return self.z[index]


class DummyModel(nn.Module):
    """
    Helper class to call the correct eval_volume_batch method regardless of what model class is being evaluated.
    """

    def __init__(self, model, zdim=None):
        super().__init__()
        if zdim:
            # multiple volumes to be evaluated, can batch-parallelize volume decoding and writing to disk
            self.decoder_fn = model.decoder.eval_volume_batch
        else:
            # single volume to be evaluated
            self.decoder_fn = model.eval_volume_batch

    def forward(self, *args, **kwargs):
        return self.decoder_fn(*args, **kwargs)


def postprocess_vols(batch_vols: torch.Tensor,
                     norm: tuple[float, float],
                     iht_downsample_scaling_correction: float,
                     lowpass_mask: np.ndarray | None = None,
                     flip: bool = False,
                     invert: bool = False) -> np.ndarray:
    """
    Apply post-volume-decoding processing steps: downsampling scaling correction, lowpass filtering, inverse fourier transform, volume handedness flipping, volume data sign inversion.

    :param batch_vols: batch of fourier space non-symmetrized volumes directly from eval_vol_batch, shape (nvols, boxsize, boxsize, boxsize)
    :param norm: tuple of floats representing mean and standard deviation of preprocessed particles used during model training
    :param iht_downsample_scaling_correction: a global scaling factor applied when forward and inverse fourier / hartley transforming.
            This is calculated and applied internally by the fft.py module as a function of array shape.
            Thus, when the volume is downsampled, a further correction is required.
    :param lowpass_mask: a binary mask applied to fourier space symmetrized volumes to low pass filter the reconstructions.
            Typically, the same mask is used for all volumes via broadcasting, thus this may be of shape (1, boxsize, boxsize, boxsize) or (nvols, boxsize, boxsize, boxsize).
    :param flip: Whether to invert the volume chirality by flipping the data order along the z axis.
    :param invert: Whether to invert the data light-on-dark vs dark-on-light convention, relative to the reconstruction returned by the decoder module.
    :return: Postprocessed volume batch in real space
    """
    # sanit check inputs
    assert batch_vols.ndim == 4, f'The volume batch must have four dimensions (batch size, boxsize, boxsize, boxsize). Found {batch_vols.shape}'
    if lowpass_mask is not None:
        assert batch_vols.shape[-3:] == lowpass_mask.shape[-3], f'The volume batch must have the same volume dimensions as the lowpass mask. Found {batch_vols.shape}, {lowpass_mask.shape.shape}'

    # convert torch tensor to numpy array for future operations
    batch_vols = batch_vols.cpu().numpy()

    # normalize the volume (mean and standard deviation) by normalization used when training the model
    batch_vols = batch_vols * norm[1] + norm[0]

    # lowpass filter with fourier space mask
    if lowpass_mask is not None:
        batch_vols = batch_vols * lowpass_mask

    # transform to real space and scale values if downsampling was applied
    batch_vols = fft.iht3_center(batch_vols)
    batch_vols *= iht_downsample_scaling_correction

    if flip:
        batch_vols = np.flip(batch_vols, 1)

    if invert:
        batch_vols *= -1

    return batch_vols


def main(args):
    # check inputs
    t1 = dt.now()
    check_z_inputs(args)

    # set the device
    device = utils.get_default_device()
    if device == torch.device('cpu'):
        args.no_amp = True
        log('Warning: pytorch AMP does not support non-CUDA (e.g. cpu) devices. Automatically disabling AMP and continuing')
        # https://github.com/pytorch/pytorch/issues/55374

    # load the model configuration
    log(args)
    cfg = utils.load_pkl(args.config)
    log('Loaded configuration:')
    pprint.pprint(cfg)

    # set parameters for volume generation
    boxsize_ht = cfg['lattice_args']['boxsize']  # image size + 1
    zdim = cfg['model_args']['zdim'] if 'zdim' in cfg['model_args'].keys() else None
    norm = cfg['dataset_args']['norm']
    angpix = cfg['angpix']
    if args.downsample:
        assert args.downsample % 2 == 0, "Boxsize must be even"
        assert args.downsample <= boxsize_ht - 1, "Must be smaller than original box size"
        angpix = angpix * (boxsize_ht - 1) / args.downsample

    # load the model
    if zdim:
        # load a VAE model
        model, lattice = TiltSeriesHetOnlyVAE.load(config=cfg,
                                                   weights=args.weights,
                                                   device=device)
    else:
        # load a non-VAE decoder-only model
        model, lattice = FTPositionalDecoder.load(config=cfg,
                                                  weights=args.weights,
                                                  device=device)

    # wrap the model in a dummy class whose purpose is to call the correct volume decoding function
    model = DummyModel(model, zdim).to(device)

    # precalculate shared parameters among all volumes
    if args.downsample:
        sym_ht_boxsize_downsampled = args.downsample + 1
        coords = lattice.get_downsample_coords(boxsize_new=sym_ht_boxsize_downsampled).to(device)
        extent = lattice.extent * (args.downsample / (boxsize_ht - 1))
        iht_downsample_scaling_correction = args.downsample ** 3 / (boxsize_ht - 1) ** 3
    else:
        coords = lattice.coords.to(device)
        extent = lattice.extent
        iht_downsample_scaling_correction = 1.
    if args.lowpass is not None:
        lowpass_mask = utils.calc_lowpass_filter_mask(boxsize=boxsize_ht,
                                                      angpix=angpix,
                                                      lowpass=args.lowpass,
                                                      device=None)
        lowpass_mask = lowpass_mask[np.newaxis, ...]
    else:
        lowpass_mask = None

    # set context managers and flags for inference mode
    model.eval()
    with torch.inference_mode():
        use_amp = not args.no_amp
        with autocast(device_type=device.type, enabled=use_amp):

            # prepare z when decoding multiple volumes (at varying z values)
            if args.z_start or args.zfile:

                # can try to parallelize
                if args.multigpu and torch.cuda.device_count() > 1:
                    raise NotImplementedError
                    # TODO implement DistributedDataParallel
                    #  DataParallel requires that all tensors to model forward are split in dim 0 as batch, but eval_volume_batch assumes coords and extent do not have batch dim
                    # log(f'Using {torch.cuda.device_count()} GPUs!')
                    # args.batch_size *= torch.cuda.device_count()
                    # log(f'Increasing batch size to {args.batch_size}')
                    # model = nn.DataParallel(model)
                elif args.multigpu:
                    log(f'WARNING: --multigpu selected, but {torch.cuda.device_count()} GPUs detected')

                # Make output directory
                if not os.path.exists(args.o):
                    os.makedirs(args.o)

                # Get z values
                if args.z_start:
                    # calculate z values to sample between z_start and z_end
                    args.z_start = np.array(args.z_start)
                    args.z_end = np.array(args.z_end)
                    z = np.repeat(np.arange(args.n, dtype=np.float32), zdim).reshape((args.n, zdim))
                    z *= ((args.z_end - args.z_start) / (args.n - 1))
                    z += args.z_start
                else:
                    if args.zfile.endswith('.pkl'):
                        z = utils.load_pkl(args.zfile)
                    else:
                        z = np.loadtxt(args.zfile, dtype=np.float32).reshape(-1, zdim)
                assert z.shape[1] == zdim
                if z.shape[0] < args.batch_size:
                    log(f'Decreasing batchsize to number of volumes: {z.shape[0]}')
                    args.batch_size = z.shape[0]

                # prepare threadpool for parallelized file writing
                pool = mp.pool.ThreadPool(processes=min(os.cpu_count(), args.batch_size))

                # construct dataset and dataloader
                z = ZDataset(z)
                z_iterator = DataLoader(z, batch_size=args.batch_size, shuffle=False)
                log(f'Generating {len(z)} volumes in batches of {args.batch_size}')
                for (i, zz) in enumerate(z_iterator):
                    log(f'    Generating volume batch {i}')
                    if args.verbose:
                        log(zz)

                    zz = zz.to(device)
                    batch_vols = model(coords, zz, extent)
                    batch_vols = postprocess_vols(batch_vols=batch_vols[:, :-1, :-1, :-1],  # exclude symmetrized +k frequency
                                                  norm=norm,
                                                  iht_downsample_scaling_correction=iht_downsample_scaling_correction,
                                                  lowpass_mask=lowpass_mask,
                                                  flip=args.flip,
                                                  invert=args.invert)

                    out_mrcs = [f'{args.o}/{args.prefix}{i * args.batch_size + j:03d}.mrc' for j in range(len(zz))]
                    pool.starmap(func=mrc.write, iterable=zip(out_mrcs,
                                                              batch_vols[:len(out_mrcs)],
                                                              itertools.repeat(None, len(out_mrcs)),
                                                              itertools.repeat(angpix, len(out_mrcs))))
                pool.close()

            # decoding a single volume
            else:
                # take z from args if decoding single z, if no z passed, then decode homogeneous model with z=None
                z = np.array(args.z).reshape(1, -1) if args.z else None
                batch_vols = model(coords, z, extent)
                batch_vols = postprocess_vols(batch_vols=batch_vols[:, :-1, :-1, :-1],  # exclude symmetrized +k frequency
                                              norm=norm,
                                              iht_downsample_scaling_correction=iht_downsample_scaling_correction,
                                              lowpass_mask=lowpass_mask,
                                              flip=args.flip,
                                              invert=args.invert)
                mrc.write(fname=args.o,
                          array=batch_vols[0],  # only one volume in the batch
                          angpix=angpix)

    td = dt.now() - t1
    log(f'Finished in {td}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
