"""
Downsample an image stack or volume by Fourier cropping
"""

import argparse
import numpy as np
import os
import math
import torch
from torch.utils import data
from torch.utils.data import DataLoader

from tomodrgn import utils, mrc, fft, dataset, starfile

log = utils.log


def add_args(_parser):
    _parser.add_argument('input', help='Input particles or volume (.mrc, .mrcs, .star, or .txt)')
    _parser.add_argument('--downsample', type=int, required=True, help='New box size in pixels, must be even')
    _parser.add_argument('--output', metavar='MRCS', type=os.path.abspath, required=True, help='Output projection stack (.mrcs)')
    _parser.add_argument('--batch-size', type=int, default=5000, help='Batch size for processing images')
    _parser.add_argument('--is-vol', action='store_true', help='Flag if input .mrc is a volume')
    _parser.add_argument('--chunk', type=int, help='Chunksize (in # of images) to split particle stack when loading and saving if full stack + downsampled stack too large for system memory')
    _parser.add_argument('--lazy', action='store_true', help='Lazily load each image on the fly if full stack too large for system memory')
    _parser.add_argument('--datadir', help='Optionally provide path to input .mrcs if loading from a .star file')
    _parser.add_argument('--write-tiltseries-starfile', action='store_true', help='If input is a star file, write a downsampled star file')
    return _parser


def mkbasedir(out: str) -> None:
    """
    Create specified output directory, and all required intermediate directories, that do not yet exist.
    :param out: path to output directory or file within desired output directory to create
    :return: None
    """
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out), exist_ok=True)


def warnexists(out: str) -> None:
    """
    Warn if a directory exists.
    :param out: directory to check existence.
    :return: None
    """
    if os.path.exists(out):
        log(f'Warning: {out} already exists. Overwriting.')


def downsample_images(batch_original: torch.Tensor,
                      start: int,
                      stop: int) -> torch.Tensor:
    """
    Downsample a batch of real space images by Fourier cropping.
    :param batch_original: tensor of real space images, shape (batchsize, boxsize_old, boxsize_old)
    :param start: image pixel index at which to begin including Fourier frequencies in Fourier cropped output
    :param stop:  image pixel index at which to stop including Fourier frequencies in Fourier cropped output
    :return: tensor of real space images after downsampling, shape (batchsize, boxsize_new, boxsize_new)
    """
    # calculate 2-D DHT
    batch_ht = fft.ht2_center_torch(batch_original)

    # crop hartley-transformed image stack
    batch_ht_cropped = batch_ht[:, start:stop, start:stop]

    # inverse hartley transform image stack
    batch_real_cropped = fft.iht2_center_torch(batch_ht_cropped)

    # fix image stack scaling (due to cropping during transform) and dtype (due to numpy type promotion during transform)
    batch_real_cropped *= (stop - start) ** 2 / batch_original.shape[-1] ** 2
    batch_real_cropped = batch_real_cropped.to(batch_original.dtype)

    return batch_real_cropped


def write_downsampled_starfile(input_starfile: str,
                               boxsize_old: int,
                               boxsize_new: int,
                               out_mrcs: str | list[str],
                               chunk_size: int | None = None) -> None:
    """
    Write a TiltSeriesStarfile describing a downsampled particle stack (optionally chunked into multiple .mrcs files) to disk.
    :param input_starfile: path to the star file used to reference original (non-downsampled) images
    :param boxsize_old: the original box size (units: px)
    :param boxsize_new: the new box size after Fourier cropping (units: px)
    :param out_mrcs: path to the downsampled mrcs file(s), referencing images in the same order as the original star file
    :param chunk_size: number of images chunked together in separate .mrcs files written to disk, if used. None means no chunking was performed.
    :return: None
    """
    # load the star file
    assert input_starfile.endswith('.star')
    star = starfile.TiltSeriesStarfile(input_starfile)

    # get new pixel size according to star file
    old_angpix = star.get_tiltseries_pixelsize()
    new_angpix = old_angpix * boxsize_old / boxsize_new

    # rescale coordinates, shifts, and pixel size in the new star file
    star.df[star.header_pose_tx] = star.df[star.header_pose_tx] * old_angpix / new_angpix
    star.df[star.header_pose_ty] = star.df[star.header_pose_ty] * old_angpix / new_angpix
    for coord_col in star.df.columns[star.df.columns.str.contains(pat='Coordinate')]:
        star.df[coord_col] = star.df[coord_col] * old_angpix / new_angpix
    star.header_ctf_angpix = new_angpix

    # update paths to .mrcs image data, forming string of form `123456@/path/to/stack.mrcs` with 1-indexed images
    if chunk_size is None:
        # wrote one .mrcs, so all particles are contained within that file
        assert type(out_mrcs) is str
        new_paths = [f'{i+1:06}@{out_mrcs}' for i in range(len(star.df))]
        out_star = f'{os.path.splitext(out_mrcs)[0]}.star'
    else:
        # wrote multiple .mrcs (containing `chunk_size` images except for the last file which contains <= `chunk_size` images)
        assert type(out_mrcs) is list
        new_paths = [f'{i+1:06}@{out_path}' for out_path in out_mrcs for i in range(chunk_size)]
        out_star = f'{os.path.splitext(os.path.splitext(out_mrcs[0])[0])[0]}.star'  # convert FOO.0.mrcs to FOO
    star.df[star.header_ptcl_image] = new_paths[:len(star.df)]

    # save the star file
    star.write(out_star)


class ImageDataset(data.Dataset):
    """
    Simple dataset class to rapidly supply and downsample particle images. Operates per-image (rather than more typical per-particle for tilt series).
    """

    def __init__(self,
                 particles: np.ndarray | list[mrc.LazyImage],
                 nimgs: int):
        self.particles = particles
        self.nimgs = nimgs

    def __len__(self) -> int:
        return self.nimgs

    def __getitem__(self,
                    index: int) -> tuple[int, np.ndarray]:
        if type(self.particles[index]) is mrc.LazyImage:
            return index, self.particles[index].get()
        else:
            return index, self.particles[index]


def main(args):
    # log the arguments
    log(args)

    # check inputs
    warnexists(args.output)
    mkbasedir(args.output)
    assert (args.output.endswith('.mrcs') or args.output.endswith('mrc')), "Must specify output in .mrc(s) file format"

    # set the device
    device = utils.get_default_device()
    torch.set_grad_enabled(False)

    # load the original particles
    log(f'Loading {args.input}')
    lazy = args.lazy
    old = dataset.load_particles(mrcs_txt_star=args.input,
                                 lazy=lazy,
                                 datadir=args.datadir)
    nimgs = len(old)
    boxsize_old = old[0].get().shape[0] if lazy else old.shape[-1]
    boxsize_new = args.downsample
    assert boxsize_new <= boxsize_old, f'New box size {boxsize_new} cannot be larger than the original box size {boxsize_old}'
    assert boxsize_new % 2 == 0, 'New box size must be even'

    # establish parameters for downsampling
    start = int(boxsize_old / 2 - boxsize_new / 2)
    stop = int(boxsize_old / 2 + boxsize_new / 2)
    if args.input.endswith('.mrcs') or args.input.endswith('.mrc'):
        old_angpix = mrc.parse_header(args.input).get_apix()
    elif args.input.endswith('.star'):
        old_angpix = starfile.TiltSeriesStarfile(args.input).get_tiltseries_pixelsize()
    elif args.input.endswith('.txt'):
        with open(args.input, 'r') as f:
            old_angpix = mrc.parse_header(f.readline()).get_apix()
    else:
        raise ValueError
    new_apix = old_angpix * boxsize_old / boxsize_new

    # Downsample volume, saving to one .mrc file
    log('Downsampling...')
    if args.is_vol:
        log(old.shape)

        # hartley transform, crop the hartley transformed volume, and return to real space
        vol_ht = fft.ht3_center(old)
        vol_ht_cropped = vol_ht[start:stop, start:stop, start:stop]
        vol_real_cropped = fft.iht3_center(vol_ht_cropped)
        log(vol_real_cropped.shape)

        # fix volume scaling (due to cropping during transform)
        vol_real_cropped *= boxsize_new ** 3 / boxsize_old ** 3

        # write to disk
        log(f'Saving {args.output}')
        mrc.write(fname=args.output,
                  array=vol_real_cropped,
                  angpix=new_apix,
                  is_vol=True)

    # Downsample images in batches, saving to one .mrcs file
    elif args.chunk is None:
        # prepare outputs
        out_mrcs = args.output
        new = np.empty((nimgs, boxsize_new, boxsize_new), dtype=np.float32)

        # iterate through image batches to apply downsampling and store in new volume array
        particle_dataset = ImageDataset(particles=old, nimgs=nimgs)
        data_generator = DataLoader(dataset=particle_dataset, batch_size=args.batch_size, shuffle=False)
        for batch_idx, batch_ptcls in data_generator:
            log(f'Processing indices {batch_idx[0]} - {batch_idx[-1]}')
            batch_ptcls.to(device)
            batch_new = downsample_images(batch_original=batch_ptcls, start=start, stop=stop)
            new[batch_idx.cpu().numpy()] = batch_new.cpu().numpy()

        # write to disk
        log(new.shape)
        log(f'Saving {args.output}')
        mrc.write(fname=out_mrcs,
                  array=new,
                  angpix=new_apix,
                  is_vol=False)

        if args.write_tiltseries_starfile:
            write_downsampled_starfile(input_starfile=args.input,
                                       boxsize_old=boxsize_old,
                                       boxsize_new=boxsize_new,
                                       out_mrcs=out_mrcs,
                                       chunk_size=args.chunk)

    # Downsample images in batches, saving chunks of N images to nchunks separate .mrcs files
    else:
        assert args.batch_size <= args.chunk
        nchunks = math.ceil(len(old) / args.chunk)
        out_mrcs = [f'.{i}'.join(os.path.splitext(args.output)) for i in range(nchunks)]
        chunk_names = [os.path.basename(x) for x in out_mrcs]
        for i in range(nchunks):
            log(f'Processing chunk {i}')
            chunk = old[i * args.chunk: (i + 1) * args.chunk]
            new = np.empty((len(chunk), boxsize_new, boxsize_new), dtype=np.float32)

            particle_dataset = ImageDataset(particles=chunk, nimgs=len(chunk))
            data_generator = DataLoader(dataset=particle_dataset, batch_size=args.batch_size, shuffle=False)
            for batch_idx, batch_ptcls in data_generator:
                log(f'Processing chunk indices {batch_idx[0]} - {batch_idx[-1]}')
                batch_new = downsample_images(batch_original=batch_ptcls, start=start, stop=stop)
                new[batch_idx.cpu().numpy()] = batch_new.cpu().numpy()

            log(new.shape)
            log(f'Saving {out_mrcs[i]}')
            mrc.write(fname=out_mrcs[i],
                      array=new,
                      angpix=new_apix,
                      is_vol=False)

        # Write a text file with all chunks
        out_txt = '{}.txt'.format(os.path.splitext(args.output)[0])
        log(f'Saving {out_txt}')
        with open(out_txt, 'w') as f:
            f.write('\n'.join(chunk_names))

        if args.write_tiltseries_starfile:
            write_downsampled_starfile(input_starfile=args.input,
                                       boxsize_old=boxsize_old,
                                       boxsize_new=boxsize_new,
                                       out_mrcs=out_mrcs,
                                       chunk_size=args.chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
