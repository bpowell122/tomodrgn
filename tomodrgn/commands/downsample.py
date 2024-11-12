"""
Downsample an image stack or volume by Fourier cropping
"""
import argparse
import math
import os
from typing import get_args

import numpy as np
import torch
import torch.utils.data

from tomodrgn import utils, mrc, fft, dataset, starfile

log = utils.log


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        # this script is called from tomodrgn.__main__ entry point, in which case a parser is already created
        pass

    parser.add_argument('input', help='Input particles or volume (.mrc, .mrcs, .star, or .txt)')

    group = parser.add_argument_group('Core arguments')
    group.add_argument('--source-software', type=str, choices=get_args(starfile.KNOWN_STAR_SOURCES), default='auto',
                       help='Manually set the software used to extract particles. Default is to auto-detect.')
    group.add_argument('--downsample', type=int, required=True, help='New box size in pixels, must be even')
    group.add_argument('--output', metavar='MRCS', type=os.path.abspath, required=True, help='Output projection stack (.mrcs)')
    group.add_argument('--batch-size', type=int, default=5000, help='Batch size for processing images')
    group.add_argument('--is-vol', action='store_true', help='Flag if input .mrc is a volume')
    group.add_argument('--chunk', type=int, help='Chunksize (in # of images) to split particle stack when loading and saving if full stack + downsampled stack too large for system memory')
    group.add_argument('--lazy', action='store_true', help='Lazily load each image on the fly if full stack too large for system memory')
    group.add_argument('--datadir', help='Optionally provide path to input .mrcs if loading from a .star file')
    group.add_argument('--write-tiltseries-starfile', action='store_true', help='If input is a star file, write a downsampled star file')

    return parser


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
                               chunk_size: int | None = None,
                               source_software: starfile.KNOWN_STAR_SOURCES = 'auto') -> None:
    """
    Write a STAR file describing a downsampled particle stack (optionally chunked into multiple .mrcs files) to disk.

    :param input_starfile: path to the star file used to reference original (non-downsampled) images
    :param boxsize_old: the original box size (units: px)
    :param boxsize_new: the new box size after Fourier cropping (units: px)
    :param out_mrcs: path to the downsampled mrcs file(s), referencing images in the same order as the original star file
    :param chunk_size: number of images chunked together in separate .mrcs files written to disk, if used. None means no chunking was performed.
        :param source_software: type of source software used to create the star file, used to indicate the appropriate star file handling class to instantiate.
            Default of 'auto' tries to infer the appropriate star file handling class based on whether ``star_path`` is an optimisation set star file.
    :return: None
    """
    # load the star file
    assert input_starfile.endswith('.star')
    star = starfile.load_sta_starfile(input_starfile, source_software=source_software)

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

    # if input is optimisation set star file, ensure output is optimisation set star file
    if type(star) is starfile.TomoParticlesStarfile:
        if '_optimisation_set' not in out_star:
            out_star = f'{os.path.splitext(out_star)[0]}_optimisation_set.star'

    # save the star file
    star.write(out_star)


class ImageDataset(torch.utils.data.Dataset):
    """
    Simple dataset class to rapidly supply and downsample particle images. Operates per-image (rather than more typical per-particle for tilt series).
    """

    def __init__(self,
                 particles: np.ndarray | list[mrc.LazyImage] | list[mrc.LazyImageStack]):
        self.particles = particles
        if type(particles[0]) is np.ndarray:
            self.ptcl_img_indices = np.asarray([i for i in range(len(particles))])
            self.nimgs = len(particles)
        elif type(particles[0]) is mrc.LazyImage:
            self.ptcl_img_indices = np.asarray([i for i in range(len(particles))])
            self.nimgs = len(particles)
        elif type(particles[0]) is mrc.LazyImageStack:
            ptcl_img_indices = []
            offset = 0
            for ptcl in particles:
                ptcl_img_indices.append(np.arange(offset, offset + ptcl.n_images))
                offset += ptcl.n_images
            self.ptcl_img_indices = ptcl_img_indices
            self.nimgs = self.ptcl_img_indices[-1][-1] + 1  # correct for 0-indexing
        else:
            raise ValueError(f'Unsupported particles type: {type(particles[0])}')

    def __len__(self) -> int:
        return len(self.ptcl_img_indices)

    def __getitem__(self,
                    index: int) -> tuple[np.ndarray, np.ndarray]:
        particles = self.particles[index] if type(self.particles[index]) is np.ndarray else self.particles[index].get()
        # cast to float32 because input data may be float16 which is not supported by cuFFT (https://github.com/pytorch/pytorch/issues/70664)
        particles = particles.astype(np.float32)
        # get the indices of where this/these images will go in the array of all images
        indices = np.asarray(self.ptcl_img_indices[index])
        return indices, particles


def collate_particle_tilts(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.tensor, torch.tensor]:
    """
    Collate a batch of particle images, optionally with a variable number of tilt images associated with each particle.
    Given input of shape ``([ind1], [a, X, X]), ([ind2], [b, X, X])`` where a and b can be any number, returns ``[ind1, ind2], [[a+b, X, X]]``.

    :param batch: list of 2-tuples of particle indices (``np.ndarray`` of shape ``(ntilts)``) and associated images (``np.ndarray`` of shape ``(ntilts, npixels, npixels)``
    :returns: 2-tuple of concatenated particle indices and concatenated particle images
    """
    collated_indices = []
    collated_images = []
    for i in range(len(batch)):
        collated_indices.append(torch.as_tensor(batch[i][0]))
        collated_images.append(torch.as_tensor(batch[i][1]))

    collated_indices = torch.hstack(collated_indices)
    if collated_images[0].ndim == 2:
        collated_images = torch.stack(collated_images)
    elif collated_images[0].ndim == 3:
        collated_images = torch.concatenate(collated_images, dim=0)
    else:
        raise ValueError(f'Unrecognized leading shape of image data: {collated_images[0].shape}')

    return collated_indices, collated_images


def main(args):
    # log the arguments
    log(args)

    # check inputs
    warnexists(args.output)
    mkbasedir(args.output)
    assert (args.output.endswith('.mrcs') or args.output.endswith('mrc')), "Must specify output in .mrc(s) file format"

    # set the device
    device = utils.get_default_device()

    # load the original particles
    log(f'Loading {args.input}')
    lazy = args.lazy
    old = dataset.load_particles(mrcs_txt_star=args.input,
                                 lazy=lazy,
                                 datadir=args.datadir,
                                 source_software=args.source_software)
    old_dtype = old.dtype if type(old) is np.ndarray else old[0].get().dtype
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
        old_angpix = starfile.load_sta_starfile(args.input, source_software=args.source_software).get_tiltseries_pixelsize()
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
        # iterate through image batches to apply downsampling and store in new volume array
        particle_dataset = ImageDataset(particles=old)
        data_generator = torch.utils.data.DataLoader(dataset=particle_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_particle_tilts)

        # prepare outputs
        out_mrcs = args.output
        new = np.empty((particle_dataset.nimgs, boxsize_new, boxsize_new), dtype=old_dtype)
        for batch_idx, batch_ptcls in data_generator:
            log(f'Processing indices {batch_idx[0]} - {batch_idx[-1]}')
            batch_ptcls.to(device)
            batch_new = downsample_images(batch_original=batch_ptcls, start=start, stop=stop)
            new[batch_idx.cpu().numpy()] = batch_new.cpu().numpy().astype(old_dtype)

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
                                       chunk_size=args.chunk,
                                       source_software=args.source_software)

    # Downsample images in batches, saving chunks of N images to nchunks separate .mrcs files
    else:
        assert args.batch_size <= args.chunk
        nchunks = math.ceil(len(old) / args.chunk)
        out_mrcs = [f'.{i}'.join(os.path.splitext(args.output)) for i in range(nchunks)]
        chunk_names = [os.path.basename(x) for x in out_mrcs]
        for i in range(nchunks):
            log(f'Processing chunk {i}')
            chunk = old[i * args.chunk: (i + 1) * args.chunk]

            particle_dataset = ImageDataset(particles=chunk)
            data_generator = torch.utils.data.DataLoader(dataset=particle_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_particle_tilts)
            new = np.empty((particle_dataset.nimgs, boxsize_new, boxsize_new), dtype=old_dtype)
            for batch_idx, batch_ptcls in data_generator:
                log(f'Processing chunk indices {batch_idx[0]} - {batch_idx[-1]}')
                batch_new = downsample_images(batch_original=batch_ptcls, start=start, stop=stop)
                new[batch_idx.cpu().numpy()] = batch_new.cpu().numpy().astype(old_dtype)

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
                                       chunk_size=args.chunk,
                                       source_software=args.source_software)


if __name__ == '__main__':
    main(add_args().parse_args())
