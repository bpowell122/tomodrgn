"""
Backproject 2-D images to form 3-D reconstruction with optional filtering and weighting
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data

from tomodrgn import utils, mrc, fft, ctf
from tomodrgn.starfile import TiltSeriesStarfile
from tomodrgn.dataset import TiltSeriesMRCData
from tomodrgn.lattice import Lattice

log = utils.log


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        # this script is called from tomodrgn.__main__ entry point, in which case a parser is already created
        pass

    parser.add_argument('particles', type=os.path.abspath, help='Input particles_imageseries.star')
    parser.add_argument('--output', type=os.path.abspath, required=True, help='Output .mrc file')
    parser.add_argument('--plot-format', type=str, choices=['png', 'svgz'], default='png', help='File format with which to save plots')

    group = parser.add_argument_group('Particle starfile loading and filtering')
    group.add_argument('--source-software', type=str, choices=('auto', 'warp_v1', 'nextpyp', 'relion_v5', 'warp_v2'), default='auto',
                       help='Manually set the software used to extract particles. Default is to auto-detect.')
    group.add_argument('--ind-ptcls', type=os.path.abspath, metavar='PKL', help='Filter starfile by particles (unique rlnGroupName values) using np array pkl as indices')
    group.add_argument('--ind-imgs', type=os.path.abspath, help='Filter starfile by particle images (star file rows) using np array pkl as indices')
    group.add_argument('--sort-ptcl-imgs', choices=('unsorted', 'dose_ascending', 'random'), default='unsorted', help='Sort the star file images on a per-particle basis by the specified criteria')
    group.add_argument('--use-first-ntilts', type=int, default=-1, help='Keep the first `use_first_ntilts` images of each particle in the sorted star file.'
                                                                        'Default -1 means to use all. Will drop particles with fewer than this many tilt images.')
    group.add_argument('--use-first-nptcls', type=int, default=-1, help='Keep the first `use_first_nptcls` particles in the sorted star file. Default -1 means to use all.')

    group = parser.add_argument_group('Dataset loading options')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths star file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD)')

    group = parser.add_argument_group('Reconstruction options')
    group.add_argument('--recon-tilt-weight', action='store_true', help='Weight images in fourier space by cosine(tilt_angle)')
    group.add_argument('--recon-dose-weight', action='store_true', help='Weight images in fourieri space per tilt per pixel by dose dependent amplitude attenuation')
    group.add_argument('--lowpass', type=float, default=None, help='Lowpass filter reconstructed volume to this resolution in Angstrom. Defaults to FSC=0.143 correlation between half-maps')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volume')

    return parser


def backproject_dataset(data: TiltSeriesMRCData,
                        lattice: Lattice = None,
                        device: torch.device = torch.device('cpu')) -> tuple[torch.tensor, torch.tensor]:
    """
    Backproject a dataset of 2-D tilt series images to a 3-D Hartley-transformed volume

    :param data: TiltSeriesMRCData object for accessing tilt images with known CTF and pose parameters
    :param lattice: Hartley-transform lattice of points for voxel grid operations
    :param device: torch device on which to perform backprojection
    :return vol_ht: torch tensor of 3-D Hartley-transformed volume without count scaling
    :return counts: torch tensor tracking weighting to be applied to each 3-D spatial frequency of vol_ht
    """
    # initialize the volumes and voxel count scaling tracker
    boxsize_ht = data.boxsize_ht
    vol_ht = torch.zeros((boxsize_ht, boxsize_ht, boxsize_ht), device=device)
    counts = torch.zeros_like(vol_ht)

    n_ptcls_backprojected = 0
    mask = lattice.get_circular_mask(boxsize_ht)

    # prepare the data loader
    batchsize = 1
    data_generator = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=False)

    for batch_images, batch_rot, batch_trans, batch_ctf_params, batch_frequency_weights, _, batch_indices in data_generator:

        # logging
        n_ptcls_backprojected += len(batch_indices)
        if n_ptcls_backprojected % 100 == 0:
            log(f'    {n_ptcls_backprojected} / {data.nptcls} particles')

        # transfer to GPU
        batch_images = batch_images.to(device)
        batch_rot = batch_rot.to(device)
        batch_trans = batch_trans.to(device)
        batch_ctf_params = batch_ctf_params.to(device)
        batch_frequency_weights = batch_frequency_weights.to(device)
        ntilts = batch_images.shape[1]

        # correct for translations
        if not torch.all(batch_trans == torch.zeros(batchsize, device=batch_trans.device)):
            batch_images = lattice.translate_ht(batch_images.view(batchsize * ntilts, -1), batch_trans.view(batchsize * ntilts, 1, 2))

        # restore separation of batch dim and tilt dim (merged into batch dim during translation)
        batch_images = batch_images.view(batchsize, ntilts, boxsize_ht * boxsize_ht)

        # correct CTF by phase flipping images
        if not torch.all(batch_ctf_params == torch.zeros(batchsize, device=batch_ctf_params.device)):
            batch_ctf_weights = ctf.compute_ctf(lattice, *torch.split(batch_ctf_params.view(batchsize * ntilts, 9)[:, 1:], 1, 1))
            batch_images *= batch_ctf_weights.sign()  # phase flip by CTF to be all positive amplitudes

        # weight by dose and tilt
        batch_images = batch_images * batch_frequency_weights

        # mask out frequencies greater than nyquist
        batch_images = batch_images.view(batchsize, ntilts, boxsize_ht * boxsize_ht)[:, :, mask]

        # backproject
        batch_images_coords = lattice.coords[mask] @ batch_rot
        for i in range(batchsize):
            for j in range(ntilts):
                add_slice(vol_ht, counts, batch_images_coords[i, j], batch_images[i, j])

    return vol_ht, counts


def add_slice(vol_ht: torch.tensor,
              counts: torch.tensor,
              ff_coord: torch.tensor,
              ff: torch.tensor) -> None:
    """
    Add one 2-D Hartley-transformed projection image as central slice to 3-D Hartley-transformed volume, modified in-place

    :param vol_ht: torch tensor of 3-D Hartley-transformed volume without count scaling (modified in-place)
    :param counts: torch tensor tracking weighting to be applied to each 3-D spatial frequency of vol_ht (modified in-place)
    :param ff_coord: 3-D lattice coordinates at which to add Hartley-transformed image data for one image, centered at 0
    :param ff: Hartley-transformed image data for one image
    """
    boxsize_ht = vol_ht.shape[0]
    d2 = int(boxsize_ht / 2)
    ff_coord = ff_coord.transpose(0, 1)
    xf, yf, zf = ff_coord.floor().long()
    xc, yc, zc = ff_coord.ceil().long()

    def add_for_corner(_xi: torch.tensor,
                       _yi: torch.tensor,
                       _zi: torch.tensor,
                       _d2: int) -> None:
        """
        Add one 2-D Hartley-transformed projection image to one of 8 integer-valued corners relative to the input interpolated lattice of the 3-D Hartley-transformed voxel lattice, modified in-place

        :param _xi: Integer lattice coordinates along x-axis, centered at 0
        :param _yi: Integer lattice coordinates along y-axis, centered at 0
        :param _zi: Integer lattice coordinates along z-axis, centered at 0
        :param _d2: Box size of 3-D volume to shift lattice coordinates from range `(-_d2, d2)` to `(0, 2*d2)`
        """
        dist = torch.stack([_xi, _yi, _zi]).float() - ff_coord
        w = 1 - dist.pow(2).sum(0).pow(.5)
        w[w < 0] = 0
        vol_ht[(_zi + _d2, _yi + _d2, _xi + _d2)] += w * ff
        counts[(_zi + _d2, _yi + _d2, _xi + _d2)] += w

    add_for_corner(xf, yf, zf, d2)
    add_for_corner(xc, yf, zf, d2)
    add_for_corner(xf, yc, zf, d2)
    add_for_corner(xf, yf, zc, d2)
    add_for_corner(xc, yc, zf, d2)
    add_for_corner(xf, yc, zc, d2)
    add_for_corner(xc, yf, zc, d2)
    add_for_corner(xc, yc, zc, d2)


def save_map(vol: torch.tensor,
             vol_path: str,
             angpix: float,
             flip: bool = False) -> None:
    """
    Inverse Hartley transform and save an input map as a .mrc file

    :param vol: torch tensor of 3-D Hartley-transformed volume
    :param vol_path: name of output .mrc file
    :param angpix: pixel size in angstroms per pixel of output .mrc file
    :param flip: if true, flip the volume along the z-axis before saving
    """
    vol = fft.iht3_center(vol[0:-1, 0:-1, 0:-1].cpu().numpy())
    if flip:
        vol = vol[::-1]
    mrc.write(vol_path,
              vol.astype('float32'),
              angpix=angpix)
    log(f'Wrote {vol_path}')


def main(args):
    assert args.output.endswith('.mrc')

    log(args)
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    # set the device
    device = utils.get_default_device()

    # load the star file
    ptcls_star = TiltSeriesStarfile(args.particles,
                                    source_software=args.source_software)
    ptcls_star.plot_particle_uid_ntilt_distribution(outpath=f'{os.path.dirname(args.output)}/{os.path.basename(ptcls_star.sourcefile)}_particle_uid_ntilt_distribution.{args.plot_format}')

    # filter star file
    ptcls_star.filter(ind_imgs=args.ind_imgs,
                      ind_ptcls=args.ind_ptcls,
                      sort_ptcl_imgs=args.sort_ptcl_imgs,
                      use_first_ntilts=args.use_first_ntilts,
                      use_first_nptcls=args.use_first_nptcls)

    # split ptcls_star by into half sets per-particle for independent backprojection
    ptcls_star.make_test_train_split(fraction_split1=0.5,
                                     show_summary_stats=False)

    # save filtered star file for future convenience (aligning latent embeddings with particles, re-extracting particles, mapbacks, etc.)
    outstar = f'{os.path.dirname(args.output)}/{os.path.splitext(os.path.basename(ptcls_star.sourcefile))[0]}_tomodrgn_preprocessed.star'
    ptcls_star.sourcefile_filtered = outstar
    ptcls_star.write(outstar)

    # load the dataset as two half-datasets for independent backprojection
    data_half1 = TiltSeriesMRCData(ptcls_star=ptcls_star,
                                   star_random_subset=1,
                                   datadir=args.datadir,
                                   lazy=args.lazy,
                                   norm=(0, 1),
                                   invert_data=args.invert_data,
                                   window=False,
                                   recon_dose_weight=args.recon_dose_weight,
                                   recon_tilt_weight=args.recon_tilt_weight,
                                   l_dose_mask=False,
                                   constant_mintilt_sampling=False,
                                   sequential_tilt_sampling=True)
    data_half2 = TiltSeriesMRCData(ptcls_star=ptcls_star,
                                   star_random_subset=2,
                                   datadir=args.datadir,
                                   lazy=args.lazy,
                                   norm=(0, 1),
                                   invert_data=args.invert_data,
                                   window=False,
                                   recon_dose_weight=args.recon_dose_weight,
                                   recon_tilt_weight=args.recon_tilt_weight,
                                   l_dose_mask=False,
                                   constant_mintilt_sampling=False,
                                   sequential_tilt_sampling=True)
    boxsize_ht = data_half1.boxsize_ht
    angpix = ptcls_star.get_tiltseries_pixelsize()

    # instantiate lattice
    lattice = Lattice(boxsize_ht, extent=boxsize_ht // 2, device=device)

    # run the backprojection
    log('Backprojecting half set 1 ...')
    vol_ht_half1, counts_half1 = backproject_dataset(data=data_half1,
                                                     lattice=lattice,
                                                     device=device)
    log('Backprojecting half set 2 ...')
    vol_ht_half2, counts_half2 = backproject_dataset(data=data_half2,
                                                     lattice=lattice,
                                                     device=device)

    # reconstruct full and half-maps
    log('Reconstructing...')
    vol_ht = vol_ht_half1 + vol_ht_half2
    counts = counts_half1 + counts_half2

    counts[counts == 0] = 1
    vol_ht /= counts
    counts_half1[counts_half1 == 0] = 1
    vol_ht_half1 /= counts_half1
    counts_half2[counts_half2 == 0] = 1
    vol_ht_half2 /= counts_half2

    # calculate map-map FSC
    threshold_correlation = 0.143
    x, fsc = utils.calc_fsc(fft.iht3_center(vol_ht_half1[0:-1, 0:-1, 0:-1].cpu().numpy()),
                            fft.iht3_center(vol_ht_half2[0:-1, 0:-1, 0:-1].cpu().numpy()),
                            mask='soft')
    threshold_resolution = x[-1] if np.all(fsc >= threshold_correlation) else x[np.argmax(fsc < threshold_correlation)]
    log(f'Map-map FSC falls below correlation {threshold_correlation} at resolution {angpix / threshold_resolution} Å ({threshold_resolution} 1/px)')
    utils.save_pkl((x, fsc), f'{args.output.split(".mrc")[0]}_FSC.pkl')

    # plot FSC
    plt.plot(x / angpix, fsc)
    plt.xlabel('Spatial frequency (1/Å)')
    plt.ylabel('Half-map FSC')
    plt.tight_layout()
    plt.savefig(f'{args.output.split(".mrc")[0]}_FSC.{args.plot_format}')
    plt.close()

    # apply lowpass filter
    lowpass_target = angpix / threshold_resolution if args.lowpass is None else args.lowpass
    log(f'Lowpass filtering to {lowpass_target} Å')
    vol_ht_filt = utils.lowpass_filter(vol_ht, angpix=angpix, lowpass=lowpass_target)

    # save volumes
    save_map(vol_ht, args.output, angpix, flip=args.flip)
    save_map(vol_ht_filt, f'{args.output.split(".mrc")[0]}_filt.mrc', angpix, flip=args.flip)
    save_map(vol_ht_half1, f'{args.output.split(".mrc")[0]}_half1.mrc', angpix, flip=args.flip)
    save_map(vol_ht_half2, f'{args.output.split(".mrc")[0]}_half2.mrc', angpix, flip=args.flip)
    log('Done!')


if __name__ == '__main__':
    main(add_args().parse_args())
