'''
Backproject 2-D images to form 3-D reconstruction with optional filtering and weighting
'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from tomodrgn import utils, mrc, fft, dataset, ctf, starfile
from tomodrgn.lattice import Lattice

log = utils.log

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles_imageseries.star')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output .mrc file')

    group = parser.add_argument_group('Dataset loading options')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths star file')
    group.add_argument('--ind', type=os.path.abspath, help='Indices of which particles to backproject corresponding tilt images (pkl)')
    group.add_argument('--first', type=int, help='Backproject the first N particles (default is all)')

    group = parser.add_argument_group('Reconstruction options')
    group.add_argument('--recon-tilt-weight', action='store_true', help='Weight images in fourier space by cosine(tilt_angle)')
    group.add_argument('--recon-dose-weight', action='store_true', help='Weight images in fourieri space per tilt per pixel by dose dependent amplitude attenuation')
    group.add_argument('--lowpass', type=float, default=None, help='Lowpass filter reconstructed volume to this resolution in Angstrom. Defaults to FSC=0.143 correlation between half-maps')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volume')

    return parser

def add_slice(V, counts, ff_coord, ff, D):
    d2 = int(D/2)
    ff_coord = ff_coord.transpose(0,1)
    xf, yf, zf = ff_coord.floor().long()
    xc, yc, zc = ff_coord.ceil().long()
    def add_for_corner(xi,yi,zi):
        dist = torch.stack([xi,yi,zi]).float() - ff_coord
        w = 1 - dist.pow(2).sum(0).pow(.5)
        w[w<0]=0
        V[(zi+d2,yi+d2,xi+d2)] += w*ff
        counts[(zi+d2,yi+d2,xi+d2)] += w
    add_for_corner(xf,yf,zf)
    add_for_corner(xc,yf,zf)
    add_for_corner(xf,yc,zf)
    add_for_corner(xf,yf,zc)
    add_for_corner(xc,yc,zf)
    add_for_corner(xf,yc,zc)
    add_for_corner(xc,yf,zc)
    add_for_corner(xc,yc,zc)
    return V, counts

def main(args):
    assert args.o.endswith('.mrc')

    log(args)
    if not os.path.exists(os.path.dirname(args.o)):
        os.makedirs(os.path.dirname(args.o))

    ## set the device
    device = utils.get_default_device()

    # load the star file
    ptcls_star = starfile.TiltSeriesStarfile.load(args.particles)

    # filter by indices and/or manual selection of first N particles
    if args.ind is not None:
        log(f'Reading supplied particle indices {args.ind}')
        ind_ptcls = utils.load_pkl(args.ind)
        if args.first is not None:
            ind_ptcls = ind_ptcls[:args.first]
    else:
        if args.first is not None:
            ind_ptcls = np.arange(args.first)
        else:
            ind_ptcls = np.arange(len(ptcls_star.get_ptcl_img_indices()))

    # split ind_ptcls by into half sets for calculation of map-map FSC
    log('Creating random split of selected indices for calculation of map-map FSC')
    ind_ptcls_half1 = np.sort(np.random.choice(ind_ptcls, size=len(ind_ptcls)//2, replace=False))
    ind_ptcls_half2 = np.sort(np.array(list(set(ind_ptcls) - set(ind_ptcls_half1))))

    data_half1 = dataset.TiltSeriesMRCData(ptcls_star,
                                           norm=(0,1),
                                           invert_data=args.invert_data,
                                           ind_ptcl=ind_ptcls_half1,
                                           window=False,
                                           datadir=args.datadir,
                                           recon_dose_weight=args.recon_dose_weight,
                                           recon_tilt_weight=args.recon_tilt_weight,
                                           l_dose_mask=False,
                                           lazy=False,
                                           sequential_tilt_sampling=True)
    ptcls_star = starfile.TiltSeriesStarfile.load(args.particles)  # re-load the star file object because loading dataset with ind filtering applies filtering in-place
    data_half2 = dataset.TiltSeriesMRCData(ptcls_star,
                                           norm=(0,1),
                                           invert_data=args.invert_data,
                                           ind_ptcl=ind_ptcls_half2,
                                           window=False,
                                           datadir=args.datadir,
                                           recon_dose_weight=args.recon_dose_weight,
                                           recon_tilt_weight=args.recon_tilt_weight,
                                           l_dose_mask=False,
                                           lazy=False,
                                           sequential_tilt_sampling=True)
    D = data_half1.D
    Apix = ptcls_star.get_tiltseries_pixelsize()

    # instantiate lattice
    lattice = Lattice(D, extent=D//2, device=device)
    mask = lattice.get_circular_mask(D//2)

    # instantiate volumes
    V_half1 = torch.zeros((D,D,D), device=device)
    counts_half1 = torch.zeros((D,D,D), device=device)
    V_half2 = torch.zeros((D,D,D), device=device)
    counts_half2 = torch.zeros((D,D,D), device=device)

    # run the backprojection
    def backproject_dataset(data, V, counts):

        B = 1
        n_ptcls_backprojected = 0
        data_generator = DataLoader(data, batch_size=B, shuffle=False)

        for batch_images, batch_rot, batch_trans, batch_ctf, batch_frequency_weights, _, batch_indices in data_generator:
            # logging
            n_ptcls_backprojected += len(batch_indices)
            if n_ptcls_backprojected % 100 == 0: log(f'    {n_ptcls_backprojected} / {data.nptcls} particles')

            # transfer to GPU
            batch_images = batch_images.to(device)
            batch_rot = batch_rot.to(device)
            batch_trans = batch_trans.to(device)
            batch_ctf = batch_ctf.to(device)
            batch_frequency_weights = batch_frequency_weights.to(device)
            ntilts = batch_images.shape[1]

            # correct for translations
            if not torch.all(batch_trans == 0):
                batch_images = lattice.translate_ht(batch_images.view(B * ntilts, -1), batch_trans.view(B * ntilts, 1, 2))

            # correct CTF by phase flipping images
            if not torch.all(batch_ctf == 0):
                freqs = lattice.freqs2d.unsqueeze(0).expand(B * ntilts, *lattice.freqs2d.shape)
                freqs = freqs / batch_ctf[:, :, 1].view(B * ntilts, 1, 1)  # convert units from 1/px to 1/Angstrom
                ctf_weights = ctf.compute_ctf(freqs, *torch.split(batch_ctf.view(B * ntilts, -1)[:, 2:], 1, 1))
                batch_images = batch_images.view(B, ntilts, D * D) * ctf_weights.view(B, ntilts, D * D).sign()

            # weight by dose and tilt
            batch_images = batch_images * batch_frequency_weights.view(B, ntilts, D * D)

            # mask out frequencies greater than nyquist
            batch_images = batch_images.view(B, ntilts, D * D)[:,:,mask]

            # backproject
            batch_images_coords = lattice.coords[mask] @ batch_rot
            for i in range(B):
                for j in range(ntilts):
                    add_slice(V, counts, batch_images_coords[i,j], batch_images[i,j], D)

        return V, counts
    log('Backprojecting half set 1 ...')
    V_half1, counts_half1 = backproject_dataset(data_half1, V_half1, counts_half1)
    log('Backprojecting half set 2 ...')
    V_half2, counts_half2 = backproject_dataset(data_half2, V_half2, counts_half2)

    # reconstruct full and half-maps
    log('Reconstructing...')
    V = V_half1 + V_half2
    counts = counts_half1 + counts_half2

    counts[counts == 0] = 1
    V /= counts
    counts_half1[counts_half1 == 0] = 1
    V_half1 /= counts_half1
    counts_half2[counts_half2 == 0] = 1
    V_half2 /= counts_half2

    # calculate map-map FSC
    threshold_correlation = 0.143
    x, fsc = utils.calc_fsc(fft.iht3_center(V_half1[0:-1,0:-1,0:-1].cpu().numpy()),
                            fft.iht3_center(V_half2[0:-1,0:-1,0:-1].cpu().numpy()),
                            mask='soft')
    threshold_resolution = x[-1] if np.all(fsc >= threshold_correlation) else x[np.argmax(fsc < threshold_correlation)]
    log(f'Map-map FSC falls below correlation {threshold_correlation} at resolution {Apix/threshold_resolution} Å ({threshold_resolution} 1/px)')
    utils.save_pkl((x, fsc), f'{args.o.split(".mrc")[0]}_FSC.pkl')

    # plot FSC
    plt.plot(x / Apix, fsc)
    plt.xlabel('Spatial frequency (1/Å)')
    plt.ylabel('Half-map FSC')
    plt.tight_layout()
    plt.savefig(f'{args.o.split(".mrc")[0]}_FSC.png')

    # apply lowpass filter
    lowpass_target = Apix/threshold_resolution if args.lowpass is None else args.lowpass
    log(f'Lowpass filtering to {lowpass_target} Å')
    V_lowpass = utils.lowpass_filter(V, angpix = Apix, lowpass = lowpass_target)

    # save volumes
    def save_map(vol, vol_path, Apix):
        vol = fft.iht3_center(vol[0:-1,0:-1,0:-1].cpu().numpy())
        if args.flip: vol = vol[::-1]
        mrc.write(vol_path, vol.astype('float32'), Apix=Apix)
        log(f'Wrote {vol_path}')
    save_map(V, args.o, Apix)
    save_map(V_lowpass, f'{args.o.split(".mrc")[0]}_filt.mrc', Apix)
    save_map(V_half1, f'{args.o.split(".mrc")[0]}_half1.mrc', Apix)
    save_map(V_half2, f'{args.o.split(".mrc")[0]}_half2.mrc', Apix)
    log('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
