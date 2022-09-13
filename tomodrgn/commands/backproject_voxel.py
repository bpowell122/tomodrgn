'''
Backproject 2-D images to form unfiltered 3-D reconstruction
'''

import argparse
import os
import time
import numpy as np
import torch

from tomodrgn import utils, mrc, fft, dataset, ctf, starfile
from tomodrgn.lattice import Lattice

log = utils.log

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles_imageseries.star')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output .mrc file')

    group = parser.add_argument_group('Dataset loading options')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths star file')
    group.add_argument('--ind', type=os.path.abspath, help='Indices to iterate over (pkl)')
    group.add_argument('--first', type=int, default=10000, help='Backproject the first N images')

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

    t1 = time.time()    
    log(args)
    if not os.path.exists(os.path.dirname(args.o)):
        os.makedirs(os.path.dirname(args.o))

    ## set the device
    device = utils.get_default_device()

    # load the star file
    ptcls_star = starfile.TiltSeriesStarfile.load(args.particles)
    df_grouped = ptcls_star.df.groupby('_rlnGroupName')
    ind_imgs = np.array([df_grouped.get_group(ptcl).index.to_numpy() for ptcl in df_grouped.groups], dtype=object)

    if args.ind is not None:
        ind_ptcls = np.array(utils.load_pkl(args.ind))
        ind_imgs = ind_imgs[ind_ptcls]
        ptcls_star.df = ptcls_star.df.iloc[ind_imgs.flatten()]

    # lazily load particle images and filter by ind.pkl, if applicable
    data = dataset.LazyMRCData(args.particles, norm=(0,1), invert_data=args.invert_data, datadir=args.datadir)
    data.particles = [data.particles[i] for i in ind_imgs.flatten().astype(int)]
    D = data.D
    Nimg = ind_imgs.flatten().shape[0]

    # Load poses (from pre-filtered dataframe)
    rots_columns = ['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']
    euler = ptcls_star.df[rots_columns].to_numpy(dtype=np.float32)
    rots = np.asarray([utils.R_from_relion(*x) for x in euler], dtype=np.float32)
    rots = torch.from_numpy(rots).to(device)

    trans_columns = ['_rlnOriginX', '_rlnOriginY']
    if np.all([trans_column in ptcls_star.headers for trans_column in trans_columns]):
        trans = ptcls_star.df[trans_columns].to_numpy(dtype=np.float32)
        trans = torch.from_numpy(trans).to(device)
    else:
        trans = None

    # Load CTF (from pre-filtered dataframe)
    ctf_columns = ['_rlnDetectorPixelSize', '_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration',
                   '_rlnAmplitudeContrast', '_rlnPhaseShift']
    if np.all([ctf_column in ptcls_star.headers for ctf_column in ctf_columns]):
        ctf_params = ptcls_star.df[ctf_columns].to_numpy(dtype=np.float32)
        ctf_params = torch.tensor(ctf_params).to(device)
    else:
        ctf_params = None
    Apix = ctf_params[0,0] if ctf_params is not None else 1

    # instantiate lattice
    lattice = Lattice(D, extent=D//2, device=device)
    mask = lattice.get_circular_mask(D//2)

    # instantiate volume
    V = torch.zeros((D,D,D), device=device)
    counts = torch.zeros((D,D,D), device=device)

    if args.first:
        args.first = min(args.first, Nimg)
        iterator = range(args.first)
    else:
        iterator = range(Nimg)

    for ii in iterator:
        if ii%100==0: log(f'image {ii}')
        r = rots[ii]
        t = trans[ii] if trans is not None else None
        ff = torch.tensor(data.get(ii), device=device).view(-1)[mask]
        if ctf_params is not None:
            freqs = lattice.freqs2d/ctf_params[ii,0]
            c = ctf.compute_ctf(freqs, *ctf_params[ii,1:]).view(-1)[mask]
            ff *= c.sign()
        if t is not None:
            ff = lattice.translate_ht(ff.view(1,-1),t.view(1,1,2), mask).view(-1)
        ff_coord = lattice.coords[mask] @ r
        add_slice(V, counts, ff_coord, ff, D)

    td = time.time()-t1
    log(f'Backprojected {len(iterator)} images in {td}s ({td/Nimg}s per image)')
    counts[counts == 0] = 1
    V /= counts
    V = fft.ihtn_center(V[0:-1,0:-1,0:-1].cpu().numpy())
    mrc.write(args.o,V.astype('float32'), Apix=Apix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
