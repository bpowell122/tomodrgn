'''
Generate projections of a 3D volume
Written by Ellen Zhong, Emily Navarret, and Joey Davis
'''

import argparse
import numpy as np
import sys, os
import time
import pickle
from scipy.ndimage.fourier import fourier_shift

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from cryodrgn import utils
from cryodrgn import mrc
from cryodrgn import fft
from cryodrgn import lie_tools
from cryodrgn import so3_grid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = utils.log
vlog = utils.vlog

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mrc', help='Input volume')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output projection stack (.mrcs)')
    parser.add_argument('--out-pose', type=os.path.abspath, required=True, help='Output poses (.pkl)')
    parser.add_argument('--out-png', type=os.path.abspath, help='Montage of first 9 projections')
    parser.add_argument('--in-pose', type=os.path.abspath, help='Optionally provide input poses instead of random poses (.pkl)')
    parser.add_argument('-N', type=int, help='Number of random projections')
    parser.add_argument('-b', type=int, default=100, help='Minibatch size (default: %(default)s)')
    parser.add_argument('--t-extent', type=float, default=5, help='Extent of image translation in pixels (default: +/-%(default)s)')
    parser.add_argument('--grid', type=int, help='Generate projections on a uniform deterministic grid on SO3. Specify resolution level')
    parser.add_argument('--is-mask', action='store_true', help='Takes max value along z instead of integrating along z, to create mask images from mask volumes')
    parser.add_argument('--tilt', type=float, help='Right-handed x-axis tilt offset in degrees')
    parser.add_argument('--tiltseries', action='store_true', help='Project dose-symmetric tilt series per random pose')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('-v','--verbose',action='store_true',help='Increaes verbosity')
    parser.add_argument('--chunk', type=int, default=10000, help='Chunksize (in # of images) to split particle stack when saving')
    return parser

class Projector:
    def __init__(self, vol, tilt=None, tiltseries=False, is_mask=False):
        nz, ny, nx = vol.shape
        assert nz==ny==nx, 'Volume must be cubic'
        x2, x1, x0 = np.meshgrid(np.linspace(-1, 1, nz, endpoint=True), 
                             np.linspace(-1, 1, ny, endpoint=True),
                             np.linspace(-1, 1, nx, endpoint=True),
                             indexing='ij')

        lattice = np.stack([x0.ravel(), x1.ravel(), x2.ravel()],1).astype(np.float32)
        self.lattice = torch.from_numpy(lattice)

        self.vol = torch.from_numpy(vol.astype(np.float32))
        self.vol = self.vol.unsqueeze(0)
        self.vol = self.vol.unsqueeze(0)

        self.nz = nz
        self.ny = ny
        self.nx = nx

        # FT is not symmetric around origin
        D = nz
        c = 2/(D-1)*(D/2) -1 
        self.center = torch.tensor([c,c,c]) # pixel coordinate for vol[D/2,D/2,D/2]

        if tilt is not None:
            assert tilt.shape == (3,3)
            tilt = torch.tensor(tilt)
        self.tilt = tilt

        if tiltseries:
            # TODO add option for different tilt scheme
            dose_symmetric_tilts = np.array([0, 3, -3, -6, 6, 9, -9, -12, 12, 15, -15, -18, 18, 21, -21, -24, 24, 27, -27,
                                         -30, 30, 33, -33, -36, 36, 39, -39, -42, 42, 45, -45, -48, 48, 51, -51, -54,
                                         54, 57, -57, -60, 60])
            tiltseries_matrices = np.zeros((dose_symmetric_tilts.shape[0], 3, 3))
            for i, tilt in enumerate(dose_symmetric_tilts):
                tiltseries_matrices[i] = utils.xrot(tilt).astype(np.float32)
            self.tiltseries_matrices = torch.cuda.FloatTensor(tiltseries_matrices)
            self.dose_symmetric_tilts = dose_symmetric_tilts
        self.tiltseries = tiltseries
        self.is_mask = is_mask

    def rotate(self, rot):
        B = rot.size(0)
        if self.tilt is not None:
            rot = self.tilt @ rot
        grid = self.lattice @ rot # B x D^3 x 3 ERROR ENDS HERE 
        grid = grid.view(-1, self.nz, self.ny, self.nx, 3)
        offset = self.center - grid[:,int(self.nz/2),int(self.ny/2),int(self.nx/2)]
        grid += offset[:,None,None,None,:]
        grid = grid.view(1, -1, self.ny, self.nx, 3)
        vol = F.grid_sample(self.vol, grid)
        vol = vol.view(B,self.nz,self.ny,self.nx)
        return vol

    def project(self, rot):
        if self.is_mask:
            return self.rotate(rot).max(dim=1)[0]
        else:
            return self.rotate(rot).sum(dim=1)
        # else:
        #     imgs = torch.empty((rot.shape[0], self.tiltseries_matrices.shape[0], self.ny, self.nx)).to(0)
        #     for i, tilt in enumerate(self.tiltseries_matrices):
        #         assert tilt.shape == (3,3), print(tilt.shape)
        #         imgs[:,i,:,:] = self.rotate(rot @ tilt).sum(dim=1)
        #     return imgs.view(-1, imgs.shape[-1], imgs.shape[-1])

   
class Poses(data.Dataset):
    def __init__(self, pose_pkl):
        poses = utils.load_pkl(pose_pkl)
        self.rots = torch.tensor(poses[0])
        self.trans = poses[1]
        self.N = len(poses[0])
        assert self.rots.shape == (self.N,3,3)
        assert self.trans.shape == (self.N,2)
        assert self.trans.max() < 1
    def __len__(self):
        return self.N
    def __getitem__(self, index):
        return self.rots[index]

class RandomRot(data.Dataset):
    def __init__(self, N):
        self.N = N
        self.rots = lie_tools.random_SO3(N)
    def __len__(self):
        return self.N
    def __getitem__(self, index):
        return self.rots[index]

class GridRot(data.Dataset):
    def __init__(self, resol):
        quats = so3_grid.grid_SO3(resol)
        self.rots = lie_tools.quaternions_to_SO3(torch.tensor(quats))
        self.N = len(self.rots)
    def __len__(self):
        return self.N
    def __getitem__(self, index):
        return self.rots[index]

def plot_projections(out_png, imgs):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    axes = axes.ravel()
    for i in range(min(len(imgs),9)):
        axes[i].imshow(imgs[i])
    plt.savefig(out_png)

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def warnexists(out):
    if os.path.exists(out):
        log('Warning: {} already exists. Overwriting.'.format(out))

def translate_img(img, t):
    '''
    img: BxYxX real space image
    t: Bx2 shift in pixels
    '''
    ff = np.fft.fft2(np.fft.fftshift(img))
    ff = fourier_shift(ff, t)
    return np.fft.fftshift(np.fft.ifft2(ff)).real

def main(args):
    for out in (args.o, args.out_png, args.out_pose):
        if not out: continue
        mkbasedir(out)
        warnexists(out)

    if args.in_pose is None and args.t_extent == 0.:
        log('Not shifting images')
    elif args.in_pose is None:
        assert args.t_extent > 0

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    log('Use cuda {}'.format(use_cuda))
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    t1 = time.time()    
    vol, _ = mrc.parse_mrc(args.mrc)
    log('Loaded {} volume'.format(vol.shape))

    if args.tilt:
        theta = args.tilt*np.pi/180
        args.tilt = np.array([[1.,0.,0.],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]]).astype(np.float32)

    projector = Projector(vol, args.tilt, args.tiltseries, args.is_mask)
    if projector.tiltseries:
        ntilts = projector.dose_symmetric_tilts.shape[0]
    if use_cuda:
        projector.lattice = projector.lattice.cuda()
        projector.vol = projector.vol.cuda()

    if args.grid is not None:
        rots = GridRot(args.grid)
        log('Generating {} rotations at resolution level {}'.format(len(rots), args.grid))
    elif args.in_pose is not None:
        rots = Poses(args.in_pose)
        log('Generating {} rotations from {}'.format(len(rots), args.grid))
    else:
        log('Generating {} random rotations'.format(args.N))
        rots = RandomRot(args.N)
    
    log('Projecting...')
     # imgs = []
    iterator = data.DataLoader(rots, batch_size=args.b)
    if projector.tiltseries:
        log('Projecting tiltseries...')
        imgs = [] #np.empty((iterator.__len__() * ntilts, vol.shape[0], vol.shape[0]))
        final_rots = np.empty((iterator.__len__() * ntilts, 3, 3))
        for i, rot in enumerate(iterator):
            vlog('Projecting {}/{}'.format((i+1)*len(rot), args.N))
            for j, tilt in enumerate(projector.tiltseries_matrices):
                pose = rot @ tilt.view(1, 3, 3)
                assert pose.shape == (1, 3, 3), print(pose.shape)
                projections = projector.project(pose)  # MEMORY ERROR STARTS HERE
                projections = projections.cpu().numpy()
                imgs.append(projections)
                final_rots[ntilts*i + j] = pose.view(3, 3).cpu().numpy()
    else:
        imgs = [] #np.empty((iterator.__len__(), vol.shape[0], vol.shape[0]))
        for i, rot in enumerate(iterator):
            vlog('Projecting {}/{}'.format((i+1)*len(rot), args.N))
            projections = projector.project(rot)  # MEMORY ERROR STARTS HERE
            projections = projections.cpu().numpy()
            imgs.append(projections)
        final_rots = rots

    td = time.time()-t1
    log('Projected {} images in {}s ({}s per image)'.format(rots.N, td, td/rots.N ))
    imgs = np.vstack(imgs).astype(np.float32)
    print(imgs.shape)

    if args.in_pose is None and args.t_extent:
        log('Shifting images between +/- {} pixels'.format(args.t_extent))
        if not projector.tiltseries:
            trans = np.random.rand(args.N,2)*2*args.t_extent - args.t_extent
        else:
            trans = np.random.rand(args.N*ntilts, 2) * 2 * args.t_extent - args.t_extent
    elif args.in_pose is not None:
        log('Shifting images by input poses')
        D = imgs.shape[-1]
        trans = rots.trans*D # convert to pixels
        trans = -trans[:,::-1] # convention for scipy
    else:
        trans = None

    if trans is not None:
        nchunks = np.ceil(imgs.shape[0] / args.chunk).astype(int)
        out_mrcs = ['.{}'.format(i).join(os.path.splitext(args.o)) for i in range(nchunks)]
        chunk_names = [os.path.basename(x) for x in out_mrcs]
        for i in range(nchunks):
            log('Processing chunk {}'.format(i))
            chunk_imgs = imgs[i * args.chunk:(i + 1) * args.chunk]
            chunk_trans = trans[i * args.chunk:(i + 1) * args.chunk]
            chunk_translated = np.asarray([translate_img(img, t) for img,t in zip(chunk_imgs, chunk_trans)], dtype=np.float32)
            log(chunk_translated.shape)
            log(f'Saving {out_mrcs[i]}')
            mrc.write(out_mrcs[i], chunk_translated, is_vol=False)
        out_txt = '{}.txt'.format(os.path.splitext(args.o)[0])
        log(f'Saving {out_txt}')
        with open(out_txt, 'w') as f:
            f.write('\n'.join(chunk_names))
        # imgs = np.asarray([translate_img(img, t) for img,t in zip(imgs,trans)], dtype=np.float32)
        # convention: we want the first column to be x shift and second column to be y shift
        # reverse columns since current implementation of translate_img uses scipy's 
        # fourier_shift, which is flipped the other way
        # convention: save the translation that centers the image
        trans = -trans[:,::-1]
        # convert translation from pixel to fraction
        D = imgs.shape[-1]
        assert D % 2 == 0
        trans /= D

    # log('Saving {}'.format(args.o))
    # mrc.write(args.o,imgs.astype(np.float32))
    log('Saving {}'.format(args.out_pose))
    with open(args.out_pose,'wb') as f:
        if args.t_extent:
            pickle.dump((final_rots,trans),f)
        else:
            pickle.dump(final_rots, f)
    if args.out_png:
        log('Saving {}'.format(args.out_png))
        plot_projections(args.out_png, imgs[:9])

if __name__ == '__main__':
    args = parse_args().parse_args()
    utils._verbose = args.verbose
    main(args)
