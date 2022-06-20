'''
Generate projections of a 3D mask to soft-edged 2D mask stack at supplied poses
'''

import argparse
import numpy as np
import os
import time
import scipy.ndimage as ndimage

import torch
import torch.nn.functional as F
import torch.utils.data as data

from cryodrgn import utils
from cryodrgn import mrc

import matplotlib
import matplotlib.pyplot as plt

log = utils.log
vlog = utils.vlog
matplotlib.use('Agg')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mrc', help='Input volume')
    parser.add_argument('--poses', type=os.path.abspath, required=True, help='Input poses parsed by cryodrgn (.pkl)')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output projection stack (.mrcs)')
    parser.add_argument('--out-png', type=os.path.abspath, help='Optionally save PNG montage of first 9 projections')
    parser.add_argument('-b', type=int, default=100, help='Minibatch size for GPU-based rotations')
    parser.add_argument('-v','--verbose',action='store_true',help='Increases verbosity')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--is-vol', action='store_true', help = 'Input .mrc is density map, will project sum of rotated vol')
    group.add_argument('--is-mask', action='store_true', help = 'Input .mrc is mask, will project max of rotated vol')
    return parser


class Projector:
    def __init__(self, vol, is_mask):
        self.is_mask = is_mask
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

    def rotate(self, rot):
        B = rot.size(0)
        grid = self.lattice @ rot
        grid = grid.view(-1, self.nz, self.ny, self.nx, 3)
        offset = self.center - grid[:,int(self.nz/2),int(self.ny/2),int(self.nx/2)]
        grid += offset[:,None,None,None,:]
        grid = grid.view(1, -1, self.ny, self.nx, 3)
        vol = F.grid_sample(self.vol, grid, align_corners=False)
        vol = vol.view(B,self.nz,self.ny,self.nx)
        return vol

    def project(self, rot):
        if self.is_mask == True:
            return self.rotate(rot).max(dim=1)[0]
        elif self.is_mask == False:
            return self.rotate(rot).sum(dim=1)

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


def translate_img_real(img, t):
    '''
    img: BxYxX real space image
    t: Bx2 shift in pixels
    order > 1 can give interpolation artifacts for step-like signals (so does phase shifting in fourier space)
    '''
    img_shifted = ndimage.shift(img, t, order=1, mode='grid-wrap', prefilter=True)
    return img_shifted


def plot_projections(out_png, imgs):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    axes = axes.ravel()
    for i in range(min(len(imgs),9)):
        axes[i].imshow(imgs[i])
    plt.savefig(out_png)


def main(args):
    if os.path.exists(args.o):
        log('Warning: {} already exists. Overwriting.'.format(args.o))

    use_cuda = torch.cuda.is_available()
    log('Use cuda {}'.format(use_cuda))
    if use_cuda:
        device = torch.device('cuda')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type(torch.FloatTensor)

    vol, _ = mrc.parse_mrc(args.mrc)
    log('Loaded {} volume'.format(vol.shape))

    if args.is_vol:
        is_mask = False
        log('Treating input volume as density map and projecting sum of voxels')
    else:
        assert args.is_mask
        is_mask = True
        log('Treating input volume as mask and projecting max of voxels')

    projector = Projector(vol, is_mask = is_mask)
    poses = Poses(args.poses)
    log('Generating {} rotations from {}'.format(len(poses), args.poses))
    imgs = np.empty((poses.N, vol.shape[0], vol.shape[0]), dtype=np.float32)

    log('Projecting volume according to input poses...')
    t1 = time.time()
    projector.lattice = projector.lattice.to(device)
    projector.vol = projector.vol.to(device)
    iterator = data.DataLoader(poses, batch_size=args.b)
    for i, rot in enumerate(iterator):
        vlog('Projecting {}/{}'.format((i+1)*len(rot), poses.N))
        projections = projector.project(rot)
        projections = projections.cpu().numpy()
        imgs[i*args.b:(i+1)*args.b] = projections
    tp = time.time()
    log('Projected {} images in {}s ({}s per image)'.format(poses.N, tp-t1, (tp-t1)/poses.N ))

    log('Shifting images according to input poses...')
    # convention: we want the first column to be x shift and second column to be y shift
    # reverse columns since current implementation of translate_img uses scipy's fourier_shift, which is flipped the other way
    D = imgs.shape[-1]
    assert D % 2 == 0
    trans = poses.trans*D # convert from fraction of box to pixels
    trans = -trans[:,::-1] # convention for scipy
    for i, (img, t) in enumerate(zip(imgs, trans)):
        imgs[i] = translate_img_real(img, t)
    tt = time.time()
    log('Translated {} images in {}s ({}s per image)'.format(poses.N, tt-tp, (tt-tp)/poses.N))

    if is_mask:
        log('Normalizing each image to [0,1]...')
        for i, img in enumerate(imgs):
            imgs[i] = (imgs[i] - imgs[i].min()) / (imgs[i].max() - imgs[i].min())

    log(f'Saving {args.o}')
    mrc.write(args.o, imgs, is_vol=False)

    if args.out_png:
        log('Saving {}'.format(args.out_png))
        plot_projections(args.out_png, imgs[:9])

if __name__ == '__main__':
    args = parse_args().parse_args()
    main(args)
