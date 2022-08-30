'''
Downsample an image stack or volume by clipping fourier frequencies
Recommended to re-extract particles at different box size in Warp rather than use this script
'''

import argparse
import numpy as np
import os
import math
import torch
from torch.utils import data
from torch.utils.data import DataLoader

from tomodrgn import utils
from tomodrgn import mrc
from tomodrgn import fft
from tomodrgn import dataset

log = utils.log

def add_args(parser):
    parser.add_argument('mrcs', help='Input particles or volume (.mrc, .mrcs, .star, or .txt)')
    parser.add_argument('-D', type=int, required=True, help='New box size in pixels, must be even')
    parser.add_argument('-o', metavar='MRCS', type=os.path.abspath, required=True, help='Output projection stack (.mrcs)')
    parser.add_argument('-b', type=int, default=5000, help='Batch size for processing images')
    parser.add_argument('--is-vol',action='store_true', help='Flag if input .mrc is a volume')
    parser.add_argument('--chunk', type=int, help='Chunksize (in # of images) to split particle stack when loading and saving if stack too large for system memory')
    parser.add_argument('--lazy', action='store_true', help='Lazily load images on the fly if stack too large for system memory')
    parser.add_argument('--relion31', action='store_true', help='Flag for relion3.1 star format')
    parser.add_argument('--datadir', help='Optionally provide path to input .mrcs if loading from a .star file')
    return parser

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def warnexists(out):
    if os.path.exists(out):
        log('Warning: {} already exists. Overwriting.'.format(out))

class ImageDataset(data.Dataset):
    '''
    Quick dataset class to shovel particle images into pytorch dataloader
    Benefit = more parallelized FFT computations using GPU
    '''
    def __init__(self, particles, N):
        self.particles = particles
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if type(self.particles[index]) is mrc.LazyImage:
            return index, self.particles[index].get()
        else:
            return index, self.particles[index]


def main(args):
    log(args)

    mkbasedir(args.o)
    warnexists(args.o)
    assert (args.o.endswith('.mrcs') or args.o.endswith('mrc')), "Must specify output in .mrc(s) file format"

    use_cuda =  torch.cuda.is_available()
    torch.set_grad_enabled(False)
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        log(f'Using GPU for downsampling')

    log(f'Loading {args.mrcs}')
    lazy = args.lazy
    old = dataset.load_particles(args.mrcs, lazy=lazy, datadir=args.datadir, relion31=args.relion31)
    N = len(old)

    oldD = old[0].get().shape[0] if lazy else old.shape[-1]
    newD = args.D
    assert newD <= oldD, f'New box size {newD} cannot be larger than the original box size {oldD}'
    assert newD % 2 == 0, 'New box size must be even'

    start = int(oldD/2 - newD/2)
    stop = int(oldD/2 + newD/2)

    def downsample_images(batch_original, start, stop):
        batch_original = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(batch_original, dim=(-1, -2))), dim=(-1, -2))
        batch_original = batch_original.real - batch_original.imag

        batch_new = batch_original[:, start:stop, start:stop]

        batch_new = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(batch_new, dim=(-1, -2))), dim=(-1, -2))
        batch_new /= (batch_new.shape[-1] * batch_new.shape[-2])
        return batch_new.real - batch_new.imag

    log('Downsampling...')
    ### Downsample volume ###
    if args.is_vol:
        oldft = fft.htn_center(old)
        log(oldft.shape)
        newft = oldft[start:stop,start:stop,start:stop]
        log(newft.shape)
        new = fft.ihtn_center(newft).astype(np.float32)
        log(f'Saving {args.o}')
        mrc.write(args.o, new, is_vol=True)

    ### Downsample images ###
    elif args.chunk is None:
        new = np.empty((N, newD, newD), dtype=np.float32)

        particle_dataset = ImageDataset(old, N)
        data_generator = DataLoader(particle_dataset, batch_size=args.b, shuffle=False)
        for batch_idx, batch_ptcls in data_generator:
            log(f'Processing indices {batch_idx[0]} - {batch_idx[-1]}')
            batch_new = downsample_images(batch_ptcls, start, stop)
            new[batch_idx.cpu().numpy()] = batch_new.cpu().numpy()

        log(new.shape)
        log('Saving {}'.format(args.o))
        mrc.write(args.o, new.astype(np.float32), is_vol=False)

    ### Downsample images, saving chunks of N images ###
    else:
        assert args.b <= args.chunk
        nchunks = math.ceil(len(old)/args.chunk)
        out_mrcs = ['.{}'.format(i).join(os.path.splitext(args.o)) for i in range(nchunks)]
        chunk_names = [os.path.basename(x) for x in out_mrcs]
        for i in range(nchunks):
            log(f'Processing chunk {i}')
            chunk = old[i*args.chunk: (i+1)*args.chunk]
            new = np.empty((len(chunk), newD, newD), dtype=np.float32)

            particle_dataset = ImageDataset(chunk, len(chunk))
            data_generator = DataLoader(particle_dataset, batch_size=args.b, shuffle=False)
            for batch_idx, batch_ptcls in data_generator:
                batch_new = downsample_images(batch_ptcls, start, stop)
                new[batch_idx.cpu().numpy()] = batch_new.cpu().numpy()

            log(new.shape)
            log(f'Saving {out_mrcs[i]}')
            mrc.write(out_mrcs[i], new, is_vol=False)

        # Write a text file with all chunks
        out_txt = '{}.txt'.format(os.path.splitext(args.o)[0])
        log(f'Saving {out_txt}')
        with open(out_txt,'w') as f:
            f.write('\n'.join(chunk_names))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
