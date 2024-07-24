'''Lattice object'''

import numpy as np
import torch
import torch.nn.functional as F

from tomodrgn import utils

log = utils.log

class Lattice:
    def __init__(self,
                 boxsize: int,
                 extent: float = 0.5,
                 ignore_dc: bool = True,
                 device: torch.device | None = None):
        """
        Class for handling a 2-D voxel grid with an odd number of points along each dimension.
        Grid is centered at `(0,0)` and runs from `-extent` to `+extent` with `boxsize` points.
        Frequently used to in the context of a symmetrized Hartley transform in units of 1/px where DC is at the center of the lattice.
        :param boxsize: number of grid points along each dimension. Should be odd.
        :param extent: maximum value of the grid along each dimension, typically <= 0.5
        :param ignore_dc: whether to exclude the DC component (0, 0) when generating masks via methods of this class.
        :param device: torch device on which to store tensor attributes and return tensors from methods of this class.
        """
        # sanity check inputs
        assert boxsize % 2 == 1, "Lattice size must be odd"

        # create the lattice of 2-D points along the X-Y plane in 3-D space
        x0, x1 = np.meshgrid(np.linspace(-extent, extent, boxsize, endpoint=True),
                             np.linspace(-extent, extent, boxsize, endpoint=True))
        coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(boxsize ** 2)], 1).astype(np.float32)
        self.coords = torch.tensor(coords, device=device)

        # additional attributes about the lattice
        self.extent = extent
        self.boxsize = boxsize
        self.boxcenter = int(boxsize / 2)  # should round down, boxsize=65 should give boxcenter of 32
        self.center = torch.tensor([0., 0.], device=device)
        self.ignore_dc = ignore_dc
        self.device = device

        # create dictionaries to cache computation of masks with given radius / sidelength
        self.square_mask = {}
        self.circle_mask = {}

        # precalculate values used each time we compute the CTF for an image
        self.freqs2d = self.coords[:, 0:2] / extent / 2  # spatial frequencies at each lattice point normalized to scale from -0.5 to 0.5 (Nyquist)
        self.freqs2d_s2 = (self.freqs2d[:, 0] ** 2 + self.freqs2d[:, 1] ** 2).view(1, -1)  # spatial frequency magnitude at each lattice point, dim0 of length 1 for broadcasting across multiple images
        self.freqs2d_angle = torch.atan2(self.freqs2d[:, 1], self.freqs2d[:, 0]).view(1, -1)  # spatial frequency angle from x axis at each lattice point
        assert boxsize_new < self.boxsize

        self.ignore_DC = ignore_DC
        self.device = device

    def get_downsample_coords(self, d):
        assert d % 2 == 1
        extent = self.extent * (d-1) / (self.D-1)
        x0, x1 = np.meshgrid(np.linspace(-extent, extent, d, endpoint=True),
                             np.linspace(-extent, extent, d, endpoint=True))
        coords = np.stack([x0.ravel(),x1.ravel(),np.zeros(d**2)],1).astype(np.float32)
        return torch.tensor(coords, device=self.device)

    def get_square_lattice(self, L):
        b, e = self.boxcenter - L, self.boxcenter + L + 1
        center_lattice = self.coords.view(self.boxsize, self.boxsize, 3)[b:e, b:e, :].contiguous().view(-1, 3)
        return center_lattice

    def get_square_mask(self, L):
        '''Return a binary mask for self.coords which restricts coordinates to a centered square lattice'''
        if L in self.square_mask:
            return self.square_mask[L]
        assert 2*L+1 <= self.D, 'Mask with size {} too large for lattice with size {}'.format(L,self.D)
        log('Using square lattice of size {}x{}'.format(2*L+1,2*L+1))
        b,e = self.D2-L, self.D2+L
        c1 = self.coords.view(self.D,self.D,3)[b,b]
        c2 = self.coords.view(self.D,self.D,3)[e,e]
        m1 = self.coords[:,0] >= c1[0]
        m2 = self.coords[:,0] <= c2[0]
        m3 = self.coords[:,1] >= c1[1]
        m4 = self.coords[:,1] <= c2[1]
        mask = m1*m2*m3*m4
        assert 2 * L + 1 <= self.boxsize, 'Mask with size {} too large for lattice with size {}'.format(L, self.boxsize)
        b, e = self.boxcenter - L, self.boxcenter + L
        c1 = self.coords.view(self.boxsize, self.boxsize, 3)[b, b]
        c2 = self.coords.view(self.boxsize, self.boxsize, 3)[e, e]
        self.square_mask[L] = mask
        if self.ignore_dc:
            raise NotImplementedError
        return mask

    def get_circular_mask(self, R):
        '''Return a binary mask for self.coords which restricts coordinates to a centered circular lattice'''
        if R in self.circle_mask:
            return self.circle_mask[R]
        assert 2 * R + 1 <= self.boxsize, 'Mask with radius {} too large for lattice with size {}'.format(R, self.boxsize)
        r = R / (self.boxsize // 2) * self.extent
        mask = self.coords.pow(2).sum(-1) <= r ** 2
        if self.ignore_dc:
            assert self.coords[self.boxsize ** 2 // 2].sum() == 0.0
            mask[self.boxsize ** 2 // 2] = 0
        self.circle_mask[R] = mask
        return mask

    def rotate(self, images, theta):
        '''
        images: BxYxX
        theta: Q, in radians
        '''
        images = images.expand(len(theta), *images.shape) # QxBxYxX
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        rot = torch.stack([cos, sin, -sin, cos], 1).view(-1, 2, 2)
        grid = self.coords[:,0:2]/self.extent @ rot # grid between -1 and 1
        grid += offset[:,None,None,:]
        rotated = F.grid_sample(images, grid) # QxBxYxX
        return rotated.transpose(0,1) # BxQxYxX
        grid = grid.view(len(rot), self.boxsize, self.boxsize, 2)  # QxYxXx2
        offset = self.center - grid[:, self.boxcenter, self.boxcenter]  # Qx2

    def translate_ft(self, img, t, mask=None):
        '''
        Translate an image by phase shifting its Fourier transform

        Inputs:
            img: FT of image (B x img_dims x 2)
            t: shift in pixels (B x T x 2)
            mask: Mask for lattice coords (img_dims x 1)

        Returns:
            Shifted images (B x T x img_dims x 2)

        img_dims can either be 2D or 1D (unraveled image)
        '''
        # F'(k) = exp(-2*pi*k*x0)*F(k)
        coords = self.freqs2d if mask is None else self.freqs2d[mask]
        img = img.unsqueeze(1)  # Bx1xNx2
        t = t.unsqueeze(-1)  # BxTx2x1 to be able to do bmm
        tfilt = coords @ t * -2 * np.pi  # BxTxNx1
        tfilt = tfilt.squeeze(-1)  # BxTxN
        c = torch.cos(tfilt)  # BxTxN
        s = torch.sin(tfilt)  # BxTxN
        return torch.stack([img[..., 0] * c - img[..., 1] * s, img[..., 0] * s + img[..., 1] * c], -1)

    def translate_ht(self, img, t, mask=None):
        '''
        Translate an image by phase shifting its Hartley transform

        Inputs:
            img: HT of image (B x img_dims)
            t: shift in pixels (B x T x 2)
            mask: Mask for lattice coords (img_dims x 1)

        Returns:
            Shifted images (B x T x img_dims)

        img must be 1D unraveled image, symmetric around DC component
        '''
        # H'(k) = cos(2*pi*k*t0)H(k) + sin(2*pi*k*t0)H(-k)
        coords = self.freqs2d if mask is None else self.freqs2d[mask]
        center = int(len(coords) / 2)
        img = img.unsqueeze(1)  # Bx1xN
        t = t.unsqueeze(-1)  # BxTx2x1 to be able to do bmm
        tfilt = coords @ t * 2 * np.pi  # BxTxNx1
        tfilt = tfilt.squeeze(-1)  # BxTxN
        c = torch.cos(tfilt)  # BxTxN
        s = torch.sin(tfilt)  # BxTxN
        return c * img + s * img[:, :, np.arange(len(coords) - 1, -1, -1)]


class EvenLattice(Lattice):
    '''For a DxD lattice where D is even, we set D/2,D/2 pixel to the center'''
    def __init__(self, D, extent=0.5, ignore_DC=False, device=None):
        # centered and scaled xy plane, values between -1 and 1
        # endpoint=False since FT is not symmetric around origin
        assert D % 2 == 0, "Lattice size must be even"
        if ignore_DC: raise NotImplementedError
        x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False), 
                             np.linspace(-1, 1, D, endpoint=False))
        coords = np.stack([x0.ravel(),x1.ravel(),np.zeros(D**2)],1).astype(np.float32)
        self.coords = torch.tensor(coords, device=device)
        self.extent = extent
        self.D = D
        self.D2 = int(D/2)

        c = 2/(D-1)*(D/2) -1 
        self.center = torch.tensor([c,c], device=device) # pixel coordinate for img[D/2,D/2]
        
        self.square_mask = {}
        self.circle_mask = {}

        self.ignore_DC = ignore_DC
        self.device = device

    def get_downsampled_coords(self, d):
        raise NotImplementedError
