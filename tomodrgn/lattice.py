"""
Classes and functions for interfacing with 2-D voxel grids representing a lattice of voxel coordinates.
"""

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
        self.boxcenter = int(boxsize / 2)  # coords center in 0-indexed space, boxsize=65 should give boxcenter of 32
        self.center = torch.tensor([0., 0.], device=device)  # coords center in coordinate space
        self.ignore_dc = ignore_dc
        self.device = device

        # create dictionaries to cache computation of masks with given radius / sidelength
        self.square_mask = {}
        self.circle_mask = {}

        # precalculate values used each time we compute the CTF for an image
        self.freqs2d = self.coords[:, 0:2] / extent / 2  # spatial frequencies at each lattice point normalized to scale from -0.5 to 0.5 (Nyquist)
        self.freqs2d_s2 = (self.freqs2d[:, 0] ** 2 + self.freqs2d[:, 1] ** 2).view(1, -1)  # spatial frequency magnitude at each lattice point, dim0 of length 1 for broadcasting across multiple images
        self.freqs2d_angle = torch.atan2(self.freqs2d[:, 1], self.freqs2d[:, 0]).view(1, -1)  # spatial frequency angle from x axis at each lattice point

    def get_downsample_coords(self,
                              boxsize_new: int) -> torch.Tensor:
        """
        Return a 2-D lattice of coordinates representing a downsampled (fourier-cropped) copy of the original lattice.
        :param boxsize_new: number of grid points along each dimension in the downsampled lattice. Should be odd.
        :return: coordinates of the downsampled lattice, shape (boxsize_new ** 2, 3)
        """
        # sanity check inputs
        assert boxsize_new % 2 == 1
        assert boxsize_new < self.boxsize

        # calculate the range of the original lattice to keep after cropping
        ind_low = self.boxcenter - boxsize_new // 2
        ind_high = self.boxcenter + boxsize_new // 2 + 1

        # crop the lattice
        downsample_coords = self.coords.view(self.boxsize, self.boxsize, 3)[ind_low:ind_high, ind_low: ind_high, :].contiguous().view(-1, 3)

        return downsample_coords

    def get_square_mask(self,
                        sidelength: int) -> torch.Tensor:
        """
        Return a binary mask for self.coords which restricts coordinates to a centered square lattice
        :param sidelength: number of grid points to include in the mask along each dimension
        :return: binary mask, shape (lattice.boxsize ** 2)
        """
        # return precomputed result if cached
        if sidelength in self.square_mask:
            return self.square_mask[sidelength]

        # sanity check inputs
        assert sidelength <= self.boxsize, f'Square mask with side length ({sidelength=}) too large for lattice with size {self.boxsize}'
        assert sidelength % 2 == 1, f'Square mask with side length ({sidelength=}) must be an odd integer to be centered on odd-length lattice'

        # calculate the range of the original lattice to keep after masking
        ind_low = self.boxcenter - sidelength // 2
        ind_high = self.boxcenter + sidelength // 2 + 1

        # create the mask as a flattened array
        mask = torch.ones((self.boxsize, self.boxsize), dtype=torch.bool, device=self.device)
        mask[:, :ind_low] = 0
        mask[:, ind_high:] = 0
        mask[:ind_low, :] = 0
        mask[ind_high:, :] = 0
        if self.ignore_dc:
            mask[self.boxcenter, self.boxcenter] = 0
        mask = mask.view(-1)

        # cache the result
        self.square_mask[sidelength] = mask

        return mask

    def get_circular_mask(self,
                          diameter: int) -> torch.Tensor:
        """
        Return a binary mask for self.coords which restricts coordinates to a centered circular lattice
        :param diameter: number of grid points to include in the mask along each dimension
        :return: binary mask, shape (lattice.boxsize ** 2)
        """
        # return precomputed result if cached
        if diameter in self.circle_mask:
            return self.circle_mask[diameter]

        # sanity check inputs
        assert diameter <= self.boxsize, f'Circular mask with diameter {diameter} too large for lattice with size {self.boxsize}'

        # calculate the range of the original lattice to keep after masking
        mask_radius = ((diameter-1) / 2) * (self.extent / self.boxcenter)

        # create the mask as a flattened array
        mask = self.coords.pow(2).sum(-1) <= mask_radius ** 2

        # ignore the DC coordinate which is the center in the coords array
        if self.ignore_dc:
            ind_dc_flattened_coords = self.boxsize ** 2 // 2
            assert self.coords[ind_dc_flattened_coords].sum() == 0.0
            mask[ind_dc_flattened_coords] = 0

        # cache the result
        self.circle_mask[diameter] = mask

        return mask

    def rotate(self,
               images: torch.Tensor,
               theta: torch.Tensor) -> torch.Tensor:
        """
        Resample a stack of images on the lattice grid rotated in-plane counterclockwise by a batch of theta angles.
        :param images: stack of images to rotate, shape (B,Y,X)
        :param theta: batch of angles in radians, shape (Q)
        :return: rotated images, shape (B,Q,Y,X)
        """
        # prepare the (transposed) rotation matrix for multiplication from the right
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        rot = torch.stack([cos, sin, -sin, cos], 1).view(-1, 2, 2)

        # rescale the lattice from (-0.5, 0.5) to (-1, 1) and rotate all lattice points counterclockwise by theta radians
        grid = self.coords[:, 0:2] / self.extent @ rot  # shape (Q, Y*X, 2)
        grid = grid.view(len(rot), self.boxsize, self.boxsize, 2)  # shape (Q, Y, X, 2)

        # correct for translational offsets
        offset = self.center - grid[:, self.boxcenter, self.boxcenter]  # shape (Q, 2)
        grid += offset[:, None, None, :]

        # resample the images on the rotated grids
        images = images.expand(len(theta), *images.shape)  # shape (Q, B, Y, X)
        # use "border" padding rather than "0" padding to avoid introducing abrupt level transitions for grid corners that are rotated out of input range (-1, 1)
        rotated = F.grid_sample(images, grid, padding_mode='border')  # shape (Q, B, Y, X)

        return rotated.transpose(0, 1)  # shape (B, Q, Y, X)

    def translate_ft(self,
                     images: torch.Tensor,
                     trans: torch.Tensor,
                     mask: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        """
        Translate an image by phase shifting its Fourier transform.
        `F'(k) = exp(-2*pi*k*x0)*F(k)`
        Note that shape img_dims below can either be 2D or 1D (unraveled image)
        :param images: fourier transform of image, shape (B, img_dims, 2)
        :param trans: shift in pixels, shape (B, T, 2)
        :param mask: optional mask for lattice coords (img_dims, 1)
        :return: translated images, shape (B, T, img_dims, 2)
        """
        # if input images are masked, the same mask must be applied to associated grid coordinates
        coords = self.freqs2d if mask is None else self.freqs2d[mask]

        # apply the translations to the grid coordinates via batched matrix multiplication of the translation matrices
        trans = trans.unsqueeze(-1)  # BxTx2x1 to be able to do bmm
        tfilt = coords @ trans * -2 * np.pi  # BxTxNx1
        tfilt = tfilt.squeeze(-1)  # BxTxN

        # calculate and apply phase shift to the images
        c = torch.cos(tfilt)  # BxTxN
        s = torch.sin(tfilt)  # BxTxN
        images_flattened = images.unsqueeze(1)  # Bx1xNx2
        return torch.stack([images_flattened[..., 0] * c - images_flattened[..., 1] * s,
                            images_flattened[..., 0] * s + images_flattened[..., 1] * c], -1)

    def translate_ht(self,
                     images: torch.Tensor,
                     trans: torch.Tensor,
                     mask: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        """
        Translate an image by phase shifting its Hartley transform
        `H'(k) = cos(2*pi*k*t0)H(k) + sin(2*pi*k*t0)H(-k)`
        img must be 1D unraveled image, symmetric around DC component
        :param images: hartley transform of image, shape (B, boxsize_ht)
        :param trans: shift in pixels, shape (B, T, 2)
        :param mask: mask for lattice coords, shape (boxsize_ht, 1)
        :return: translated images, shape (B, T, boxsize_ht)
        """
        # if input images are masked, the same mask must be applied to associated grid coordinates
        coords = self.freqs2d if mask is None else self.freqs2d[mask]

        # apply the translations to the grid coordinates via batched matrix multiplication of the translation matrices
        trans = trans.unsqueeze(-1)  # BxTx2x1 to be able to do bmm
        tfilt = coords @ trans * 2 * np.pi  # BxTxNx1
        tfilt = tfilt.squeeze(-1)  # BxTxN

        # calculate and apply phase shift to the images
        c = torch.cos(tfilt)  # BxTxN
        s = torch.sin(tfilt)  # BxTxN
        images = images.unsqueeze(1)  # Bx1xN
        return c * images + s * images[:, :, np.arange(len(coords) - 1, -1, -1)]


class EvenLattice:

    def __init__(self,
                 boxsize: int,
                 extent: float = 0.5,
                 ignore_dc: bool = False,
                 device: torch.device | None = None):
        """
        Class for handling a 2-D voxel grid with an even number of points along each dimension.
        Grid is centered at `(0,0)` and runs from `-extent` (inclusive) to `+extent` (exclusive) with `boxsize` points.
        :param boxsize: number of grid points along each dimension. Should be even.
        :param extent: maximum value of the grid along each dimension, typically <= 0.5
        :param ignore_dc: whether to exclude the DC component (0, 0) when generating masks via methods of this class.
        :param device: torch device on which to store tensor attributes and return tensors from methods of this class.
        """

        # sanity check inputs
        assert boxsize % 2 == 0, "Lattice size must be even"
        if ignore_dc:
            raise NotImplementedError

        # create the lattice of 2-D points along the X-Y plane in 3-D space
        # centered and scaled xy plane, values between -1 and 1
        # endpoint=False since FT is not symmetric around origin
        x0, x1 = np.meshgrid(np.linspace(-1, 1, boxsize, endpoint=False),
                             np.linspace(-1, 1, boxsize, endpoint=False))
        coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(boxsize ** 2)], 1).astype(np.float32)
        self.coords = torch.tensor(coords, device=device)

        # additional attributes about the lattice
        self.extent = extent
        self.boxsize = boxsize
        self.boxcenter = int(boxsize / 2)  # coords center in 0-indexed space, boxsize=64 should give boxcenter of 32
        self.center = torch.tensor([0., 0.], device=device)  # coords center in coordinate space
        self.ignore_dc = ignore_dc
        self.device = device

        # create dictionaries to cache computation of masks with given radius / sidelength
        self.square_mask = {}
        self.circle_mask = {}

    def get_downsampled_coords(self, d):
        raise NotImplementedError
