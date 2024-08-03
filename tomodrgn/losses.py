"""
Classes and functions for handling model loss calculations
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tomodrgn.models import TiltSeriesHetOnlyVAE


class EquivarianceLoss(nn.Module):
    """
    Equivariance loss for encoder over SO(2) subgroup.
    """

    def __init__(self,
                 model: TiltSeriesHetOnlyVAE,
                 boxsize: int):
        super().__init__()
        self.model = model
        self.boxsize = boxsize

    def forward(self,
                batch_images: torch.Tensor,
                batch_encoding: torch.Tensor) -> torch.Tensor:
        """
        Rotate each image by a random theta, encode the rotated image, and compute difference in latent encoding versus unrotated image.
        :param batch_images: batch of images to be rotated and encoded, shape (batchsize, ntilts, boxsize_ht, boxsize_ht)
        :param batch_encoding: encodings from corresponding non-rotated images, shape (batchsize, zdim)
        :return: MSE difference in latent encodings, shape (1)
        """
        # get a random rotation angle for each image in the batch
        batchsize, ntilts, boxsize_ht, boxsize_ht = batch_images.shape
        batch_images = batch_images.view(batchsize * ntilts, boxsize_ht, boxsize_ht)
        theta = torch.rand(batchsize * ntilts) * 2 * np.pi

        # rotate each image by the corresponding random angle in-plane
        batch_images = torch.unsqueeze(batch_images, 1)
        batch_images_rot = self.rotate(batch_images, theta)
        batch_images_rot = torch.squeeze(batch_images_rot)

        # encode the rotated images
        batch_images_rot_enc = self.model.encode(batch_images_rot)[0]

        # compute the MSE between input unrotated image encodings and rotated image encodings
        diffs = (batch_encoding - batch_images_rot_enc).pow(2).view(batchsize, -1).sum(-1)
        return diffs.mean()

    def rotate(self,
               batch_images: torch.Tensor,
               batch_thetas: torch.Tensor) -> torch.Tensor:
        """
        Rotate each image by theta radians counterclockwise.
        :param batch_images: batch of images to be rotated, shape (B, 1, X, Y)
        :param batch_thetas: batch of theta angles corresponding to each image, shape (B)
        :return: batch of rotated images, shape (B, 1, X, Y)
        """
        # prepare the (transposed) rotation matrix for multiplication from the right
        cos = torch.cos(batch_thetas)
        sin = torch.sin(batch_thetas)
        rot_transposed = torch.stack([cos, sin, -sin, cos], 1).view(-1, 2, 2)

        # rotate all lattice points counterclockwise by theta radians
        grid = self.model.lattice.coords[:, 0:2] @ rot_transposed
        grid = grid.view(-1, self.D, self.D, 2)

        # use "border" padding rather than "0" padding to avoid introducing abrupt level transitions for grid corners that are rotated out of input range (-1, 1)
        return F.grid_sample(batch_images, grid, padding_mode='border')
