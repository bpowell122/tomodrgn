"""
Classes and functions to retrieve and optionally update image poses (defined as rotations and translations) via neural networks backpropogation.
"""
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from tomodrgn import lie_tools, utils, starfile


class PoseTracker(nn.Module):
    """
    A class to track and optionally update rotations (and optionally translations) describing the alignments of a stack of 2-D projection images in relation to a 3-D reference frame.
    """
    def __init__(self,
                 rots_np: np.ndarray,
                 trans_np: np.ndarray | None = None,
                 boxsize: int = None,
                 emb_type: Literal['s2s2', 'quat'] | None = None):

        # initialize the parent class
        super().__init__()

        # assign essential class attributes
        rots = torch.tensor(rots_np).float()
        trans = torch.tensor(trans_np).float() if trans_np is not None else None
        self.rots = rots
        self.trans = trans
        self.use_trans = trans_np is not None
        self.boxsize = boxsize

        # define the type of embedding used to encode rotations and translations
        self.emb_type = emb_type
        if emb_type is None:
            # no embedding used; individual rotations are returned as fixed 3x3 rotation matrices, translations are returned as fixed 1x2 offsets
            self.rots_emb = None
            self.trans_emb = None
        else:
            # embeddings used: translations can be returned from learnable nn.Embedding lookup table mapping 1x2 offset to 2-D embedding
            if self.use_trans:
                trans_emb = nn.Embedding(num_embeddings=trans.shape[0], embedding_dim=2, sparse=True)
                trans_emb.weight.data.copy_(trans)
            else:
                trans_emb = None

            # embeddings used: rotations can be returned from learnable nn.Embedding lookup table mapping 3x3 rotation matrix to 6-D or 4-D embedding (s2s2 or quat)
            if emb_type == 's2s2':
                rots_emb = nn.Embedding(num_embeddings=rots.shape[0], embedding_dim=6, sparse=True)
                rots_emb.weight.data.copy_(lie_tools.SO3_to_s2s2(rots))
            elif emb_type == 'quat':
                rots_emb = nn.Embedding(num_embeddings=rots.shape[0], embedding_dim=4, sparse=True)
                rots_emb.weight.data.copy_(lie_tools.SO3_to_quaternions(rots))
            else:
                raise RuntimeError(f'Embedding type {emb_type} not recognized')

            self.rots_emb = rots_emb
            self.trans_emb = trans_emb

    @classmethod
    def load(cls,
             star: starfile.TiltSeriesStarfile,
             boxsize: int,
             emb_type: Literal['s2s2', 'quat'] | None = None):
        """
        Return a PoseTracker instance given a particle imageseries star file.

        :param star: loaded and pre-filtered imageseries star file
        :param boxsize: box size of input images in pixels
        :param emb_type: type of embedding for SO(3) rotation matrices if refining poses
        :return: PoseTracker instance
        """
        # rotations
        euler = star.df[star.headers_rot].to_numpy()
        rot = np.asarray([utils.rot_3d_from_relion(*x) for x in euler])

        # parse translations (if present)
        if all(header_trans in star.df.columns for header_trans in star.headers_trans):
            trans = star.df[star.headers_trans].to_numpy()
        else:
            trans = None

        return cls(rots_np=rot,
                   trans_np=trans,
                   boxsize=boxsize,
                   emb_type=emb_type)

    def save(self,
             out_pkl: str) -> None:
        """
        Write poses to a pickle file.
        Particularly useful if learning poses (i.e., emb_type is not None).
        Writes poses as tuple of arrays: (euler1, euler2, euler3) in units of degrees, and (shift_x, shift_y) in units of box size fraction

        :param out_pkl: path of output pkl file to write
        :return: None
        """
        if self.emb_type == 'quat':
            r = lie_tools.quaternions_to_SO3(self.rots_emb.weight.data).cpu().numpy()
        elif self.emb_type == 's2s2':
            r = lie_tools.s2s2_to_SO3(self.rots_emb.weight.data).cpu().numpy()
        else:
            r = self.rots.cpu().numpy()

        if self.use_trans:
            if self.emb_type is None:
                t = self.trans.cpu().numpy()
            else:
                t = self.trans_emb.weight.data.cpu().numpy()
            t /= self.boxsize  # convert from pixels to extent
            poses = (r, t)
        else:
            poses = (r,)

        utils.save_pkl(data=poses, out_pkl=out_pkl)

    def get_pose(self,
                 ind: int | np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the (optionally embedded) rotations and translations of the images at specified indices

        :param ind: index or indices (numpy array or torch tensor) of images for which to return rotations and translations
        :return: rotations as 3x3 rotation matrices shape(len(ind), 3, 3); rotations as 1x2 translation matrices (shape len(ind), 2) or None
        """
        if self.emb_type is None:
            rot = self.rots[ind]
            tran = self.trans[ind] if self.use_trans else None
        elif self.emb_type == 's2s2':
            rot = lie_tools.s2s2_to_SO3(self.rots_emb(ind))
            tran = self.trans_emb(ind) if self.use_trans else None
        elif self.emb_type == 'quat':
            rot = lie_tools.quaternions_to_SO3(self.rots_emb(ind))
            tran = self.trans_emb(ind) if self.use_trans else None
        else:
            raise RuntimeError  # should not reach here
        return rot, tran
