"""
Classes for creating, loading, training, and evaluating pytorch models.
"""

import itertools
import os
from multiprocessing import Pool
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torchinfo
from einops import repeat, pack
from einops.layers.torch import Reduce, Rearrange
from torch.amp import autocast

from tomodrgn import utils, lattice, set_transformer, fft, mrc

log = utils.log


class TiltSeriesHetOnlyVAE(nn.Module):
    # TODO move sequential tilt sampling from dataset.__getitem__ to encoder A -> B transition
    # TODO add torch.compile here and to other common functions (calc ctf? eval vol batch? ffts?) https://discuss.pytorch.org/t/how-should-i-use-torch-compile-properly/179021
    """
    A module to encode multiple tilt images of a particle to a learned low-dimensional latent space embedding,
    then decode spatial frequency coordinates to corresponding voxel amplitudes conditioned on the per-particle latent embedding.
    """

    # No pose inference
    def __init__(self,
                 in_dim: int,
                 hidden_layers_a: int,
                 hidden_dim_a: int,
                 out_dim_a: int,
                 ntilts: int,
                 hidden_layers_b: int,
                 hidden_dim_b: int,
                 zdim: int,
                 hidden_layers_decoder: int,
                 hidden_dim_decoder: int,
                 lat: lattice.Lattice,
                 activation: nn.ReLU | nn.LeakyReLU = nn.ReLU,
                 enc_mask: torch.Tensor | None = None,
                 pooling_function: Literal['concatenate', 'max', 'mean', 'median', 'set_encoder'] = 'mean',
                 feat_sigma: float = 0.5,
                 num_seeds: int = 1,
                 num_heads: int = 4,
                 layer_norm: bool = False,
                 pe_type: Literal['geom_ft', 'geom_full', 'geom_lowf', 'geom_nohighf', 'linear_lowf', 'gaussian', 'none'] = 'geom_lowf',
                 pe_dim: int | None = None):

        # initialize the parent nn.Module class
        super().__init__()

        # save attributes used elsewhere
        self.enc_mask = enc_mask
        self.in_dim = in_dim
        self.ntilts = ntilts
        self.zdim = zdim
        self.lat = lat

        # instantiate the encoder module
        self.encoder = TiltSeriesEncoder(in_dim=in_dim,
                                         hidden_layers_a=hidden_layers_a,
                                         hidden_dim_a=hidden_dim_a,
                                         out_dim_a=out_dim_a,
                                         hidden_layers_b=hidden_layers_b,
                                         hidden_dim_b=hidden_dim_b,
                                         out_dim=zdim * 2,
                                         activation=activation,
                                         ntilts=ntilts,
                                         pooling_function=pooling_function,
                                         num_seeds=num_seeds,
                                         num_heads=num_heads,
                                         layer_norm=layer_norm)

        # instantiate the decoder module
        self.decoder = FTPositionalDecoder(boxsize_ht=lat.boxsize,
                                           in_dim=3 + zdim,
                                           hidden_layers=hidden_layers_decoder,
                                           hidden_dim=hidden_dim_decoder,
                                           activation=activation,
                                           pe_type=pe_type,
                                           pe_dim=pe_dim,
                                           feat_sigma=feat_sigma)

    @classmethod
    def load(cls,
             config: str | dict,
             weights: str,
             device: torch.device = torch.device('cpu')):
        """
        Constructor method to create an TiltSeriesHetOnlyVAE object from a config.pkl.

        :param config: Path to config.pkl or loaded config.pkl
        :param weights: Path to weights.pkl
        :param device: `torch.device` object
        :return: TiltSeriesHetOnlyVAE instance, Lattice instance
        """
        # load the config dict if not preloaded
        cfg: dict = utils.load_pkl(config) if type(config) is str else config

        # create the Lattice object
        lat = lattice.Lattice(boxsize=cfg['lattice_args']['boxsize'],
                              extent=cfg['lattice_args']['extent'],
                              device=device)

        # create the TiltSeriesHetOnlyVAE object
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[cfg['model_args']['activation']]
        model = TiltSeriesHetOnlyVAE(in_dim=cfg['model_args']['in_dim'],
                                     hidden_layers_a=cfg['model_args']['qlayersA'],
                                     hidden_dim_a=cfg['model_args']['qdimA'],
                                     out_dim_a=cfg['model_args']['out_dimA'],
                                     ntilts=cfg['model_args']['ntilts'],
                                     hidden_layers_b=cfg['model_args']['qlayersB'],
                                     hidden_dim_b=cfg['model_args']['qdimB'],
                                     zdim=cfg['model_args']['zdim'],
                                     hidden_layers_decoder=cfg['model_args']['players'],
                                     hidden_dim_decoder=cfg['model_args']['pdim'],
                                     lat=lat,
                                     activation=activation,
                                     enc_mask=cfg['model_args']['enc_mask'],
                                     pooling_function=cfg['model_args']['pooling_function'],
                                     feat_sigma=cfg['model_args']['feat_sigma'],
                                     num_seeds=cfg['model_args']['num_seeds'],
                                     num_heads=cfg['model_args']['num_heads'],
                                     layer_norm=cfg['model_args']['layer_norm'],
                                     pe_type=cfg['model_args']['pe_type'],
                                     pe_dim=cfg['model_args']['pe_dim'], )

        # load weights if provided
        if weights is not None:
            ckpt = torch.load(weights, weights_only=True, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])

        # move the model to the requested device
        model.to(device)

        return model, lat

    def encode(self,
               batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input batch of particle's tilt images to a corresponding batch of latent embeddings.
        Input images are masked by `self.enc_mask` if provided.

        :param batch: batch of particle tilt images to encode, shape (batch, ntilts, boxsize*boxsize)
        :return: `mu`: batch of mean values parameterizing latent embedding as Gaussian, shape (batch, zdim).
                `logvar`: batch of log variance values parameterizing latent embedding as Gaussian, shape (batch, zdim).
        """
        if self.enc_mask is not None:
            batch = batch[:, :, self.enc_mask]  # B x ntilts x D*D[mask]
        z = self.encoder(batch)  # B x zdim*2

        return z[:, :self.zdim], z[:, self.zdim:]  # B x zdim

    def decode(self,
               coords: torch.Tensor,
               z: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of lattice coordinates concatenated with the corresponding latent embedding to infer the associated voxel intensities.

        :param coords: 3-D spatial frequency coordinates (e.g. from Lattice.coords) concatenated with the
                shape (batch, ntilts * boxsize_ht * boxsize_ht [mask], 3).
        :param z: latent embedding per-particle, shape (batch, zdim)
        :return: Decoded voxel intensities at the specified 3-D spatial frequencies.
        """
        return self.decoder(coords, z)

    def print_model_info(self) -> None:
        """
        Wrapper around torchinfo to display summary of model layers, input and output tensor shapes, and number of trainable parameters.
        Note that the predicted model size assumes float32 input tensors and model weights, which is an overestimate by ~2x if AMP is enabled.

        :return: None
        """
        # print the encoder module which we know input size exactly due to fixed ntilt sampling
        torchinfo.summary(self.encoder,
                          input_size=(self.ntilts, self.in_dim),
                          batch_dim=0,
                          col_names=('input_size', 'output_size', 'num_params'),
                          depth=3)
        # print the decoder module which will be a conservative overestimate of input size without lattice masking
        # this is because each particle has a potentially unique lattice mask and therefore this is an upper bound on model size
        torchinfo.summary(self.decoder,
                          input_data=[torch.rand(self.ntilts * self.lat.boxsize * self.lat.boxsize, 3) - 0.5, torch.rand(self.zdim)],
                          batch_dim=0,
                          col_names=('input_size', 'output_size', 'num_params'),
                          depth=3)


class FTPositionalDecoder(nn.Module):
    """
    A module to decode a (batch of tilts of) spatial frequency coordinates spanning (-0.5, 0.5) to the corresponding spatial frequency amplitude.
    The output may optionally be conditioned on a latent embedding `z`.
    """

    def __init__(self,
                 boxsize_ht: int,
                 in_dim: int = 3,
                 hidden_layers: int = 3,
                 hidden_dim: int = 256,
                 activation: torch.nn.ReLU | torch.nn.LeakyReLU = torch.nn.ReLU,
                 pe_type: Literal['geom_ft', 'geom_full', 'geom_lowf', 'geom_nohighf', 'linear_lowf', 'gaussian', 'none'] = 'geom_lowf',
                 pe_dim: int | None = None,
                 feat_sigma: float = 0.5):
        """
        Create the FTPositionalDecoder module.

        :param boxsize_ht: fourier-symmetrized box width in pixels (typically odd, 1px larger than input image)
        :param in_dim: number of dimensions of input coordinate lattice, typically 3 (x,y,z) + zdim
        :param hidden_layers: number of intermediate hidden layers in the decoder module
        :param hidden_dim: number of features in each hidden layer in the decoder module
        :param activation: activation function to be applied after each layer, either `torch.nn.ReLU` or `torch.nn.LeakyReLU`
        :param pe_type: the type of positional encoding to map each spatial frequency coordinate (x,y,z) to a higher dimensional representation to be passed to the decoder.
        :param pe_dim: the dimension of the higher dimensional representation of the positional encoding, typically automatically set to half of the box size to sample up to Nyquist.
        :param feat_sigma: the scale of random frequency vectors sampled from a gaussian for pe_type gaussian.
        """
        # initialize the parent nn.Module class
        super().__init__()

        # sanity check inputs
        assert in_dim >= 3

        # determine the latent dimensionality as the input dimensionality minus the expected three spatial dimensions
        self.zdim = in_dim - 3

        # commonly referenced boxsize-related constants
        self.D = boxsize_ht
        self.D2 = boxsize_ht // 2
        self.DD = 2 * (boxsize_ht // 2)

        # what type of positional encoding to use
        self.pe_type = pe_type

        # the dimensionality of the positional encoding defaults to half of the input coordinate box size
        self.pe_dim = self.D2 if pe_dim is None else pe_dim

        # the final number of features the decoder network receives as input per 3-D coordinate to evaluate
        # each of the 3 spatial frequency axes will have pe_dim features expressed as both sin and cos, finally concatenated with latent embedding
        self.in_dim = 3 * self.pe_dim * 2 + self.zdim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

        # construct the decoder ResidLinearMLP module
        self.decoder = ResidLinearMLP(in_dim=self.in_dim,
                                      nlayers=hidden_layers,
                                      hidden_dim=hidden_dim,
                                      out_dim=2,
                                      activation=activation)

        if self.pe_type == 'geom_ft':
            # option 1: 2/D to 1
            freqs = torch.arange(self.pe_dim, dtype=torch.float)
            self.freqs = self.DD * np.pi * (2. / self.DD) ** (freqs / (self.pe_dim - 1))
        elif self.pe_type == 'geom_full':
            # option 2: 2/D to 2pi
            freqs = torch.arange(self.pe_dim, dtype=torch.float)
            self.freqs = self.DD * np.pi * (1. / self.DD / np.pi) ** (freqs / (self.pe_dim - 1))
        elif self.pe_type == 'geom_lowf':
            # option 3: 2/D*2pi to 2pi
            freqs = torch.arange(self.pe_dim, dtype=torch.float)
            self.freqs = self.D2 * (1. / self.D2) ** (freqs / (self.pe_dim - 1))
        elif self.pe_type == 'geom_nohighf':
            # option 4: 2/D*2pi to 1
            freqs = torch.arange(self.pe_dim, dtype=torch.float)
            self.freqs = self.D2 * (2. * np.pi / self.D2) ** (freqs / (self.pe_dim - 1))
        elif self.pe_type == "gaussian":
            # We construct 3 * self.pe_dim random vector frequences, to match the original positional encoding:
            # In the positional encoding we produce self.pe_dim features for each of the x,y,z dimensions,
            # whereas in gaussian encoding we produce self.pe_dim features each with random x,y,z components
            #
            # Each of the random feats is the sine/cosine of the dot product of the coordinates with a frequency
            # vector sampled from a gaussian with std of feat_sigma
            freqs = torch.randn((3 * self.pe_dim, 3), dtype=torch.float) * feat_sigma * self.D2
            # make rand_feats a parameter so that it is saved in the checkpoint, but do not perform SGD on it
            self.freqs = nn.Parameter(freqs, requires_grad=False)
        elif self.pe_type == 'linear':
            # construct a linear increase in frequency, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2
            self.freqs = torch.arange(1, self.D2 + 1, dtype=torch.float)
        else:
            raise ValueError(f'Invalid pe_type {pe_type}')

    @classmethod
    def load(cls,
             config: str | dict,
             weights: str,
             device: torch.device = torch.device('cpu')):
        """
        Constructor method to create an FTPositionalDecoder object from a config.pkl.

        :param config: Path to config.pkl or loaded config.pkl
        :param weights: Path to weights.pkl
        :param device: `torch.device` object
        :return: FTPositionalDecoder instance, Lattice instance
        """
        # load the config dict if not preloaded
        cfg: dict = utils.load_pkl(config) if type(config) is str else config

        # create the Lattice object
        lat = lattice.Lattice(boxsize=cfg['lattice_args']['boxsize'],
                              extent=cfg['lattice_args']['extent'],
                              device=device)

        # create the FTPositionalDecoder object
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[cfg['model_args']['activation']]
        model = FTPositionalDecoder(boxsize_ht=cfg['lattice_args']['boxsize'],
                                    in_dim=3,
                                    hidden_layers=cfg['model_args']['players'],
                                    hidden_dim=cfg['model_args']['pdim'],
                                    activation=activation,
                                    pe_type=cfg['model_args']['pe_type'],
                                    pe_dim=cfg['model_args']['pe_dim'],
                                    feat_sigma=cfg['model_args']['feat_sigma'])

        # load weights if provided
        if weights is not None:
            ckpt = torch.load(weights, weights_only=True, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])

        # move the model to the requested device
        model.to(device)

        return model, lat

    def positional_encoding(self,
                            coords: torch.Tensor) -> torch.Tensor:
        """
        Expand coordinates in the Fourier basis with variably spaced wavelengths

        :param coords: Tensor or NestedTensor shape (batch, ntilts * boxsize_ht * boxsize_ht [mask], 3).
        :return: Positionally encoded spatial coordinates
        """
        if self.pe_type == 'gaussian':
            # expand freqs with singleton dimension along the batch dimensions, e.g. dim (1, ..., 1, n_feats, 3)
            freqs = self.freqs.view(1, 1, -1, 3)  # 1 x 1 x 3*D2 x 3
            # calculate features as torch.dot(k, freqs): compute x,y,z components of k then sum x,y,z components
            kxkykz = coords[..., None, :] * freqs  # B x N*D*D[mask] x 3*D2 x 3
            k = kxkykz.sum(-1)  # B x N*D*D[mask] x 3*D2
            s = torch.sin(k)  # B x N*D*D[mask] x 3*D2
            c = torch.cos(k)  # B x N*D*D[mask] x 3*D2
            x = torch.cat([s, c], dim=-1)  # B x N*D*D[mask] x 3*D == B x N*D*D[mask] x in_dim - zdim
        else:
            # expand freqs with singleton dimension along the batch dimensions, e.g. dim (1, ..., 1, n_feats)
            freqs = self.freqs.view(1, 1, 1, -1)  # 1 x 1 x 1 x D2
            # calculate the features as freqs scaled by coords
            k = coords[..., None] * freqs  # B x N*D*D[mask] x 3 x D2
            s = torch.sin(k)  # B x N*D*D[mask] x 3 x D2
            c = torch.cos(k)  # B x N*D*D[mask] x 3 x D2
            x = torch.cat([s, c], -1)  # B x N*D*D[mask] x 3 x D
            x = x.view(coords.size(0), -1, self.in_dim - self.zdim)  # B x N*D*D[mask] x in_dim-zdim

        return x

    @staticmethod
    def cat_z(coords: torch.Tensor,
              z: torch.Tensor) -> torch.Tensor:
        """
        Concatenate each 3-D spatial coordinate (from a particular particle's particular tilt image) with the latent embedding assigned to that particle.

        :param coords: 3-D spatial frequency coordinates at which to decode corresponding voxel intensity, possibly NestedTensor, shape (batch, ntilts * boxsize**2 [mask], self.in_dim - zdim)
        :param z: latent embedding for each particle, shape (batch, zdim)
        :return: concatenated coordinates and latent embedding tensors, possibly NestedTensor, shape (batch, ntilts * boxsize**2 [mask], self.in_dim)
        """
        # confirm coords and z have same batch size (axis over which they will eventualy be aligned)
        if z.ndim == 1:
            # coords should have batch dimension populated from positional_encoding, but z might not
            z = z.unsqueeze(0)
        assert coords.size(0) == z.size(0)
        assert z.ndim == 2

        # repeat z along a new axis corresponding to spatial frequency coordinates for each particle's images
        z = repeat(z, 'batch zdim -> batch repeat_ntilts_npixels zdim', repeat_ntilts_npixels=coords.shape[1])

        # concatenate coords with z along the last axis (coordinate + zdim value)
        coords_z, _ = pack([coords, z], 'batch ntilts_npixels *')

        return coords_z

    def forward(self,
                coords: torch.Tensor,
                z: torch.Tensor | None = None) -> torch.Tensor:
        """
        Decode a batch of lattice coordinates concatenated with the corresponding latent embedding to Hartley Transform spatial frequency amplitudes.

        :param coords: masked 3-D spatial frequency coordinates (e.g. from Lattice.coords), shape (batch, ntilts * boxsize_ht * boxsize_ht [mask], 3)
        :param z: latent embedding per-particle, shape (batch, zdim)
        :return: Decoded voxel intensities at the specified 3-D spatial frequencies.
        """
        # evaluate model on all pixel coordinates for each image
        # TODO revisit whether it is worth leveraging Hermitian symmetry to only evaluate half of each image
        full_image = self.decode(coords=coords, z=z)

        # return hartley information (FT real - FT imag) for each image
        image = full_image[..., 0] - full_image[..., 1]

        return image

    def decode(self,
               coords: torch.Tensor,
               z: torch.Tensor | None = None) -> torch.Tensor:
        """
        Decode a batch of lattice coordinates concatenated with the corresponding latent embedding to Fourier Transform spatial frequency amplitudes.

        :param coords: 3-D spatial frequency coordinates (e.g. from Lattice.coords) concatenated with the shape (batch, ntilts * boxsize_ht * boxsize_ht [mask], 3).
        :param z: latent embedding per-particle, shape (batch, zdim)
        :return: Decoded voxel intensities at the specified 3-D spatial frequencies.
        """
        # sanity check inputs coordinates are within range (-0.5, 0.5)
        assert (coords.abs() - 0.5 < 1e-4).all(), f'coords.max(): {coords.max()}; coords.min(): {coords.min()}'

        # TODO reimplement hermetian symmetry to evaluate +z coords only?

        # positionally encode x,y,z 3-D coordinate
        coords_pe = self.positional_encoding(coords)

        # concatenate each spatial coordinate with the appropriate latent embedding to be conditioned on
        if z is not None:
            coords_pe = self.cat_z(coords_pe, z)

        # evaluate the model
        result = self.decoder(coords_pe)

        return result

    def eval_volume_batch(self,
                          coords: torch.Tensor,
                          z: torch.Tensor | None,
                          extent: float) -> torch.Tensor:
        """
        Evaluate the model on 3-D volume coordinates given an optional (batch of) latent coordinate.

        :param coords: lattice coords on the x-y plane, shape (boxsize_ht**2, 3)
        :param z: latent embedding associated with the volume to decode, shape (batchsize, zdim)
        :param extent: maximum value of the grid along each dimension, typically <= 0.5 to constrain points to range (-0.5, 0.5)
        :return: batch of decoded volumes directly output from the model (Fourier space, symmetrized) with no postprocessing applied, shape (batchsize, boxsize_ht, boxsize_ht, boxsize_ht).
        """

        # sanity check inputs
        assert extent <= 0.5
        if z is not None:
            assert z.ndim == 2

        # get key array sizes
        batchsize = len(z) if z is not None else 1
        boxsize_ht = int(coords.shape[0] ** 0.5)

        # preallocate array to store batch of decoded volumes
        batch_vol_f = torch.zeros((batchsize, boxsize_ht, boxsize_ht, boxsize_ht), dtype=coords.dtype, device=coords.device)

        # evaluate the batch of volumes slice-by-slice along z axis to avoid potential memory overflows from evaluating D**3 voxels at once
        zslices = torch.linspace(-extent, extent, boxsize_ht, dtype=batch_vol_f.dtype)
        for (i, zslice) in enumerate(zslices):
            # create the z slice to evaluate
            xy_coords = coords + torch.tensor([0, 0, zslice], dtype=coords.dtype, device=coords.device)

            # only evaluate coords within `extent` radius (if extent==0.5, nyquist limit in reciprocal space)
            slice_mask = xy_coords.pow(2).sum(dim=-1) <= extent ** 2
            if torch.sum(slice_mask) == 0:
                # no coords left to evaluate after masking, skip to next zslice in next loop iteration
                continue
            xy_coords = xy_coords[slice_mask]

            # expand coords to batch size
            batch_xy_coords = torch.stack(batchsize * [xy_coords])

            # reconstruct the slice by passing it through the model
            batch_xy_slice = self.forward(coords=batch_xy_coords, z=z)

            # fill the corresponding slice in the 3-D volume with these amplitudes
            batch_vol_f[:, i, slice_mask.view(boxsize_ht, boxsize_ht)] = batch_xy_slice.to(batch_vol_f.dtype)

        return batch_vol_f

    @staticmethod
    def postprocess_volume_batch(batch_vols: torch.Tensor,
                                 norm: tuple[float, float],
                                 iht_downsample_scaling_correction: float,
                                 lowpass_mask: np.ndarray | None = None,
                                 flip: bool = False,
                                 invert: bool = False) -> np.ndarray:
        """
        Apply post-volume-decoding processing steps: downsampling scaling correction, lowpass filtering, inverse fourier transform, volume handedness flipping, volume data sign inversion.

        :param batch_vols: batch of fourier space non-symmetrized volumes directly from eval_vol_batch, shape (nvols, boxsize, boxsize, boxsize)
        :param norm: tuple of floats representing mean and standard deviation of preprocessed particles used during model training
        :param iht_downsample_scaling_correction: a global scaling factor applied when forward and inverse fourier / hartley transforming.
                This is calculated and applied internally by the fft.py module as a function of array shape.
                Thus, when the volume is downsampled, a further correction is required.
        :param lowpass_mask: a binary mask applied to fourier space symmetrized volumes to low pass filter the reconstructions.
                Typically, the same mask is used for all volumes via broadcasting, thus this may be of shape (1, boxsize, boxsize, boxsize) or (nvols, boxsize, boxsize, boxsize).
        :param flip: Whether to invert the volume chirality by flipping the data order along the z axis.
        :param invert: Whether to invert the data light-on-dark vs dark-on-light convention, relative to the reconstruction returned by the decoder module.
        :return: Postprocessed volume batch in real space
        """
        # sanity check inputs
        assert batch_vols.ndim == 4, f'The volume batch must have four dimensions (batch size, boxsize, boxsize, boxsize). Found {batch_vols.shape}'
        if lowpass_mask is not None:
            assert batch_vols.shape[-3:] == lowpass_mask.shape[-3:], f'The volume batch must have the same volume dimensions as the lowpass mask. Found {batch_vols.shape}, {lowpass_mask.shape}'

        # convert torch tensor to numpy array for future operations
        batch_vols = batch_vols.cpu().numpy()

        # normalize the volume (mean and standard deviation) by normalization used when training the model
        batch_vols = batch_vols * norm[1] + norm[0]

        # lowpass filter with fourier space mask
        if lowpass_mask is not None:
            batch_vols = batch_vols * lowpass_mask

        # transform to real space and scale values if downsampling was applied
        batch_vols = fft.iht3_center(batch_vols)
        batch_vols *= iht_downsample_scaling_correction

        if flip:
            batch_vols = np.flip(batch_vols, 1)

        if invert:
            batch_vols *= -1

        return batch_vols


class TiltSeriesEncoder(nn.Module):
    """
    A module to encode multiple (tilt) images of a particle to a latent embedding.
    The module comprises two submodules: `encoder_a` and `encoder_b`.
    Encoder_a is a ResidLinearMLP module that embeds a single tilt image of a particle with `in_dim` features (pixels) to an embedding with `out_dim_a` features.
    Encoder_b is a ResidLinearMLP module that pools all per-tilt-image embeddings from `encoder_a` by a `pooling_function`,
    and embeds this pooled representation to a single latent embedding with `out_dim` features.
    """

    def __init__(self,
                 in_dim: int,
                 hidden_layers_a: int = 3,
                 hidden_dim_a: int = 256,
                 out_dim_a: int = 128,
                 hidden_layers_b: int = 3,
                 hidden_dim_b: int = 256,
                 out_dim: int = 64 * 2,
                 activation: torch.nn.ReLU | torch.nn.LeakyReLU = torch.nn.ReLU,
                 ntilts: int = 41,
                 pooling_function: Literal['concatenate', 'max', 'mean', 'median', 'set_encoder'] = 'concatenate',
                 num_seeds: int = 1,
                 num_heads: int = 4,
                 layer_norm: bool = False):
        """
        Create the TiltSeriesEncoder module.

        :param in_dim: number of input features to the module (typically the number of pixels in a masked image)
        :param hidden_layers_a: number of intermediate hidden layers in the encoder_a submodule
        :param hidden_dim_a: number of features in each hidden layer in the encoder_a submodule
        :param out_dim_a: number of output features from the encoder_a submodule (sometimes referred to as the intermediate latent embedding dimensionality)
        :param hidden_layers_b: number of intermediate hidden layers in the encoder_b submodule
        :param hidden_dim_b: number of features in each hidden layer in the encoder_b submodule
        :param out_dim: number of output features from the encoder_b submodule (sometimes referred to as the latent embedding dimensionality or zdim)
        :param activation: activation function to be applied after each layer, either `torch.nn.ReLU` or `torch.nn.LeakyReLU`
        :param ntilts: the number of tilt images per particle, used in setting the number of input features to `encoder_b`
        :param pooling_function: the method used to pool the per-tilt-image intermediate latent representations prior to `encoder_b`
        :param num_seeds: number of seed vectors to use in the set encoder module. Generally should be set to 1
        :param num_heads: number of heads for multihead attention in the set encoder module. Generally should be set to a power of 2
        :param layer_norm: whether to apply layer normalization in the set encoder module
        """
        # initialize the parent nn.Module class
        super().__init__()

        # assign attributes from parameters used at instance creation
        self.in_dim = in_dim
        self.hidden_layers_a = hidden_layers_a
        self.hidden_dim_a = hidden_dim_a
        self.out_dim_a = out_dim_a
        self.hidden_layers_b = hidden_layers_b
        self.hidden_dim_b = hidden_dim_b
        self.out_dim = out_dim
        self.activation = activation
        self.ntilts = ntilts
        self.pooling_function = pooling_function
        self.num_seeds = num_seeds
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        assert hidden_layers_a >= 0  # possible to have no hidden layers, just a direct mapping from input to output
        assert hidden_layers_b >= 0  # possible to have no hidden layers, just a direct mapping from input to output
        if pooling_function in ['concatenate', 'set_encoder']:
            # other pooling functions do not use ntilts so not relevant
            assert ntilts > 1  # having ntilts == 1 is very likely to cause problems with squeezing and broadcasting tensors

        # encoder1 encodes each identically-masked tilt image, independently
        self.encoder_a = ResidLinearMLP(in_dim, hidden_layers_a, hidden_dim_a, out_dim_a, activation)

        # encoder2 merges ntilts-concatenated encoder1 information and encodes further to latent space via one of
        # ('concatenate', 'max', 'mean', 'median', 'set_encoder')
        if self.pooling_function == 'concatenate':
            self.pooling_layer = nn.Sequential(Rearrange('batch tilt pixels -> batch (tilt pixels)'),
                                               activation())
            self.encoder_b = ResidLinearMLP(in_dim=out_dim_a * ntilts,
                                            nlayers=hidden_layers_b,
                                            hidden_dim=hidden_dim_b,
                                            out_dim=out_dim,
                                            activation=activation)

        elif self.pooling_function == 'max':
            self.pooling_layer = nn.Sequential(Reduce('batch tilt pixels -> batch pixels', 'max'),
                                               activation())
            self.encoder_b = ResidLinearMLP(in_dim=out_dim_a,
                                            nlayers=hidden_layers_b,
                                            hidden_dim=hidden_dim_b,
                                            out_dim=out_dim,
                                            activation=activation)

        elif self.pooling_function == 'mean':
            self.pooling_layer = nn.Sequential(Reduce('batch tilt pixels -> batch pixels', 'mean'),
                                               activation())
            self.encoder_b = ResidLinearMLP(in_dim=out_dim_a,
                                            nlayers=hidden_layers_b,
                                            hidden_dim=hidden_dim_b,
                                            out_dim=out_dim,
                                            activation=activation)

        elif self.pooling_function == 'median':
            self.pooling_layer = nn.Sequential(MedianPool1d(pooling_axis=-2),
                                               activation())
            self.encoder_b = ResidLinearMLP(in_dim=out_dim_a,
                                            nlayers=hidden_layers_b,
                                            hidden_dim=hidden_dim_b,
                                            out_dim=out_dim,
                                            activation=activation)

        elif self.pooling_function == 'set_encoder':
            self.pooling_layer = nn.Sequential(set_transformer.SetTransformer(dim_input=out_dim_a,
                                                                              num_outputs=num_seeds,
                                                                              dim_output=out_dim,
                                                                              dim_hidden=hidden_dim_b,
                                                                              num_heads=num_heads,
                                                                              ln=layer_norm),
                                               Rearrange('batch 1 latent -> batch latent'))
            self.encoder_b = nn.Identity()

        else:
            raise ValueError

    def forward(self, batch):
        """
        Pass data forward through the module.

        :param batch: Input data tensor, shape (batch, ntilts, boxsize*boxsize[enc_mask])
        :return: Output data tensor, shape (batch, zdim*2)
        """
        # pass each tilt independently through encoder_a (parallelizing along both batch and ntilts dimensions)
        batch_tilts_intermediate = self.encoder_a(batch)
        # pool each tilt's intermediate latent embedding along the ntilts dimension
        batch_tilts_pooled = self.pooling_layer(batch_tilts_intermediate)
        # obtain parameters for per-particle latent embedding mean and log(variance) as a single concatenated tensor output
        z = self.encoder_b(batch_tilts_pooled)

        return z

    def reparameterize(self,
                       mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparamaterization trick to allow backpropogation through semi-randomly sampled latent embedding.
        Sampling a latent embedding from a gaussian parameterized by mu and logvar is an operation without an associated gradient with respect to inputs, and therefore breaks training.
        We reparamaterize such that the latent embedding is deterministically calculated as `z = epsilon * standard_deviation + mean`.
        This representation "outsources" the randomness from `z` itself to `epsilon`, and allows gradient calculation through `z` to `standard_deviation` and `mean`, and onward to earlier layers.

        :param mu: mean parameterizing latent embeddings `z`, shape (batch, zdim)
        :param logvar: log(variance) parameterizing latent embeddings `z`, shape (batch, zdim)
        :return: reparameterized latent embeddings `z`, shape (batch, zdim)
        """
        # no need for reparameterization at inference time; better to be deterministic
        if not self.training:
            return mu

        # otherwise perform reparameterization
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class ResidLinearMLP(nn.Module):
    """
    Multiple connected Residual Blocks as a Multi-Layer Perceptron.
    """

    def __init__(self,
                 in_dim: int,
                 nlayers: int,
                 hidden_dim: int,
                 out_dim: int,
                 activation: torch.nn.ReLU | torch.nn.LeakyReLU):
        """
        Create the ResidLinearMLP module.

        :param in_dim: number of input features to the module
        :param nlayers: number of intermediate hidden layers in the module
        :param hidden_dim: number of features in each hidden layer
        :param out_dim: number of output features from the module
        :param activation: activation function to be applied after each layer, either `torch.nn.ReLU` or `torch.nn.LeakyReLU`
        """
        # intialize the parent nn.Module class
        super().__init__()

        # create the first layer to receive input
        # the layer will be a ResidLinear layer if possible, otherwise a Linear layer
        if in_dim == hidden_dim:
            layers = [ResidLinear(in_dim, hidden_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(activation())

        # append the hidden layers to the module as ResidLinear layers
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())

        # append the output layer to the module without a final activation function
        # the layer will be a ResidLinear layer if possible, otherwise a Linear layer
        if out_dim == hidden_dim:
            layers.append(ResidLinear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(hidden_dim, out_dim))

        # create the overall module as a sequential pass through the defined layers
        self.main = nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Pass data forward through the module.

        :param x: Input data tensor.
        :return: Output data tensor.
        """
        return self.main(x)


class ResidLinear(nn.Module):
    """
    A Residual Block layer consisting of a single linear layer with an identity skip connection.
    Note that the identity mapping requires that the number of input and output features are the same for element-wise addition.
    References: https://arxiv.org/abs/1512.03385
    """

    def __init__(self,
                 nin: int,
                 nout: int):
        """
        Create the ResidLinear layer.

        :param nin: number of input features to the linear layer
        :param nout: number of output features to the linear layer
        """
        super().__init__()
        self.linear = nn.Linear(nin, nout)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Pass data forward through the layer.

        :param x: Input data tensor.
        :return: Output data tensor.
        """
        # output is element-wise addition of the output of the linear layer given x with (identity-mapped) input x
        z = self.linear(x) + x
        return z


class MedianPool1d(nn.Module):
    """
    Median pool module.
    Primarily exists due to limitations in pre-existing layer-based definitions of median.
     * torch.nn does not have a MedianPool module
     * einops does not support a callable (such as `torch.median`) when defining a Reduce layer.
    """

    def __init__(self, pooling_axis=-2):
        """
        Create the MedianPool1d layer.

        :param pooling_axis: the tensor axis over which to take the median
        """
        super().__init__()
        self.pooling_axis = pooling_axis

    def forward(self, x: torch.Tensor):
        """
        Pass data forward through the layer.

        :param x: Input data tensor.
        :return: Output data tensor.
        """
        with autocast(device_type=x.device.type, enabled=False):  # torch.quantile and torch.median do not support fp16 so casting to fp32 in case AMP is used
            x = x.to(dtype=torch.float32, device=x.device)
            x = x.quantile(dim=self.pooling_axis, q=0.5)
            return x


class DataParallelPassthrough(torch.nn.DataParallel):
    """
    Class to wrap underlying module in DataParallel for GPU-parallelized computations, but allow accessing underlying module attributes and methods.
    Intended use: `model = DataParallelPassthrough(model)`
    """

    def __getattr__(self,
                    name: str):
        """
        Get the requested attribute or method from the parent DataParallel module if it exists, otherwise from the wrapped module.

        :param name: name of the attribute or method to request
        :return: the requested attribute or method
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class VolumeGenerator:
    """
    Convenience class to generate volume(s) from a trained tomoDRGN model.
    Supports evaluating homogeneous (`train_nn`) or heterogeneous (`train_vae`) models.
    """

    def __init__(self,
                 config: str | dict[str, dict],
                 weights_path: str | None = None,
                 model: TiltSeriesHetOnlyVAE | FTPositionalDecoder | None = None,
                 lat: lattice.Lattice | None = None,
                 amp: bool = True):
        """
        Instantiate a `VolumeGenerator` object.
        :param config: path to trained model `config.pkl` from `train_vae.py` or `train_nn.py`, or the corresponding preloaded config dictionary
        :param weights_path: path to trained model `weights.*.pkl` from `train_vae.py` or `train_nn.py`.
                Prefer specifying `weights_path` over `model` when a model is not yet loaded in memory.
        :param model: preloaded model from which to evaluate volumes.
                Prefer specifying `model` over `weights_path` when a model is already loaded in memory. Must specify `lat` with `model`.
        :param lat: preloaded tomodrgn lattice object. Must specify `model` with `lat`.
        :param amp: Enable or disable use of mixed-precision model inference
        """
        # set the device
        device = utils.get_default_device()
        if device == torch.device('cpu'):
            amp = False
            log('Warning: pytorch AMP does not support non-CUDA (e.g. cpu) devices. Automatically disabling AMP and continuing')
            # https://github.com/pytorch/pytorch/issues/55374

        # load the model configuration
        if type(config) is str:
            cfg = utils.load_pkl(config)
        else:
            assert type(config) is dict
            cfg = config

        # set parameters for volume generation
        boxsize_ht = int(cfg['lattice_args']['boxsize'])  # image size + 1
        zdim = int(cfg['model_args']['zdim']) if 'zdim' in cfg['model_args'].keys() else None
        norm = cfg['dataset_args']['norm']
        angpix = float(cfg['angpix'])

        # load the model and lattice
        if weights_path is not None:
            assert model is None, 'Cannot specify both weights_path` and `model`'
            if zdim:
                # load a VAE model
                model, lat = TiltSeriesHetOnlyVAE.load(config=cfg,
                                                       weights=weights_path,
                                                       device=device)
            else:
                # load a non-VAE decoder-only model
                model, lat = FTPositionalDecoder.load(config=cfg,
                                                      weights=weights_path,
                                                      device=device)
        elif model is not None:
            # using a pre-loaded model and lattice
            assert weights_path is None, 'Cannot specify both weights_path` and `model`'
            assert type(model) in [TiltSeriesHetOnlyVAE, FTPositionalDecoder], f'Unrecognized model type: {type(model)}'
        else:
            raise ValueError('Must specify one of weights_path` or `model`')

        # define the volume evaluation and postprocessing functions associated with the model
        if zdim:
            eval_volume_batch = model.decoder.eval_volume_batch
            postprocess_volume_batch = model.decoder.postprocess_volume_batch
        else:
            eval_volume_batch = model.eval_volume_batch
            postprocess_volume_batch = model.postprocess_volume_batch

        self.device = device
        self.model_boxsize_ht = boxsize_ht
        self.model_angpix = angpix
        self.norm = norm
        self.zdim = zdim
        self.model = model
        self.lat = lat
        self.amp = amp
        self.eval_volume_batch = eval_volume_batch
        self.postprocess_volume_batch = postprocess_volume_batch

    def generate_volumes(self,
                         z: np.ndarray | str | None,
                         out_dir: str,
                         out_name: str = 'vol',
                         downsample: int | None = None,
                         lowpass: float | None = None,
                         flip: bool = False,
                         invert: bool = False,
                         batch_size: int = 1) -> None:
        """
        Generate volumes at specified latent embeddings and save to specified output directory.
        Generated volume filename format is `out_dir / out_name _ {i:03d if multiple volumes} .mrc`

        :param z: latent embeddings to evaluate as numpy array of shape `(nptcls, zdim)`, or path to a .txt or .pkl file containing array of shape (nptcls, zdim).
                `None` if evaluating a homogeneous tomodrgn model.
        :param out_dir: path to output directory in which to save output mrc file(s)
        :param out_name: string to prepend to output .mrc file name(s)
        :param downsample: downsample reconstructed volumes to this box size (units: px) by Fourier cropping, None means to skip downsampling
        :param lowpass: lowpass filter reconstructed volumes to this resolution (units: Å), None means to skip lowpass filtering
        :param flip: flip the chirality of the reconstructed volumes by inverting along the z axis
        :param invert: invert the data sign of the reconstructed volumes (light-on-dark vs dark-on-light)
        :param batch_size: batch size to parallelize volume generation (32-64 works well for box64 volumes)
        :return: None
        """
        # sanity check inputs for evaluating model and postprocessing
        self._check_inputs(downsample=downsample,
                           out_dir=out_dir)

        # prepare inputs for evaluating model and postprocessing
        coords, extent, iht_downsample_scaling_correction, angpix, lowpass_mask, z = self._prepare_inputs(out_dir=out_dir,
                                                                                                          downsample=downsample,
                                                                                                          lowpass=lowpass,
                                                                                                          z=z)

        # set context managers and flags for inference mode evaluation loop
        self.model.eval()
        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
                if z is None:
                    batch_vols = self.eval_volume_batch(coords=coords,
                                                        z=z,
                                                        extent=extent)
                    batch_vols = self.postprocess_volume_batch(batch_vols=batch_vols[:, :-1, :-1, :-1],  # exclude symmetrized +k frequency
                                                               norm=self.norm,
                                                               iht_downsample_scaling_correction=iht_downsample_scaling_correction,
                                                               lowpass_mask=lowpass_mask,
                                                               flip=flip,
                                                               invert=invert)
                    out_mrc = f'{out_dir}/{out_name}.mrc'
                    mrc.write(fname=out_mrc,
                              array=batch_vols[0],
                              angpix=angpix)

                else:
                    # batch size cannot be larger than the number of latent embeddings to evaluate, or is 1 if homogeneous model
                    batch_size = min(batch_size, z.shape[0])
                    # construct z iterator
                    z_iterator = torch.split(z, split_size_or_sections=batch_size, dim=0)
                    num_batches = len(z_iterator)
                    log(f'Generating {len(z)} volumes in batches of {batch_size}')

                    # prepare threadpool for parallelized file writing
                    with Pool(min(os.cpu_count(), batch_size)) as p:
                        for (i, z_batch) in enumerate(z_iterator):
                            log(f'    Generating volume batch {i+1} / {num_batches}')
                            z_batch = torch.as_tensor(z_batch, device=self.device)
                            batch_vols = self.eval_volume_batch(coords=coords,
                                                                z=z_batch,
                                                                extent=extent)
                            batch_vols = self.postprocess_volume_batch(batch_vols=batch_vols[:, :-1, :-1, :-1],  # exclude symmetrized +k frequency
                                                                       norm=self.norm,
                                                                       iht_downsample_scaling_correction=iht_downsample_scaling_correction,
                                                                       lowpass_mask=lowpass_mask,
                                                                       flip=flip,
                                                                       invert=invert)
                            out_mrcs = [f'{out_dir}/{out_name}_{i * batch_size + j:03d}.mrc' for j in range(len(z_batch))]
                            p.starmap(func=mrc.write, iterable=zip(out_mrcs,
                                                                   batch_vols[:len(out_mrcs)],
                                                                   itertools.repeat(None, len(out_mrcs)),
                                                                   itertools.repeat(angpix, len(out_mrcs))))

    def _check_inputs(self,
                      out_dir: str,
                      downsample: int | None = None) -> None:
        """
        Check inputs are expected types, shapes, and within expected bounds for model evaluation.
        :param out_dir: path to output directory in which to save output mrc file(s)
        :param downsample: downsample reconstructed volumes to this box size (units: px) by Fourier cropping, None means to skip downsampling
        :return: None
        """
        # create output directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # downsample checks
        if downsample:
            assert downsample % 2 == 0, "Boxsize must be even"
            assert downsample <= self.model_boxsize_ht - 1, "Must be smaller than original box size"

    def _prepare_inputs(self,
                        out_dir: str,
                        downsample: int | None = None,
                        lowpass: float | None = None,
                        z: np.ndarray | str | None = None) -> tuple[torch.Tensor, float, float, float, np.ndarray, torch.Tensor | None]:
        """
        Prepare inputs for repeatedly evaluating a tomoDRGN model to produce and postprocess and ensemble of volumes.
            generate 2d plane of coords to evaluate with appropriate extent and eventual volume scaling factor for downsampling
            generate lowpass filter mask
            calculate downsample scaling factor
            load z values to evaluate (if any, and write to disk if given array)
        :return:
        """
        # generate 2-D XY plane of coordinates to eventually evaluate (as cube of coords by repeating along z axis) with appropriate extent and IHT scaling factor for downsampling
        if downsample:
            boxsize_ht = downsample + 1
            coords = self.lat.get_downsample_coords(boxsize_new=boxsize_ht).to(self.device)
            extent = self.lat.extent * (downsample / (self.model_boxsize_ht - 1))
            iht_downsample_scaling_correction = downsample ** 3 / (self.model_boxsize_ht - 1) ** 3
            angpix = self.model_angpix * (self.model_boxsize_ht - 1) / downsample
        else:
            boxsize_ht = self.model_boxsize_ht
            coords = self.lat.coords.to(self.device)
            extent = self.lat.extent
            iht_downsample_scaling_correction = 1.
            angpix = self.model_angpix

        # generate array to be multiplied into volumes as lowpass filter (note this takes place after removing symmetrized +k frequency row/column of pixels, and operates on np array so device None)
        if lowpass is not None:
            lowpass_mask = utils.calc_lowpass_filter_mask(boxsize=boxsize_ht-1,
                                                          angpix=angpix,
                                                          lowpass=lowpass,
                                                          device=None)
        else:
            lowpass_mask = np.ones((boxsize_ht-1, boxsize_ht-1, boxsize_ht-1),
                                   dtype=bool)
        # unsqueeze along batch dimension
        lowpass_mask = lowpass_mask[np.newaxis, ...]

        # prepare array of latent values to evaluate, if provided
        if z is not None:
            # load latent if not preloaded
            if type(z) is str:
                if z.endswith('.pkl'):
                    z = utils.load_pkl(z)
                elif z.endswith('.txt'):
                    z = np.loadtxt(z)
                else:
                    raise ValueError(f'Unrecognized file type for loading z: {z}')
            elif type(z) is np.ndarray:
                pass
            else:
                raise ValueError(f'Unrecognized data structure for z: {type(z)}')

            # confirm latent array shape
            if z.ndim == 1:
                z = z.reshape(1, -1)
            elif z.ndim > 2:
                raise ValueError(f'zdim must be 1 or 2, got {z.ndim}')

            # confirm latent dimensionality matches value loaded from config
            assert z.shape[1] == self.zdim

            # save the z values used to generate the volumes in the output directory for future convenience
            zfile = f'{out_dir}/z_values.txt'
            np.savetxt(zfile, z)
            log(f'Saved latent embeddings to decode to {zfile}')

            # convert z to tensor
            z = torch.as_tensor(z, dtype=torch.float32)

        return coords, extent, iht_downsample_scaling_correction, angpix, lowpass_mask, z


def mlp_ascii(input_dim: int, hidden_dims: list[int], output_dim: int):
    """
    Create ASCII art of a fully connected multi-layer perceptron.
    Constraints: number of digits in each dimension cannot exceed 5.
    Sample usage: ``print_mlp_ascii(input_dim=12345, hidden_dims=[128, 128, 128], output_dim=2)``

    :param input_dim: dimensionality of input layer
    :param hidden_dims: list of dimensionalities of hidden layers
    :param output_dim: dimensionality of output layer
    :return: None
    """
    output = []
    art_height = 9  # 9 lines tall
    node_symbol = '\u25CF'
    node_connection_symbol = '\u292C'
    for i in range(art_height):
        if i in (0, 2, 6, 8):
            # draw the conneciton lines and nodes
            if i in (0, 8):
                # top or bottom line, number of nodes = len(hidden_dims)
                line = f'        {node_symbol} ' + ''.join([f'--- {node_symbol} ' for _ in range(len(hidden_dims) - 1)]) + '       '
            else:
                # central two lines, number of nodes = len(hidden_dims + 2)
                line = f'  {node_symbol} ---' + ''.join([f' {node_symbol} ---' for _ in range(len(hidden_dims))]) + f' {node_symbol}  '
        elif i in (1, 3, 5, 7):
            # draw the connection lines and colons only
            if i == 1:
                # top line, number of connections with X = len(hidden_dims)-1
                line = '     /     ' + ''.join([f'{node_connection_symbol}     ' for _ in range(len(hidden_dims) - 1)]) + '\\     '
            elif i == 3:
                # central two lines, number of connections with X = len(hidden_dims)-1 and need to draw colons connecting nodes
                line = '  :  \\ ' + ''.join([f' :  {node_connection_symbol} ' for _ in range(len(hidden_dims) - 1)]) + ' :  /  :  '
            elif i == 5:
                # central two lines, number of connections with X = len(hidden_dims)-1 and need to draw colons connecting nodes
                line = '  :  / ' + ''.join([f' :  {node_connection_symbol} ' for _ in range(len(hidden_dims) - 1)]) + ' :  \\  :  '
            else:
                # bottom line, number of connections with X = len(hidden_dims)-1
                line = '     \\     ' + ''.join([f'{node_connection_symbol}     ' for _ in range(len(hidden_dims) - 1)]) + '/     '
        else:
            # draw the center padded node numbers
            # f'{input_dim:^5}'
            line = f'{input_dim:^5} ' + ''.join([f'{hidden_dim:^5} ' for hidden_dim in hidden_dims]) + f'{output_dim:^5}'
        output.append(line)

    return output


def print_tiltserieshetonlyvae_ascii(model: TiltSeriesHetOnlyVAE):
    """
    Print an ASCII art representation of a TiltSeriesHetOnlyVAE model

    :param model: the model to represent
    :return: None
    """
    enca = mlp_ascii(input_dim=model.encoder.in_dim,
                     hidden_dims=[model.encoder.hidden_dim_a for _ in range(model.encoder.hidden_layers_a)],
                     output_dim=model.encoder.out_dim_a)
    enca.insert(0, f'{"ENCODER A - PER IMAGE":^{len(enca[-1])}}')

    encb = mlp_ascii(input_dim=model.encoder.out_dim_a * model.encoder.ntilts,
                     hidden_dims=[model.encoder.hidden_dim_b for _ in range(model.encoder.hidden_layers_b)],
                     output_dim=model.zdim * 2)
    encb.insert(0, f'{"ENCODER B - PER PARTICLE":^{len(encb[-1])}}')

    dec = mlp_ascii(input_dim=model.decoder.in_dim,
                    hidden_dims=[model.decoder.hidden_dim for _ in range(model.decoder.hidden_layers)],
                    output_dim=2)
    dec.insert(0, f'{"DECODER - PER VOXEL":^{len(dec[-1])}}')

    output = [f'{enca_line} {encb_line} {dec_line}' for enca_line, encb_line, dec_line in zip(enca, encb, dec)]
    for line in output:
        print(line)
