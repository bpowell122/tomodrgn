"""
Classes for creating, loading, training, and evaluating pytorch models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Literal
from einops import rearrange
from einops.layers.torch import Reduce, Rearrange
import torchinfo

from tomodrgn import fft, utils, lattice, set_transformer


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
                                           in_dim=3+zdim,
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
        lat = lattice.Lattice(boxsize=cfg['lattice_args']['D'],
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
                                     pe_dim=cfg['model_args']['pe_dim'],)

        # load weights if provided
        if weights is not None:
            ckpt = torch.load(weights)
            model.load_state_dict(ckpt['model_state_dict'])

        # move the model to the requested device
        model.to(device)

        return model, lat

    def encode(self,
               batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input batch of particle's tilt images to a corresponding batch of latent embeddings.
        Input images are masked by `self.enc_mask` if provided.
        :param batch: batch of particle tilt images to encode, shape (batch, ntilts, boxsize, boxsize)
        :return: `mu`: batch of mean values parameterizing latent embedding as Gaussian, shape (batch, zdim).
                `logvar`: batch of log variance values parameterizing latent embedding as Gaussian, shape (batch, zdim).
        """
        # input batch is of shape B x ntilts x D x D, need to flatten last two dimensions to apply flat encoder mask and to pass into flat model input layer
        batch = rearrange(batch, 'batch tilts boxsize_y boxsize_x -> batch tilts (boxsize_y boxsize_x)')  # B x ntilts x D*D
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
                Coords may be a NestedTensor (along the batch dimension, ragged along the ntilts dimension)
                due to typically unequal numbers of coordinates retained per-particle or per-tilt-image after upstream masking.
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
        print(torchinfo.summary(self.encoder,
                                input_size=(self.ntilts, self.in_dim),
                                batch_dim=0,
                                col_names=('input_size', 'output_size', 'num_params'),
                                depth=5))
        # print the decoder module which will be a conservative overestimate of input size without lattice masking
        # this is because each particle has a potentially unique lattice mask and therefore this is an upper bound on model size
        print(torchinfo.summary(self.decoder,
                                input_data=torch.rand(self.ntilts * self.boxsize_ht * self.boxsize_ht, 3 + self.zdim) - 0.5,
                                batch_dim=0,
                                col_names=('input_size', 'output_size', 'num_params'),
                                depth=5))


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
            k = k.reshape(coords.size(0), -1, 3, self.pe_dim)  # B x N*D*D[mask] x 3 x D2 (where .size(0) and -1 are required for NestedTensor)
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
        # z = repeat(z, 'batch zdim -> batch repeat_ntilts repeat_pixels zdim', repeat_ntilts=coords.shape[1], repeat_pixels=coords.shape[2])
        # unbind returns a view into the coords NestedTensor; iterating over the batch dim and calling len() to get the number of pixels per particle
        z_expanded = torch.nested.as_nested_tensor([z_ptcl.repeat(len(coords_ptcl), 1) for z_ptcl, coords_ptcl in zip(z, coords.unbind())])

        # concatenate coords with z along the last axis (coordinate + zdim value)
        # coords_z, _ = pack([coords, z], 'batch ntilts coords *')
        coords_z = torch.cat([coords, z_expanded], -1)

        return coords_z

    def forward(self,
                coords: torch.Tensor,
                z: torch.Tensor = None) -> torch.Tensor:
        """
        Decode a batch of lattice coordinates concatenated with the corresponding latent embedding to Hartley Transform spatial frequency amplitudes.
        :param coords: 3-D spatial frequency coordinates (e.g. from Lattice.coords) concatenated with the
                shape (batch, ntilts * boxsize_ht * boxsize_ht [mask], 3).
                Coords may be a NestedTensor (along the batch dimension, ragged along the ntilts dimension)
                due to typically unequal numbers of coordinates retained per-particle or per-tilt-image after upstream masking.
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
        :param coords: 3-D spatial frequency coordinates (e.g. from Lattice.coords) concatenated with the
                shape (batch, ntilts * boxsize_ht * boxsize_ht [mask], 3).
                Coords may be a NestedTensor (along the batch dimension, ragged along the ntilts dimension)
                due to typically unequal numbers of coordinates retained per-particle or per-tilt-image after upstream masking.
        :param z: latent embedding per-particle, shape (batch, zdim)
        :return: Decoded voxel intensities at the specified 3-D spatial frequencies.
        """
        # sanity check inputs coordinates are within range (-0.5, 0.5)
        assert (coords.abs() - 0.5 < 1e-4).all(), f'coords.max(): {coords.max()}; coords.min(): {coords.min()}'

        # convention: only evaluate the -z points
        zflipper = torch.where(coords[..., 2] > 0, -1, 1)
        coords_flipped = coords * zflipper  # avoids "modifying tensor in-place" warnings+errors

        # positionally encode x,y,z 3-D coordinate
        coords_pe = self.positional_encoding(coords_flipped)

        # concatenate each spatial coordinate with the appropriate latent embedding to be conditioned on
        if z:
            coords_pe = self.cat_z(coords_pe, z)

        # evaluate the model
        result = self.decoder(coords_pe)

        # replace with complex conjugate to get correct values for original lattice positions
        result_flipped = torch.cat([result[..., 0], result[..., 1] * zflipper], -1)

        return result_flipped

    def eval_volume(self,
                    coords: torch.Tensor,
                    boxsize_ht: int,
                    extent: float,
                    norm: tuple[float, float],
                    z: torch.Tensor | None = None) -> np.ndarray:
        """
        Evaluate the model on a DxDxD volume, optionally conditioned on a latent embedding z
        :param coords: lattice coords on the x-y plane, shape (boxsize_ht**2, 3)
        :param boxsize_ht:  number of grid points along each dimension. Should be odd.
        :param extent: maximum value of the grid along each dimension, typically <= 0.5 to constrain points to range (-0.5, 0.5)
        :param norm: tuple of floats representing mean and standard deviation of preprocessed particles used during model training
        :param z: latent embedding associated with the volume to decode, shape (1, zdim)
        :return: reconstructed volume, shape (boxsize, boxsize, boxsize)
        """
        # sanity check inputs
        assert extent <= 0.5

        # prepare for model evaluation
        assert not self.training
        with torch.inference_mode():

            # evaluate the volume by slicing through z to avoid memory overflows
            vol_f = torch.zeros((boxsize_ht, boxsize_ht, boxsize_ht), dtype=np.float32, device=coords.device)
            for i, dz in enumerate(np.linspace(-extent, extent, boxsize_ht, endpoint=True, dtype=np.float32)):
                # create the z slice to evaluate
                xy_coords = coords + torch.tensor([0, 0, dz], device=coords.device)

                # only evaluate coords within `extent` limit (if extent==0.5, nyquist limit in reciprocal space)
                keep = xy_coords.pow(2).sum(dim=1) <= extent ** 2
                xy_coords = xy_coords[keep]

                # reconstruct the slice by passing it through the model
                xy_slice = self.forward(coords=xy_coords, z=z)

                # fill the corresponding slice in the 3-D volume with these amplitudes
                slice_ = torch.zeros((boxsize_ht ** 2), dtype=vol_f.dtype, device=vol_f.device)
                slice_[keep] = xy_slice.to(slice_.dtype)
                vol_f[i] = slice_.view(boxsize_ht, boxsize_ht)

            # convert the volume from tensor to ndarray
            vol_f = vol_f.cpu().numpy()
            # normalize the volume (mean and standard deviation) by normalization used when training the model
            vol_f = vol_f * norm[1] + norm[0]
            # inverse DHT,  remove last +k freq that is present due to symmetrizing of DHT
            vol = fft.iht3_center(vol_f[:-1, :-1, :-1])
        return vol

    def eval_volume_batch(self,
                          coords: torch.Tensor,
                          z: torch.Tensor,
                          keep: torch.Tensor,
                          norm: np.ndarray) -> np.ndarray:
        """
        Evaluate the model on a batch of DxDxD volumes sharing pre-defined masked coords, D, extent
        :param coords: batch of lattice coords on the x-y plane, shape (batch, boxsize_ht**2, 3)
        :param z: latent embedding associated with the volume to decode, shape (batch, zdim)
        :param keep: mask of which coords to keep per z-plane (satisfying extent) (B=1 x D(z) x D**2(xy))
        :param norm: mean and standard deviation of preprocessed particles used during model training, shape (batch=0, 2)
        :return: batch of reconstructed volume, shape (batch, boxsize, boxsize, boxsize)
        """
        assert not self.training
        with torch.inference_mode():
            # TODO add amp context manager?
            # evaluate the volume by zslice to avoid memory overflows
            batch_size, boxsize_ht, _, _ = coords.shape
            batch_vol_f = torch.zeros((batch_size, boxsize_ht, boxsize_ht, boxsize_ht), dtype=coords.dtype, device=coords.device)
            for i in range(boxsize_ht):
                # get the coords across all volumes (batchdim) of one slice `i` to evaluate, premasked by the `keep` extent
                batch_xy_coords = coords[:, i, keep[0, i], :]

                # reconstruct the slices by passing them through the model
                batch_xy_slice = self.forward(coords=batch_xy_coords, z=z)

                # fill the corresponding slices in the batch of 3-D volumes with these amplitudes
                batch_slice_ = torch.zeros((batch_size, boxsize_ht ** 2), dtype=batch_vol_f.dtype, device=batch_vol_f.device)
                batch_slice_[:, keep[0, i]] = batch_xy_slice.to(batch_slice_.dtype)
                batch_vol_f[:, i] = batch_slice_.view(batch_size, boxsize_ht, boxsize_ht)

            # convert the volume from tensor to ndarray
            batch_vol_f = batch_vol_f.cpu().numpy()
            # normalize the volume (mean and standard deviation) by normalization used when training the model
            batch_vol_f = batch_vol_f * norm[0, 1] + norm[0, 0]
            # inverse DHT,  remove last +k freq that is present due to symmetrizing of DHT
            batch_vol = fft.iht3_center(batch_vol_f[:, :-1, :-1, :-1])
        return batch_vol


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
                 out_dim: int = 64*2,
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

    def forward(self, x):
        """
        Pass data forward through the layer.
        :param x: Input data tensor.
        :return: Output data tensor.
        """
        with autocast(enabled=False):  # torch.quantile and torch.median do not support fp16 so casting to fp32 in case AMP is used
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
