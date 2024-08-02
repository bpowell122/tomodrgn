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
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf', enc_dim=None, feat_sigma=None):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_type = enc_type
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 2, activation)

        if enc_type == "gaussian":
            # We construct 3 * self.enc_dim random vector frequences, to match the original positional encoding:
            # In the positional encoding we produce self.enc_dim features for each of the x,y,z dimensions,
            # whereas in gaussian encoding we produce self.enc_dim features each with random x,y,z components
            #
            # Each of the random feats is the sine/cosine of the dot product of the coordinates with a frequency
            # vector sampled from a gaussian with std of feat_sigma
            rand_freqs = torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * feat_sigma
            # make rand_feats a parameter so it is saved in the checkpoint, but do not perform SGD on it
            self.rand_freqs = nn.Parameter(rand_freqs, requires_grad=False)
        else:
            self.rand_feats = None
    
    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        if self.enc_type == "gaussian":
            return self.random_fourier_encoding(coords)
        freqs = torch.arange(self.enc_dim, dtype=torch.float, device=coords.device)
        if self.enc_type == 'geom_ft':
            freqs = self.DD*np.pi*(2./self.DD)**(freqs/(self.enc_dim-1)) # option 1: 2/D to 1 
        elif self.enc_type == 'geom_full':
            freqs = self.DD*np.pi*(1./self.DD/np.pi)**(freqs/(self.enc_dim-1)) # option 2: 2/D to 2pi
        elif self.enc_type == 'geom_lowf':
            freqs = self.D2*(1./self.D2)**(freqs/(self.enc_dim-1)) # option 3: 2/D*2pi to 2pi 
        elif self.enc_type == 'geom_nohighf':
            freqs = self.D2*(2.*np.pi/self.D2)**(freqs/(self.enc_dim-1)) # option 4: 2/D*2pi to 1 
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError('Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x N x 3 x 1

        k = coords[..., 0:3, :] * freqs  # B x N x 3 x D2
        s = torch.sin(k)  # B x N x 3 x D2
        c = torch.cos(k)  # B x N x 3 x D2
        x = torch.cat([s, c], -1)  # B x N x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x N x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def random_fourier_encoding(self, coords):
        assert self.rand_freqs is not None
        # k = coords . rand_freqs
        # expand rand_freqs with singleton dimension along the batch dimensions
        # e.g. dim (1, ..., 1, n_rand_feats, 3)
        freqs = self.rand_freqs.view(1, 1, -1, 3) * self.D2     # 1 x 1 x 3*D2 x 3

        # compute x,y,z components of k then sum x,y,z components
        k = coords[..., None, 0:3] * freqs                 # 1 x B*N[mask] x 3*D2 x 3
        k = k.sum(-1)                                      # 1 x B*N[mask] x 3*D2
        x = torch.zeros((*k.shape, 2), dtype=k.dtype, device=k.device)  # preallocate memory to slightly lower allocation requirement
        x[...,0] = torch.sin(k)                            # 1 x B*N[mask] x 3*D2 x 2
        x[...,1] = torch.cos(k)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)   # 1 x B*N[mask] x 3*D2*2

        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2+1, dtype=torch.float, device=coords.device)
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, lattice):
        '''
        lattice: 1 x B*ntilts*N[mask] x 3+zdim, useful when images are different sizes to avoid ragged tensors
        '''
        # evaluate model on all pixel coordinates for each image
        full_image = self.decode(lattice)

        # return hartley information (FT real - FT imag) for each image
        image = full_image[...,0] - full_image[...,1]

        return image


    def decode(self, lattice):
        '''Return FT transform'''
        assert (lattice[...,0:3].abs() - 0.5 < 1e-4).all(), \
            f'lattice[...,0:3].max(): {lattice[...,0:3].max().to(torch.float32)}; ' \
            f'lattice[...,0:3].min(): {lattice[...,0:3].min().to(torch.float32)}'
        # convention: only evaluate the -z points
        w = lattice[...,2] > 0.0
        w_zflipper = torch.ones_like(lattice, device=lattice.device, requires_grad=False)
        w_zflipper[..., 0:3][w] *= -1
        lattice = lattice * w_zflipper  # avoids "modifying tensor in-place" warnings+errors
        # lattice[..., 0:3][w] = -lattice[..., 0:3][w]  # negate lattice coordinates where z > 0
        result = self.decoder(self.positional_encoding_geom(lattice))
        result[...,1][w] *= -1 # replace with complex conjugate to get correct values for original lattice positions
        return result

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume
        
        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization 
            zval: value of latent (zdim) ... or 1 x zdim from eval_vol?
        '''
        assert extent <= 0.5
        if zval is not None:
            zdim = len(zval)
            z = torch.tensor(zval, dtype=torch.float32, device=coords.device)

        vol_f = torch.zeros((D, D, D), dtype=coords.dtype, device=coords.device, requires_grad=False)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)):
            x = coords + torch.tensor([0, 0, dz], device=coords.device)
            keep = x.pow(2).sum(dim=1) <= extent ** 2
            x = x[keep]
            if zval is not None:
                x = torch.cat((x, z.expand(x.shape[0], zdim)), dim=-1)
            with torch.no_grad():
                if dz == 0.0:
                    y = self.forward(x)
                else:
                    y = self.decode(x)
                    y = y[..., 0] - y[..., 1]
                slice_ = torch.zeros((D ** 2), dtype=coords.dtype, device=coords.device, requires_grad=False)
                slice_[keep] = y.to(slice_.dtype)
                vol_f[i] = slice_.view(D, D)
        vol_f = vol_f.cpu().numpy()
        vol_f = vol_f * norm[1] + norm[0]
        vol = fft.iht3_center(vol_f[:-1, :-1, :-1])  # remove last +k freq for inverse FFT
        return vol

    def eval_volume_batch(self, coords_zz, keep, norm):
        '''
        Evaluate the model on a batch of DxDxD volumes sharing pre-defined masked coords, D, extent

        Inputs:
            coords_zz: pre-batched and z-concatenated lattice coords (B x D(z) x D**2(xy) x 3+zdim)
            keep: mask of which coords to keep per z-plane (satisfying extent) (B=1 x D(z) x D**2(xy))
            norm: data normalization (B=1 x 2)
        '''
        batch_size, D, _, _ = coords_zz.shape
        batch_vol_f = torch.zeros((batch_size, D, D, D), dtype=coords_zz.dtype, device=coords_zz.device, requires_grad=False)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i in range(D):
            with torch.no_grad():
                x = coords_zz[:, i, keep[0, i], :]
                y = self.decode(x)
                y = y[..., 0] - y[..., 1]
                slice_ = torch.zeros((batch_size, D ** 2), dtype=coords_zz.dtype, device=coords_zz.device, requires_grad=False)
                slice_[:, keep[0, i]] = y.to(slice_.dtype)
                batch_vol_f[:, i] = slice_.view(batch_size, D, D)
        batch_vol_f = batch_vol_f * norm[0, 1] + norm[0, 0]
        batch_vol = fft.iht3_center_torch(batch_vol_f[:, :-1, :-1, :-1])  # remove last +k freq for inverse FFT
        return batch_vol


class TiltSeriesEncoder(nn.Module):
    def __init__(self, in_dim, nlayersA, hidden_dimA, out_dimA, nlayersB,
                 hidden_dimB, out_dim, activation, ntilts, pooling_function,
                 num_seeds, num_heads, layer_norm):
        super(TiltSeriesEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dimA = out_dimA
        self.ntilts = ntilts
        self.pooling_function = pooling_function
        assert nlayersA + nlayersB > 2

        # encoder1 encodes each identically-masked tilt image, independently
        self.encoder1 = ResidLinearMLP(in_dim, nlayersA, hidden_dimA, out_dimA, activation)

        # encoder2 merges ntilts-concatenated encoder1 information and encodes further to latent space via one of
        # ('concatenate', 'max', 'mean', 'median', 'set_encoder')
        if self.pooling_function == 'concatenate':
            self.encoder2 = ResidLinearMLP(in_dim = out_dimA * ntilts, nlayers = nlayersB, hidden_dim = hidden_dimB,
                                           out_dim = out_dim, activation = activation)

        elif self.pooling_function == 'max':
            self.encoder2 = ResidLinearMLP(in_dim = out_dimA, nlayers = nlayersB, hidden_dim = hidden_dimB,
                                           out_dim = out_dim, activation = activation)

        elif self.pooling_function == 'mean':
            self.encoder2 = ResidLinearMLP(in_dim=out_dimA, nlayers=nlayersB, hidden_dim=hidden_dimB,
                                           out_dim=out_dim, activation=activation)

        elif self.pooling_function == 'median':
            self.encoder2 = ResidLinearMLP(in_dim=out_dimA, nlayers=nlayersB, hidden_dim=hidden_dimB,
                                           out_dim=out_dim, activation=activation)

        elif self.pooling_function == 'set_encoder':
            self.encoder2 = set_transformer.SetTransformer(dim_input = out_dimA, num_outputs = num_seeds, dim_output = out_dim,
                                                           dim_hidden = hidden_dimB, num_heads = num_heads, ln = layer_norm)

        else:
            raise ValueError


    def forward(self, batch):
        # input: B x ntilts x D*D[lattice_circular_mask]
        batch_tilts_intermediate = self.encoder1(batch)

        # input: B x ntilts x out_dim_A
        if self.pooling_function != 'set_encoder':

            if self.pooling_function == 'concatenate':
                batch_pooled_tilts = batch_tilts_intermediate.view(batch.shape[0], self.out_dimA * self.ntilts)

            elif self.pooling_function == 'max':
                batch_pooled_tilts = batch_tilts_intermediate.max(dim = 1)[0]
                batch_pooled_tilts = torch.nn.functional.relu(batch_pooled_tilts)

            elif self.pooling_function == 'mean':
                batch_pooled_tilts = batch_tilts_intermediate.mean(dim = 1)
                batch_pooled_tilts = torch.nn.functional.relu(batch_pooled_tilts)

            elif self.pooling_function == 'median':
                with autocast(enabled=False):  # torch.quantile and torch.median do not support fp16 so casting to fp32 assuming AMP is used
                    batch_pooled_tilts = batch_tilts_intermediate.to(torch.float32).quantile(dim = 1, q = 0.5)
                    batch_pooled_tilts = torch.nn.functional.relu(batch_pooled_tilts)

            else:
                raise ValueError

            # input: B x out_dim_A
            z = self.encoder2(batch_pooled_tilts)  # reshape to encode all tilts of one ptcl together

        else:
            # with autocast(enabled=False):  # set encoder appears numerically unstable in half-precision
            z = self.encoder2(batch_tilts_intermediate)
            z = z.squeeze(0)

        return z  # B x zdim


class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(ResidLinearMLP, self).__init__()
        layers = [ResidLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(ResidLinear(hidden_dim, out_dim) if out_dim == hidden_dim else nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResidLinear(nn.Module):
    def __init__(self, nin, nout):
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(nin, nout)
        #self.linear = nn.utils.weight_norm(nn.Linear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z

        
class DataParallelPassthrough(torch.nn.DataParallel):
    """
    Class to wrap underlying module in DataParallel for GPU-parallelized computations, but allow accessing underlying module attributes and methods
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
