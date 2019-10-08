'''Pytorch models'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fft
import lie_tools
import so3_grid
import utils

log = utils.log

class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(self, lattice, # Lattice object
            qlayers, qdim, 
            players, pdim,
            in_dim, zdim = 1, 
            encode_mode = 'resid',
            enc_mask = None,
            enc_type = 'linear_lowf',
            domain = 'fourier'):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, zdim*2)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(in_dim, 
                            qlayers, # nlayers
                            qdim,  # hidden_dim
                            zdim*2, # out_dim
                            nn.ReLU) 
        elif encode_mode == 'mlp':
            self.encoder = MLP(in_dim, 
                            qlayers, 
                            qdim, # hidden_dim
                            zdim*2, # out_dim
                            nn.ReLU) #in_dim -> hidden_dim
        elif encode_mode == 'tilt':
            self.encoder = TiltEncoder(in_dim,
                            qlayers,
                            qdim,
                            zdim*2,
                            nn.ReLU)
        else:
            raise RuntimeError('Encoder mode {} not recognized'.format(encode_mode))
        self.encode_mode = encode_mode
        if enc_type == 'none':
            model = ResidLinearMLP if domain == 'hartley' else FTSliceDecoder
            self.decoder = model(3+zdim,
                            lattice.D, # lattice size
                            players, # nlayers
                            pdim, # hidden dim
                            nn.ReLU)
        else:
            model = PositionalDecoder if domain == 'hartley' else FTPositionalDecoder 
            self.decoder = model(3+zdim, # input dim
                            lattice.D, # lattice size
                            players, # nlayers
                            pdim, # hidden dim
                            nn.ReLU,
                            enc_type=enc_type) #R3 -> R1
   
    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def encode(self, *img):
        img = (x.view(x.shape[0],-1) for x in img)
        if self.enc_mask is not None:
            img = (x[:,self.enc_mask] for x in img)
        z = self.encoder(*img)
        return z[:,:self.zdim], z[:,self.zdim:]

    def cat_z(self, coords, z):
        '''
        coords: Bx...x3
        z: Bxzdim
        '''
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1]*(coords.ndimension()-2)), self.zdim)
        z = torch.cat((coords,z.expand(*coords.shape[:-1],self.zdim)),dim=-1)
        return z

    def decode(self, coords, z, mask=None):
        '''
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        '''
        return self.decoder(self.cat_z(coords,z))

class PositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf'):
        super(PositionalDecoder, self).__init__()
        assert in_dim >= 3 
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_dim = self.D2
        self.enc_type = enc_type
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 1, activation)
     
    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
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

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2+1, dtype=torch.float) 
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

    def forward(self, coords):
        '''Input should be coordinates from [-.5,.5]'''
        assert (coords[...,0:3].abs() <= 0.5).all()
        return self.decoder(self.positional_encoding_geom(coords))

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume
        
        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization 
            zval: value of latent (zdim x 1)
        '''
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-0.5,0.5,D,endpoint=True)):
            x = coords/extent/2 + torch.tensor([0,0,dz])
            if zval is not None:
                x = torch.cat((x,z), dim=-1)
            with torch.no_grad():
                y = self.forward(x)
                y = y.view(D,D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f*norm[1]+norm[0]
        vol = fft.ihtn_center(vol_f[0:-1,0:-1,0:-1]) # remove last +k freq for inverse FFT
        return vol

class FTPositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf'):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_type = enc_type
        #self.enc_dim = max(100,self.D2)
        self.enc_dim = self.D2
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 2, activation)
    
    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
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

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2+1, dtype=torch.float) 
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
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        # if ignore_DC = False, then the size of the lattice will be odd (since it
        # includes the origin), so we need to evaluate one additional pixel
        c = lattice.shape[-2]//2 # top half
        cc = c + 1 if lattice.shape[-2] % 2 == 1 else c # include the origin
        assert abs(lattice[...,0:3].mean()) < 1e-8, '{} != 0.0'.format(lattice[...,0:3].mean())
        image = torch.empty(lattice.shape[:-1]) 
        top_half = self.decode(lattice[...,0:cc,:])
        image[..., 0:cc] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,cc:] = (top_half[...,0] + top_half[...,1])[...,np.arange(c-1,-1,-1)]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        assert (lattice[...,0:3].abs() <= 0.5).all()
        # convention: only evalute the -z points
        w = lattice[...,2] > 0.0
        lattice[...,0:3][w] = -lattice[...,0:3][w] # negate lattice coordinates where z > 0
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
            zval: value of latent (zdim x 1)
        '''
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-0.5,0.5,D,endpoint=True)):
            x = coords/extent/2 + torch.tensor([0,0,dz])
            if zval is not None:
                x = torch.cat((x,z), dim=-1)
            with torch.no_grad():
                if dz == 0.0:
                    y = self.forward(x)
                else:
                    y = self.decode(x)
                    y = y[...,0] - y[...,1]
                y = y.view(D,D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f*norm[1]+norm[0]
        vol_f = utils.zero_sphere(vol_f)
        vol = fft.ihtn_center(vol_f[:-1,:-1,:-1]) # remove last +k freq for inverse FFT
        return vol

class FTSliceDecoder(nn.Module):
    '''
    Evaluate a central slice out of a 3D FT of a model, returns representation in
    Hartley reciprocal space

    Exploits the symmetry of the FT where F*(x,y) = F(-x,-y) and only
    evaluates half of the lattice. The decoder is f(x,y,z) => real, imag
    '''
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation):
        '''D: image width or height'''
        super(FTSliceDecoder, self).__init__()
        self.decoder = ResidLinearMLP(in_dim, nlayers, hidden_dim, 2, activation)
        D2 = int(D/2)

        ### various pixel indices to keep track of for forward_even
        self.center = D2*D + D2 
        self.extra = np.arange((D2+1)*D, D**2, D) # bottom-left column without conjugate pair
        # evalute the top half of the image up through the center pixel 
        # and extra bottom-left column (todo: just evaluate a D-1 x D-1 image so 
        # we don't have to worry about this)
        self.all_eval = np.concatenate((np.arange(self.center+1), self.extra))
        
        # pixel indices for the top half of the image up to (but not incl) 
        # the center pixel and excluding the top row and left-most column
        i, j = np.meshgrid(np.arange(1,D),np.arange(1,D2+1))
        self.top = (j*D+i).ravel()[:-D2]

        # pixel indices for bottom half of the image after the center pixel
        # excluding left-most column and given in reverse order
        i, j =np.meshgrid(np.arange(1,D),np.arange(D2,D))
        self.bottom_rev = (j*D+i).ravel()[D2:][::-1].copy()

        self.D = D
        self.D2 = D2

    def forward(self, lattice):
        '''
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        assert lattice.shape[-2] % 2 == 1
        c = lattice.shape[-2]//2 # center pixel
        assert lattice[...,c,0:3].sum() == 0.0, '{} != 0.0'.format(lattice[...,c,0:3].sum())
        assert abs(lattice[...,0:3].mean()) < 1e-8, '{} != 0.0'.format(lattice[...,0:3].mean())
        image = torch.empty(lattice.shape[:-1]) 
        top_half = self.decode(lattice[...,0:c+1,:])
        image[..., 0:c+1] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,c+1:] = (top_half[...,0] + top_half[...,1])[...,np.arange(c-1,-1,-1)]
        return image

    def forward_even(self, lattice):
        '''Extra bookkeeping with extra row/column for an even sized DFT'''
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,self.all_eval,:])
        image[..., self.all_eval] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,self.bottom_rev] = top_half[...,self.top,0] + top_half[...,self.top,1]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        # convention: only evalute the -z points
        w = lattice[...,2] > 0.0
        lattice[...,0:3][w] = -lattice[...,0:3][w] # negate lattice coordinates where z > 0
        result = self.decoder(lattice)
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
            zval: value of latent (zdim x 1)
        '''
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True)):
            x = coords + torch.tensor([0,0,dz])
            if zval is not None:
                x = torch.cat((x,z), dim=-1)
            with torch.no_grad():
                y = self.decode(x)
                y = y[...,0] - y[...,1]
                y = y.view(D,D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f*norm[1]+norm[0]
        vol_f = utils.zero_sphere(vol_f)
        vol = fft.ihtn_center(vol_f[:-1,:-1,:-1]) # remove last +k freq for inverse FFT
        return vol

class VAE(nn.Module):
    def __init__(self, 
            lattice,
            qlayers, qdim, 
            players, pdim,
            encode_mode = 'mlp',
            no_trans = False,
            enc_mask = None
            ):
        super(VAE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.in_dim = lattice.D*lattice.D if enc_mask is None else enc_mask.sum()
        self.enc_mask = enc_mask
        assert qlayers > 2
        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, qdim)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(self.in_dim, 
                            qlayers-2, # -2 bc we add 2 more layers in the homeomorphic encoer
                            qdim,  # hidden_dim
                            qdim, # out_dim
                            nn.ReLU) #in_dim -> hidden_dim
        elif encode_mode == 'mlp':
            self.encoder = MLP(self.in_dim, 
                            qlayers-2, 
                            qdim, # hidden_dim
                            qdim, # out_dim
                            nn.ReLU) #in_dim -> hidden_dim
        else:
            raise RuntimeError('Encoder mode {} not recognized'.format(encode_mode))
        # predict rotation and translation in two completely separate NNs
        #self.so3_encoder = SO3reparameterize(qdim) # hidden_dim -> SO(3) latent variable
        #self.trans_encoder = ResidLinearMLP(nx*ny, 5, qdim, 4, nn.ReLU)

        # or predict rotation/translations from intermediate encoding
        self.so3_encoder = SO3reparameterize(qdim, 1, qdim) # hidden_dim -> SO(3) latent variable
        self.trans_encoder = ResidLinearMLP(qdim, 1, qdim, 4, nn.ReLU)

        self.decoder = FTSliceDecoder(3, self.D, players, pdim, nn.ReLU)
        self.no_trans = no_trans

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def encode(self, img):
        '''img: BxDxD'''
        img = img.view(img.size(0),-1)
        if self.enc_mask is not None:
            img = img[:,self.enc_mask]
        enc = nn.ReLU()(self.encoder(img))
        z_mu, z_std = self.so3_encoder(enc)
        if self.no_trans:
            tmu, tlogvar = (None, None)
        else:
            z = self.trans_encoder(enc)
            tmu, tlogvar = z[:,:2], z[:,2:]
        return z_mu, z_std, tmu, tlogvar

    def eval_volume(self, norm):
        return self.decoder.eval_volume(self.lattice.coords, self.D, self.lattice.extent, norm)

    def decode(self, rot):
        # transform lattice by rot.T
        x = self.lattice.coords @ rot # R.T*x
        y_hat = self.decoder(x)
        y_hat = y_hat.view(-1, self.D, self.D)
        return y_hat

    def forward(self, img):
        z_mu, z_std, tmu, tlogvar = self.encode(img)
        rot, w_eps = self.so3_encoder.sampleSO3(z_mu, z_std)
        # transform lattice by rot and predict image
        y_hat = self.decode(rot)
        if not self.no_trans:
            # translate image by t
            B = img.size(0)
            t = self.reparameterize(tmu, tlogvar)
            t = t.unsqueeze(1) # B x 1 x 2
            img = self.lattice.translate_ht(img.view(B,-1), t)
            img = img.view(B,self.D, self.D)
        return y_hat, img, z_mu, z_std, w_eps, tmu, tlogvar

class TiltVAE(nn.Module):
    def __init__(self, 
            lattice, tilt,
            qlayers, qdim, 
            players, pdim,
            no_trans=False,
            enc_mask=None
            ):
        super(TiltVAE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.in_dim = lattice.D*lattice.D if enc_mask is None else enc_mask.sum()
        self.enc_mask = enc_mask
        assert qlayers > 3
        self.encoder = ResidLinearMLP(self.in_dim,
                                      qlayers-3,
                                      qdim,
                                      qdim,
                                      nn.ReLU)
        self.so3_encoder = SO3reparameterize(2*qdim, 3, qdim) # hidden_dim -> SO(3) latent variable
        self.trans_encoder = ResidLinearMLP(2*qdim, 2, qdim, 4, nn.ReLU)
        self.decoder = FTSliceDecoder(3, self.D, players, pdim, nn.ReLU)
        assert tilt.shape == (3,3), 'Rotation matrix input required'
        self.tilt = torch.tensor(tilt)
        self.no_trans = no_trans

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def eval_volume(self, norm):
        return self.decoder.eval_volume(self.lattice.coords, self.D, self.lattice.extent, norm)

    def encode(self, img, img_tilt):
        img = img.view(img.size(0), -1)
        img_tilt = img_tilt.view(img_tilt.size(0), -1)
        if self.enc_mask is not None:
            img = img[:,self.enc_mask]
            img_tilt = img_tilt[:,self.enc_mask]
        enc1 = self.encoder(img)
        enc2 = self.encoder(img_tilt)
        enc = torch.cat((enc1,enc2), -1) # then nn.ReLU?
        z_mu, z_std = self.so3_encoder(enc)
        rot, w_eps = self.so3_encoder.sampleSO3(z_mu, z_std)
        if self.no_trans:
            tmu, tlogvar, t = (None,None,None)
        else:
            z = self.trans_encoder(enc)
            tmu, tlogvar = z[:,:2], z[:,2:]
            t = self.reparameterize(tmu, tlogvar)
        return z_mu, z_std, w_eps, rot, tmu, tlogvar, t

    def forward(self, img, img_tilt):
        B = img.size(0)
        z_mu, z_std, w_eps, rot, tmu, tlogvar, t = self.encode(img, img_tilt)
        if not self.no_trans:
            t = t.unsqueeze(1) # B x 1 x 2
            img = self.lattice.translate_ht(img.view(B,-1), -t)
            img_tilt = self.lattice.translate_ht(img_tilt.view(B,-1), -t)
            img = img.view(B, self.D, self.D)
            img_tilt = img_tilt.view(B, self.D, self.D)

        # rotate lattice by rot.T
        x = self.lattice.coords @ rot # R.T*x
        y_hat = self.decoder(x)
        y_hat = y_hat.view(-1, self.D, self.D)

        # tilt series pair
        x = self.lattice.coords @ self.tilt @ rot
        y_hat2 = self.decoder(x)
        y_hat2 = y_hat2.view(-1, self.D, self.D)
        return y_hat, y_hat2, img, img_tilt, z_mu, z_std, w_eps, tmu, tlogvar

# fixme: this is half-deprecated (not used in TiltVAE, but still used in tilt BNB)
class TiltEncoder(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(TiltEncoder, self).__init__()
        assert nlayers > 2
        self.encoder1 = ResidLinearMLP(in_dim, nlayers-2, hidden_dim, hidden_dim, activation)
        self.encoder2 = ResidLinearMLP(hidden_dim*2, 2, hidden_dim, out_dim, activation)
        self.in_dim = in_dim

    def forward(self, x, x_tilt):
        x_enc = self.encoder1(x)
        x_tilt_enc = self.encoder1(x_tilt)
        z = self.encoder2(torch.cat((x_enc,x_tilt_enc),-1))
        return z

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

    def forward(self, x):
        z = self.linear(x) + x
        return z

class MLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(MLP, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
      
# Adapted from soumith DCGAN
class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(ConvEncoder, self).__init__()
        ndf = hidden_dim
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, out_dim, 4, 1, 0, bias=False),
            # state size. out_dims x 1 x 1
        )
    def forward(self, x):
        x = x.view(-1,1,64,64)
        x = self.main(x)
        return x.view(x.size(0), -1) # flatten

class SO3reparameterize(nn.Module):
    '''Reparameterize R^N encoder output to SO(3) latent variable'''
    def __init__(self, input_dims, nlayers=None, hidden_dim=None):
        super().__init__()
        if nlayers is not None:
            self.main = ResidLinearMLP(input_dims, nlayers, hidden_dim, 9, nn.ReLU)
        else:
            self.main = nn.Linear(input_dims, 9)

        # start with big outputs
        #self.s2s2map.weight.data.uniform_(-5,5)
        #self.s2s2map.bias.data.uniform_(-5,5)

    def sampleSO3(self, z_mu, z_std):
        '''
        Reparameterize SO(3) latent variable
        # z represents mean on S2xS2 and variance on so3, which enocdes a Gaussian distribution on SO3
        # See section 2.5 of http://ethaneade.com/lie.pdf
        '''
        # resampling trick
        if not self.training:
            return z_mu, z_std
        eps = torch.randn_like(z_std)
        w_eps = eps*z_std
        rot_eps = lie_tools.expmap(w_eps)
        #z_mu = lie_tools.quaternions_to_SO3(z_mu)
        rot_sampled = z_mu @ rot_eps
        return rot_sampled, w_eps

    def forward(self, x):
        z = self.main(x)
        z1 = z[:,:3].double()
        z2 = z[:,3:6].double()
        z_mu = lie_tools.s2s2_to_SO3(z1,z2).float()
        logvar = z[:,6:]
        z_std = torch.exp(.5*logvar) # or could do softplus
        return z_mu, z_std

        

