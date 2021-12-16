import numpy as np
import torch
from torch.utils import data
import os
import multiprocessing as mp
from multiprocessing import Pool

from . import fft
from . import mrc
from . import utils
from . import starfile

log = utils.log

def load_particles(mrcs_txt_star, lazy=False, datadir=None, relion31=False):
    '''
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files, or a cryosparc particles.cs file

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    '''
    if mrcs_txt_star.endswith('.txt'):
        particles = mrc.parse_mrc_list(mrcs_txt_star, lazy=lazy)
    elif mrcs_txt_star.endswith('.star'):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir, lazy=lazy)
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(mrcs_txt_star) # assume .mrcs files are in the same director as the starfile
                particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir, lazy=lazy)
            else: raise RuntimeError(e)
    elif mrcs_txt_star.endswith('.cs'):
        particles = starfile.csparc_get_particles(mrcs_txt_star, datadir, lazy)
    else:
        particles, _ = mrc.parse_mrc(mrcs_txt_star, lazy=lazy)
    return particles


class LazyMRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file -- images loaded on the fly
    '''
    def __init__(self, mrcfile, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None, relion31=False, window_r=0.85):
        assert not keepreal, 'Not implemented error'
        particles = load_particles(mrcfile, True, datadir=datadir, relion31=relion31)
        if ind is not None:
            particles = [particles[x] for x in ind]
        N = len(particles)
        ny, nx = particles[0].get().shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))
        self.particles = particles
        self.N = N
        self.D = ny + 1 # after symmetrizing HT
        self.invert_data = invert_data
        if norm is None:
            norm = self.estimate_normalization()
        self.norm = norm
        self.window = window_mask(ny, window_r, .99) if window else None

    def estimate_normalization(self, n=1000):
        n = min(n,self.N)
        imgs = np.asarray([fft.ht2_center(self.particles[i].get()) for i in range(0,self.N, self.N//n)])
        if self.invert_data: imgs *= -1
        imgs = fft.symmetrize_ht(imgs)
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        log('Normalizing HT by {} +/- {}'.format(*norm))
        return norm

    def get(self, i):
        img = self.particles[i].get()
        if self.window is not None:
            img *= self.window
        img = fft.ht2_center(img).astype(np.float32)
        if self.invert_data: img *= -1
        img = fft.symmetrize_ht(img)
        img = (img - self.norm[0])/self.norm[1]
        return img

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.get(index), index

def window_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32), 
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r-in_rad)/(out_rad-in_rad)))
    return mask

class MRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file
    '''
    def __init__(self, mrcfile, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None, relion31=False, max_threads=16, window_r=0.85):
        if keepreal:
            raise NotImplementedError
        if ind is not None:
            particles = load_particles(mrcfile, True, datadir=datadir, relion31=relion31)
            particles = np.array([particles[i].get() for i in ind])
        else:
            particles = load_particles(mrcfile, False, datadir=datadir, relion31=relion31)
        N, ny, nx = particles.shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))

        # Real space window
        if window:
            log(f'Windowing images with radius {window_r}')
            particles *= window_mask(ny, window_r, .99)

        # compute HT
        log('Computing FFT')
        max_threads = min(max_threads, mp.cpu_count())
        if max_threads > 1:
            log(f'Spawning {max_threads} processes')
            with Pool(max_threads) as p:
                particles = np.asarray(p.map(fft.ht2_center, particles), dtype=np.float32)
        else:
            particles = np.asarray([fft.ht2_center(img) for img in particles], dtype=np.float32)
            log('Converted to FFT')
            
        if invert_data: particles *= -1

        # symmetrize HT
        log('Symmetrizing image data')
        particles = fft.symmetrize_ht(particles)

        # normalize
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))

        self.particles = particles
        self.N = N
        self.D = particles.shape[1] # ny + 1 after symmetrizing HT
        self.norm = norm
        self.keepreal = keepreal
        if keepreal:
            self.particles_real = particles_real
            log('Normalized real space images by {}'.format(particles_real.std()))
            self.particles_real /= particles_real.std()

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self, index):
        return self.particles[index]

class PreprocessedMRCData(data.Dataset):
    '''
    '''
    def __init__(self, mrcfile, norm=None, ind=None):
        particles = load_particles(mrcfile, False)
        if ind is not None:
            particles = particles[ind]
        log(f'Loaded {len(particles)} {particles.shape[1]}x{particles.shape[1]} images')
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))
        self.particles = particles
        self.N = len(particles)
        self.D = particles.shape[1] # ny + 1 after symmetrizing HT
        self.norm = norm

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self, index):
        return self.particles[index]

class TiltMRCData(data.Dataset):
    '''
    Class representing an .mrcs tilt series pair
    '''
    def __init__(self, mrcfile, mrcfile_tilt, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None, window_r=0.85):
        if ind is not None:
            particles_real = load_particles(mrcfile, True, datadir)
            particles_tilt_real = load_particles(mrcfile_tilt, True, datadir)
            particles_real = np.array([particles_real[i].get() for i in ind], dtype=np.float32)
            particles_tilt_real = np.array([particles_tilt_real[i].get() for i in ind], dtype=np.float32)
        else:
            particles_real = load_particles(mrcfile, False, datadir)
            particles_tilt_real = load_particles(mrcfile_tilt, False, datadir)

        N, ny, nx = particles_real.shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))
        assert particles_tilt_real.shape == (N, ny, nx), "Tilt series pair must have same dimensions as untilted particles"
        log('Loaded {} {}x{} tilt pair images'.format(N, ny, nx))

        # Real space window
        if window:
            m = window_mask(ny, window_r, .99)
            particles_real *= m
            particles_tilt_real *= m 

        # compute HT
        particles = np.asarray([fft.ht2_center(img) for img in particles_real]).astype(np.float32)
        particles_tilt = np.asarray([fft.ht2_center(img) for img in particles_tilt_real]).astype(np.float32)
        if invert_data: 
            particles *= -1
            particles_tilt *= -1

        # symmetrize HT
        particles = fft.symmetrize_ht(particles)
        particles_tilt = fft.symmetrize_ht(particles_tilt)

        # normalize
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        particles_tilt = (particles_tilt - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))

        self.particles = particles
        self.particles_tilt = particles_tilt
        self.norm = norm
        self.N = N
        self.D = particles.shape[1]
        self.keepreal = keepreal
        if keepreal:
            self.particles_real = particles_real
            self.particles_tilt_real = particles_tilt_real

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], self.particles_tilt[index], index

    def get(self, index):
        return self.particles[index], self.particles_tilt[index]

# TODO: LazyTilt

class LazyTiltSeriesMRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file -- particleseries of tilt-related images loaded on the fly
    Currently supports initializing mrcfile from .star exported by warp when generating particleseries
    '''
    def __init__(self, mrcfile, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None, relion31=False, window_r=0.85):
        assert not keepreal, 'Not implemented error'
        particles_df = starfile.TiltSeriesStarfile.load(mrcfile)
        nptcls, ntilts = particles_df.get_tiltseries_shape()
        expanded_ind_base = np.array([np.arange(i * ntilts, (i + 1) * ntilts) for i in range(nptcls)])  # 2D array [(ind_ptcl, inds_imgs)]
        if ind is not None:
            particles = particles_df.get_particles(datadir=datadir, lazy=True)  # shape(n_unique_ptcls * n_tilts, D, D)
            expanded_ind = expanded_ind_base[ind].reshape(-1) # 1D array [(ind_imgs_selected)]
            particles = [particles[i] for i in expanded_ind] # ind must be relative to each unique ptcl, not each unique image
            nptcls = int(len(particles)/ntilts)
        else:
            particles = particles_df.get_particles(datadir=datadir, lazy=True)
            expanded_ind = expanded_ind_base.reshape(-1)

        nimgs = len(particles)
        ny, nx = particles[0].get().shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Preparing to lazily load {} {}x{}x{} subtomo particleseries on-the-fly'.format(nptcls, ntilts, ny, nx))

        # particles = particles.reshape(nptcls, ntilts, ny_ht, nx_ht)  # reshape to 4-dim ptcl stack for DataLoader
        particles = [particles[ntilts*i : ntilts*(i+1)] for i in range(nptcls)] # reshape to list (of all ptcls) of lists (of tilt images for each ptcl)

        self.particles = particles
        self.nptcls = nptcls
        self.ntilts = ntilts
        self.nimgs = nimgs
        self.D = ny + 1 # what particles.shape[-1] will be after symmetrizing HT
        self.keepreal = keepreal
        self.expanded_ind = expanded_ind
        self.invert_data = invert_data
        if norm is None:
            norm = self.estimate_normalization()
        self.norm = norm
        self.window = window_mask(ny, window_r, .99) if window else None

    def estimate_normalization(self, n=1000): # samples the i%ntilts (randomish) tilt of nptcls//n randomly selected ptcls
        n = min(n,self.nimgs)
        random_ptcls_for_normalization = np.random.choice(np.arange(self.nptcls), self.nptcls//n, replace=False)
        imgs = np.asarray([fft.ht2_center(self.particles[i][i%self.ntilts].get()) for i in random_ptcls_for_normalization])
        if self.invert_data: imgs *= -1
        imgs = fft.symmetrize_ht(imgs)
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        log('Normalizing HT by {} +/- {}'.format(*norm))
        return norm

    def get(self, i):
        ptcl = np.asarray([ii.get() for ii in self.particles[i]], dtype=np.float32) # calls cryodrgn.mrc.LazyImage.get() to load each tilt_ii of particle_i
        if self.window is not None: ptcl *= self.window
        ptcl = np.asarray([fft.ht2_center(img) for img in ptcl], dtype=np.float32)
        if self.invert_data: ptcl *= -1
        ptcl = fft.symmetrize_ht(ptcl)
        ptcl = (ptcl - self.norm[0])/self.norm[1]
        return ptcl

    def __len__(self):
        return self.nptcls

    def __getitem__(self, index):
        return self.get(index), index

class TiltSeriesMRCData(data.Dataset):
    '''
    Class representing an .mrcs particleseries of tilt-related images
    Currently supports initializing mrcfile from .star exported by warp when generating particleseries
    '''

    def __init__(self, mrcfile, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None,
                 max_threads=16, window_r=0.85, do_dose_weighting=False, dose_override=None):
        particles_df = starfile.TiltSeriesStarfile.load(mrcfile)
        nptcls, ntilts = particles_df.get_tiltseries_shape()
        expanded_ind_base = np.array([np.arange(i * ntilts, (i + 1) * ntilts) for i in range(nptcls)])  # 2D array [(ind_ptcl, inds_imgs)]
        if ind is not None:
            particles = particles_df.get_particles(datadir=datadir, lazy=True)  # shape(n_unique_ptcls * n_tilts, D, D)
            expanded_ind = expanded_ind_base[ind].reshape(-1) # 1D array [(ind_imgs_selected)]
            particles = np.array([particles[i].get() for i in expanded_ind], dtype=np.float32) #note that ind must be relative to each unique ptcl, not each unique image
            nptcls = int(particles.shape[0]/ntilts)
        else:
            particles = particles_df.get_particles(datadir=datadir, lazy=False)
            expanded_ind = expanded_ind_base.reshape(-1)

        nimgs, ny, nx = particles.shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{}x{} subtomo particleseries'.format(nptcls, ntilts, ny, nx))

        # Real space window
        if window:
            m = window_mask(ny, window_r, .99)
            particles *= m

        # compute HT
        log('Computing FFT')
        max_threads = min(max_threads, mp.cpu_count())
        if max_threads > 1:
            log(f'Spawning {max_threads} processes')
            with Pool(max_threads) as p:
                particles = np.asarray(p.map(fft.ht2_center, particles), dtype=np.float32)
        else:
            particles = np.asarray([fft.ht2_center(img) for img in particles], dtype=np.float32)
        log('Converted to FFT')

        if invert_data:
            particles *= -1

        # symmetrize HT
        particles = fft.symmetrize_ht(particles)
        _, ny_ht, nx_ht = particles.shape

        # calculate and apply dose-weighting
        if do_dose_weighting:
            # for now, restricting users to single dose increment between each sequential tilt
            # assumes tilt images are ordered by collection sequence, i.e. ordered by increasing dose
            log('Calculating dose-weighting matrix')

            pixel_size = particles_df.get_tiltseries_pixelsize()  # angstroms per pixel
            voltage = particles_df.get_tiltseries_voltage()  # kV
            if dose_override is not None:
                dose_per_A2_per_tilt = particles_df.get_tiltseries_dose_per_A2_per_tilt()  # electrons per square angstrom per tilt micrograph
                log(f'Dose/A2/tilt extracted from star file: {dose_per_A2_per_tilt}')
            else:
                dose_per_A2_per_tilt = dose_override
                log(f'Dose/A2/tilt override supplied by user: {dose_per_A2_per_tilt}')

            # code adapted from Grigorieff lab summovie_1.0.2/src/core/electron_dose.f90
            # see also Grant and Grigorieff, eLife (2015) DOI: 10.7554/eLife.06980

            dose_weights = np.zeros((ntilts, ny_ht, nx_ht))
            fourier_pixel_sizes = 1.0 / (np.array([nx, ny]) / 2) # in units of 1/px
            box_center_indices = (np.array([nx, ny]) / 2).astype(int)
            critical_dose_at_dc = 2**31 # shorthand way to ensure dc component is always weighted ~1
            voltage_scaling_factor = 1.0 if voltage == 300 else 0.8 # 1.0 for 300kV, 0.8 for 200kV microscopes

            for k in range(ntilts):
                dose_at_start_of_tilt = k * dose_per_A2_per_tilt
                dose_at_end_of_tilt = (k+1) * dose_per_A2_per_tilt

                for j in range(ny_ht):
                    y = ((j - box_center_indices[1]) * fourier_pixel_sizes[1])

                    for i in range(nx_ht):
                        x = ((i - box_center_indices[0]) * fourier_pixel_sizes[0])

                        if ((i,j) == box_center_indices).all():
                            current_critical_dose = critical_dose_at_dc
                        else:
                            spatial_frequency = np.sqrt(x**2 + y**2) / pixel_size # units of 1/A
                            current_critical_dose = (0.24499 * spatial_frequency**(-1.6649) + 2.8141) * voltage_scaling_factor # eq 3 from DOI: 10.7554/eLife.06980

                        current_optimal_dose = 2.51284 * current_critical_dose

                        if (abs(dose_at_end_of_tilt - current_optimal_dose) < abs(dose_at_start_of_tilt - current_optimal_dose)):
                            dose_weights[k, j, i] = np.exp((-0.5 * dose_at_end_of_tilt) / current_critical_dose) # eq 5 from DOI: 10.7554/eLife.06980
                        else:
                            dose_weights[k,j,i] = 0.0
            assert dose_weights.min() >= 0.0
            assert dose_weights.max() <= 1.0
        else:
            log('Dose weighting not performed; all frequencies will be equally weighted for loss')
            dose_weights = np.ones((ntilts, ny_ht, nx_ht))

        # normalize
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))

        particles = particles.reshape(nptcls, ntilts, ny_ht, nx_ht)  # reshape to 4-dim ptcl stack for DataLoader

        self.particles = particles
        self.norm = norm
        self.nptcls = nptcls
        self.ntilts = ntilts
        self.D = particles.shape[-1]
        self.keepreal = keepreal
        self.expanded_ind = expanded_ind
        self.dose_weights = dose_weights
        if keepreal: raise NotImplementedError
            # self.particles_real = particles_real

    def __len__(self):
        return self.nptcls

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self, index):
        return self.particles[index]