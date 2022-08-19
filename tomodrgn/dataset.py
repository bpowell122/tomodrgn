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
from . import dose
from . import lattice

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
    def __init__(self, mrcfile, norm=None, invert_data=False, ind=None, window=True, datadir=None,
                 window_r=0.85, do_dose_weighting=False, dose_override=None, do_tilt_weighting=False):
        log('Parsing metadata...')
        particles_df = starfile.TiltSeriesStarfile.load(mrcfile)
        nptcls, ntilts = particles_df.get_tiltseries_shape()
        expanded_ind_base = np.array([np.arange(i * ntilts, (i + 1) * ntilts) for i in range(nptcls)])  # 2D array [(ind_ptcl, inds_imgs)]

        log('Loading particles...')
        if ind is not None:
            lazyparticles = particles_df.get_particles(datadir=datadir, lazy=True)  # shape(n_unique_ptcls * n_tilts, D, D)
            expanded_ind = expanded_ind_base[ind].reshape(-1) # 1D array [(ind_imgs_selected)]
            lazyparticles = [lazyparticles[i] for i in expanded_ind] # ind must be relative to each unique ptcl, not each unique image
            nptcls = int(len(lazyparticles)/ntilts)
        else:
            lazyparticles = particles_df.get_particles(datadir=datadir, lazy=True)
            expanded_ind = expanded_ind_base.reshape(-1)

        nimgs = len(lazyparticles)
        ny, nx = lazyparticles[0].get().shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Preparing to lazily load {} {}x{}x{} subtomo particleseries on-the-fly'.format(nptcls, ntilts, ny, nx))

        # calculate dose-weighting matrix
        ny_ht, nx_ht = ny+1, nx+1
        if do_dose_weighting:
            # for now, restricting users to single dose increment between each sequential tilt
            # assumes tilt images are ordered by collection sequence, i.e. ordered by increasing dose
            log('Calculating dose-weighting matrix')
            dose_weights = dose.calculate_dose_weights(particles_df, dose_override, ntilts, ny_ht, nx_ht, nx, ny)
            spatial_frequencies = dose.get_spatial_frequencies(particles_df, ny_ht, nx_ht, nx, ny)
        else:
            log('Dose weighting not performed; all frequencies will be equally weighted')
            dose_weights = np.ones((ntilts, ny_ht, nx_ht))
            spatial_frequencies = dose.get_spatial_frequencies(particles_df, ny_ht, nx_ht, nx, ny)

        # weight by cosine of tilt angle following relion1.4 convention for sample thickness
        if do_tilt_weighting:
            log('Using cosine(tilt_angle) weighting')
            cosine_weights = particles_df.get_tiltseries_cosine_weight(ntilts)
            dose_weights *= cosine_weights.reshape(ntilts,1,1)
        else:
            log('Cosing(tilt_angle) weighting not performed; all tilt angles will be equally weighted')
        cumulative_weights = dose_weights

        # particles = particles.reshape(nptcls, ntilts, ny_ht, nx_ht)  # reshape to 4-dim ptcl stack for DataLoader
        lazyparticles = [lazyparticles[ntilts*i : ntilts*(i+1)] for i in range(nptcls)] # reshape to list (of all ptcls) of lists (of tilt images for each ptcl)

        self.particles = lazyparticles
        self.nptcls = nptcls
        self.ntilts = ntilts
        self.nimgs = nimgs
        self.D = ny + 1 # what particles.shape[-1] will be after symmetrizing HT
        self.expanded_ind = expanded_ind
        self.invert_data = invert_data
        self.cumulative_weights = cumulative_weights
        self.spatial_frequencies = spatial_frequencies
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

    def __init__(self, mrcfile, norm=None, invert_data=False, ind=None, window=True, datadir=None,
                 window_r=0.85, do_dose_weighting=False, dose_override=None, do_tilt_weighting=False):
        log('Parsing metadata...')
        particles_df = starfile.TiltSeriesStarfile.load(mrcfile)
        nptcls, ntilts = particles_df.get_tiltseries_shape()
        expanded_ind_base = np.array([np.arange(i * ntilts, (i + 1) * ntilts) for i in range(nptcls)])  # 2D array [(ind_ptcl, inds_imgs)]

        log('Loading particles...')
        if ind is not None:
            lazyparticles = particles_df.get_particles(datadir=datadir, lazy=True)  # shape(n_unique_ptcls * n_tilts, D, D)
            expanded_ind = expanded_ind_base[ind].reshape(-1) # 1D array [(ind_imgs_selected)]
            lazyparticles = [lazyparticles[i] for i in expanded_ind]
            nptcls = int(len(lazyparticles)/ntilts)
            D = lazyparticles[0].shape[0]

            # preallocating numpy array for in-place loading, fourier transform, fourier transform centering, etc
            particles = np.empty((len(expanded_ind), D + 1, D + 1), dtype=np.float32)
            for i, img in enumerate(lazyparticles): particles[i, :-1, :-1] = img.get()

        else:
            particles = particles_df.get_particles(datadir=datadir, lazy=False)
            expanded_ind = expanded_ind_base.reshape(-1)

        nimgs, ny, nx = np.subtract(particles.shape, (0, 1, 1))
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{}x{} subtomo particleseries'.format(nptcls, ntilts, ny, nx))

        # Real space window
        if window:
            m = window_mask(ny, window_r, .99)
            particles[:,:-1,:-1] *= m

        # compute HT
        log('Computing FFT')
        for i, img in enumerate(particles):
            particles[i,:-1,:-1] = fft.ht2_center(img[:-1,:-1])
        log('Converted to FFT')

        if invert_data:
            particles *= -1

        # symmetrize HT
        particles = fft.symmetrize_ht(particles, preallocated=True)
        _, ny_ht, nx_ht = particles.shape

        # calculate dose-weighting matrix
        if do_dose_weighting:
            log('Calculating dose-weighting matrix')
            dose_weights = dose.calculate_dose_weights(particles_df, dose_override, ntilts, ny_ht, nx_ht, nx, ny)
            spatial_frequencies = dose.get_spatial_frequencies(particles_df, ny_ht, nx_ht, nx, ny)
        else:
            log('Dose weighting not performed; all frequencies will be equally weighted')
            dose_weights = np.ones((ntilts, ny_ht, nx_ht))
            spatial_frequencies = dose.get_spatial_frequencies(particles_df, ny_ht, nx_ht, nx, ny)

        # weight by cosine of tilt angle following relion1.4 convention for sample thickness
        if do_tilt_weighting:
            log('Using cosine(tilt_angle) weighting')
            cosine_weights = particles_df.get_tiltseries_cosine_weight(ntilts)
            dose_weights *= cosine_weights.reshape(ntilts,1,1)
        else:
            log('Cosine(tilt_angle) weighting not performed; all tilt angles will be equally weighted')
        cumulative_weights = dose_weights

        # normalize
        if norm is None:
            random_ptcls_for_normalization = np.random.choice(np.arange(nimgs), nimgs // 100, replace=False)
            norm = [np.mean(particles[random_ptcls_for_normalization]), np.std(particles[random_ptcls_for_normalization])]
            norm[0] = 0
        # particles = (particles - norm[0])/norm[1]
        particles /= norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))

        particles = particles.reshape(nptcls, ntilts, ny_ht, nx_ht)  # reshape to 4-dim ptcl stack for DataLoader

        self.particles = particles
        self.norm = norm
        self.nptcls = nptcls
        self.ntilts = ntilts
        self.D = particles.shape[-1]
        self.expanded_ind = expanded_ind
        self.cumulative_weights = cumulative_weights
        self.spatial_frequencies = spatial_frequencies

    def __len__(self):
        return self.nptcls

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self, index):
        return self.particles[index]


class TiltSeriesDatasetMaster(data.Dataset):
    '''
    Metaclass responsible for instantiating and querying per-particle object dataset objects
    Currently supports initializing mrcfile from .star exported by warp when generating particleseries
    '''

    def __init__(self, mrcfile, norm=None, invert_data=False, ind_ptcl=None, window=True, datadir=None, window_r=0.85,
                 recon_dose_weight=False, recon_tilt_weight=False, dose_override=None, l_dose_mask=False, lazy=True):
        log('Parsing metadata...')
        ptcls_star = starfile.TiltSeriesStarfile.load(mrcfile)

        # evaluate command line arguments affecting preprocessing
        if ind_ptcl is not None:
            log('Filtering by supplied indices...')
            ptcls_unique_list = ptcls_star.df['_rlnGroupName'].unique()[ind_ptcl].astype(str)
        else:
            ptcls_unique_list = ptcls_star.df['_rlnGroupName'].unique().astype(str)
        log(f'Will window images with radius {window_r}')
        if recon_dose_weight:
            log('Will calculate weights due to incremental dose; frequencies will be weighted by exposure-dependent amplitude attenuation for pixelwise loss')
        else:
            log('Will not perform dose weighting; all frequencies will be equally weighted for pixelwise loss')
        if recon_tilt_weight:
            log('Will calculate weights due to tilt angle; frequencies will be weighted by cosine(tilt_angle) for pixelwise loss')
        else:
            log('Will not perform tilt weighting; all tilt angles will be equally weighted for pixelwise loss')

        # load and preprocess all particles
        log('Loading and preprocessing particles...')
        ptcls_unique_objects = {}
        nptcls = len(ptcls_unique_list)
        ntilts_range = [0, 0]
        weights_dict = {}
        dec_mask_dict = {}
        utils.print_progress_bar(0, nptcls, prefix='Progress:', length=50)
        for i, ptcl_id in enumerate(ptcls_unique_list):
            utils.print_progress_bar(i + 1, nptcls, prefix='Progress:', length=50)
            ptcls_star_subset = starfile.TiltSeriesStarfile(ptcls_star.headers, ptcls_star.df[ptcls_star.df['_rlnGroupName'] == ptcl_id])

            if lazy:
                ptcl = LazyTiltSeriesDatasetPerParticle(ptcls_star_subset, datadir=datadir)
            else:
                ptcl = TiltSeriesDatasetPerParticle(ptcls_star_subset, invert_data=invert_data, window=window, datadir=datadir, window_r=window_r)
            ptcls_unique_objects[ptcl_id] = ptcl

            # check if ptcls_star_subset has corresponding precalculated weights; cache in dict as {input bytes: ntilts x D x D array}
            weights_key = ptcls_star_subset.df[['_rlnCtfScalefactor', '_rlnCtfBfactor']].to_numpy(dtype=float).data.tobytes()
            if weights_key not in weights_dict.keys():
                tilt_weights = np.ones((ptcl.ntilts, 1, 1))
                dose_weights = np.ones((ptcl.ntilts, ptcl.D, ptcl.D))
                dec_mask = np.ones((ptcl.ntilts, ptcl.D, ptcl.D))

                if recon_tilt_weight:
                    tilt_weights = ptcls_star_subset.df['_rlnCtfScalefactor'].to_numpy(dtype=float).reshape(ptcl.ntilts, 1, 1)

                if recon_dose_weight or l_dose_mask:
                    dose_weights = dose.calculate_dose_weights(ptcls_star_subset, dose_override, ptcl.ntilts, ptcl.D, ptcl.D, ptcl.D - 1, ptcl.D - 1)

                    if l_dose_mask:
                        dec_mask = dose_weights != 0.0

                    if recon_dose_weight:
                        pass
                    else:
                        dose_weights = np.ones((ptcl.ntilts, ptcl.D, ptcl.D))

                weights_dict[weights_key] = dose_weights * tilt_weights
                dec_mask_dict[weights_key] = dec_mask
            ptcl.weights_key = weights_key

            # update min and max number of tilts across all particles
            if ptcl.ntilts > any(ntilts_range):
                ntilts_range[0] = ptcl.ntilts
                if ptcl.ntilts > ntilts_range[1]:
                    ntilts_range[1] = ptcl.ntilts

        # check particle boxsize consistency
        D = list(set([ptcls_unique_objects[ptcl_id].D for ptcl_id in ptcls_unique_list]))
        assert(len(D) == 1), f'All particles must have the same boxsize! Found boxsizes: {[d-1 for d in D]}'
        D = D[0]
        log(f'Loaded {nptcls} {D-1}x{D-1} subtomo particleseries with {ntilts_range[0]} to {ntilts_range[1]} tilts')

        # check how many weighting schemes were found
        log(f'Found {len(weights_dict.keys())} different weighting schemes')

        # calculate spatial frequencies matrix
        spatial_frequencies = dose.get_spatial_frequencies(ptcls_star.df, D, D, D-1, D-1)

        # normalize by subsampled mean and stdev
        if norm is None:
            random_ptcls_for_normalization = np.random.choice(ptcls_unique_list, min(1000, nptcls // 10), replace=False)
            if lazy:
                random_data_for_normalization = np.array([img.get() for ptcl_id in random_ptcls_for_normalization for img in ptcls_unique_objects[ptcl_id].images])
            else:
                random_data_for_normalization = np.array([ptcls_unique_objects[ptcl_id].images for ptcl_id in random_ptcls_for_normalization])
            norm = [np.mean(random_data_for_normalization), np.std(random_data_for_normalization)]
            norm[0] = 0
        if not lazy:
            for ptcl_id in ptcls_unique_objects:
                ptcls_unique_objects[ptcl_id].images = (ptcls_unique_objects[ptcl_id].images - norm[0]) / norm[1]
            log(f'Normalized HT by {norm[0]} +/- {norm[1]}')

        self.nptcls = nptcls
        self.ptcls = ptcls_unique_objects
        self.ptcls_list = ptcls_unique_list
        self.norm = norm
        self.D = D
        self.ntilts_range = ntilts_range
        self.weights_dict = weights_dict
        self.spatial_frequencies = spatial_frequencies
        self.ntilts_training = ntilts_range[0]
        self.star = ptcls_star
        self.recon_tilt_weight = recon_tilt_weight
        self.recon_dose_weight = recon_dose_weight
        self.dec_mask_dict = dec_mask_dict
        self.lazy = lazy
        self.window = window
        self.window_r = window_r
        self.invert_data = invert_data

    def __len__(self):
        return self.nptcls

    def __getitem__(self, idx_ptcl):
        # randomly select self.ntilts_training tilt images for a given particle to train
        ptcl_id = self.ptcls_list[idx_ptcl]
        ptcl = self.ptcls[ptcl_id]
        tilt_inds = np.asarray(np.random.choice(ptcl.ntilts, self.ntilts_training, replace=False))

        if self.lazy:
            images = ptcl.images
            images = np.asarray([image.get() for image in images])[tilt_inds]
            if self.window:
                m = window_mask(self.D - 1, self.window_r, .99)
                images *= m
            for i, img in enumerate(images):
                images[i] = fft.ht2_center(img)
            if self.invert_data: images *= -1
            images = fft.symmetrize_ht(images)
            images = (images - self.norm[0]) / self.norm[1]

        else:
            images = ptcl.images [tilt_inds]

        rot = ptcl.rot[tilt_inds]
        trans = 0 if ptcl.trans is None else ptcl.trans[tilt_inds]  # tells train loop to skip translation block, bc no translation params provided and collate_fn does not allow returning None
        ctf = 0 if ptcl.ctf is None else ptcl.ctf[tilt_inds]  # tells train loop to skip ctf weighting block, bc no ctf params provided and collate_fn does not allow returning None
        weights = self.weights_dict[ptcl.weights_key][tilt_inds]
        dec_mask = self.dec_mask_dict[ptcl.weights_key][tilt_inds]

        return images, rot, trans, ctf, weights, dec_mask, ptcl_id

    def get(self, index):
        return self.ptcls[index]

class TiltSeriesDatasetPerParticle(data.Dataset):
    '''
    Class responsible for instantiating a single per-particle object dataset
    Currently supports initializing mrcfile from .star exported by warp when generating particleseries
    '''
    def __init__(self, ptcls_star_subset, invert_data=False, window=True, datadir=None, window_r=0.85):

        images = ptcls_star_subset.get_particles(datadir=datadir, lazy=False)

        ntilts, ny, nx = np.subtract(images.shape, (0, 1, 1))  # get_particles(lazy=False) returns preallocated ny+1, nx+1 array
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"

        # Real space window
        if window:
            m = window_mask(ny, window_r, .99)
            images[:,:-1,:-1] *= m

        # compute HT
        for i, img in enumerate(images):
            images[i,:-1,:-1] = fft.ht2_center(img[:-1,:-1])

        if invert_data: images *= -1

        # symmetrize HT
        images = fft.symmetrize_ht(images, preallocated=True)
        _, ny_ht, nx_ht = images.shape

        # parse rotations
        euler = np.zeros((ntilts, 3), dtype=np.float32)
        euler[:, 0] = ptcls_star_subset.df['_rlnAngleRot']
        euler[:, 1] = ptcls_star_subset.df['_rlnAngleTilt']
        euler[:, 2] = ptcls_star_subset.df['_rlnAnglePsi']
        rot = np.asarray([utils.R_from_relion(*x) for x in euler], dtype=np.float32)

        # parse translations (default none for warp-exported particleseries)
        trans = np.zeros((ntilts, 2), dtype=np.float32)
        if '_rlnOriginX' in ptcls_star_subset.headers and '_rlnOriginY' in ptcls_star_subset.headers:
            trans[:, 0] = ptcls_star_subset.df['_rlnOriginX']
            trans[:, 1] = ptcls_star_subset.df['_rlnOriginY']
        else:
            trans = None

        # parse ctf parameters
        ctf_params = np.zeros((ntilts, 9), dtype=np.float32)
        ctf_columns = ['_rlnDetectorPixelSize','_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration',
                 '_rlnAmplitudeContrast', '_rlnPhaseShift']
        if np.all([ctf_column in ptcls_star_subset.headers for ctf_column in ctf_columns]):
            ctf_params[:, 0] = nx  # first column is real space box size
            for i, header in enumerate(ctf_columns):
                ctf_params[:, i + 1] = ptcls_star_subset.df[header]
        else:
            ctf_params = None

        self.images = images
        self.ntilts = ntilts
        self.D = images.shape[-1]
        self.weights_key = None
        self.rot = rot
        self.trans = trans
        self.ctf = ctf_params

class LazyTiltSeriesDatasetPerParticle(data.Dataset):
    '''
    Class responsible for lazily instantiating a single per-particle object dataset
    Currently supports initializing mrcfile from .star exported by warp when generating particleseries
    '''
    def __init__(self, ptcls_star_subset, datadir=None):

        images = ptcls_star_subset.get_particles(datadir=datadir, lazy=True)
        ntilts = len(images)
        ny, nx = images[0].get().shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"

        # parse rotations
        euler = np.zeros((ntilts, 3), dtype=np.float32)
        euler[:, 0] = ptcls_star_subset.df['_rlnAngleRot']
        euler[:, 1] = ptcls_star_subset.df['_rlnAngleTilt']
        euler[:, 2] = ptcls_star_subset.df['_rlnAnglePsi']
        rot = np.asarray([utils.R_from_relion(*x) for x in euler], dtype=np.float32)

        # parse translations (default none for warp-exported particleseries)
        trans = np.zeros((ntilts, 2), dtype=np.float32)
        if '_rlnOriginX' in ptcls_star_subset.headers and '_rlnOriginY' in ptcls_star_subset.headers:
            trans[:, 0] = ptcls_star_subset.df['_rlnOriginX']
            trans[:, 1] = ptcls_star_subset.df['_rlnOriginY']
        else:
            trans = None

        # parse ctf parameters
        ctf_params = np.zeros((ntilts, 9), dtype=np.float32)
        ctf_columns = ['_rlnDetectorPixelSize','_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration',
                 '_rlnAmplitudeContrast', '_rlnPhaseShift']
        if np.all([ctf_column in ptcls_star_subset.headers for ctf_column in ctf_columns]):
            ctf_params[:, 0] = nx  # first column is real space box size
            for i, header in enumerate(ctf_columns):
                ctf_params[:, i + 1] = ptcls_star_subset.df[header]
        else:
            ctf_params = None

        self.images = images
        self.ntilts = ntilts
        self.D = nx + 1
        self.weights_key = None
        self.rot = rot
        self.trans = trans
        self.ctf = ctf_params
