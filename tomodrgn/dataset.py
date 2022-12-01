import numpy as np
from torch.utils import data
import os
import multiprocessing as mp
from multiprocessing import Pool

from tomodrgn import fft, mrc, utils, starfile, dose, ctf

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


class TiltSeriesMRCData(data.Dataset):
    '''
    Metaclass responsible for instantiating and querying per-particle object dataset objects
    Currently supports initializing mrcfile from .star exported by warp when generating particleseries
    '''

    def __init__(self, mrcfile, norm=None, invert_data=False, ind_ptcl=None, window=True, datadir=None, window_r=0.85,
                 recon_dose_weight=False, recon_tilt_weight=False, dose_override=None, l_dose_mask=False, lazy=True,
                 sequential_tilt_sampling=False):

        log('Parsing star file...')
        ptcls_star = starfile.TiltSeriesStarfile.load(mrcfile)

        # evaluate command line arguments affecting preprocessing
        ptcls_unique_list = ptcls_star.df['_rlnGroupName'].unique().astype(str)
        log(f'Found {len(ptcls_unique_list)} particles')
        if ind_ptcl is not None:
            log('Filtering particles by supplied indices...')
            ptcls_unique_list = ptcls_unique_list[ind_ptcl]
            ptcls_star.df = ptcls_star.df[ptcls_star.df['_rlnGroupName'].isin(ptcls_unique_list)]
            ptcls_star.df = ptcls_star.df.reset_index(drop=True)
            assert len(ptcls_star.df['_rlnGroupName'].unique().astype(str)) == len(ind_ptcl), 'Make sure particle indices file does not contain duplicates'
            log(f'Found {len(ptcls_unique_list)} particles after filtering')
        ptcls_to_imgs_ind = ptcls_star.get_ptcl_img_indices()  # either instantiate for the first time or update after ind_ptcl filtering
        ntilts_set = set(ptcls_star.df.groupby('_rlnGroupName').size().values)
        ntilts_min = min(ntilts_set)
        ntilts_max = max(ntilts_set)
        log(f'Found {ntilts_min} (min) to {ntilts_max} (max) tilt images per particle')
        log(f'Will sample tilt images per particle: {"sequentially" if sequential_tilt_sampling else "randomly"}')
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
        log('Loading particles...')
        particles = ptcls_star.get_particles(datadir=datadir, lazy=lazy)
        nx = ptcls_star.get_image_size(datadir=datadir)
        nimgs = len(particles)
        nptcls = len(ptcls_unique_list)
        if not lazy:
            log('Preprocessing particles...')
            assert particles.shape[-1] == particles.shape[-2], "Images must be square"
            assert nx % 2 == 0, "Image size must be even"

            # Real space window
            if window:
                m = window_mask(nx, window_r, .99)
                particles[:, :-1, :-1] *= m

            # compute HT
            log('Computing FFT')
            for i, img in enumerate(particles):
                particles[i, :-1, :-1] = fft.ht2_center(img[:-1, :-1])
            log('Converted to FFT')

            if invert_data:
                particles *= -1

            # symmetrize HT
            particles = fft.symmetrize_ht(particles, preallocated=True)
            _, ny_ht, nx_ht = particles.shape

            # normalize
            if norm is None:
                random_imgs_for_normalization = np.random.choice(np.arange(nimgs), size=nimgs//100, replace=False)
                norm = [np.mean(particles[random_imgs_for_normalization]), np.std(particles[random_imgs_for_normalization])]
                norm[0] = 0
            particles = (particles - norm[0]) / norm[1]
            log(f'Normalized HT by {norm[0]} +/- {norm[1]}')
            log(f'Finished loading and preprocessing {nptcls} {nx}x{nx} subtomo particleseries in memory')
        else:
            log('Will lazily load and preprocess particles on the fly')

        # parse rotations
        log('Loading rotations from star file')
        euler = np.zeros((nimgs, 3), dtype=np.float32)
        euler[:, 0] = ptcls_star.df['_rlnAngleRot']
        euler[:, 1] = ptcls_star.df['_rlnAngleTilt']
        euler[:, 2] = ptcls_star.df['_rlnAnglePsi']
        log('Euler angles (Rot, Tilt, Psi):')
        log(euler[0])
        log('Converting to rotation matrix:')
        rot = np.asarray([utils.R_from_relion(*x) for x in euler], dtype=np.float32)
        log(rot[0])

        # parse translations (if present, default none for warp-exported particleseries)
        log('Loading translations from star file (if any)')
        trans = np.zeros((nimgs, 2), dtype=np.float32)
        if '_rlnOriginX' in ptcls_star.headers and '_rlnOriginY' in ptcls_star.headers:
            trans[:, 0] = ptcls_star.df['_rlnOriginX'].to_numpy(dtype=np.float32)
            trans[:, 1] = ptcls_star.df['_rlnOriginY'].to_numpy(dtype=np.float32)
            log('Translations (pixels):')
            log(trans[0])
        else:
            trans = None
            log('Translations not found in star file. Reconstruction will not have translations applied.')

        # Loading CTF parameters from star file (if present)
        ctf_params = np.zeros((nimgs, 9), dtype=np.float32)
        ctf_columns = ['_rlnDetectorPixelSize', '_rlnDefocusU', '_rlnDefocusV',
                       '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration',
                       '_rlnAmplitudeContrast', '_rlnPhaseShift']
        if np.all([ctf_column in ptcls_star.df.columns for ctf_column in ctf_columns]):
            ctf_params[:, 0] = nx  # first column is real space box size
            for i, column in enumerate(ctf_columns):
                ctf_params[:, i + 1] = ptcls_star.df[column].to_numpy(dtype=np.float32)
            ctf.print_ctf_params(ctf_params[0])
        else:
            ctf_params = None
            log('CTF parameters not found in star file. Reconstruction will not have CTF applied.')

        # Calculating weighting schemes
        weight_mask_keys = []  # len(ptcls), associates small key per particle
        weights_dict = {}      # associates small keys with large weighting matrices
        dec_mask_dict = {}     # associates small keys with large masking matrices
        apix = ptcls_star.get_tiltseries_pixelsize()
        spatial_frequencies = dose.get_spatial_frequencies(apix, nx + 1)
        for name, group in ptcls_star.df.groupby('_rlnGroupName'):

            ntilts = len(group)
            weight_informing_cols = ['_rlnCtfScalefactor', '_rlnCtfBfactor']
            if np.all([col in group.columns for col in weight_informing_cols]):
                weights_key = group[weight_informing_cols].to_numpy(dtype=float).data.tobytes()
            else:
                weights_key = ntilts  # HACKY, will miscall starfiles with discontinuous tilt series but no scalefactor/bfactor columns

            if weights_key not in weights_dict.keys():
                tilt_weights = np.ones((ntilts, 1, 1))
                dose_weights = np.ones((ntilts, nx+1, nx+1))
                dec_mask = np.ones((ntilts, nx+1, nx+1))

                if recon_tilt_weight:
                    tilt_weights = group['_rlnCtfScalefactor'].to_numpy(dtype=float).reshape(ntilts, 1, 1)

                if recon_dose_weight or l_dose_mask:
                    voltage = ptcls_star.get_tiltseries_voltage()
                    if dose_override is None:
                        dose_series = starfile.get_tiltseries_dose_per_A2_per_tilt(group, ntilts)
                    else:
                        # increment scalar dose_override across ntilts
                        dose_series = dose_override * np.arange(1, ntilts + 1)
                    dose_weights = dose.calculate_dose_weights(dose_series, spatial_frequencies, voltage, nx+1)

                    if l_dose_mask:
                        dec_mask = dose_weights != 0.0

                    if recon_dose_weight:
                        pass
                    else:
                        dose_weights = np.ones((ntilts, nx+1, nx+1))

                weights_dict[weights_key] = dose_weights * tilt_weights
                dec_mask_dict[weights_key] = dec_mask
            weight_mask_keys.append(weights_key)

        log(f'Found {len(weights_dict.keys())} different weighting schemes')

        self.nimgs = nimgs
        self.nptcls = nptcls
        self.ntilts_range = [ntilts_min, ntilts_max]
        self.ntilts_training = ntilts_min

        self.ptcls = particles
        self.ptcls_to_imgs_ind = ptcls_to_imgs_ind
        self.D = nx+1
        self.lazy = lazy
        self.window = window
        self.window_r = window_r
        self.invert_data = invert_data
        self.sequential_tilt_sampling = sequential_tilt_sampling
        self.norm = norm if not lazy else self.lazy_particles_estimate_normalization()

        self.star = ptcls_star
        self.ptcls_list = ptcls_unique_list

        self.recon_tilt_weight = recon_tilt_weight
        self.recon_dose_weight = recon_dose_weight
        self.weight_mask_keys = weight_mask_keys
        self.dec_mask_dict = dec_mask_dict
        self.weights_dict = weights_dict
        self.spatial_frequencies = spatial_frequencies

        self.rot = rot
        self.trans = trans
        self.ctf_params = ctf_params

    def __len__(self):
        return self.nptcls

    def __getitem__(self, idx_ptcl):
        # get correct image indices for image, pose, and ctf params (indexed against entire dataset)
        ptcl_img_ind = self.ptcls_to_imgs_ind[idx_ptcl].flatten().astype(int)
        if self.sequential_tilt_sampling:
            zero_indexed_ind = np.arange(self.ntilts_training)  # take first ntilts_training images for deterministic loading/debugging
        else:
            zero_indexed_ind = np.asarray(np.random.choice(len(ptcl_img_ind), size=self.ntilts_training, replace=False))
        ptcl_img_ind = ptcl_img_ind[zero_indexed_ind]

        if self.lazy:
            images = np.asarray([self.ptcls[i].get() for i in ptcl_img_ind])
            if self.window:
                m = window_mask(self.D - 1, self.window_r, .99)
                images *= m
            for i, img in enumerate(images):
                images[i] = fft.ht2_center(img)
            if self.invert_data: images *= -1
            images = fft.symmetrize_ht(images)
            images = (images - self.norm[0]) / self.norm[1]
        else:
            images = self.ptcls[ptcl_img_ind]

        rot = self.rot[ptcl_img_ind]
        trans = 0 if self.trans is None else self.trans[ptcl_img_ind]  # tells train loop to skip translation block, bc no translation params provided and collate_fn does not allow returning None
        ctf = 0 if self.ctf_params is None else self.ctf_params[ptcl_img_ind]  # tells train loop to skip ctf weighting block, bc no ctf params provided and collate_fn does not allow returning None
        weights = self.weights_dict[self.weight_mask_keys[idx_ptcl]][zero_indexed_ind]
        dec_mask = self.dec_mask_dict[self.weight_mask_keys[idx_ptcl]][zero_indexed_ind]

        return images, rot, trans, ctf, weights, dec_mask, idx_ptcl

    def get(self, index):
        return self.ptcls[index]

    def lazy_particles_estimate_normalization(self):
        n = min(10000, self.nimgs)
        random_imgs_for_normalization = np.random.choice(self.nimgs, size=n, replace=False)
        imgs = np.asarray([self.ptcls[i].get() for i in random_imgs_for_normalization])
        if self.window:
            m = window_mask(self.D-1, self.window_r, 0.99)
            imgs *= m
        for i, img in enumerate(imgs):
            imgs[i] = fft.ht2_center(img)
        if self.invert_data: imgs *= -1
        imgs = fft.symmetrize_ht(imgs)
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        log(f'Normalizing HT by {norm[0]} +/- {norm[1]}')
        return norm
