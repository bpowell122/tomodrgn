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
    mask = np.where((r < out_rad) & (r > in_rad),
                    (1 + np.cos((r-in_rad)/(out_rad-in_rad) * np.pi)) / 2,
                    1)
    mask = np.where(r < out_rad,
                    mask,
                    0)
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

    def __init__(self, ptcls_star, norm=None, invert_data=False, ind_ptcl=None, window=True, datadir=None, window_r=0.8,
                 window_r_outer=0.9, recon_dose_weight=False, recon_tilt_weight=False, dose_override=None, l_dose_mask=False,
                 lazy=True, sequential_tilt_sampling=False, ind_img=None):

        # filter by test/train split per-image
        if ind_img is not None:
            ptcls_star.df = ptcls_star.df.iloc[ind_img].reset_index(drop=True)

        # evaluate command line arguments affecting preprocessing
        ptcls_unique_list = ptcls_star.df[ptcls_star.header_uid].unique()
        log(f'Found {len(ptcls_unique_list)} particles')
        if ind_ptcl is not None:
            log('Filtering particles by supplied indices...')
            if type(ind_ptcl) is str and ind_ptcl.endswith('.pkl'):
                ind_ptcl = utils.load_pkl(ind_ptcl)
            ptcls_unique_list = ptcls_unique_list[ind_ptcl]
            ptcls_star.df = ptcls_star.df[ptcls_star.df[ptcls_star.header_uid].isin(ptcls_unique_list)]
            ptcls_star.df = ptcls_star.df.reset_index(drop=True)
            assert len(ptcls_star.df[ptcls_star.header_uid].unique().astype(str)) == len(ind_ptcl), 'Make sure particle indices file does not contain duplicates'
            log(f'Found {len(ptcls_unique_list)} particles after filtering')
        ptcls_to_imgs_ind = ptcls_star.get_ptcl_img_indices()  # either instantiate for the first time or update after ind_ptcl filtering
        ntilts_set = set(ptcls_star.df.groupby(ptcls_star.header_uid, sort=False).size().values)
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
        particles = ptcls_star.get_particles_stack(datadir=datadir, lazy=lazy)
        nx = ptcls_star.get_image_size(datadir=datadir)
        nimgs = len(particles)
        nptcls = len(ptcls_unique_list)
        if not lazy:
            log('Preprocessing particles...')
            assert particles.shape[-1] == particles.shape[-2], "Images must be square"
            assert nx % 2 == 0, "Image size must be even"

            # Real space window
            if window:
                log('Windowing particles...')
                m = window_mask(nx, window_r, window_r_outer)
                particles[:, :-1, :-1] *= m

            # compute HT
            log('Computing FFT...')
            for i, img in enumerate(particles):
                particles[i, :-1, :-1] = fft.ht2_center(img[:-1, :-1])
            log('Converted to FFT')

            if invert_data:
                log('Inverting data sign...')
                particles *= -1

            # symmetrize HT
            log('Symmetrizing HT...')
            particles = fft.symmetrize_ht(particles, preallocated=True)
            _, ny_ht, nx_ht = particles.shape

            # normalize
            if norm is None:
                log('Calculating normalization factor...')
                random_imgs_for_normalization = np.random.choice(np.arange(nimgs), size=nimgs//100, replace=False)
                norm = [np.mean(particles[random_imgs_for_normalization]), np.std(particles[random_imgs_for_normalization])]
                norm[0] = 0
            # particles = (particles - norm[0]) / norm[1]
            particles -= norm[0]  # zero mean
            particles /= norm[1]  # unit stdev, separate line required to avoid redundant memory allocation
            log(f'Normalized HT by {norm[0]} +/- {norm[1]}')
            log(f'Finished loading and preprocessing {nptcls} {nx}x{nx} subtomo particleseries in memory')
        else:
            log('Will lazily load and preprocess particles on the fly')

        # parse rotations
        log('Loading rotations from star file')
        euler = ptcls_star.df[ptcls_star.headers_rot].to_numpy(dtype=np.float32)
        log('Euler angles (Rot, Tilt, Psi):')
        log(euler[0])
        log('Converting to rotation matrix:')
        rot = np.asarray([utils.R_from_relion(*x) for x in euler], dtype=np.float32)
        log(rot[0])

        # parse translations (if present, default none for warp-exported particleseries)
        log('Loading translations from star file (if any)')
        if all(header_trans in ptcls_star.headers for header_trans in ptcls_star.headers_trans):
            trans = ptcls_star.df[ptcls_star.headers_trans].to_numpy(dtype=np.float32)
            log('Translations (pixels):')
            log(trans[0])
        else:
            trans = None
            log('Translations not found in star file. Reconstruction will not have translations applied.')

        # Loading CTF parameters from star file (if present)
        ctf_params = np.zeros((nimgs, 9), dtype=np.float32)
        if all(header_ctf in ptcls_star.headers for header_ctf in ptcls_star.headers_ctf):
            ctf_params[:, 0] = nx  # first column is real space box size
            for i, column in enumerate(ptcls_star.headers_ctf):
                ctf_params[:, i + 1] = ptcls_star.df[column].to_numpy(dtype=np.float32)
            ctf.print_ctf_params(ctf_params[0])
        else:
            ctf_params = None
            log('CTF parameters not found in star file. Reconstruction will not have CTF applied.')

        # Getting relevant properties and precalculating appropriate arrays/masks for possible masking/weighting
        angpix = ptcls_star.get_tiltseries_pixelsize()
        voltage = ptcls_star.get_tiltseries_voltage()
        spatial_frequencies = dose.calculate_spatial_frequencies(angpix, nx+1)
        spatial_frequencies_critical_dose = dose.calculate_critical_dose_per_frequency(spatial_frequencies, voltage)
        circular_mask = dose.calculate_circular_mask(nx+1)

        # Saving relevant values as attributes of the class for future use
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
        self.window_r_outer = window_r_outer
        self.invert_data = invert_data
        self.sequential_tilt_sampling = sequential_tilt_sampling
        self.norm = norm if norm is not None else self.lazy_particles_estimate_normalization()

        self.star = ptcls_star
        self.ptcls_list = ptcls_unique_list

        self.recon_tilt_weight = recon_tilt_weight
        self.recon_dose_weight = recon_dose_weight
        self.l_dose_mask = l_dose_mask
        self.angpix = angpix
        self.voltage = voltage
        self.spatial_frequencies = spatial_frequencies
        self.spatial_frequencies_critical_dose = spatial_frequencies_critical_dose
        self.circular_mask = circular_mask

        self.rot = rot
        self.trans = trans
        self.ctf_params = ctf_params

    def __len__(self):
        return self.nptcls

    def __getitem__(self, idx_ptcl):
        # get correct image indices for image, pose, and ctf params (indexed against entire dataset)
        ptcl_img_ind = self.ptcls_to_imgs_ind[idx_ptcl].astype(int)

        # determine the number of images and related parameters to get
        if self.ntilts_training is None:
            ntilts_training = len(ptcl_img_ind)
        else:
            ntilts_training = self.ntilts_training

        # determine the order in which to return the images and related parameters
        if self.sequential_tilt_sampling:
            zero_indexed_ind = np.arange(ntilts_training)  # take first ntilts_training images for deterministic loading/debugging
        else:
            zero_indexed_ind = np.asarray(np.random.choice(len(ptcl_img_ind), size=ntilts_training, replace=False))
        ptcl_img_ind = ptcl_img_ind[zero_indexed_ind]

        # load and preprocess the images to be returned
        if self.lazy:
            images = np.asarray([self.ptcls[i].get() for i in ptcl_img_ind])
            if self.window:
                m = window_mask(self.D - 1, self.window_r, self.window_r_outer)
                images *= m
            for i, img in enumerate(images):
                images[i] = fft.ht2_center(img)
            if self.invert_data: images *= -1
            images = fft.symmetrize_ht(images)
            images = (images - self.norm[0]) / self.norm[1]
        else:
            images = self.ptcls[ptcl_img_ind]

        # get metadata to be used in calculating what to return
        cumulative_doses = self.star.df[self.star.header_dose].iloc[ptcl_img_ind].to_numpy(dtype=np.float32)
        if self.star.header_tilt:
            tilts = self.star.df[self.star.header_tilt].iloc[ptcl_img_ind].to_numpy(dtype=np.float32)
        else:
            if self.recon_tilt_weight:
                raise NotImplementedError

        # get the associated metadata to be returned
        rot = self.rot[ptcl_img_ind]
        trans = 0 if self.trans is None else self.trans[ptcl_img_ind]  # tells train loop to skip translation block, bc no translation params provided and collate_fn does not allow returning None
        ctf = 0 if self.ctf_params is None else self.ctf_params[ptcl_img_ind]  # tells train loop to skip ctf weighting block, bc no ctf params provided and collate_fn does not allow returning None
        if self.recon_dose_weight or self.l_dose_mask:
            dose_weights = dose.calculate_dose_weights(self.spatial_frequencies_critical_dose, cumulative_doses)
            decoder_mask = dose.calculate_dose_mask(dose_weights, self.circular_mask) if self.l_dose_mask else np.repeat(self.circular_mask[np.newaxis,:,:], ntilts_training, axis=0)
        else:
            dose_weights = np.ones((self.ntilts_training, self.D, self.D))
            decoder_mask = np.repeat(self.circular_mask[np.newaxis,:,:], ntilts_training, axis=0)
        if self.recon_tilt_weight:
            tilt_weights = np.cos(tilts)
        else:
            tilt_weights = np.ones((ntilts_training))
        decoder_weights = dose.combine_dose_tilt_weights(dose_weights, tilt_weights)

        return images, rot, trans, ctf, decoder_weights, decoder_mask, idx_ptcl

    def get(self, index):
        return self.ptcls[index]

    def lazy_particles_estimate_normalization(self):
        n = min(10000, self.nimgs)
        random_imgs_for_normalization = np.random.choice(self.nimgs, size=n, replace=False)
        imgs = np.asarray([self.ptcls[i].get() for i in random_imgs_for_normalization])
        if self.window:
            m = window_mask(self.D-1, self.window_r, self.window_r_outer)
            imgs *= m
        for i, img in enumerate(imgs):
            imgs[i] = fft.ht2_center(img)
        if self.invert_data: imgs *= -1
        imgs = fft.symmetrize_ht(imgs)
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        log(f'Normalizing HT by {norm[0]} +/- {norm[1]}')
        return norm

    @classmethod
    def load(self, config):

        '''Instantiate a dataset from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl

        Returns:
            TiltSeriesMRCData instance
        '''

        config = utils.load_pkl(config) if type(config) is str else config
        data = TiltSeriesMRCData(config['dataset_args']['particles'],
                                 norm=config['dataset_args']['norm'],
                                 invert_data=config['dataset_args']['invert_data'],
                                 ind_ptcl=config['dataset_args']['ind'],
                                 window=config['dataset_args']['window'],
                                 datadir=config['dataset_args']['datadir'],
                                 window_r=config['dataset_args']['window_r'],
                                 window_r_outer=config['dataset_args']['window_r_outer'],
                                 recon_dose_weight=config['training_args']['recon_dose_weight'],
                                 recon_tilt_weight=config['training_args']['recon_tilt_weight'],
                                 dose_override=config['training_args']['dose_override'],
                                 l_dose_mask=config['model_args']['l_dose_mask'],
                                 lazy=config['training_args']['lazy'],
                                 sequential_tilt_sampling=config['dataset_args']['sequential_tilt_sampling'])
        return data
