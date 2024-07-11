import numpy as np
from torch.utils import data
import os
import multiprocessing as mp
from multiprocessing import Pool

from tomodrgn import fft, mrc, utils, starfile, dose, ctf

log = utils.log


def load_particles(mrcs_txt_star: str,
                   lazy: bool = False,
                   datadir: str = None) -> np.ndarray | list[mrc.LazyImage]:
    """
    Load particle stack from a .mrcs file, a .star file, or a .txt file containing paths to .mrcs files
    :param mrcs_txt_star: path to .mrcs, .star, or .txt file referencing images to load
    :param lazy: return numpy array if True, or return list of LazyImages
    :param datadir: relative or absolute path to overwrite path to particle image .mrcs specified in the STAR file
    :return: numpy array of particle images of shape (n_images, boxsize+1, boxsize+1), or list of LazyImage objects
    """
    if mrcs_txt_star.endswith('.txt'):
        particles = mrc.parse_mrc_list(mrcs_txt_star,
                                       lazy=lazy)
    elif mrcs_txt_star.endswith('.star'):
        star = starfile.TiltSeriesStarfile(mrcs_txt_star)
        particles = star.get_particles_stack(particles_block_name=star.block_particles,
                                             particles_path_column=star.header_ptcl_image,
                                             datadir=datadir,
                                             lazy=lazy)
    elif mrcs_txt_star.endswith('.mrcs'):
        particles, _ = mrc.parse_mrc(mrcs_txt_star, lazy=lazy)
    else:
        raise ValueError(f'Unrecognized file type: {mrcs_txt_star}')
    return particles


def window_mask(boxsize: int,
                in_rad: float = 0.75,
                out_rad: float = 0.95) -> np.ndarray:
    """
    Create a 2-D circular mask with a soft edge falling as a cosine from `in_rad` to `out_rad`.
    Mask is defined as a circle inscribed within the (square) image box.
    :param boxsize: the image box width in pixels
    :param in_rad: the fraction of the image box radius at which to begin a falling cosine edge
    :param out_rad: the fraction of the image box radius at which to end a falling cosine edge
    :return: 2-D mask as a 2-D numpy array of shape (boxsize, boxsize) of dtype float
    """

    # sanity check inputs
    assert boxsize % 2 == 0, f'Image box size must be an even number (box size: {boxsize})'
    assert 0 <= in_rad <= 1 * 2**0.5, f'Window inner radius must be between 0 and sqrt(2) (inner radius: {in_rad})'
    assert 0 <= out_rad <= 1 * 2**0.5, f'Window outer radius must be between 0 and sqrt(2) (outer radius: {out_rad})'
    assert in_rad < out_rad, f'Window inner radius must be less than window outer radius (inner radius: {in_rad}, outer radius: {out_rad})'

    # create a mesh of 2-D grid points
    x0, x1 = np.meshgrid(np.linspace(-1, 1, boxsize, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, boxsize, endpoint=False, dtype=np.float32))

    # calculate distance from box center at every point in the grid
    r = (x0**2 + x1**2)**.5

    # create the mask, fill regions between in_rad and out_rad with falling cosine edge, otherwise fill with 1
    mask = np.where((r < out_rad) & (r > in_rad),
                    (1 + np.cos((r-in_rad)/(out_rad-in_rad) * np.pi)) / 2,
                    1)
    # fill mask regions at and outside of out_rad with 0
    mask = np.where(r < out_rad,
                    mask,
                    0)
    return mask


class TiltSeriesMRCData(data.Dataset):
    """
    Class for loading and accessing image, pose, ctf, and weighting data associated with a series of tilt images of particles.
    """

    def __init__(self,
                 ptcls_star: starfile.TiltSeriesStarfile,
                 datadir: str = None,
                 lazy: bool = True,
                 ind_ptcl: str | np.ndarray = None,
                 ind_img: np.ndarray = None,
                 norm: tuple[float, float] = None,
                 invert_data: bool = False,
                 window: bool = True,
                 window_r: float = 0.75,
                 window_r_outer: float = 0.95,
                 recon_dose_weight: bool = False,
                 recon_tilt_weight: bool = False,
                 l_dose_mask: bool = False,
                 sequential_tilt_sampling: bool = False,):

        # filter by test/train split per-image
        if ind_img is not None:
            ptcls_star.df = ptcls_star.df.iloc[ind_img].reset_index(drop=True)

        # evaluate command line arguments affecting preprocessing
        ptcls_unique_list = ptcls_star.df[ptcls_star.header_ptcl_uid].unique()
        log(f'Found {len(ptcls_unique_list)} particles')
        if ind_ptcl is not None:
            log('Filtering particles by supplied indices...')
            if type(ind_ptcl) is str and ind_ptcl.endswith('.pkl'):
                ind_ptcl = utils.load_pkl(ind_ptcl)
            ptcls_unique_list = ptcls_unique_list[ind_ptcl]
            ptcls_star.df = ptcls_star.df[ptcls_star.df[ptcls_star.header_ptcl_uid].isin(ptcls_unique_list)]
            ptcls_star.df = ptcls_star.df.reset_index(drop=True)
            assert len(ptcls_star.df[ptcls_star.header_ptcl_uid].unique().astype(str)) == len(ind_ptcl), 'Make sure particle indices file does not contain duplicates'
            log(f'Found {len(ptcls_unique_list)} particles after filtering')
        ptcls_to_imgs_ind = ptcls_star.get_ptcl_img_indices()
        ntilts_set = set(ptcls_star.df.groupby(ptcls_star.header_ptcl_uid, sort=False).size().values)
        ntilts_min = min(ntilts_set)
        ntilts_max = max(ntilts_set)
        log(f'Found {ntilts_min} (min) to {ntilts_max} (max) tilt images per particle')
        log(f'Will sample tilt images per particle: {"sequentially" if sequential_tilt_sampling else "randomly"}')
        if window:
            log(f'Will window images in real space with falling cosine edge between inner radius {window_r} and outer radius {window_r_outer}')
        if recon_dose_weight:
            log('Will calculate weights due to incremental dose; frequencies will be weighted by exposure-dependent amplitude attenuation for pixelwise loss')
        else:
            log('Will not perform dose weighting; all frequencies will be equally weighted for pixelwise loss')
        if recon_tilt_weight:
            log('Will calculate weights due to tilt angle; frequencies will be weighted by cosine(tilt_angle) for pixelwise loss')
        else:
            log('Will not perform tilt weighting; all tilt angles will be equally weighted for pixelwise loss')

        # load all particles
        log('Loading particles...')
        particles = ptcls_star.get_particles_stack(datadir=datadir, lazy=lazy)
        nx = ptcls_star.get_image_size(datadir=datadir)
        nimgs = len(particles)
        nptcls = len(ptcls_unique_list)

        # prepare the real space circular window mask
        if window:
            self.real_space_2d_mask = window_mask(nx, window_r, window_r_outer)
        else:
            self.real_space_2d_mask = None

        # preprocess particles if not lazy
        if not lazy:
            log('Preprocessing particles...')
            assert particles.shape[-1] == particles.shape[-2], "Images must be square"
            assert nx % 2 == 0, "Image size must be even"

            # Real space window
            if window:
                log('Windowing particles...')
                particles[:, :-1, :-1] *= self.real_space_2d_mask

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
        if all(header_trans in ptcls_star.df.headers for header_trans in ptcls_star.headers_trans):
            trans = ptcls_star.df[ptcls_star.headers_trans].to_numpy(dtype=np.float32)
            log('Translations (pixels):')
            log(trans[0])
        else:
            trans = None
            log('Translations not found in star file. Reconstruction will not have translations applied.')

        # Loading CTF parameters from star file (if present)
        ctf_params = None
        if not ptcls_star.image_ctf_corrected:
            if all(header_ctf in ptcls_star.df.headers for header_ctf in ptcls_star.headers_ctf):
                ctf_params = np.zeros((nimgs, 9), dtype=np.float32)
                ctf_params[:, 0] = nx  # first column is real space box size
                for i, column in enumerate(ptcls_star.headers_ctf):
                    ctf_params[:, i + 1] = ptcls_star.df[column].to_numpy(dtype=np.float32)
                ctf.print_ctf_params(ctf_params[0])
            else:
                log('CTF parameters not found in star file. During training, reconstructed Fourier central slices will not have CTF applied.')
        else:
            log('Particles exported by detected STAR file source software are pre-CTF corrected. During training, reconstructed Fourier central slices will not have CTF applied.')

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
                images *= self.real_space_2d_mask
            for i, img in enumerate(images):
                images[i] = fft.ht2_center(img)
            if self.invert_data: images *= -1
            images = fft.symmetrize_ht(images)
            images = (images - self.norm[0]) / self.norm[1]
        else:
            images = self.ptcls[ptcl_img_ind]

        # get metadata to be used in calculating what to return
        cumulative_doses = self.star.df[self.star.header_ptcl_dose].iloc[ptcl_img_ind].to_numpy(dtype=images.dtype)

        # get the associated metadata to be returned
        rot = self.rot[ptcl_img_ind]
        trans = 0 if self.trans is None else self.trans[ptcl_img_ind]  # tells train loop to skip translation block, bc no translation params provided and collate_fn does not allow returning None
        ctf_params = 0 if self.ctf_params is None else self.ctf_params[ptcl_img_ind]  # tells train loop to skip ctf weighting block, bc no ctf params provided and collate_fn does not allow returning None
        if self.recon_dose_weight or self.l_dose_mask:
            dose_weights = dose.calculate_dose_weights(self.spatial_frequencies_critical_dose, cumulative_doses)
            decoder_mask = dose.calculate_dose_mask(dose_weights, self.circular_mask) if self.l_dose_mask else np.repeat(self.circular_mask[np.newaxis,:,:], ntilts_training, axis=0)
        else:
            dose_weights = np.ones((self.ntilts_training, self.D, self.D))
            decoder_mask = np.repeat(self.circular_mask[np.newaxis,:,:], ntilts_training, axis=0)
        if self.recon_tilt_weight:
            tilts = self.star.df[self.star.header_ptcl_tilt].iloc[ptcl_img_ind].to_numpy(dtype=images.dtype)
            tilt_weights = np.cos(tilts)
        else:
            tilt_weights = np.ones((ntilts_training))
        decoder_weights = dose.combine_dose_tilt_weights(dose_weights, tilt_weights)

        return images, rot, trans, ctf_params, decoder_weights, decoder_mask, idx_ptcl

    def get(self, index):
        return self.ptcls[index]

    def lazy_particles_estimate_normalization(self):
        n = min(10000, self.nimgs)
        random_imgs_for_normalization = np.random.choice(self.nimgs, size=n, replace=False)
        imgs = np.asarray([self.ptcls[i].get() for i in random_imgs_for_normalization])
        if self.window:
            imgs *= self.real_space_2d_mask
        for i, img in enumerate(imgs):
            imgs[i] = fft.ht2_center(img)
        if self.invert_data:
            imgs *= -1
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
