"""
Classes and functions for interfacing with particle image data and associated starfile metadata.
"""

import numpy as np
from torch.utils import data

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
    assert 0 <= in_rad <= 1 * 2 ** 0.5, f'Window inner radius must be between 0 and sqrt(2) (inner radius: {in_rad})'
    assert 0 <= out_rad <= 1 * 2 ** 0.5, f'Window outer radius must be between 0 and sqrt(2) (outer radius: {out_rad})'
    assert in_rad < out_rad, f'Window inner radius must be less than window outer radius (inner radius: {in_rad}, outer radius: {out_rad})'

    # create a mesh of 2-D grid points
    x0, x1 = np.meshgrid(np.linspace(-1, 1, boxsize, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, boxsize, endpoint=False, dtype=np.float32))

    # calculate distance from box center at every point in the grid
    r = (x0 ** 2 + x1 ** 2) ** .5

    # create the mask, fill regions between in_rad and out_rad with falling cosine edge, otherwise fill with 1
    mask = np.where((r < out_rad) & (r > in_rad),
                    (1 + np.cos((r - in_rad) / (out_rad - in_rad) * np.pi)) / 2,
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
                 norm: tuple[float, float] | None = None,
                 invert_data: bool = False,
                 window: bool = True,
                 window_r: float = 0.75,
                 window_r_outer: float = 0.95,
                 recon_dose_weight: bool = False,
                 recon_tilt_weight: bool = False,
                 l_dose_mask: bool = False,
                 sequential_tilt_sampling: bool = False, ):

        # set attributes known immediately at creation time
        self.star = ptcls_star
        self.datadir = datadir
        self.lazy = lazy
        self.window = window
        self.window_r = window_r
        self.window_r_outer = window_r_outer
        self.invert_data = invert_data
        self.sequential_tilt_sampling = sequential_tilt_sampling
        self.recon_tilt_weight = recon_tilt_weight
        self.recon_dose_weight = recon_dose_weight
        self.l_dose_mask = l_dose_mask
        self.norm = norm

        # filter particles by image indices and particle indices
        self.ptcls_list = self._filter_ptcls_star(ind_img=ind_img,
                                                  ind_ptcl=ind_ptcl)
        self.nimgs = len(self.star.df)
        self.nptcls = len(self.ptcls_list)

        # get mapping of particle indices to image indices using star file ordering
        ptcls_to_imgs_ind = ptcls_star.get_ptcl_img_indices()
        self.ptcls_to_imgs_ind = ptcls_to_imgs_ind

        # get distribution of number of tilt images per particle across star file
        ntilts_min, ntilts_max = self._get_ntilts_distribution()
        self.ntilts_range = [ntilts_min, ntilts_max]
        self.ntilts_training = ntilts_min

        # load the particle images
        particles, norm, real_space_2d_mask = self._load_particles()
        self.ptcls = particles
        self.norm = norm if norm is not None else self.lazy_particles_estimate_normalization()
        self.real_space_2d_mask = real_space_2d_mask

        # load the poses
        rot, trans = self._load_pose_params()
        self.rot = rot
        self.trans = trans

        # load the CTF parameters
        ctf_params = self._load_ctf_params()
        self.ctf_params = ctf_params
        self.boxsize_ht = self.ctf_params[0, 0] + 1

        # get critical dose for each spatial frequency
        spatial_frequencies_critical_dose, hartley_2d_mask = self._get_spatial_frequencies_critical_dose()
        self.spatial_frequencies_critical_dose = spatial_frequencies_critical_dose
        self.hartley_2d_mask = hartley_2d_mask

    def __len__(self):
        return self.nptcls

    def __getitem__(self, idx_ptcl):
        # get correct image indices for image, pose, and ctf params (indexed against entire dataset)
        ptcl_img_ind = self.ptcls_to_imgs_ind[idx_ptcl].astype(int)

        # determine the order in which to return the images and related parameters
        if self.sequential_tilt_sampling:
            zero_indexed_ind = np.arange(self.ntilts_training)  # take first ntilts_training images for deterministic loading/debugging
        else:
            zero_indexed_ind = np.asarray(np.random.choice(len(ptcl_img_ind), size=self.ntilts_training, replace=False))
        ptcl_img_ind = ptcl_img_ind[zero_indexed_ind]

        # load and preprocess the images to be returned
        if self.lazy:
            images = np.asarray([self.ptcls[i].get() for i in ptcl_img_ind])
            if self.window:
                images *= self.real_space_2d_mask
            for i, img in enumerate(images):
                images[i] = fft.ht2_center(img)
            if self.invert_data:
                images *= -1
            images = fft.symmetrize_ht(images)
            images = (images - self.norm[0]) / self.norm[1]
        else:
            images = self.ptcls[ptcl_img_ind]

        # get metadata to be used in calculating what to return
        cumulative_doses = self.star.df[self.star.header_ptcl_dose].iloc[ptcl_img_ind].to_numpy(dtype=images.dtype)

        # get the associated metadata to be returned
        rot = self.rot[ptcl_img_ind]
        # collate_fn does not allow returning None, so return 0 to tell downstream usages to skip translating image
        trans = 0 if self.trans is None else self.trans[ptcl_img_ind]
        # collate_fn does not allow returning None, so return 0 to tell downstream usages to skip applying CTF to image
        ctf_params = 0 if self.ctf_params is None else self.ctf_params[ptcl_img_ind]
        if self.recon_dose_weight or self.l_dose_mask:
            dose_weights = dose.calculate_dose_weights(self.spatial_frequencies_critical_dose, cumulative_doses).astype(images.dtype)
            decoder_mask = dose.calculate_dose_mask(dose_weights, self.hartley_2d_mask) if self.l_dose_mask else np.repeat(self.hartley_2d_mask[np.newaxis, :, :], self.ntilts_training, axis=0)
        else:
            dose_weights = np.ones((self.ntilts_training, self.boxsize_ht, self.boxsize_ht), dtype=images.dtype)
            decoder_mask = np.repeat(self.hartley_2d_mask[np.newaxis, :, :], self.ntilts_training, axis=0)
        if self.recon_tilt_weight:
            tilts = self.star.df[self.star.header_ptcl_tilt].iloc[ptcl_img_ind].to_numpy(dtype=images.dtype)
            tilt_weights = np.cos(tilts)
        else:
            tilt_weights = np.ones(self.ntilts_training)
        decoder_weights = dose.combine_dose_tilt_weights(dose_weights, tilt_weights).astype(np.float32)

        return images, rot, trans, ctf_params, decoder_weights, decoder_mask, idx_ptcl

    def get(self, index):
        return self.ptcls[index]

    def _filter_ptcls_star(self,
                           ind_img: np.ndarray | str,
                           ind_ptcl: np.ndarray | str) -> np.ndarray:
        """
        Filter the TiltSeriesStarfile associated with this TiltSeriesMRCData by image indices (rows) and particle indices (groups of rows).
        :param ind_img: numpy array or path to numpy array of integer row indices to preserve, shape (N)
        :param ind_ptcl: numpy array or path to numpy array of integer particle indices to preserve, shape (N)
        :return: numpy array of UIDs assocated with preserved particles, shape: (nptcls)
        """

        # how many particles does the star file initially contain
        ptcls_unique_list = self.star.df[self.star.header_ptcl_uid].unique().to_numpy()
        log(f'Found {len(ptcls_unique_list)} particles in input star file')

        # filter by image (row of dataframe)
        if ind_img is not None:
            log('Filtering particle images by supplied indices')

            if type(ind_img) is str and ind_img.endswith('.pkl'):
                ind_ptcl = utils.load_pkl(ind_ptcl)
            else:
                raise ValueError(f'Expected .pkl file for {ind_img=}')

            assert min(ind_img) >= 0
            assert max(ind_img) <= len(self.star.df)

            self.star.df = self.star.df.iloc[ind_img].reset_index(drop=True)

        # filter by particle (group of rows sharing common header_ptcl_uid)
        if ind_ptcl is not None:
            log('Filtering particles by supplied indices')

            if type(ind_ptcl) is str and ind_ptcl.endswith('.pkl'):
                ind_ptcl = utils.load_pkl(ind_ptcl)
            else:
                raise ValueError(f'Expected .pkl file for {ind_ptcl=}')

            assert min(ind_ptcl) >= 0
            assert max(ind_ptcl) <= len(ptcls_unique_list)

            ptcls_unique_list = ptcls_unique_list[ind_ptcl]
            self.star.df = self.star.df[self.star.df[self.star.header_ptcl_uid].isin(ptcls_unique_list)]
            self.star.df = self.star.df.reset_index(drop=True)

            assert len(self.star.df[self.star.header_ptcl_uid].unique().to_numpy()) == len(ind_ptcl), 'Make sure particle indices file does not contain duplicates'
            log(f'Found {len(ptcls_unique_list)} particles after filtering')

        return ptcls_unique_list

    def _load_pose_params(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load pose parameters from TiltSeriesStarfile associated with this TiltSeriesMRCData object to numpy array.
        :return: rot: numpy array of rotation matrices, shape (nimgs, 3, 3)
        :return: trans: numpy array of translation vectors in pixels, shape (nimgs, 2)
        """
        # parse rotations
        log('Loading rotations from star file')
        euler = self.star.df[self.star.headers_rot].to_numpy(dtype=np.float32)
        log('Euler angles (Rot, Tilt, Psi):')
        log(euler[0])
        log('Converting to rotation matrix:')
        rot = np.asarray([utils.R_from_relion(*x) for x in euler], dtype=np.float32)
        log(rot[0])

        # parse translations (if present, default none for warp-exported particleseries)
        log('Loading translations from star file (if any)')
        if all(header_trans in self.star.df.headers for header_trans in self.star.headers_trans):
            trans = self.star.df[self.star.headers_trans].to_numpy(dtype=np.float32)
            log('Translations (pixels):')
            log(trans[0])
        else:
            trans = None
            log('Translations not found in star file. Reconstruction will not have translations applied.')

        return rot, trans

    def _load_ctf_params(self) -> np.ndarray | None:
        """
        Load CTF parameters from TiltSeriesMRCData-associated TiltSeriesStarfile object to numpy array.
        If CTF parameters are not present in star file, or if images are not CTF corrected, returns None.
        CTF parameters are organized as columns: box_size, pixel_size, defocus_u, defocus_v, defocus_angle, voltage, spherical_aberration, amplitude_contrast, phase_shift.
        :return: ctf_params as either numpy array with shape (nimgs, 9) or None
        """
        ctf_params = None

        if self.star.image_ctf_corrected:
            log('Particles exported by detected STAR file source software are pre-CTF corrected. During training, reconstructed Fourier central slices will not have CTF applied.')
            return ctf_params

        if not all(header_ctf in self.star.df.headers for header_ctf in self.star.headers_ctf):
            log('CTF parameters not found in star file. During training, reconstructed Fourier central slices will not have CTF applied.')
            return ctf_params

        num_images = len(self.star.df)
        boxsize = self.star.get_image_size(self.datadir)
        ctf_params = np.zeros((num_images, 9), dtype=np.float32)
        ctf_params[:, 0] = boxsize  # first column is real space box size
        for i, column in enumerate(self.star.headers_ctf):
            ctf_params[:, i + 1] = self.star.df[column].to_numpy(dtype=np.float32)
        ctf.print_ctf_params(ctf_params[0])
        return ctf_params

    def _load_particles(self) -> tuple[np.ndarray | list[mrc.LazyImage], tuple[float, float] | None, np.ndarray]:
        """
        Load the particles referenced in the TiltSeriesStarfile associated with the TiltSeriesMRCData object.
        :return: particles: numpy array of preprocessed particles with shape (nimgs, real_boxsize+1, real_boxsize+1) or list of LazyImage objects
        :return: norm: tuple of floats representing mean and standard deviation of preprocessed particles
        :return: real_space_2d_mask: numpy array of soft-edged 2-D mask applied to particles in real space during preprocessing
        """

        # load the image stack
        log(f'Loading particles with {self.lazy=}...')
        particles = self.star.get_particles_stack(datadir=self.datadir,
                                                  lazy=self.lazy)
        nx = self.star.get_image_size(datadir=self.datadir)
        nimgs = len(particles)

        # prepare the real space circular window mask
        if self.window:
            real_space_2d_mask = window_mask(nx,
                                             self.window_r,
                                             self.window_r_outer)
        else:
            real_space_2d_mask = None

        # preprocess particles in memory if not lazily loading
        if not self.lazy:
            log('Preprocessing particles...')
            assert particles.shape[-1] == particles.shape[-2], "Images must be square"
            assert nx % 2 == 0, "Image size must be even"

            # apply soft circular real space window
            if self.window:
                log('Windowing particles...')
                particles[:, :-1, :-1] *= real_space_2d_mask

            # convert real space particles to real-valued reciprocal space via hartley transform
            log('Converting to reciprocal space via Hartley transform...')
            for i, img in enumerate(particles):
                particles[i, :-1, :-1] = fft.ht2_center(img[:-1, :-1])

            if self.invert_data:
                log('Inverting data sign...')
                particles *= -1

            # symmetrize HT
            log('Symmetrizing Hartley transform...')
            particles = fft.symmetrize_ht(particles,
                                          preallocated=True)
            _, ny_ht, nx_ht = particles.shape

            # normalize HT to zero mean and unit standard deviation
            if self.norm is None:
                log('Calculating normalization factor...')
                # using a random subset of 1% of all images to calculate normalization factors
                random_imgs_for_normalization = np.random.choice(np.arange(nimgs),
                                                                 size=nimgs // 100,
                                                                 replace=False)
                norm = [np.mean(particles[random_imgs_for_normalization]),
                        np.std(particles[random_imgs_for_normalization])]
                norm[0] = 0
            else:
                norm = self.norm
            particles -= norm[0]  # zero mean
            particles /= norm[1]  # unit stdev, separate line required to avoid redundant memory allocation
            log(f'Normalized HT by {norm[0]} +/- {norm[1]}')

            log(f'Finished loading and preprocessing subtomo particleseries in memory')
        else:
            norm = self.norm
            log('Will lazily load and preprocess particles on the fly')

        return particles, norm, real_space_2d_mask

    def lazy_particles_estimate_normalization(self) -> list[float]:
        """
        Estimate mean and standard deviation of particles when lazy is True.
        :return: norm: list of floats representing mean and standard deviation to apply to lazy-loaded particles
        """
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
        norm[0] = 0.0
        log(f'Normalizing HT by {norm[0]} +/- {norm[1]}')
        return norm

    def _get_ntilts_distribution(self) -> tuple[int, int]:
        """
        Calculate the distribution of tilt images per particle across the TiltSeriesStarfile
        :return: ntilts_min: minimum number of tilt images associated with any particle in the dataset
        :return: ntilts_max: maximum number of tilt images associated with any particle in the dataset
        """
        ntilts_set = set(self.star.df.groupby(self.star.header_ptcl_uid, sort=False).size().values)
        ntilts_min = min(ntilts_set)
        ntilts_max = max(ntilts_set)
        self.star.plot_particle_uid_ntilt_distribution()
        log(f'Found {ntilts_min} (min) to {ntilts_max} (max) tilt images per particle')

        return ntilts_min, ntilts_max

    def _get_spatial_frequencies_critical_dose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the critical dose at all spatial frequencies sampled by the dataset's box size and pixel size.
        :return: spatial_frequencies_critical_dose: numpy array of critical dose at each spatial frequency, shape (real_boxsize+1, real_boxsize+1)
        :return: hartley_mask: numpy array of binary mask of spatial frequencies to consider when drawing upon the critical dose array, shape (real_boxsize+1, real_boxsize+1)
        """
        angpix = self.star.get_tiltseries_pixelsize()
        voltage = self.star.get_tiltseries_voltage()

        spatial_frequencies = dose.calculate_spatial_frequencies(angpix, self.boxsize_ht).astype(np.float32)
        spatial_frequencies_critical_dose = dose.calculate_critical_dose_per_frequency(spatial_frequencies, voltage).astype(np.float32)
        hartley_mask = dose.calculate_circular_mask(self.boxsize_ht)

        return spatial_frequencies_critical_dose, hartley_mask

    @classmethod
    def load(cls,
             config: str | dict[str, dict]):
        """
        Instantiate a dataset from a config.pkl
        :param config: path to config.pkl or preloaded config.pkl dictionary as generated by tomodrgn
        :return: TiltSeriesMRCData instance
        """

        config = utils.load_pkl(config) if type(config) is str else config
        star = starfile.TiltSeriesStarfile(config['dataset_args']['particles'])
        return TiltSeriesMRCData(ptcls_star=star,
                                 datadir=config['dataset_args']['datadir'],
                                 lazy=config['training_args']['lazy'],
                                 ind_ptcl=config['dataset_args']['ind'],
                                 ind_img=config['dataset_args']['ind_img'],
                                 norm=config['dataset_args']['norm'],
                                 invert_data=config['dataset_args']['invert_data'],
                                 window=config['dataset_args']['window'],
                                 window_r=config['dataset_args']['window_r'],
                                 window_r_outer=config['dataset_args']['window_r_outer'],
                                 recon_dose_weight=config['training_args']['recon_dose_weight'],
                                 recon_tilt_weight=config['training_args']['recon_tilt_weight'],
                                 l_dose_mask=config['model_args']['l_dose_mask'],
                                 sequential_tilt_sampling=config['dataset_args']['sequential_tilt_sampling'])
