"""
Classes and functions for interfacing with particle image data and associated starfile metadata.
"""

import numpy as np
from copy import deepcopy
from torch.utils import data

from tomodrgn import fft, mrc, utils, starfile, dose, ctf, lattice


def load_particles(mrcs_txt_star: str,
                   lazy: bool = False,
                   datadir: str = None) -> np.ndarray | list[mrc.LazyImage]:
    """
    Load particle stack from a .mrcs file, a .star file, or a .txt file containing paths to .mrcs files
    :param mrcs_txt_star: path to .mrcs, .star, or .txt file referencing images to load
    :param lazy: whether to load particle images now in memory (False) or later on-the-fly (True)
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
                 star_random_subset: int = -1,
                 datadir: str = None,
                 lazy: bool = True,
                 norm: tuple[float, float] | None = None,
                 invert_data: bool = False,
                 window: bool = True,
                 window_r: float = 0.75,
                 window_r_outer: float = 0.95,
                 recon_dose_weight: bool = False,
                 recon_tilt_weight: bool = False,
                 l_dose_mask: bool = False,
                 constant_mintilt_sampling: bool = False,
                 sequential_tilt_sampling: bool = False):

        # set attributes known immediately at creation time
        self.star = deepcopy(ptcls_star)
        self.star_random_subset = star_random_subset
        self.datadir = datadir
        self.lazy = lazy
        self.window = window
        self.window_r = window_r
        self.window_r_outer = window_r_outer
        self.invert_data = invert_data
        self.constant_mintilt_sampling = constant_mintilt_sampling
        self.sequential_tilt_sampling = sequential_tilt_sampling
        self.recon_tilt_weight = recon_tilt_weight
        self.recon_dose_weight = recon_dose_weight
        self.l_dose_mask = l_dose_mask
        self.norm = norm

        # filter particle images by random subset within each particle
        if self.star_random_subset == -1:
            pass
        elif self.star_random_subset == 1:
            self.star.df = self.star.df.drop(self.star.df.loc[self.star.df[self.star.header_image_random_split] != 1].index).reset_index(drop=True)
        elif self.star_random_subset == 2:
            self.star.df = self.star.df.drop(self.star.df.loc[self.star.df[self.star.header_image_random_split] != 2].index).reset_index(drop=True)
        else:
            raise ValueError(f'Random star subset label not supported: {self.star_random_subset}')

        # filter particles by image indices and particle indices
        self.ptcls_list = self.star.df[self.star.header_ptcl_uid].unique()
        self.nimgs = len(self.star.df)
        self.nptcls = len(self.ptcls_list)

        # get mapping of particle indices to image indices using star file ordering
        ptcls_to_imgs_ind = self.star.get_ptcl_img_indices()
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
        self.boxsize_ht = int(self.ctf_params[0, 0] + 1)

        # get dose weights and masks for each spatial frequency
        self.cumulative_doses = self.star.df[self.star.header_ptcl_dose].to_numpy(dtype=np.float32)
        spatial_frequency_dose_weights, spatial_frequency_dose_masks = self._precalculate_dose_weights_masks()
        self.spatial_frequency_dose_weights = spatial_frequency_dose_weights
        self.spatial_frequency_dose_masks = spatial_frequency_dose_masks

        # get tilt weights for each image
        self.tilts = self.star.df[self.star.header_ptcl_tilt].to_numpy(dtype=np.float32)
        self.spatial_frequency_tilt_weights = self._precalculate_tilt_weights()

    def __len__(self):
        return self.nptcls

    def __getitem__(self, idx_ptcl):
        # get correct image indices for image, pose, and ctf params (indexed against entire dataset)
        ptcl_img_ind = self.ptcls_to_imgs_ind[idx_ptcl].astype(int)

        # determine the order in which to return the images and related parameters
        if self.constant_mintilt_sampling:
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
            for (i, img) in enumerate(images):
                images[i] = fft.ht2_center(img)
            if self.invert_data:
                images *= -1
            images = fft.symmetrize_ht(images)
            images = (images - self.norm[0]) / self.norm[1]
        else:
            images = self.ptcls[ptcl_img_ind]

        # get the associated metadata to be returned
        rot = self.rot[ptcl_img_ind]
        # collate_fn does not allow returning None, so return 0 to tell downstream usages to skip translating image
        trans = 0 if self.trans is None else self.trans[ptcl_img_ind]
        # collate_fn does not allow returning None, so return 0 to tell downstream usages to skip applying CTF to image
        ctf_params = 0 if self.ctf_params is None else self.ctf_params[ptcl_img_ind]

        # get weighting and masking metadata to be returned
        tilt_weights = np.asarray([self.spatial_frequency_tilt_weights.get(tilt) for tilt in self.tilts[ptcl_img_ind]])
        dose_weights = np.asarray([self.spatial_frequency_dose_weights.get(cumulative_dose) for cumulative_dose in self.cumulative_doses[ptcl_img_ind]])
        decoder_mask = np.asarray([self.spatial_frequency_dose_masks.get(cumulative_dose) for cumulative_dose in self.cumulative_doses[ptcl_img_ind]])
        decoder_weights = np.asarray(dose.combine_dose_tilt_weights(dose_weights, tilt_weights))

        return images, rot, trans, ctf_params, decoder_weights, decoder_mask, idx_ptcl

    def get(self, index):
        return self.ptcls[index]

    def _load_pose_params(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load pose parameters from TiltSeriesStarfile associated with this TiltSeriesMRCData object to numpy array.
        :return: rot: numpy array of rotation matrices, shape (nimgs, 3, 3)
        :return: trans: numpy array of translation vectors in pixels, shape (nimgs, 2)
        """
        # parse rotations
        utils.log('Loading rotations from star file')
        euler = self.star.df[self.star.headers_rot].to_numpy(dtype=np.float32)
        utils.log(f'First image Euler angles (Rot, Tilt, Psi): {euler[0]}')
        utils.log('Converting to rotation matrix:')
        rot = np.asarray([utils.rot_3d_from_relion(*x) for x in euler], dtype=np.float32)
        utils.log(f'First image rotation matrix: {rot[0]}')

        # parse translations (if present, default none for warp-exported particleseries)
        utils.log('Loading translations from star file (if any)')
        if all(header_trans in self.star.df.columns for header_trans in self.star.headers_trans):
            trans = self.star.df[self.star.headers_trans].to_numpy(dtype=np.float32)
            utils.log('Translations (pixels):')
            utils.log(f'First image translation matrix: {trans[0]}')
        else:
            trans = None
            utils.log('Translations not found in star file. Reconstruction will not have translations applied.')

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
            utils.log('Particles exported by detected STAR file source software are pre-CTF corrected. During training, reconstructed Fourier central slices will not have CTF applied.')
            return ctf_params

        if not all(header_ctf in self.star.df.columns for header_ctf in self.star.headers_ctf):
            utils.log('CTF parameters not found in star file. During training, reconstructed Fourier central slices will not have CTF applied.')
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
        utils.log(f'Loading particles with {self.lazy=}...')
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
            utils.log('Preprocessing particles...')
            assert particles.shape[-1] == particles.shape[-2], "Images must be square"
            assert nx % 2 == 0, "Image size must be even"

            # apply soft circular real space window
            if self.window:
                utils.log('Windowing particles...')
                particles[:, :-1, :-1] *= real_space_2d_mask

            # convert real space particles to real-valued reciprocal space via hartley transform
            utils.log('Converting to reciprocal space via Hartley transform...')
            for i, img in enumerate(particles):
                particles[i, :-1, :-1] = fft.ht2_center(img[:-1, :-1])

            if self.invert_data:
                utils.log('Inverting data sign...')
                particles *= -1

            # symmetrize HT
            utils.log('Symmetrizing Hartley transform...')
            particles = fft.symmetrize_ht(particles,
                                          preallocated=True)
            _, ny_ht, nx_ht = particles.shape

            # normalize HT to zero mean and unit standard deviation
            if self.norm is None:
                utils.log('Calculating normalization factor...')
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
            utils.log(f'Normalized HT by mean offset {norm[0]} and standard deviation scaling {norm[1]}')

            utils.log(f'Finished loading and preprocessing subtomo particleseries in memory')
        else:
            norm = self.norm
            utils.log('Will lazily load and preprocess particles on the fly')

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
        for (i, img) in enumerate(imgs):
            imgs[i] = fft.ht2_center(img)
        if self.invert_data:
            imgs *= -1
        imgs = fft.symmetrize_ht(imgs)
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0.0
        utils.log(f'Normalizing HT by {norm[0]} +/- {norm[1]}')
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
        utils.log(f'Found {ntilts_min} (min) to {ntilts_max} (max) tilt images per particle')

        return ntilts_min, ntilts_max

    def _precalculate_dose_weights_masks(self) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
        """
        Precalculate the spatial frequency weights and masks based on fixed dose exposure curves.
        :return: frequency_weights_dose: dict mapping cumulative dose to numpy array of relative weights at each spatial frequency, shape (boxsize_ht ** 2)
        :return: frequency_masks_dose: dict mapping cumulative dose to numpy array of mask of spatial frequencies to evaluate, shape (boxsize_ht ** 2)
        """
        # get the unique set of dose values for which to calculate 2D weights and masks
        unique_doses = np.unique(self.cumulative_doses)

        # calculate the critical dose per spatial frequency given pixel size, voltage, and box size
        angpix = self.star.get_tiltseries_pixelsize()
        voltage = self.star.get_tiltseries_voltage()
        spatial_frequencies = dose.calculate_spatial_frequencies(angpix, self.boxsize_ht).astype(np.float32)
        spatial_frequencies_critical_dose = dose.calculate_critical_dose_per_frequency(spatial_frequencies, voltage).astype(np.float32)

        # calculate the 2-D spatial frequency weights for each dose and cache result
        unique_dose_weights = dose.calculate_dose_weights(spatial_frequencies_critical_dose, unique_doses).astype(np.float32)
        if self.recon_dose_weight:
            frequency_weights_dose = {cumulative_dose: frequency_weights_per_dose.ravel()
                                      for cumulative_dose, frequency_weights_per_dose in zip(unique_doses, unique_dose_weights)}
        else:
            frequency_weights_dose = {cumulative_dose: np.ones((self.boxsize_ht * self.boxsize_ht), dtype=np.float32)
                                      for cumulative_dose in unique_doses}

        # calculate the 2-D spatial frequency masks for each dose and cache result
        hartley_2d_mask = lattice.Lattice(boxsize=self.boxsize_ht, extent=0.5, ignore_dc=True).get_circular_mask(diameter=self.boxsize_ht).numpy().reshape(self.boxsize_ht, self.boxsize_ht)
        if self.l_dose_mask:
            frequency_masks_dose = {cumulative_dose: dose.calculate_dose_mask(frequency_weights_per_dose, hartley_2d_mask).ravel()
                                    for cumulative_dose, frequency_weights_per_dose in zip(unique_doses, unique_dose_weights)}
        else:
            frequency_masks_dose = {cumulative_dose: hartley_2d_mask.ravel() for cumulative_dose in unique_doses}

        return frequency_weights_dose, frequency_masks_dose

    def _precalculate_tilt_weights(self) -> dict[float, np.ndarray]:
        """
        Precalculate the spatial frequency weights and masks based on global stage tilt.
        :return: frequency_weights_tilt: dict mapping stage tilt to numpy array of relative weights at each spatial frequency, shape (1)
        """
        # get the unique set of tilt values for which to calculate 2D weights
        unique_tilts = np.unique(self.tilts)

        # calculate the per-image weights for each tilt and cache result
        if self.recon_tilt_weight:
            frequency_weights_tilt = {tilt: dose.calculate_tilt_weights(tilt) for tilt in unique_tilts}
        else:
            frequency_weights_tilt = {tilt: np.array(1) for tilt in unique_tilts}

        return frequency_weights_tilt

    @classmethod
    def load(cls,
             config: str | dict[str, dict]):
        """
        Instantiate a dataset from a config.pkl
        :param config: path to config.pkl or preloaded config.pkl dictionary as generated by tomodrgn
        :return: TiltSeriesMRCData instance
        """

        config = utils.load_pkl(config) if type(config) is str else config
        ptcls_star = starfile.TiltSeriesStarfile(config['starfile_args']['sourcefile_filtered'],
                                                 source_software=config['starfile_args']['source_software'])
        return TiltSeriesMRCData(ptcls_star=ptcls_star,
                                 star_random_subset=config['dataset_args']['star_random_subset'],
                                 datadir=config['dataset_args']['datadir'],
                                 lazy=config['dataset_args']['lazy'],
                                 norm=config['dataset_args']['norm'],
                                 invert_data=config['dataset_args']['invert_data'],
                                 window=config['dataset_args']['window'],
                                 window_r=config['dataset_args']['window_r'],
                                 window_r_outer=config['dataset_args']['window_r_outer'],
                                 recon_dose_weight=config['training_args']['recon_dose_weight'],
                                 recon_tilt_weight=config['training_args']['recon_tilt_weight'],
                                 l_dose_mask=config['model_args']['l_dose_mask'],
                                 constant_mintilt_sampling=config['dataset_args']['constant_mintilt_sampling'],
                                 sequential_tilt_sampling=config['dataset_args']['sequential_tilt_sampling'])
