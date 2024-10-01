"""
Classes and functions for interfacing with particle image data and associated starfile metadata.
"""

import ast
import einops
import numpy as np
from copy import deepcopy
from torch.utils import data

from tomodrgn import fft, mrc, utils, starfile, dose, ctf, lattice


def load_particles(mrcs_txt_star: str,
                   lazy: bool = False,
                   datadir: str = None) -> np.ndarray | list[mrc.LazyImage]:
    """
    Load particle stack from a .mrcs file, a .star file, or a .txt file containing paths to .mrcs files

    :param mrcs_txt_star: path to .mrcs, .star, or .txt file referencing images to load.
            If using a star file, should be an image-series star file (if using Warp/M or NextPYP), or an optimisation set star file (if using WarpTools or RELION v5)').
    :param lazy: whether to load particle images now in memory (False) or later on-the-fly (True)
    :param datadir: relative or absolute path to overwrite path to particle image .mrcs specified in the STAR file
    :return: numpy array of particle images of shape (n_images, boxsize+1, boxsize+1), or list of LazyImage objects
    """
    if mrcs_txt_star.endswith('.txt'):
        particles = mrc.parse_mrc_list(mrcs_txt_star,
                                       lazy=lazy)
    elif mrcs_txt_star.endswith('.star'):
        star = starfile.load_sta_starfile(mrcs_txt_star)
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
    Class for loading and accessing image, pose, ctf, and weighting data associated with a tomodrgn.starfile.TiltSeriesStarfile instance.
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
        self.real_space_2d_mask = real_space_2d_mask
        self.norm = norm if norm is not None else self.lazy_particles_estimate_normalization()

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
            # always return ntilts_training number of images from each particle
            # loading different particles will always return the same number of tilt images
            if self.sequential_tilt_sampling:
                zero_indexed_ind = np.arange(self.ntilts_training)  # take first ntilts_training images for deterministic loading/debugging
            else:
                zero_indexed_ind = np.asarray(np.random.choice(len(ptcl_img_ind), size=self.ntilts_training, replace=False))
            ptcl_img_ind = ptcl_img_ind[zero_indexed_ind]
        else:
            # always return all images associated with each image
            # loading different particles can return different numbers of tilt images
            if self.sequential_tilt_sampling:
                pass
            else:
                np.random.shuffle(ptcl_img_ind)

        # load and preprocess the images to be returned
        if self.lazy:
            images = np.asarray([self.ptcls[i].get(low_memory=True) for i in ptcl_img_ind])
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
        imgs = np.asarray([self.ptcls[i].get(low_memory=True) for i in random_imgs_for_normalization])
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
                                 recon_dose_weight=config['dataset_args']['recon_dose_weight'],
                                 recon_tilt_weight=config['dataset_args']['recon_tilt_weight'],
                                 l_dose_mask=config['dataset_args']['l_dose_mask'],
                                 constant_mintilt_sampling=config['dataset_args']['constant_mintilt_sampling'],
                                 sequential_tilt_sampling=config['dataset_args']['sequential_tilt_sampling'])


class TomoParticlesMRCData(data.Dataset):
    """
    Class for loading and accessing image, pose, ctf, and weighting data associated with a tomodrgn.starfile.TiltSeriesStarfile instance.
    """

    def __init__(self,
                 ptcls_star: starfile.TomoParticlesStarfile,
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

        # filter particle images by random image subset within each particle
        if self.star_random_subset == -1:
            # keeping all images of all particles, regardless of split1 / split2 label
            pass
        elif self.star_random_subset == 1:
            # only keep included images assigned to split1 (set images assigned to split2 to NOT include)
            halfset_visible_frames = []
            for ptcl_visible_frames, ptcl_train_test_split in self.star.df[[self.star.header_ptcl_visible_frames, self.star.header_image_random_split]].to_numpy():
                ptcl_visible_frames = np.where(ptcl_train_test_split == 1, 1, 0)
                halfset_visible_frames.append(ptcl_visible_frames)
            self.star.df[self.star.header_ptcl_visible_frames] = halfset_visible_frames
        elif self.star_random_subset == 2:
            # only keep included images assigned to split2 (set images assigned to split1 to NOT include)
            halfset_visible_frames = []
            for ptcl_visible_frames, ptcl_train_test_split in self.star.df[[self.star.header_ptcl_visible_frames, self.star.header_image_random_split]].to_numpy():
                ptcl_visible_frames = np.where(ptcl_train_test_split == 2, 1, 0)
                halfset_visible_frames.append(ptcl_visible_frames)
            self.star.df[self.star.header_ptcl_visible_frames] = halfset_visible_frames
        else:
            raise ValueError(f'Random star subset label not supported: {self.star_random_subset}. Must be either `1` or `2`')
        # drop particles that now have 0 visible frames (keeping these will likely cause downstream problems loading images / assigning z to particles with no data)
        ptcl_inds_to_drop = self.star.df[self.star.df[self.star.header_ptcl_visible_frames].apply(np.sum) == 0].index.to_numpy()
        self.star.df = self.star.df.drop(ptcl_inds_to_drop, axis=0).reset_index(drop=True)

        # filter particles by image indices and particle indices
        self.ptcls_list = self.star.df.index.to_numpy()
        self.nimgs = self.star.df[self.star.header_ptcl_visible_frames].apply(np.sum).sum()
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
        self.real_space_2d_mask = real_space_2d_mask
        self.norm = norm if norm is not None else self.lazy_particles_estimate_normalization()

        # load the poses
        rot, trans = self._load_pose_params()
        self.rot = rot
        self.trans = trans

        # load the CTF parameters
        ctf_params = self._load_ctf_params()
        self.ctf_params = ctf_params
        self.boxsize_ht = int(self.star.df[self.star.header_ptcl_box_size].to_numpy()[0]) + 1

        # get dose weights and masks for each spatial frequency
        self.cumulative_doses = np.concatenate([self.star.tomograms_star.blocks[f'data_{tomo_name}'][self.star.header_tomo_dose].to_numpy(dtype=np.float32)
                                                for tomo_name in self.star.df[self.star.header_ptcl_tomogram]])
        spatial_frequency_dose_weights, spatial_frequency_dose_masks = self._precalculate_dose_weights_masks()
        self.spatial_frequency_dose_weights = spatial_frequency_dose_weights
        self.spatial_frequency_dose_masks = spatial_frequency_dose_masks

        # get tilt weights for each image
        self.tilts = np.concatenate([self.star.tomograms_star.blocks[f'data_{tomo_name}'][self.star.header_tomo_tilt].to_numpy(dtype=np.float32)
                                     for tomo_name in self.star.df[self.star.header_ptcl_tomogram]])
        self.spatial_frequency_tilt_weights = self._precalculate_tilt_weights()

    def __len__(self):
        return self.nptcls

    def __getitem__(self, idx_ptcl):
        # get correct image indices for image, pose, and ctf params (indexed against entire dataset)
        ptcl_img_ind = self.ptcls_to_imgs_ind[idx_ptcl].astype(int)

        # determine the order in which to return the images and related parameters
        if self.constant_mintilt_sampling:
            # always return ntilts_training number of images from each particle
            # loading different particles will always return the same number of tilt images
            if self.sequential_tilt_sampling:
                zero_indexed_ind = np.arange(self.ntilts_training)  # take first ntilts_training images for deterministic loading/debugging
            else:
                zero_indexed_ind = np.asarray(np.random.choice(len(ptcl_img_ind), size=self.ntilts_training, replace=False))
            ptcl_img_ind = ptcl_img_ind[zero_indexed_ind]
        else:
            # always return all images associated with each image
            # loading different particles can return different numbers of tilt images
            if self.sequential_tilt_sampling:
                pass
            else:
                np.random.shuffle(ptcl_img_ind)

        # load and preprocess the images to be returned
        if self.lazy:
            images = self.ptcls[idx_ptcl].get(low_memory=False)[ptcl_img_ind]
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
        Load pose parameters from TomoParticlesStarfile associated with this TomoParticlesMRCData object to numpy array.
        The optimisation set structuring of metadata means we need to combine the per-tilt-series geometry in the TomoTomogramsStarfile with per-particle poses in the TomoParticlesStarfile.
        # TODO include fine pose shifts from TomoTrajectoriesFile

        :return: rot: numpy array of rotation matrices, shape (nimgs, 3, 3)
        :return: trans: numpy array of translation vectors in pixels, shape (nimgs, 2)
        """
        from scipy.spatial.transform import Rotation

        rots = []
        trans = []
        # iterate through tomograms
        for tomo_name, ptcl_group_df in self.star.df.groupby(self.star.header_ptcl_tomogram, sort=False):
            # TOMOGRAMS
            # get tomogram table for this tomogram
            tomogram_df = self.star.tomograms_star.blocks[f'data_{tomo_name}']

            # get (n_tilts, 4) arrays for first three rows of tomogram projection matrices
            projection_matrices_x = np.stack(tomogram_df[self.star.header_tomo_proj_x].apply(lambda x: ast.literal_eval(x)))
            projection_matrices_y = np.stack(tomogram_df[self.star.header_tomo_proj_y].apply(lambda x: ast.literal_eval(x)))
            projection_matrices_z = np.stack(tomogram_df[self.star.header_tomo_proj_z].apply(lambda x: ast.literal_eval(x)))

            # get (n_tilts, 3, 4) tomogram projection_matrices
            tilt_projection_matrices = np.asarray([projection_matrices_x, projection_matrices_y, projection_matrices_z])
            tilt_projection_matrices = einops.rearrange(tilt_projection_matrices, pattern='projxyz ntilts d -> ntilts projxyz d')

            # get (n_tilts, 3, 3) tomogram rotation matrices
            tilt_rotation_matrices = tilt_projection_matrices[:, :, :-1]

            # PARTICLE ROTATIONS
            # get (n_particles, 3, 3) particle rotation_matrices
            ptcl_euler_angles = ptcl_group_df[self.star.headers_rot].to_numpy()
            ptcl_rotation_matrices = Rotation.from_euler(seq='ZYZ', angles=ptcl_euler_angles, degrees=True).inv().as_matrix()  # equivalent to tomodrgn.utils.rot_3d_from_relion in TiltSeriesMRCData

            # get (n_particles, n_tilts, 3, 3) per particle per tilt rotation matrices
            ptcl_rotation_matrices = einops.rearrange(ptcl_rotation_matrices, 'nptcls i j -> nptcls 1 i j')
            ptcl_rotation_matrices = tilt_rotation_matrices @ ptcl_rotation_matrices  # (ntilts, 3, 3) @ (nptcls, 1, 3, 3) -> (nptcls, ntilts, 3, 3)
            ptcl_rotation_matrices = einops.rearrange(ptcl_rotation_matrices, 'nptcls ntilts i j -> (nptcls ntilts) i j')

            # PARTICLE TRANSLATIONS
            # get (n_particles, 3) particle translation matrices (units: px)
            ptcl_trans_matrices = ptcl_group_df[self.star.headers_trans].to_numpy()

            # get (n_particles, n_tilts, 3) per particle per tilt translation matrices
            ptcl_trans_matrices = einops.rearrange(ptcl_trans_matrices, 'nptcls i -> nptcls 1 i 1')
            ptcl_trans_matrices = tilt_rotation_matrices @ ptcl_trans_matrices  # (ntilts, 3, 3) @ (nptcls, 1, 3, 1) -> (nptcls, ntilts, 3, 1)
            ptcl_trans_matrices = einops.rearrange(ptcl_trans_matrices, 'nptcls ntilts i 1 -> (nptcls ntilts) i')

            # only keeping translations in X and Y
            ptcl_trans_matrices = ptcl_trans_matrices[:, :2]

            # SAVE POSES
            # get (n_particles * ntilts) mask of which images are loaded for each particle
            ptcl_visible_frames = np.hstack(ptcl_group_df[self.star.header_ptcl_visible_frames]).astype(bool)

            # save per-particle-per-tilt rotations from this tomogram
            rots.append(ptcl_rotation_matrices[ptcl_visible_frames].astype(np.float32))

            # save per-particle-per-tilt translations from this tomogram
            trans.append(ptcl_trans_matrices[ptcl_visible_frames].astype(np.float32))

        rots = np.concatenate(rots)
        trans = np.concatenate(trans)

        return rots, trans

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

        # TODO figure out how to calculate CTF parameters per particle given particle coordinate in tomogram and tomogram projection geometry to each tilt micrograph
        raise NotImplementedError('TomoDRGN does not yet support images that are not pre-CTF corrected from RELION-5 style star files')

    def _load_particles(self) -> tuple[np.ndarray | list[mrc.LazyImage], tuple[float, float] | None, np.ndarray]:
        """
        Load the particles referenced in the TomoParticlesStarfile associated with the TomoParticlesMRCData object.

        :return: particles: numpy array of preprocessed particles with shape (nimgs, real_boxsize+1, real_boxsize+1) or list of LazyImage objects
        :return: norm: tuple of floats representing mean and standard deviation of preprocessed particles
        :return: real_space_2d_mask: numpy array of soft-edged 2-D mask applied to particles in real space during preprocessing
        """

        # load the image stack
        utils.log(f'Loading particles with {self.lazy=}...')
        particles = self.star.get_particles_stack(datadir=self.datadir,
                                                  lazy=self.lazy)
        nx = particles[0].shape_image[-1] if self.lazy else particles.shape[-1] - 1

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
                random_imgs_for_normalization = np.random.choice(np.arange(self.nimgs),
                                                                 size=self.nimgs // 100,
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

    def _get_ntilts_distribution(self):
        """
        Calculate the distribution of tilt images per particle across the TomoParticlesStarfile
        :return: ntilts_min: minimum number of tilt images associated with any particle in the dataset
        :return: ntilts_max: maximum number of tilt images associated with any particle in the dataset
        """
        ntilts_per_ptcl = self.star.df[self.star.header_ptcl_visible_frames].apply(np.sum)  # number of included images per particle
        unique_ntilts_per_ptcl, ptcl_counts_per_unique_ntilt = np.unique(ntilts_per_ptcl, return_counts=True)
        ntilts_min = min(unique_ntilts_per_ptcl)
        ntilts_max = max(unique_ntilts_per_ptcl)
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
        :return: TomoParticlesMRCData instance
        """

        config = utils.load_pkl(config) if type(config) is str else config
        ptcls_star = starfile.TomoParticlesStarfile(config['starfile_args']['sourcefile_filtered'],
                                                    source_software=config['starfile_args']['source_software'])
        return TomoParticlesMRCData(ptcls_star=ptcls_star,
                                    star_random_subset=config['dataset_args']['star_random_subset'],
                                    datadir=config['dataset_args']['datadir'],
                                    lazy=config['dataset_args']['lazy'],
                                    norm=config['dataset_args']['norm'],
                                    invert_data=config['dataset_args']['invert_data'],
                                    window=config['dataset_args']['window'],
                                    window_r=config['dataset_args']['window_r'],
                                    window_r_outer=config['dataset_args']['window_r_outer'],
                                    recon_dose_weight=config['dataset_args']['recon_dose_weight'],
                                    recon_tilt_weight=config['dataset_args']['recon_tilt_weight'],
                                    l_dose_mask=config['dataset_args']['l_dose_mask'],
                                    constant_mintilt_sampling=config['dataset_args']['constant_mintilt_sampling'],
                                    sequential_tilt_sampling=config['dataset_args']['sequential_tilt_sampling'])

    def lazy_particles_estimate_normalization(self) -> list[float]:
        """
        Estimate mean and standard deviation of particles when lazy is True.

        :return: norm: list of floats representing mean and standard deviation to apply to lazy-loaded particles
        """
        n = min(500, self.nptcls)
        random_ptcls_for_normalization = np.random.choice(self.nptcls, size=n, replace=False)
        imgs = np.asarray([self.ptcls[i].get(low_memory=False) for i in random_ptcls_for_normalization])
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


def load_sta_dataset(ptcls_star: starfile.TiltSeriesStarfile | starfile.TomoParticlesStarfile,
                     *args,
                     **kwargs) -> TiltSeriesMRCData | TomoParticlesMRCData:
    """
    Loads a tomodrgn particles dataset class (either ``TiltSeriesMRCData`` or ``TomoParticlesMRCData``) given an instantiated star file handler class.
    Loads particle image data, pose parameters, CTF parameters, dose and tilt weighting parameters, etc.
    This is the preferred way of creating a tomodrgn dataset class instance.

    :param ptcls_star: pre-existing starfile object used to obtain file paths and metadata in creating the returned dataset object.
    :return: The created dataset object (either ``TiltSeriesMRCData`` or ``TomoParticlesMRCData``)
    """

    if type(ptcls_star) is starfile.TiltSeriesStarfile:
        return TiltSeriesMRCData(ptcls_star, *args, **kwargs)
    elif type(ptcls_star) is starfile.TomoParticlesStarfile:
        return TomoParticlesMRCData(ptcls_star, *args, **kwargs)
    else:
        raise ValueError(f'Unrecognized input star file type: {type(ptcls_star)}. '
                         f'Must be one of ``tomodrgn.starfile.TiltSeriesStarfile`` or ``tomodrgn.starfile.TomoParticlesStarfile``.')
