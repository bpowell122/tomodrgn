"""
Common utility functions pertaining to both data processing and script execution
"""
from datetime import datetime as dt
import os
import sys
import numpy as np
import pickle
import collections
import functools
from typing import Any
from tomodrgn import mrc, fft
from scipy import ndimage
from scipy.spatial.transform import Rotation
import subprocess
import torch

_verbose = False


def prefix_paths(mrcs: list[str],
                 datadir: str) -> list[str]:
    """
    Test which of various modifications to the image .mrcs files correctly locates the files on disk.
    Tries no modification; prepending `datadir` to the basename of each image; prepending `datadir` to the full path of each image.
    :param mrcs: list of strings corresponding to the path to each image file specified in the star file (expected format: the `path_to_mrc` part of `index@path_to_mrc`)
    :param datadir: str corresponding to absolute or relative path to prepend to `mrcs`
    :return: list of strings corresponding to the confirmed path to each image file
    """

    filename_patterns = [
        mrcs,
        [f'{datadir}/{os.path.basename(x)}' for x in mrcs],
        [f'{datadir}/{x}' for x in mrcs],
    ]

    for filename_pattern in filename_patterns:
        if all([os.path.isfile(file) for file in set(filename_pattern)]):
            return filename_pattern

    raise FileNotFoundError(f'Not all files (or possibly no files) could be found using any of the filename patterns: {[filename_pattern[0] for filename_pattern in filename_patterns]}')


def log(msg: str | Exception) -> None:
    """
    Write a string to STDOUT with a standardized datetime format.
    :param msg: Text to write to STDOUT.
    :return: None
    """
    print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
    sys.stdout.flush()


def vlog(msg: str) -> None:
    """
    Write a string to STDOUT with a standardized datetime format only if `_verbose` is True.
    :param msg: Text to write to STDOUT.
    :return: None
    """
    if _verbose:
        print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
        sys.stdout.flush()


def flog(msg: str,
         outfile: str) -> None:
    """
    Write a string to `outfile` with a standardized datetime format.
    :param msg: Text to write to `outfile`.
    :param outfile: Name of log file to write.
    :return: None
    """
    msg = '{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(msg)
    sys.stdout.flush()
    try:
        with open(outfile, 'a') as f:
            f.write(msg + '\n')
    except Exception as e:
        log(e)


class Memoized(object):
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.abc.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


def load_pkl(pkl: str) -> Any:
    """
    Convenience function to read a binary pickle file.
    Only performs one read of the pickle file (so will not return multiple values if the file was created with repeated calls to `pickle.dump`)
    :param pkl: Path to file to load from disk.
    :return: Unpickled object.
    """
    with open(pkl, 'rb') as f:
        x = pickle.load(f)
    return x


def save_pkl(data: Any,
             out_pkl: str) -> None:
    """
    Convenience function to write `data` to a binary pickle file.
    :param data: Data to be written to disk.
    :param out_pkl: Path to file to write to disk.
    :return: None
    """
    if os.path.exists(out_pkl):
        vlog(f'Warning: {out_pkl} already exists. Overwriting.')
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)


def rot_3d_from_eman(a: float,
                     b: float,
                     y: float) -> np.ndarray:
    """
    Convert Euler angles from EMAN to a 3-D rotation matrix.
    :param a: Rotation angle A in degrees.
    :param b: Rotation angle B in degrees.
    :param y: Rotation angle Y in degrees.
    :return: 3-D rotation matrix, shape (3, 3)
    """
    a *= np.pi / 180.
    b *= np.pi / 180.
    y *= np.pi / 180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    rot_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    rot_b = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    rot_y = np.array(([cy, -sy, 0], [sy, cy, 0], [0, 0, 1]))
    rot = np.dot(np.dot(rot_y, rot_b), rot_a)
    # handling EMAN convention mismatch for where the origin of an image is (bottom right vs top right)
    rot[0, 1] *= -1
    rot[1, 0] *= -1
    rot[1, 2] *= -1
    rot[2, 1] *= -1
    return rot


def rot_3d_from_relion(a: float,
                       b: float,
                       y: float) -> np.ndarray:
    """
    Convert Euler angles from RELION to a 3-D rotation matrix.
    :param a: Rotation angle A in degrees (typically _rlnAngleRot).
    :param b: Rotation angle B in degrees (typically _rlnAngleTilt).
    :param y: Rotation angle Y in degrees (typically _rlnAnglePsi).
    :return: 3-D rotation matrix, shape (3, 3)
    """
    a *= np.pi / 180.
    b *= np.pi / 180.
    y *= np.pi / 180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    rot_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    rot_b = np.array([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]])
    rot_y = np.array(([cy, -sy, 0], [sy, cy, 0], [0, 0, 1]))
    rot = np.dot(np.dot(rot_y, rot_b), rot_a)
    rot[0, 1] *= -1
    rot[1, 0] *= -1
    rot[1, 2] *= -1
    rot[2, 1] *= -1
    return rot


def rot_3d_from_relion_scipy(euler_: np.ndarray,
                             degrees: bool = True) -> np.ndarray:
    """
    Convert a Nx3 array of RELION euler angles to rotation matrices using Scipy Rotations class.
    :param euler_: Array of Euler angles to convert to rotation matrices.
    :param degrees: Whether angles in `euler_` are expressed in degrees.
    :return: Nx3x3 array of rotation matrices.
    """
    euler = euler_.copy()
    if euler.shape == (3,):
        euler = euler.reshape(1, 3)
    euler[:, 0] += 90
    euler[:, 2] -= 90
    f = np.ones((3, 3))
    f[0, 1] = -1
    f[1, 0] = -1
    f[1, 2] = -1
    f[2, 1] = -1
    rot = Rotation.from_euler('zxz', euler, degrees=degrees).as_matrix() * f
    return rot


def rot_3d_to_relion_scipy(rot: np.ndarray,
                           degrees: bool = True) -> np.ndarray:
    """
    Convert an array of Nx3x3 rotation matrices to RELION euler angles using Scipy Rotations class.
    :param rot: Nx3x3 array of rotation matrices.
    :param degrees: Whether to express the output Euler angles in degrees
    :return: Nx3 array of RELION euler angles
    """
    if rot.shape == (3, 3):
        rot = rot.reshape(1, 3, 3)
    assert len(rot.shape) == 3, "Input must have dim Nx3x3"
    f = np.ones((3, 3))
    f[0, 1] = -1
    f[1, 0] = -1
    f[1, 2] = -1
    f[2, 1] = -1
    euler = Rotation.from_matrix(rot * f).as_euler('zxz', degrees=True)
    euler[:, 0] -= 90
    euler[:, 2] += 90
    euler += 180
    euler %= 360
    euler -= 180
    if not degrees:
        euler *= np.pi / 180
    return euler


def xrot(tilt_deg: float) -> np.ndarray:
    """
    Return rotation matrix associated with rotation over the x-axis.
    :param tilt_deg: Rotation angle in degrees.
    :return: 3x3 rotation matrix.
    """
    theta = tilt_deg * np.pi / 180
    tilt = np.array([[1., 0., 0.],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return tilt


def yrot(tilt_deg: float) -> np.ndarray:
    """
    Return rotation matrix associated with rotation over the y-axis
    :param tilt_deg: Rotation angle in degrees.
    :return: 3x3 rotation matrix.
    """
    theta = tilt_deg * np.pi / 180
    tilt = np.array([[np.cos(theta), 0., np.sin(theta)],
                     [0., 1., 0.],
                     [-np.sin(theta), 0., np.cos(theta)]])
    return tilt


@Memoized
def _zero_sphere_helper(boxsize: int) -> np.ndarray:
    """
    Calculate a mask of coordinates corresponding to a sphere inscribed within a cube.
    :param boxsize: The number of discrete points along each cube edge
    :return: mask array corresponding to values within the sphere, shape (boxsize, boxsize, boxsize)
    """
    xx = np.linspace(-1, 1, boxsize, endpoint=True if boxsize % 2 == 1 else False)
    z, y, x = np.meshgrid(xx, xx, xx)
    coords = np.stack((x, y, z), -1)
    r = np.sum(coords ** 2, axis=-1) ** .5
    return np.where(r > 1, 0, 1)


def zero_sphere(vol: np.ndarray) -> np.ndarray:
    """
    Set volume values outside of a sphere inscribed within a cube to zero.
    :param vol: Volume array to be modified, shape (boxsize, boxsize, boxsize).
    :return: Volume array after setting corner values to zero, shape (boxsize, boxsize, boxsize).
    """
    assert len(set(vol.shape)) == 1, 'volume must be a cube'
    boxsize = vol.shape[0]
    tmp = _zero_sphere_helper(boxsize)
    vlog('Zeroing {} pixels'.format(np.sum(tmp)))
    vol *= tmp
    return vol


def calc_fsc(vol1: np.ndarray | str,
             vol2: np.ndarray | str,
             mask: str | None = None,
             dilate: int = 3,
             dist: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the FSC between two volumes with optional masking.
    :param vol1: path to volume1.mrc, boxsize D,D,D, or ndarray of vol1 voxels
    :param vol2: path to volume2.mrc, boxsize D,D,D, or ndarray of vol2 voxels
    :param mask: mask to apply to volumes, one of [None, 'sphere', 'tight', 'soft', path to mask.mrc with boxsize D,D,D]
    :param dilate: for soft mask, number of pixels to expand auto-determined tight mask
    :param dist: for soft mask, number of pixels over which to apply soft edge
    :return: x: spatial resolution in units of (1/px). fsc: the Fourier Shell Correlation at each resolution shell specified by `x`.
    """
    # load masked volumes in real space
    if isinstance(vol1, np.ndarray):
        pass
    else:
        vol1, _ = mrc.parse_mrc(vol1)
        vol2, _ = mrc.parse_mrc(vol2)
    assert vol1.shape == vol2.shape

    # define fourier grid and label into shells
    boxsize = vol1.shape[0]
    x = np.arange(-boxsize // 2, boxsize // 2)
    x0, x1, x2 = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(x0 ** 2 + x1 ** 2 + x2 ** 2)
    r_max = boxsize // 2  # sphere inscribed within volume box
    r_step = 1  # int(np.min(r[r>0]))
    bins = np.arange(0, r_max + r_step, r_step)  # since np.arange does not include final point `D//2`, need one more shell to calculate FSC at Nyquist
    bin_labels = np.searchsorted(bins, r, side='left')  # bin_label=0 is DC, bin_label=r_max+r_step is highest included freq, bin_label=r_max+2*r_step is frequencies excluded by D//2 spherical mask

    # prepare mask
    if mask is None:
        mask = np.ones_like(vol1)
    elif mask == 'sphere':
        mask = np.where(r <= boxsize // 2, True, False)
    elif mask == 'tight':
        mask = np.where(vol1 >= np.percentile(vol1, 99.99) / 2, True, False)
    elif mask == 'soft':
        mask = np.where(vol1 >= np.percentile(vol1, 99.99) / 2, True, False)
        mask = ndimage.morphology.binary_dilation(mask, iterations=dilate)
        distance_to_mask = ndimage.morphology.distance_transform_edt(~mask)
        distance_to_mask = np.where(distance_to_mask > dist, dist, distance_to_mask) / dist
        mask = np.cos((np.pi / 2) * distance_to_mask)
    elif mask.endswith('.mrc'):
        assert os.path.exists(os.path.abspath(mask))
        mask, _ = mrc.parse_mrc(mask)
    else:
        raise ValueError

    # apply mask in real space
    assert mask.shape == vol1.shape, f'Mask shape {mask.shape} does not match volume shape {vol1.shape}'
    vol1 *= mask
    vol2 *= mask

    # FFT volumes
    vol1_ft = fft.fft3_center(vol1)
    vol2_ft = fft.fft3_center(vol2)

    # calculate the FSC via labeled shells (frequencies > Nyquist share a bin_label that is excluded by `index=bins`)
    num = ndimage.sum(np.real(vol1_ft * np.conjugate(vol2_ft)), labels=bin_labels, index=bins)
    den1 = ndimage.sum(np.abs(vol1_ft) ** 2, labels=bin_labels, index=bins)
    den2 = ndimage.sum(np.abs(vol2_ft) ** 2, labels=bin_labels, index=bins)
    fsc = num / np.sqrt(den1 * den2)

    x = bins / boxsize  # x axis should be spatial frequency in 1/px
    return x, fsc


def calculate_lowpass_filter_mask(boxsize: int,
                                  angpix: float,
                                  lowpass: float,
                                  device: torch.device | None = None) -> np.ndarray | torch.Tensor:
    """
    Calculate a binary mask to later be multiplied into a fourier space volume to lowpass filter said volume.
    Useful to pre-cache a lowpass filter mask that will be used repeatedly (e.g. evaluating many volumes with `eval_vol.py`)
    :param boxsize: number of voxels along one side of the Fourier space symmetrized (DC is box center) volume
    :param angpix: Pixel size of the volume in Ångstroms per pixel.
    :param lowpass: Resolution to filter volume in Ångstroms.
    :param device: device of the volume to be lowpass filtered. None corresponds to a numpy array, otherwise specify a torch.Tensor device to produce a Tensor mask on said device.
    :return: Volume binary mask as a numpy array or torch tensor, shape (boxsize, boxsize, boxsize)
    """
    # calculate frequencies along one axis in units of 1/px
    if boxsize % 2 == 0:
        # even-sized fourier space box
        lowest_freq = 1 / boxsize
        freqs_1d_px = np.arange(start=-1 / 2, stop=1 / 2, step=lowest_freq)
    else:
        # odd-sized box (from fft.symmetrize_ht)
        lowest_freq = 1 / (boxsize - 1)
        freqs_1d_px = np.arange(start=-1 / 2, stop=1 / 2 + lowest_freq, step=lowest_freq)

    # calculate frequencies along one axis in units of 1/Å
    freqs_1d_angstrom = freqs_1d_px / angpix

    # create 3D mask binarized at lowpass frequency
    freqs_x, freqs_y, freqs_z = np.meshgrid(freqs_1d_angstrom, freqs_1d_angstrom, freqs_1d_angstrom)
    lowpass_mask = np.where(np.power(freqs_x ** 2 + freqs_y ** 2 + freqs_z ** 2, 0.5) <= 1 / lowpass,
                            True,
                            False)

    # move mask to the correct device
    if device:
        lowpass_mask = torch.tensor(lowpass_mask).to(device)

    return lowpass_mask


def lowpass_filter(vol_ft: np.ndarray | torch.Tensor,
                   angpix: float,
                   lowpass: float) -> np.ndarray | torch.Tensor:
    """
    Lowpass-filters a volume in reciprocal space.
    :param vol_ft: Fourier space symmetrized (DC is box center) volume as numpy array or torch tensor, shape (boxsize, boxsize, boxsize)
    :param angpix: Pixel size of the volume in Ångstroms per pixel.
    :param lowpass: Resolution to filter volume in Ångstroms.
    :return: Lowpass filtered fourier space symmetrized volume, shape (boxsize, boxsize, boxsize).
    """
    # calculate the binary mask corresponding to the desired lowpass filter
    device = vol_ft.device if type(vol_ft) is torch.Tensor else None
    lowpass_mask = calculate_lowpass_filter_mask(boxsize=vol_ft.shape[0],
                                                 angpix=angpix,
                                                 lowpass=lowpass,
                                                 device=device)
    # multiply the binary mask into the volume
    vol_ft = vol_ft * lowpass_mask
    return vol_ft


def check_memory_usage() -> list[str]:
    """
    Get the current VRAM memory usage of each visible GPU.
    Tries using torch functionality to get memory usage.
    If installed torch does not have this method available, then uses subprocess call to nvidia-smi and parses output.
    :return: VRAM usage of each visible GPU as a pre-formatted string.
    """
    try:
        usage = [torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())]
        return [f'{(total - free) // 1024 ** 2} MiB / {total // 1024 ** 2} MiB' for free, total in usage]
    except AttributeError:
        gpu_memory_usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'], encoding='utf-8').strip().split('\n')
        return [f'{gpu.split(", ")[0]} / {gpu.split(", ")[1]}' for gpu in gpu_memory_usage]


def check_git_revision_hash(repo_path: str) -> str:
    """
    Get the git revision hash of a git repository.
    Uses subprocess call to git and parses output.
    :param repo_path: Path to repository for which to check revision hash.
    :return: Git revision hash of HEAD in `repo_path`.
    """
    return subprocess.check_output(['git', '--git-dir', repo_path, 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_default_device() -> torch.device:
    """
    Determine whether a CUDA-capable GPU is availble or not for torch operations.
    :return: `torch.device('cuda')` if CUDA-capable GPU is available, else `torch.device('cpu')`.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    log(f'Use cuda {use_cuda}')
    if not use_cuda:
        log('WARNING: No GPUs detected')
    return device


def print_progress_bar(curr_iteration: int,
                       num_iterations: int,
                       prefix: str = '',
                       suffix: str = '',
                       decimals: int = 1,
                       length: int = 100,
                       fill: str = '*',
                       end_character: str = "\r") -> None:
    """
    Display an ASCII progress bar in STDOUT.
    Display format is: `PREFIX |*****-----| 50.0% SUFFIX`, updated in-place at STDOUT.
    Intended method of use is calling this function repeatedly within a loop with updated `curr_iteration`.
    The final call in which `curr_iteration == num_iterations` updates the progress bar (in place) and adds a newline for subsequent output.
    :param curr_iteration: the current step number through the process being monitored.
    :param num_iterations: the total number of steps in the process being monitored.
    :param prefix: text to be prepended before the progress bar.
    :param suffix: text to by appended after the progress bar.
    :param decimals: the number of decimals to show in the % completion text of the progress bar.
    :param length: the number of characters defining the length of the progress bar.
            Does not change to preserve a constant display width if prefix, suffix, or decimals change.
    :param fill: the character used to denote the completed portion of the progress bar.
    :param end_character: the character at the end of the progress bar, typically carraige return or newline.
    :return: None
    """
    # this function is most often called in a for loop where the value of `iteration` runs from 0 to `total-1`
    # therefore, offset total to be 0-indexed such that the final iteration has `iteration`==`total`
    num_iterations = num_iterations - 1

    # prepare the progress bar
    percent = f'{100 * curr_iteration / float(num_iterations):3.{decimals}f}'
    filled_length = length * curr_iteration // num_iterations
    bar = fill * filled_length + '-' * (length - filled_length)

    # print the progress bar and flush stdout to ensure it displays correctly
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=end_character)
    sys.stdout.flush()

    # add a newline for the final iteration so that the next text to STDOUT does not overwrite the progress bar
    if curr_iteration == num_iterations:
        print()


@Memoized
def first_n_factors(target: int,
                    n: int = 1,
                    lower_bound: int | None = None) -> list[int]:
    """
    Return a list of the first `n` multiplicative factors of a number `target` where no factor is smaller than `lower_bound`.
    Useful to reshape arrays or tensors.
    For example, enables splitting a tensor with batch_dim == 1 to produce batch_dim > 1, as required for multi-gpu DataParallel.
    :param target: The number for which to calculate multiplicative factors.
    :param n: The number of multiplicative factors to return.
    :param lower_bound: The minimum multiplicative factor to return.
    :return: list of multiplicative factors of target sorted by increasing value
    """
    # initialize the list of factors (which will always include 1 by definition)
    factors = []
    # define the efficient upper search limit as the square root of the target number
    upper_bound = target ** 0.5
    # initialize the factor from which to search increasing values
    i = 2 if lower_bound is None else lower_bound
    while (len(factors) < n + 1) and (i <= upper_bound):
        if not target % i:
            factors.append(i)
        i += 1
    if not factors:
        factors = [1]
    return factors
