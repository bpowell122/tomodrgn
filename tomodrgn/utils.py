from datetime import datetime as dt
import os, sys
import numpy as np
import pickle
import collections
import functools
from tomodrgn import mrc, fft
from scipy import ndimage
import subprocess
import torch

_verbose = False

def log(msg):
    print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
    sys.stdout.flush()

def vlog(msg):
    if _verbose:
        print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
        sys.stdout.flush()

def flog(msg, outfile):
    msg = '{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(msg)
    sys.stdout.flush()
    try:
        with open(outfile,'a') as f:
            f.write(msg+'\n')
    except Exception as e:
        log(e)

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
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
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

def load_pkl(pkl):
    with open(pkl,'rb') as f:
        x = pickle.load(f)
    return x

def save_pkl(data, out_pkl, mode='wb'):
    if mode == 'wb' and os.path.exists(out_pkl):
        vlog(f'Warning: {out_pkl} already exists. Overwriting.')
    with open(out_pkl, mode) as f:
        pickle.dump(data, f)

def R_from_eman(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    # handling EMAN convention mismatch for where the origin of an image is (bottom right vs top right)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1
    return R

def R_from_relion(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1
    return R

def R_from_relion_scipy(euler_, degrees=True):
    '''Nx3 array of RELION euler angles to rotation matrix'''
    from scipy.spatial.transform import Rotation as RR
    euler = euler_.copy()
    if euler.shape == (3,):
        euler = euler.reshape(1,3)
    euler[:,0] += 90
    euler[:,2] -= 90
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    rot = RR.from_euler('zxz', euler, degrees=degrees).as_matrix()*f
    return rot

def R_to_relion_scipy(rot, degrees=True):
    '''Nx3x3 rotation matrices to RELION euler angles'''
    from scipy.spatial.transform import Rotation as RR
    if rot.shape == (3,3):
        rot = rot.reshape(1,3,3)
    assert len(rot.shape) == 3, "Input must have dim Nx3x3"
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    euler = RR.from_matrix(rot*f).as_euler('zxz', degrees=True)
    euler[:,0] -= 90
    euler[:,2] += 90
    euler += 180
    euler %= 360
    euler -= 180
    if not degrees:
        euler *= np.pi/180
    return euler

def xrot(tilt_deg):
    '''Return rotation matrix associated with rotation over the x-axis'''
    theta = tilt_deg*np.pi/180
    tilt = np.array([[1.,0.,0.],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return tilt

def yrot(tilt_deg):
    '''Return rotation matrix associated with rotation over the y-axis'''
    theta = tilt_deg*np.pi/180
    tilt = np.array([[np.cos(theta), 0., np.sin(theta)],
                     [0., 1., 0.],
                     [-np.sin(theta), 0., np.cos(theta)]])
    return tilt

@memoized
def _zero_sphere_helper(D):
    xx = np.linspace(-1, 1, D, endpoint=True if D % 2 == 1 else False)
    z,y,x = np.meshgrid(xx,xx,xx)
    coords = np.stack((x,y,z),-1)
    r = np.sum(coords**2,axis=-1)**.5
    return np.where(r>1)

def zero_sphere(vol):
    '''Zero values of @vol outside the sphere'''
    assert len(set(vol.shape)) == 1, 'volume must be a cube'
    D = vol.shape[0]
    tmp = _zero_sphere_helper(D)
    vlog('Zeroing {} pixels'.format(len(tmp[0])))
    vol[tmp] = 0
    return vol

def calc_fsc(vol1, vol2, mask = 'none', dilate = 3, dist = 10):
    '''
    Function to calculate the FSC between two volumes
    vol1: path to volume1.mrc, boxsize D,D,D, or ndarray of vol1 voxels
    vol2: path to volume2.mrc, boxsize D,D,D, or ndarray of vol2 voxels
    mask: one of ['none', 'sphere', 'tight', 'soft', path to mask.mrc with boxsize D,D,D]
    dilate: int of px to expand tight mask when creating soft mask
    dist: int of px over which to apply soft edge when creating soft mask
    '''
    # load masked volumes in real space
    if type(vol1) == np.ndarray:
        pass
    else:
        vol1, _ = mrc.parse_mrc(vol1)
        vol2, _ = mrc.parse_mrc(vol2)
    assert vol1.shape == vol2.shape

    # define fourier grid and label into shells
    D = vol1.shape[0]
    x = np.arange(-D // 2, D // 2)
    x0, x1, x2 = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(x0 ** 2 + x1 ** 2 + x2 ** 2)
    r_max = D // 2  # sphere inscribed within volume box
    r_step = 1  # int(np.min(r[r>0]))
    bins = np.arange(0, r_max + r_step, r_step)  # since np.arange does not include final point `D//2`, need one more shell to calculate FSC at Nyquist
    bin_labels = np.searchsorted(bins, r, side='left')  # bin_label=0 is DC, bin_label=r_max+r_step is highest included freq, bin_label=r_max+2*r_step is frequencies excluded by D//2 spherical mask

    # prepare mask
    if mask == 'none':
        mask = np.ones_like(vol1)
    elif mask == 'sphere':
        mask = np.where(r <= D//2, True, False)
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
    else: raise ValueError

    # apply mask in real space
    assert mask.shape == vol1.shape, f'Mask shape {mask.shape} does not match volume shape {vol1.shape}'
    vol1 *= mask
    vol2 *= mask

    # FFT volumes
    vol1_ft = fft.fftn_center(vol1)
    vol2_ft = fft.fftn_center(vol2)


    # calculate the FSC via labeled shells (frequencies > Nyquist share a bin_label that is excluded by `index=bins`)
    num = ndimage.sum(np.real(vol1_ft * np.conjugate(vol2_ft)), labels=bin_labels, index=bins)
    den1 = ndimage.sum(np.abs(vol1_ft) ** 2, labels=bin_labels, index=bins)
    den2 = ndimage.sum(np.abs(vol2_ft) ** 2, labels=bin_labels, index=bins)
    fsc = num / np.sqrt(den1 * den2)

    x = bins / D  # x axis should be spatial frequency in 1/px
    return x, fsc


def lowpass_filter(vol_ft, angpix, lowpass):
    # get real space box width, and correct for FFT 1-px padding if applied
    D = vol_ft.shape[0]
    assert D % 2 == 1, f'Fourier transformed volume must have odd box length, found box length: {D}'

    # calculate frequencies along one axis in units of 1/Å
    lowest_freq = 1 / (D - 1)
    freqs_1d_px = np.arange(start=-1 / 2, stop=1 / 2 + lowest_freq, step=lowest_freq)
    freqs_1d_angstrom = freqs_1d_px / angpix

    # create 3D mask binarized at lowpass frequency
    freqs_x, freqs_y, freqs_z = np.meshgrid(freqs_1d_angstrom, freqs_1d_angstrom, freqs_1d_angstrom)
    lowpass_mask = np.where(np.power(freqs_x ** 2 + freqs_y ** 2 + freqs_z ** 2, 0.5) <= 1 / lowpass,
                            True,
                            False)

    # apply mask to volume to zero-weight higher frequency components
    if torch.is_tensor(vol_ft):
        lowpass_mask = torch.tensor(lowpass_mask).to(vol_ft.device)
    vol_ft = vol_ft * lowpass_mask

    return vol_ft


def check_memory_usage():
    try:
        usage = [torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())]
        return [f'{(total - free) // 1024**2} MiB / {total // 1024**2} MiB' for free, total in usage]
    except AttributeError:
        gpu_memory_usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'], encoding='utf-8').strip().split('\n')
        return [f'{gpu.split(", ")[0]} / {gpu.split(", ")[1]}' for gpu in gpu_memory_usage]
    except:
        return ['error checking memory usage']


def check_git_revision_hash(repo_path):
    return subprocess.check_output(['git', '--git-dir', repo_path, 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_default_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    log(f'Use cuda {use_cuda}')
    if not use_cuda:
        log('WARNING: No GPUs detected')
    return device


def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*', end_character = "\r"):
    percent = f'{100 * iteration / float(total):3.{decimals}f}'
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = end_character)
    sys.stdout.flush()
    if iteration == total:
        print()

def first_n_factors(target, n = 1, lower_bound = None):
    '''
    calculate the first `n` factors of a number `target`, with factors no smaller than `lower_bound`
    useful to calculate an internal "batch size" when dealing with otherwise ragged tensors that can be freely reshaped
    enables splitting batch along batch_dim > 1, as required for multi-gpu DataParallel
    '''
    factors = []
    upper_bound = target ** 0.5
    i = 2 if lower_bound is None else lower_bound
    while (len(factors) < n) and (i <= upper_bound):
        if not target % i: factors.append(i)
        i += 1
    if factors == []: factors = [1]
    return factors
