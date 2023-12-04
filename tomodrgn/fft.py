import numpy as np
import torch


### FOURIER TRANSFORM - 2D ###

def fft2_center(img):
    '''
    Batch-vectorized 2-D DFT of an even-box sized image(s) as numpy array
    Note: numpy promotes default input dtype np.float32 to output dtype np.complex128
    :param img: np.ndarray of images, shape ([N], D, D)
    :return: np.ndarray of images DFT, shape ([N], D, D)
    '''
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img,axes=(-1,-2))),axes=(-1,-2))


def fft2_center_torch(img):
    '''
    Batch-vectorized 2-D DFT of an even-box sized image(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32
    :param img: torch.tensor of images DFT, shape ([N], D, D)
    :return: torch.tensor of images, shape ([N], D, D)
    '''
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(img,dim=(-1,-2))),dim=(-1,-2))

def ifft2_center(img):
    '''
    Batch-vectorized 2-D inverse DFT of an even-box sized image(s) as numpy array
    Note: numpy promotes default input dtype np.float32 to output dtype np.complex128
    :param img: np.ndarray of images DFT, shape ([N], D, D)
    :return: np.ndarray of images, shape ([N], D, D)
    '''
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(img,axes=(-1,-2))),axes=(-1,-2))


def ifft2_center_torch(img):
    '''
    Batch-vectorized 2-D inverse DFT of an even-box sized image(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32
    :param img: torch.tensor of images DFT, shape ([N], D, D)
    :return: torch.tensor of images, shape ([N], D, D)
    '''
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(img,dim=(-1,-2))),dim=(-1,-2))


### FOURIER TRANSFORM - 3D ###

def fft3_center(V):
    '''
    Batch-vectorized 3-D DFT of even-box sized volume(s) as numpy array
    Note: numpy promotes default input dtype np.float32 to output dtype np.complex128
    :param V: np.ndarray of volumes, shape ([N], D, D, D)
    :return: np.ndarray of volumes DFT, shape ([N], D, D, D)
    '''
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(V, axes=(-1,-2,-3)), axes=(-1,-2,-3)), axes=(-1,-2,-3))

def fft3_center_torch(V):
    '''
    Batch-vectorized 3-D DFT of even-box sized volume(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32
    :param V: torch.tensor of volumes, shape ([N], D, D, D)
    :return: torch.tensor of volumes DFT, shape ([N], D, D, D)
    '''
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(V, dim=(-1,-2,-3)), dim=(-1,-2,-3)), dim=(-1,-2,-3))

def ifft3_center(V):
    '''
    Batch-vectorized 3-D inverse DFT of even-sized volume(s) as numpy array
    Note: numpy promotes input dtype np.float32 to output dtype np.complex128
    :param V: np.ndarray of volumes DFT, shape ([N], D, D, D)
    :return: np.ndarray of volumes, shape ([N], D, D, D)
    '''
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(V, axes=(-1,-2,-3)), axes=(-1,-2,-3)), axes=(-1,-2,-3))

def ifft3_center_torch(V):
    '''
    Batch-vectorized 3-D inverse DFT of even-sized volume(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32
    :param V: torch.tensor of volumes DFT, shape ([N], D, D, D)
    :return: torch.tensor of volumes, shape ([N], D, D, D)
    '''
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(V, dim=(-1,-2,-3)), dim=(-1,-2,-3)), dim=(-1,-2,-3))


### HARTREE TRANSFORM - 2D ###

def ht2_center(img):
    '''
    Batch-vectorized 2-D DHT of an even-box sized image(s) as numpy array
    Uses relationship DHT(x) == DFT(x).real - DFT(x).imag == (DFT(x) * (1 + i)).real
    Note: numpy promotes default input dtype np.float32 to output dtype np.float64
    :param img: np.ndarray of images, shape ([N], D, D)
    :return: np.ndarray of images DFT, shape ([N], D, D)
    '''
    f = fft2_center(img)
    return f.real-f.imag

def ht2_center_torch(img):
    '''
    Batch-vectorized 2-D DHT of an even-box sized image(s) as torch tensor
    Uses relationship DHT(x) == DFT(x).real - DFT(x).imag == (DFT(x) * (1 + i)).real
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32
    :param img: torch.tensor of images, shape ([N], D, D)
    :return: torch.tensor of images DFT, shape ([N], D, D)
    '''
    f = fft2_center_torch(img)
    return f.real-f.imag


def iht2_center(img):
    '''
    Batch-vectorized 2-D inverse DHT of an even-box sized image(s) as numpy array
    By definition, is equivalent to forward DHT up to normalization scale factor (of N samples, typically D*D here)
    Note: numpy promotes default input dtype np.float32 to output dtype np.float64
    :param img: np.ndarray of images, shape ([N], D, D)
    :return: np.ndarray of images DFT, shape ([N], D, D)
    '''
    img = fft2_center(img)
    img /= (img.shape[-1]*img.shape[-2])
    return img.real - img.imag


def iht2_center_torch(img):
    '''
    Batch-vectorized 2-D inverse DHT of an even-box sized image(s) as torch tensor
    By definition, is equivalent to forward DHT up to normalization scale factor (of N samples, typically D*D here)
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32
    :param img: torch.tensor of images, shape ([N], D, D)
    :return: torch.tensor of images DFT, shape ([N], D, D)
    '''
    img = fft2_center_torch(img)
    img /= (img.shape[-1]*img.shape[-2])
    return img.real - img.imag


### HARTREE TRANSFORM - 3D ###

def ht3_center(V):
    '''
    Batch-vectorized 3-D DHT of even-box sized volume(s) as numpy array
    Uses relationship DHT(x) == DFT(x).real - DFT(x).imag == (DFT(x) * (1 + i)).real
    Note: numpy promotes default input dtype np.float32 to output dtype np.float64
    :param V: np.ndarray of volumes, shape ([N], D, D, D)
    :return: np.ndarray of volumes DFT, shape ([N], D, D, D)
    '''
    f = fft3_center(V)
    return f.real-f.imag

def ht3_center_torch(V):
    '''
    Batch-vectorized 3-D DHT of even-box sized volume(s) as torch tensor
    Uses relationship DHT(x) == DFT(x).real - DFT(x).imag == (DFT(x) * (1 + i)).real
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32
    :param V: torch.tensor of volumes, shape ([N], D, D, D)
    :return: torch.tensor of volumes DFT, shape ([N], D, D, D)
    '''
    f = fft3_center_torch(V)
    return f.real-f.imag

def iht3_center(V):
    '''
    Batch-vectorized 3-D inverse DHT of even-sized volume(s) as numpy array
    Note: numpy promotes input dtype np.float32 to output dtype np.complex128
    :param V: np.ndarray of volumes DHT, shape ([N], D, D, D)
    :return: np.ndarray of volumes, shape ([N], D, D, D)
    '''
    V = fft3_center(V)
    V /= (V.shape[-1] * V.shape[-2] * V.shape[-3])
    return V.real - V.imag

def iht3_center_torch(V):
    '''
    Batch-vectorized 3-D inverse DHT of even-sized volume(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32
    :param V: torch.tensor of volumes DHT, shape ([N], D, D, D)
    :return: torch.tensor of volumes, shape ([N], D, D, D)
    '''
    V = fft3_center_torch(V)
    V /= (V.shape[-1] * V.shape[-2] * V.shape[-3])
    return V.real - V.imag


### MISCELLANEOUS ###

def symmetrize_ht(ht, preallocated=False):
    '''
    Mirror-pads an even-box DHT image (stack) such that DC is at D//2,D//2 and highest frequency is at both D//2,0 and D//2,D
    '''
    if preallocated:
        D = ht.shape[-1] - 1
        sym_ht = ht
    else:
        if len(ht.shape) == 2:
            ht = ht.reshape(1,*ht.shape)
        assert len(ht.shape) == 3
        D = ht.shape[-1]
        B = ht.shape[0]
        sym_ht = np.empty((B,D+1,D+1),dtype=ht.dtype)
        sym_ht[:,0:-1,0:-1] = ht
    assert D % 2 == 0
    sym_ht[:,-1,:] = sym_ht[:,0] # last row is the first row
    sym_ht[:,:,-1] = sym_ht[:,:,0] # last col is the first col
    sym_ht[:,-1,-1] = sym_ht[:,0,0]
    if len(sym_ht) == 1:
        sym_ht = sym_ht[0]
    return sym_ht