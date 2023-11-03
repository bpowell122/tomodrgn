import numpy as np
import torch

def fft2_center(img):
    '''
    2-D DFT of an even-box sized real-space image or image stack
    Note: numpy promotes default input dtype np.float32 to output dtype np.complex128
    :param img: np.ndarray of images, shape ([N], D, D)
    :return: np.ndarray of images DFT, shape ([N], D, D)
    '''
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img,axes=(-1,-2))),axes=(-1,-2))


def fft2_center_torch(img):
    '''
    2-D DFT of an even-box sized real-space image or image stack
    :param img: torch.tensor of images, shape ([N], D, D)
    :return: torch.tensor of images DFT, shape ([N], D, D)
    '''
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(img,axes=(-1,-2))),axes=(-1,-2))


def ht2_center(img):
    '''
    2-D DHT of an even-box sized real-space image or image stack
    Note: numpy promotes default input dtype np.float32 to output dtype np.float64
    :param img: np.ndarray of images, shape ([N], D, D)
    :return: np.ndarray of images DFT, shape ([N], D, D)
    '''
    f = fft2_center(img)
    return f.real-f.imag

def ht2_center_torch(img):
    '''
    2-D DHT of an even-box sized real-space image or image stack
    :param img: torch.tensor of images, shape ([N], D, D)
    :return: torch.tensor of images DFT, shape ([N], D, D)
    '''
    f = fft2_center_torch(img)
    return f.real-f.imag


def iht2_center(img):
    '''
    2-D inverse DHT
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
    2-D inverse DHT
    By definition, is equivalent to forward DHT up to normalization scale factor (of N samples, typically D*D here)
    :param img: torch.tensor of images, shape ([N], D, D)
    :return: torch.tensor of images DFT, shape ([N], D, D)
    '''
    img = fft2_center_torch(img)
    img /= (img.shape[-1]*img.shape[-2])
    return img.real - img.imag


def fftn_center(V):
    '''
    N-D DFT of even-box sized volume
    Inner-most fftshift updates real space origin position / data ordering for a less jagged FFT
    Outer-most fftshift updates reciprocal space origin to center of box
    Note: numpy promotes default input dtype np.float32 to output dtype np.complex128
    '''
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(V)))


def ifftn_center(V):
    '''
    N-D inverse DFT
    Note: preserves input dtype
    Note: numpy promotes input dtype np.float32 to output dtype np.complex128
    '''
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(V)))


def htn_center(V):
    '''
    N-D DHT of even-box sized real-space volume with low-frequency components centered
    Uses relationship DHT(x) == DFT(x).real - DFT(x).imag == (DFT(x) * (1 + i)).real
    Note: numpy promotes default input dtype np.float32 to output dtype np.float64
    '''
    f = fftn_center(V)
    return f.real - f.imag

def ihtn_center(V):
    '''
    N-D inverse DHT
    By definition, is equivalent to forward DHT up to normalization scale factor (of N samples, typically D*D*D here)
    Note: numpy promotes default input dtype np.float32 to output dtype np.float64
    '''
    h = htn_center(V)
    h *= 1 / (np.prod(V.shape))
    return h


def ihtn_center_torch(V):
    V = torch.fft.fftshift(V)
    V = torch.fft.fftn(V)
    V = torch.fft.fftshift(V)
    V /= torch.numel(V)
    return V.real - V.imag


def symmetrize_ht(ht, preallocated=False):
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
