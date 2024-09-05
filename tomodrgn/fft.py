"""
Fourier and Hartley transform functions for numpy arrays and pytorch tensors
"""

import numpy as np
import torch


def fft2_center(possibly_batched_imgs: np.ndarray) -> np.ndarray:
    """
    Batch-vectorized 2-D DFT of an even-box sized image(s) as numpy array
    Note: numpy promotes default input dtype np.float32 to output dtype np.complex128

    :param possibly_batched_imgs: np.ndarray of images, shape ([N], D, D)
    :return: np.ndarray of images DFT, shape ([N], D, D)
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(possibly_batched_imgs, axes=(-1, -2))), axes=(-1, -2))


def fft2_center_torch(possibly_batched_imgs: torch.Tensor) -> torch.Tensor:
    """
    Batch-vectorized 2-D DFT of an even-box sized image(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32

    :param possibly_batched_imgs: torch.tensor of images DFT, shape ([N], D, D)
    :return: torch.tensor of images, shape ([N], D, D)
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(possibly_batched_imgs, dim=(-1, -2))), dim=(-1, -2))


def ifft2_center(possibly_batched_imgs: np.ndarray) -> np.ndarray:
    """
    Batch-vectorized 2-D inverse DFT of an even-box sized image(s) as numpy array
    Note: numpy promotes default input dtype np.float32 to output dtype np.complex128

    :param possibly_batched_imgs: np.ndarray of images DFT, shape ([N], D, D)
    :return: np.ndarray of images, shape ([N], D, D)
    """
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(possibly_batched_imgs, axes=(-1, -2))), axes=(-1, -2))


def ifft2_center_torch(possibly_batched_imgs: torch.Tensor) -> torch.Tensor:
    """
    Batch-vectorized 2-D inverse DFT of an even-box sized image(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32

    :param possibly_batched_imgs: torch.tensor of images DFT, shape ([N], D, D)
    :return: torch.tensor of images, shape ([N], D, D)
    """
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(possibly_batched_imgs, dim=(-1, -2))), dim=(-1, -2))


def fft3_center(possibly_batched_vols: np.ndarray) -> np.ndarray:
    """
    Batch-vectorized 3-D DFT of even-box sized volume(s) as numpy array
    Note: numpy promotes default input dtype np.float32 to output dtype np.complex128

    :param possibly_batched_vols: np.ndarray of volumes, shape ([N], D, D, D)
    :return: np.ndarray of volumes DFT, shape ([N], D, D, D)
    """
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(possibly_batched_vols, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))


def fft3_center_torch(possibly_batched_vols: torch.Tensor) -> torch.Tensor:
    """
    Batch-vectorized 3-D DFT of even-box sized volume(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32

    :param possibly_batched_vols: torch.tensor of volumes, shape ([N], D, D, D)
    :return: torch.tensor of volumes DFT, shape ([N], D, D, D)
    """
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(possibly_batched_vols, dim=(-1, -2, -3)), dim=(-1, -2, -3)), dim=(-1, -2, -3))


def ifft3_center(possibly_batched_vols: np.ndarray) -> np.ndarray:
    """
    Batch-vectorized 3-D inverse DFT of even-sized volume(s) as numpy array
    Note: numpy promotes input dtype np.float32 to output dtype np.complex128

    :param possibly_batched_vols: np.ndarray of volumes DFT, shape ([N], D, D, D)
    :return: np.ndarray of volumes, shape ([N], D, D, D)
    """
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(possibly_batched_vols, axes=(-1, -2, -3)), axes=(-1, -2, -3)), axes=(-1, -2, -3))


def ifft3_center_torch(possibly_batched_vols: torch.Tensor) -> torch.Tensor:
    """
    Batch-vectorized 3-D inverse DFT of even-sized volume(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32

    :param possibly_batched_vols: torch.tensor of volumes DFT, shape ([N], D, D, D)
    :return: torch.tensor of volumes, shape ([N], D, D, D)
    """
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(possibly_batched_vols, dim=(-1, -2, -3)), dim=(-1, -2, -3)), dim=(-1, -2, -3))


def ht2_center(possibly_batched_imgs: np.ndarray) -> np.ndarray:
    """
    Batch-vectorized 2-D DHT of an even-box sized image(s) as numpy array
    Uses relationship DHT(x) == DFT(x).real - DFT(x).imag == (DFT(x) * (1 + i)).real
    Note: numpy promotes default input dtype np.float32 to output dtype np.float64

    :param possibly_batched_imgs: np.ndarray of images, shape ([N], D, D)
    :return: np.ndarray of images DFT, shape ([N], D, D)
    """
    f = fft2_center(possibly_batched_imgs)
    return np.asarray(f.real - f.imag, dtype=possibly_batched_imgs.dtype)


def ht2_center_torch(possibly_batched_imgs: torch.Tensor) -> torch.Tensor:
    """
    Batch-vectorized 2-D DHT of an even-box sized image(s) as torch tensor
    Uses relationship DHT(x) == DFT(x).real - DFT(x).imag == (DFT(x) * (1 + i)).real
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32

    :param possibly_batched_imgs: torch.tensor of images, shape ([N], D, D)
    :return: torch.tensor of images DFT, shape ([N], D, D)
    """
    f = fft2_center_torch(possibly_batched_imgs)
    return torch.as_tensor(f.real - f.imag, dtype=possibly_batched_imgs.dtype, device=possibly_batched_imgs.device)


def iht2_center(possibly_batched_imgs: np.ndarray) -> np.ndarray:
    """
    Batch-vectorized 2-D inverse DHT of an even-box sized image(s) as numpy array
    By definition, is equivalent to forward DHT up to normalization scale factor (of N samples, typically D*D here)
    Note: numpy promotes default input dtype np.float32 to output dtype np.float64

    :param possibly_batched_imgs: np.ndarray of images, shape ([N], D, D)
    :return: np.ndarray of images DFT, shape ([N], D, D)
    """
    f = fft2_center(possibly_batched_imgs)
    f /= (f.shape[-1] * f.shape[-2])
    return np.asarray(f.real - f.imag, dtype=possibly_batched_imgs.dtype)


def iht2_center_torch(possibly_batched_imgs: torch.Tensor) -> torch.Tensor:
    """
    Batch-vectorized 2-D inverse DHT of an even-box sized image(s) as torch tensor
    By definition, is equivalent to forward DHT up to normalization scale factor (of N samples, typically D*D here)
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32

    :param possibly_batched_imgs: torch.tensor of images, shape ([N], D, D)
    :return: torch.tensor of images DFT, shape ([N], D, D)
    """
    f = fft2_center_torch(possibly_batched_imgs)
    f /= (f.shape[-1] * f.shape[-2])
    return torch.as_tensor(f.real - f.imag, dtype=possibly_batched_imgs.dtype, device=possibly_batched_imgs.device)


def ht3_center(possibly_batched_vols: np.ndarray) -> np.ndarray:
    """
    Batch-vectorized 3-D DHT of even-box sized volume(s) as numpy array
    Uses relationship DHT(x) == DFT(x).real - DFT(x).imag == (DFT(x) * (1 + i)).real
    Note: numpy promotes default input dtype np.float32 to output dtype np.float64

    :param possibly_batched_vols: np.ndarray of volumes, shape ([N], D, D, D)
    :return: np.ndarray of volumes DFT, shape ([N], D, D, D)
    """
    f = fft3_center(possibly_batched_vols)
    return np.asarray(f.real - f.imag, dtype=possibly_batched_vols.dtype)


def ht3_center_torch(possibly_batched_vols: torch.Tensor) -> torch.Tensor:
    """
    Batch-vectorized 3-D DHT of even-box sized volume(s) as torch tensor
    Uses relationship DHT(x) == DFT(x).real - DFT(x).imag == (DFT(x) * (1 + i)).real
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32

    :param possibly_batched_vols: torch.tensor of volumes, shape ([N], D, D, D)
    :return: torch.tensor of volumes DFT, shape ([N], D, D, D)
    """
    f = fft3_center_torch(possibly_batched_vols)
    return torch.as_tensor(f.real - f.imag, dtype=possibly_batched_vols.dtype, device=possibly_batched_vols.device)


def iht3_center(possibly_batched_vols: np.ndarray) -> np.ndarray:
    """
    Batch-vectorized 3-D inverse DHT of even-sized volume(s) as numpy array
    Note: numpy promotes input dtype np.float32 to output dtype np.complex128

    :param possibly_batched_vols: np.ndarray of volumes DHT, shape ([N], D, D, D)
    :return: np.ndarray of volumes, shape ([N], D, D, D)
    """
    f = fft3_center(possibly_batched_vols)
    f /= (f.shape[-1] * f.shape[-2] * f.shape[-3])
    return np.asarray(f.real - f.imag, dtype=possibly_batched_vols.dtype)


def iht3_center_torch(possibly_batched_vols: torch.Tensor) -> torch.Tensor:
    """
    Batch-vectorized 3-D inverse DHT of even-sized volume(s) as torch tensor
    Note: torch maintains default input dtype torch.float32 to output dtype torch.float32

    :param possibly_batched_vols: torch.tensor of volumes DHT, shape ([N], D, D, D)
    :return: torch.tensor of volumes, shape ([N], D, D, D)
    """
    f = fft3_center_torch(possibly_batched_vols)
    f /= (f.shape[-1] * f.shape[-2] * f.shape[-3])
    return torch.as_tensor(f.real - f.imag, dtype=possibly_batched_vols.dtype, device=possibly_batched_vols.device)


def symmetrize_ht(batched_ht_imgs: np.ndarray,
                  preallocated: bool = False) -> np.ndarray:
    """
    Mirror-pads an even-box DHT image (stack) such that DC is at D//2,D//2 and highest frequency is at both D//2,0 and D//2,D

    :param batched_ht_imgs: DHT-transformed image stack (such as from `fft.ht3_center`), shape ([N], D, D) or ([N], D+1, D+1)
    :param preallocated: whether the input `batched_ht_imgs` last two dimensions are each padded one pixel larger than the input image to accomodate mirroring.
            Symmetrizing the centered DHT involves mirroring the left-most column and top-most row of each image to the right-most column and bottom-most row.
    :return: np.ndarray of symmetrized and centered DHT-transformed images, shape ([N], D+1, D+1)
    """
    if preallocated:
        boxsize = batched_ht_imgs.shape[-1] - 1
        sym_ht = batched_ht_imgs
    else:
        if len(batched_ht_imgs.shape) == 2:
            batched_ht_imgs = batched_ht_imgs.reshape(1, *batched_ht_imgs.shape)
        assert len(batched_ht_imgs.shape) == 3
        boxsize = batched_ht_imgs.shape[-1]
        batchsize = batched_ht_imgs.shape[0]
        sym_ht = np.empty((batchsize, boxsize + 1, boxsize + 1), dtype=batched_ht_imgs.dtype)
        sym_ht[:, 0:-1, 0:-1] = batched_ht_imgs
    assert boxsize % 2 == 0
    sym_ht[:, -1, :] = sym_ht[:, 0]  # last row is the first row
    sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, -1, -1] = sym_ht[:, 0, 0]
    if len(sym_ht) == 1:
        sym_ht = sym_ht[0]
    return sym_ht
