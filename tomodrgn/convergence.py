"""
Functions to aid estimation of model training convergence.
"""
import numpy as np
from typing import Literal

from tomodrgn import utils


def fsc_referencevol_to_manyvols(reference_vol: str,
                                 vol_paths: list[str],
                                 fsc_mask: Literal['sphere', 'tight', 'soft', 'none'] | str | None,
                                 include_dc: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Fourier Shell Correlation (FSC) between one reference volume and many query volumes.
    :param reference_vol: path to the reference volume on disk
    :param vol_paths: list of paths to the query volumes on disk
    :param fsc_mask: type of real space mask to apply to each volume, determined using the reference volume.
            `Sphere` is a binary spherical mask inscribed within the volume cube.
            `Tight` is a binary mask determined at the 99.99th percentile of the volume.
            `Soft` is a soft (non-binary) mask created by dilating the tight mask 3 pixels and applying a further 10 pixel falling cosine edge.
            `None` or `'none'` (note the string) both result in no mask being applied to the volume.
    :param include_dc: Whether to return the full array of frequencies and FSCs, or truncate the DC component prior to returning the array.
    :return: array of resolution values in units 1/px and shape (boxsize // 2);
            array of FSC curves with shape (len(vol_paths), boxsize // 2);
            array of FSC metrics with shape (3, len(vol_paths)): resolution crossing FSC 0.143, resolution crossing FSC 0.5, integral under FSC before 0.143 crossing.
    """
    fsc_resolutions_point143 = []
    fsc_resolutions_point5 = []
    fsc_integrals = []
    fscs = []
    resolution = []
    for vol_path in vol_paths:
        print(f'Processing volume: {vol_path}')
        resolution, fsc = utils.calc_fsc(vol1=reference_vol,
                                         vol2=vol_path,
                                         mask=fsc_mask)
        if not include_dc:
            resolution = resolution[1:]
            fsc = fsc[1:]
        fscs.append(fsc)

        # find the resolution at which FSC crosses 0.5 correlation
        if np.all(fsc >= 0.5):
            threshold_resolution = float(resolution[-1])
            fsc_resolutions_point5.append(threshold_resolution)
        else:
            threshold_resolution = float(resolution[np.argmax(fsc < 0.5)])
            fsc_resolutions_point5.append(threshold_resolution)

        # find the resolution at which FSC crosses 0.143 correlation
        if np.all(fsc >= 0.143):
            threshold_resolution = float(resolution[-1])
            threshold_index = resolution.shape[0]
            fsc_resolutions_point143.append(threshold_resolution)
        else:
            threshold_resolution = float(resolution[np.argmax(fsc < 0.143)])
            threshold_index = np.argmax(fsc < 0.143)
            fsc_resolutions_point143.append(threshold_resolution)

        # calculate the integral of correlation against resolution
        fsc_integral = np.trapz(fsc[:threshold_index], resolution[:threshold_index])
        fsc_integrals.append(fsc_integral)

    fscs = np.asarray(fscs)
    fsc_metrics = np.array([fsc_resolutions_point143, fsc_resolutions_point5, fsc_integrals])

    return resolution, fscs, fsc_metrics
