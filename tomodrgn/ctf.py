"""
Functions for Contrast Transfer Function calculation, correction, and display
"""
from __future__ import annotations

import numpy as np
import torch

from tomodrgn import utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tomodrgn.lattice import Lattice


def compute_ctf(lattice: Lattice,
                angpix: torch.Tensor,
                dfu: torch.Tensor,
                dfv: torch.Tensor,
                dfang: torch.Tensor,
                volt: torch.Tensor,
                cs: torch.Tensor,
                w: torch.Tensor,
                phase_shift: torch.Tensor = 0,
                bfactor: float = None):
    """
    Calculates the 2-D CTF given spatial frequencies per image and a batch of CTF parameters.
    Implementation of eq. 7 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8412055/ with additional reference to https://github.com/jianglab/ctfsimulation/blob/master/ctf_simulation.py
        `CTF = sin( -π * z * λ * |k|**2 + (π/2) * Cs * λ**3 * |k|**4 - ps )`

        `z = 1/2 * (𝑧1 + 𝑧2) + 1/2 * (𝑧1 − 𝑧2) * cos( 2 * ( 𝛼𝑘 − 𝛼𝑧 ) )`

        `ps = phase_shift + arcsin(w)`
    :param lattice: Lattice object for accessing pre-calculated spatial frequency magnitude and angle from x-axis
    :param angpix: pixel size in angstroms for each image, shape (nimgs, 1)
    :param dfu: defocus U in angstroms for each image, shape (nimgs, 1)
    :param dfv: defocus V in angstroms for each image, shape (nimgs, 1)
    :param dfang: defocus angle in degrees for each image, shape (nimgs, 1)
    :param volt: microscope voltage in kV for each image, shape (nimgs, 1)
    :param cs: sphrerical aberration in mm for each image, shape (nimgs, 1)
    :param w: amplitude contrast ratio for each image, shape (nimgs, 1)
    :param phase_shift: phase shift in degrees for each image, shape (nimgs, 1)
    :param bfactor: envelope function bfactor for each image, shape (nimgs, 1)
    :return: CTF evaluated at given spatial frequencies using input parameters, shape (nimgs, kx * ky)
    """

    # adjust units
    dfang = dfang * torch.tensor(np.pi / 180, dtype=dfang.dtype)  # need in radians
    volt = volt * torch.tensor(10 ** 3, dtype=volt.dtype)  # need in volts
    cs = cs * torch.tensor(10 ** 7, dtype=cs.dtype)  # need in angstrom
    phase_shift = phase_shift * np.pi / torch.tensor(180, dtype=phase_shift.dtype)  # need in radians
    angpix2 = angpix ** 2

    # calculate electron wavelength
    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / (volt + 0.97845e-6 * volt ** 2) ** .5

    # calculate non-enveloped CTF
    df = 0.5 * (dfu + dfv) + 0.5 * (dfu - dfv) * torch.cos(2 * (lattice.freqs2d_angle - dfang))
    phase_shift = phase_shift + torch.arcsin(w)
    gamma = -torch.pi * df * lam * (lattice.freqs2d_s2 / angpix2) + torch.pi / 2 * cs * lam ** 3 * (lattice.freqs2d_s2 / angpix2) ** 2 - phase_shift
    ctf = torch.sin(gamma)

    # apply CTF envelope
    if bfactor is not None:
        ctf *= torch.exp(-bfactor * (lattice.freqs2d_s2 / angpix2) / 4)

    return ctf


def print_ctf_params(params: np.ndarray | torch.Tensor) -> None:
    assert len(params) == 9
    utils.log('Image size (pix)  : {}'.format(params[0]))
    utils.log('Å/pix             : {}'.format(params[1]))
    utils.log('DefocusU (Å)      : {}'.format(params[2]))
    utils.log('DefocusV (Å)      : {}'.format(params[3]))
    utils.log('Dfang (deg)       : {}'.format(params[4]))
    utils.log('voltage (kV)      : {}'.format(params[5]))
    utils.log('cs (mm)           : {}'.format(params[6]))
    utils.log('w                 : {}'.format(params[7]))
    utils.log('Phase shift (deg) : {}'.format(params[8]))
