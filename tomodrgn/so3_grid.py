"""
Implementation of Yershova et al. "Generating uniform incremental grids on SO(3) using the Hopf fribration"
See source code implementation: https://lavalle.pl/software/so3/so3.html
"""

import numpy as np
import healpy as hp


def grid_s1(resol: int) -> np.ndarray:
    """
    Sample S1 at a given resolution (uses uniform sampling).
    The base resolution has 6 samples.

    :param resol: resolution at which to sample S1
    :return: array of sampled points
    """
    number_points = 6 * 2 ** resol
    interval = 2 * np.pi / number_points
    grid = np.arange(number_points) * interval + interval / 2
    return grid


def grid_s2(resol: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample S2 at a given resolution (uses HEALPix sampling).
    The base resolution has 12 samples.

    :param resol: resolution at which to sample S2
    :return: array of sampled points
    """
    nside = 2 ** resol
    npix = 12 * nside * nside
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    return theta, phi


def hopf_to_quat(theta: np.ndarray,
                 phi: np.ndarray,
                 psi: np.ndarray) -> np.ndarray:
    """
    Transform Hopf coordinates to quaternions

    :param theta: [0, pi), parameterizes spherical coordinates on S2 together with phi
    :param phi: [0, 2pi), parameterizes spherical coordinates on S2 together with theta
    :param psi: [0, 2pi), parameterizes circle S1
    :return: quaternions
    """
    ct = np.cos(theta / 2)
    st = np.sin(theta / 2)
    quat = np.array([ct * np.cos(psi / 2),
                     ct * np.sin(psi / 2),
                     st * np.cos(phi + psi / 2),
                     st * np.sin(phi + psi / 2)])
    return quat.T.astype(np.float32)


def grid_SO3(resol: int) -> np.ndarray:
    """
    Sample points on SO(3) at the specified resolution. Relies on sampling S1 and coset space S2.

    :param resol: resolution at which to sample SO(3)
    :return: array of sampled points as quaternions
    """
    theta, phi = grid_s2(resol)
    psi = grid_s1(resol)
    quat = hopf_to_quat(theta=np.repeat(theta, repeats=len(psi)),  # repeats each element by len(psi)
                        phi=np.repeat(phi, repeats=len(psi)),  # repeats each element by len(psi)
                        psi=np.tile(psi, reps=len(theta)))  # tiles the array len(theta) times
    return quat


def base_SO3_grid() -> np.ndarray:
    """
    Return the base resolution SO(3) grid

    :return: array of sampled points
    """
    return grid_SO3(resol=1)


####################
# Neighbor finding #
####################


def get_s1_neighbor(mini: int,
                    curr_res: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the 2 nearest neighbors on S1 at the next resolution level

    :param mini:
    :param curr_res: current grid resolution
    :return: psi of nearest neighbors on S1, indices of nearest neighbors on S1
    """
    npix = 6 * 2 ** (curr_res + 1)
    dt = 2 * np.pi / npix
    # return np.array([2*mini, 2*mini+1])*dt + dt/2
    # the fiber bundle grid on SO3 is weird
    # the next resolution level's nearest neighbors in SO3 are not 
    # necessarily the nearest neighbor grid points in S1
    # include the 13 neighbors for now... eventually learn/memoize the mapping 
    ind = np.arange(2 * mini - 1, 2 * mini + 3)
    if ind[0] < 0:
        ind[0] += npix
    return ind * dt + dt / 2, ind


def get_s2_neighbor(mini: int,
                    curr_res: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the 4 nearest neighbors on S2 at the next resolution level

    :param mini:
    :param curr_res: current grid resolution
    :return: theta and phi of nearest neighbors on S2, indices of nearest neighbors on S2
    """
    nside = 2 ** (curr_res + 1)
    ind = np.arange(4) + 4 * mini
    return hp.pix2ang(nside, ind, nest=True), ind


def get_base_ind(ind: int) -> tuple[int, int]:
    """
    Return the corresponding S2 and S1 grid index for an index on the base SO3 grid

    :param ind: number of points on the SO3 grid
    :return: corresponding S2 and S1 grid indices
    """
    psii = ind % 12
    thetai = ind // 12
    return thetai, psii


def get_neighbor(quat: np.ndarray,
                 s2i: int,
                 s1i: int,
                 curr_res: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the 8 nearest neighbors on SO3 at the next resolution level

    :param quat: rotation quaternions
    :param s2i: index of the current resolution level of S2
    :param s1i: index of the current resolution level of S1
    :param curr_res: current grid resolution
    :return: nearest neighbor quaternions, nearest neighbor indices on SO3
    """
    (theta, phi), s2_nexti = get_s2_neighbor(s2i, curr_res)
    psi, s1_nexti = get_s1_neighbor(s1i, curr_res)
    quat_n = hopf_to_quat(np.repeat(theta, len(psi)),
                          np.repeat(phi, len(psi)),
                          np.tile(psi, len(theta)))
    ind = np.array([np.repeat(s2_nexti, len(psi)),
                    np.tile(s1_nexti, len(theta))])
    ind = ind.T
    # find the 8 nearest neighbors of 16 possible points
    # need to check distance from both +q and -q
    dists = np.minimum(np.sum((quat_n - quat) ** 2, axis=1), np.sum((quat_n + quat) ** 2, axis=1))
    ii = np.argsort(dists)[:8]
    return quat_n[ii], ind[ii]
