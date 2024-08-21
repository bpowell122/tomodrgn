"""
Sample latent embeddings along latent space principal components at specified points.
"""

import argparse
import os
import pickle

import numpy as np
from scipy.spatial.distance import cdist

from tomodrgn import analysis, utils
log = utils.log


def add_args(_parser):
    _parser.add_argument('z', help='Input latent embeddings z.pkl file')
    _parser.add_argument('-o', type=os.path.abspath, required=True, help='Output directory for pc.X.txt files containing latent embeddings sampled along each PC')
    _parser.add_argument('--dim', type=int, help='Optionally specify which PC to calculate trajectory (1-based indexing) (default: all)')
    _parser.add_argument('-n', type=int, default=10, help='Number of points to sample along each PC')
    _parser.add_argument('--lim', nargs=2, type=float, help='Start and end point to sample along each PC (default: 5/95th percentile of each PC)')
    _parser.add_argument('--use-percentile-spacing', action='store_true', help='Sample equally spaced percentiles along the PC instead of equally spaced points along the PC')
    return _parser


def analyze_data_support(z: np.ndarray,
                         traj: np.ndarray,
                         cutoff: int = 3) -> np.ndarray:
    """
    Count the number of neighbors in reference array `z` within Euclidean distance `cutoff` of query points `traj`.
    :param z: array of latent embeddings, shape (nptcls, zdim)
    :param traj: trajectory of points through latent embedding space, shape (args.n, zdim)
    :param cutoff: Euclidean distance within which to consider points in `z` as neighbors of points in `traj`
    :return: array of neighbor counts at each point in `traj`, shape (args.n)
    """
    d = cdist(traj, z)
    count = (d < cutoff).sum(axis=1)
    return count


def main(args):
    # log args, create output directory
    log(args)
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # load the latent embeddings and calculate PCA
    z = pickle.load(open(args.z, 'rb'))
    zdim = z.shape[1]
    pc, pca = analysis.run_pca(z)

    # specify which principal components to generate trajactories, use 1-based indexing
    dims = [args.dim] if args.dim else list(range(1, zdim + 1))
    lim = args.lim if args.lim else (5, 95)

    for dim in dims:
        # generate trajectories
        print(f'Sampling along PC{dim}')
        if args.use_percentile_spacing:
            pc_values = np.percentile(pc[:, dim - 1], np.linspace(lim[0], lim[1], args.n))

        else:
            start = np.percentile(pc[:, dim - 1], lim[0])
            stop = np.percentile(pc[:, dim - 1], lim[1])
            pc_values = np.linspace(start, stop, num=args.n)
        print(f'Lower and upper sampling bound along this PC: {pc_values[0]}, {pc_values[-1]}')
        z_trajectory = analysis.get_pc_traj(pca=pca,
                                            dim=dim,
                                            sampling_points=pc_values)

        # count the number of adjacent neighbors at each point along the trajectory
        print('Neighbor count at each sampled point along this trajectory:')
        print(analyze_data_support(z, z_trajectory))

        # save the latent embeddings along the trajectory
        out = f'{args.o}/pc{dim}.txt'
        print(f'Saving latent embeddings along this trajectory: {out}')
        np.savetxt(fname=out, X=z_trajectory)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
