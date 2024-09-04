"""
Clean an analyzed train_vae output directory of various types of outputs.
"""

import argparse
import os
import glob
import re

from tomodrgn import utils

log = utils.log


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        # this script is called from tomodrgn.__main__ entry point, in which case a parser is already created
        pass

    parser.add_argument('workdir', type=os.path.abspath, help='Training directory containing training outputs to be cleaned')
    parser.add_argument('--weights', action='store_true', help='Remove weights.N.pkl files directly within training directory, excluding those with a matching analyze.N or convergence.N subfolder')
    parser.add_argument('--zfiles', action='store_true', help='Remove z.N.pkl files within directly training directory, excluding those with a matching analyze.N or convergence.N subfolder')
    parser.add_argument('--volumes', action='store_true', help='Remove *.mrc volumes recursively within training directory. '
                                                               'Note that volumes can be regenerated with config.pkl, weights.pkl, and appropriate z file.')
    parser.add_argument('--test', action='store_true', help='Don\'t actually delete any files, just list the files that would be deleted')

    return parser


def main(args):
    # log the args
    log(args)

    # get the list of epoch values to preserve
    regex = re.compile(r'.*\..[0-9]+')
    subfolders = [f.name for f in os.scandir(args.workdir) if os.path.isdir(f) and regex.match(f.name)]
    epochs_to_preserve = set([int(subfolder.split('.')[-1]) for subfolder in subfolders])
    log(f'Found subfolders with names matching the following epochs: {epochs_to_preserve}; preserving associated z.pkl and weights.pkl files.')

    # process args for paths to remove
    paths_to_remove = []
    if args.weights:
        # remove all weights files except those containing an epoch to preserve in the filename, assuming format `weights.N.pkl`
        regex = re.compile(r'weights\.[0-9]+\.pkl')
        weights_paths = [f.path for f in os.scandir(args.workdir) if os.path.isfile(f) and regex.match(f.name)]
        paths_to_remove.append([weights_path for weights_path in weights_paths if int(weights_path.split('.')[-2]) not in epochs_to_preserve])
    if args.zfiles:
        # remove all z files except those containing an epoch to preserve in the filename, assuming format `z.N.*.pkl`
        regex = re.compile(r'z\.[0-9]+\..*\.pkl')
        zfile_paths = [f.path for f in os.scandir(args.workdir) if os.path.isfile(f) and regex.match(f.name)]
        paths_to_remove.append([zfile_path for zfile_path in zfile_paths if int(zfile_path.split('.')[-3]) not in epochs_to_preserve])
    if args.volumes:
        # remove all .mrc files recursively within workdir
        vol_paths = glob.glob(f'{args.workdir}/**/*.mrc', recursive=True)
        paths_to_remove.append(vol_paths)

    # return if test, or remove files at paths if not
    if args.test:
        log('Found (but not deleting) the following paths:')
        for path_group in paths_to_remove:
            print(path_group)
    else:
        for path_group in paths_to_remove:
            for path in path_group:
                os.remove(path)


if __name__ == '__main__':
    main(add_args().parse_args())
