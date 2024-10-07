"""
Filter a .star file by selected particle or image indices, optionally per-tomogram
"""
import argparse
import copy
import os
from typing import Literal

import numpy as np

from tomodrgn import starfile, utils

log = utils.log


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        # this script is called from tomodrgn.__main__ entry point, in which case a parser is already created
        pass

    parser.add_argument('input', help='Input .star file')

    group = parser.add_argument_group('Core arguments')
    group.add_argument('--starfile-type', type=str, default='imageseries', choices=('imageseries', 'volumeseries', 'optimisation_set'),
                       help='Type of star file to filter. Select imageseries if rows correspond to particle images. Select volumeseries if rows correspond to particle volumes. '
                            'Select optimisation_set if passing in an optimisation set star file.')
    group.add_argument('--action', choices=('keep', 'drop'), default='keep', help='keep or remove particles associated with ind.pkl')
    group.add_argument('--tomogram', type=str, help='optionally select by individual tomogram name (if `all` then writes individual star files per tomogram')
    group.add_argument('--tomo-id-col', type=str, default='_rlnMicrographName', help='Name of column in input starfile with unique values per tomogram')
    group.add_argument('-o', required=True, help='Output .star file (treated as output base name suffixed by tomogram name if specifying `--tomogram`).'
                                                 'The output star file name must contain the string `_optimisation_set` if the input star file is of --starfile-type optimisation_set')

    group = parser.add_argument_group('Index-based filtering arguments')
    group.add_argument('--ind', help='selected indices array (.pkl)')
    group.add_argument('--ind-type', choices=('particle', 'image'), default='particle', help='use indices to filter by particle (multiple images) or by image (individual images). '
                                                                                             'Only relevant for imageseries star files filtered using ``--ind``')

    group = parser.add_argument_group('Class-label-based filtering arguments')
    group.add_argument('--labels', type=os.path.abspath, help='path to labels array (.pkl). The labels.pkl must contain a 1-D numpy array of integer class labels '
                                                              'with length matching the number of particles referenced in the star file to be filtered.')
    group.add_argument('--labels-sel', type=int, nargs='+', help='space-separated list of integer class labels to be selected (to be kept or dropped in accordance with ``--action``)')

    return parser


def check_args_compatible(args: argparse.Namespace) -> None:
    # only one of --ind or --labels can be specified
    if args.ind is not None:
        assert args.labels is None, 'Cannot specify both `--ind` and `--labels-labels`'
    elif args.labels is not None:
        assert args.ind is None, 'Cannot specify both `--ind` and `--labels-labels`'

    # if --ind-type is image, --ind must be specified
    if args.ind_type == 'image':
        assert args.ind is not None, 'Filtering with --ind-type image is only supported when filtering with --ind'

    # if --labels is provided, number of selected labels must be greater than 0
    if args.labels is not None:
        assert len(args.labels_sel) > 0


def filter_image_series_starfile(star_path: str,
                                 ind_path: str,
                                 labels_path: str,
                                 labels_sel: list[int],
                                 ind_type: Literal['particle', 'image'] = 'particle',
                                 ind_action: Literal['keep', 'drop'] = 'keep') -> starfile.TiltSeriesStarfile:
    """
    Filter an imageseries star file by specified indices in-place.

    :param star_path: path to image series star file on disk
    :param ind_path: path to indices pkl file on disk
    :param labels_path: path to labels pkl file on disk
    :param labels_sel: space-separated list of integer class labels to be selected (to be kept or dropped in accordance with ``ind_action``)
    :param ind_type: should indices be interpreted per particle (multiple images, i.e. multiple rows of df) or per image (individual images, i.e. individual row of df)
    :param ind_action: are specified indices being kept in the output star file, or dropped from the output star file
    :return: filtered TiltSeriesStarfile object
    """
    # load the star file
    star = starfile.TiltSeriesStarfile(star_path)
    ptcl_img_indices = star.get_ptcl_img_indices()
    log(f'Input star file contains {len(ptcl_img_indices)} particles consisting of {len(np.hstack(ptcl_img_indices))} images.')

    # establish indices to drop
    if ind_path:
        ind = utils.load_pkl(ind_path)
        # determine the appropriate set of indices to pass to .filter to be preserved
        if ind_type == 'particle':
            if ind_action == 'drop':
                # invert indices on particle level (groups of rows)
                ind_ptcls = np.array([i for i in range(len(ptcl_img_indices)) if i not in ind])
            elif ind_action == 'keep':
                ind_ptcls = ind
            else:
                raise ValueError
            ind_imgs = None

            # validate indices
            assert ind_ptcls.max() < len(ptcl_img_indices), 'A supplied index exceeds the number of unique particles detected'
            assert ind_ptcls.min() >= 0, 'A supplied index is negative (which is not a valid index)'
            assert len(set(ind_ptcls)) == len(ind_ptcls), 'An index was specified multiple times (which is not supported)'

        elif ind_type == 'image':
            if ind_action == 'drop':
                # invert indices on image level (individual rows)
                ind_imgs = np.array([i for i in np.hstack(ptcl_img_indices) if i not in ind])
            elif ind_action == 'keep':
                ind_imgs = ind
            else:
                raise ValueError
            ind_ptcls = None

            # validate indices
            assert ind_imgs.max() < len(np.hstack(ptcl_img_indices)), 'A supplied index exceeds the number of images detected'
            assert ind_imgs.min() >= 0, 'A supplied index is negative (which is not a valid index)'
            assert len(set(ind_imgs)) == len(ind_imgs), 'An index was specified multiple times (which is not supported)'
        else:
            raise ValueError

    elif labels_path:
        labels = utils.load_pkl(labels_path)

        # validate labels pkl
        assert len(labels) == len(ptcl_img_indices), f'The length of the labels array ({len(labels)} does not match the number of particles in the star file ({len(ptcl_img_indices)})'

        # validate selected labels
        labels_sel = list(set(labels_sel))
        labels_sel.sort()
        for label_sel in labels_sel:
            assert label_sel in labels, f'The selected label {label_sel} was not found in the supplied labels array'

        # generate ind_sel from labels_sel
        ind_sel = np.asarray([i for i, label in enumerate(labels) if label in labels_sel])

        # determine the appropriate set of indices to pass to .filter to be preserved
        if ind_action == 'drop':
            # invert indices on particle level (groups of rows)
            ind_ptcls = np.array([i for i in range(len(ptcl_img_indices)) if i not in ind_sel])
        elif ind_action == 'keep':
            ind_ptcls = ind_sel
        else:
            raise ValueError
        ind_imgs = None

    else:
        raise ValueError('One of --ind or --labels must be specified')

    # apply filtering
    star.filter(ind_imgs=ind_imgs,
                ind_ptcls=ind_ptcls)

    log(f'Filtered star file has {len(star.get_ptcl_img_indices())} particles consisting of {len(star.df)} images.')

    return star


def filter_volume_series_starfile(star_path: str,
                                  ind_path: str,
                                  labels_path: str,
                                  labels_sel: list[int],
                                  ind_action: Literal['keep', 'drop'] = 'keep') -> starfile.GenericStarfile:
    """
    Filter a volumeseries star file by specified indices in-place.

    :param star_path: path to volume series star file on disk
    :param ind_path: path to indices pkl file on disk
    :param labels_path: path to labels pkl file on disk
    :param labels_sel: space-separated list of integer class labels to be selected (to be kept or dropped in accordance with ``ind_action``)
    :param ind_action: are specified indices being kept in the output star file, or dropped from the output star file
    :return: filtered GenericStarfile object
    """
    # load the star file
    if starfile.is_starfile_optimisation_set(star_path):
        star = starfile.TomoParticlesStarfile(star_path)
        ptcl_block_name = star.block_particles
        df = star.df
    else:
        star = starfile.GenericStarfile(star_path)
        ptcl_block_name = star.identify_particles_data_block()
        df = star.blocks[ptcl_block_name]
    log(f'Input star file contains {len(df)} particles.')

    # establish indices to drop
    if ind_path is not None:
        ind_ptcls = utils.load_pkl(ind_path)
        if ind_action == 'drop':
            ind_ptcls_to_drop = ind_ptcls
        elif ind_action == 'keep':
            # invert indices on particle level (individual rows)
            ind_ptcls_to_drop = np.array([i for i in df.index.to_numpy() if i not in ind_ptcls])
        else:
            raise ValueError

        # validate indices
        assert ind_ptcls.max() < len(df), 'A supplied index exceeds the number of unique particles detected'
        assert ind_ptcls.min() >= 0, 'A supplied index is negative (which is not a valid index)'
        assert len(set(ind_ptcls)) == len(ind_ptcls), 'An index was specified multiple times (which is not supported)'

    elif labels_path is not None:
        labels = utils.load_pkl(labels_path)

        # validate labels pkl
        assert len(labels) == len(df), f'The length of the labels array ({len(labels)} does not match the number of particles in the star file ({len(df)})'

        # validate selected labels
        labels_sel = list(set(labels_sel))
        labels_sel.sort()
        for label_sel in labels_sel:
            assert label_sel in labels, f'The selected label {label_sel} was not found in the supplied labels array'

        # generate ind_sel from labels_sel
        ind_sel = np.asarray([i for i, label in enumerate(labels) if label in labels_sel])

        # determine the appropriate set of indices to pass to .filter to be preserved
        if ind_action == 'drop':
            ind_ptcls_to_drop = ind_sel
        elif ind_action == 'keep':
            # invert indices on particle level (individual rows)
            ind_ptcls_to_drop = np.array([i for i in df.index.to_numpy() if i not in ind_sel])
        else:
            raise ValueError

    else:
        raise ValueError('One of --ind or --labels must be specified')

    # apply filtering
    df = df.drop(ind_ptcls_to_drop).reset_index(drop=True)
    star.blocks[ptcl_block_name] = df

    log(f'Filtered star file contains {len(df)} particles.')

    return star


def main(args):
    # log inputs
    log(args)

    # check that selected arguments are mutually compatible
    check_args_compatible(args)

    # filter using the appropriate type of star file
    if args.starfile_type == 'imageseries':
        star = filter_image_series_starfile(star_path=args.input,
                                            ind_path=args.ind,
                                            ind_type=args.ind_type,
                                            labels_path=args.labels,
                                            labels_sel=args.labels_sel,
                                            ind_action=args.action, )
    elif args.starfile_type == 'volumeseries' or args.starfile_type == 'optimisation_set':
        star = filter_volume_series_starfile(star_path=args.input,
                                             ind_path=args.ind,
                                             labels_path=args.labels,
                                             labels_sel=args.labels_sel,
                                             ind_action=args.action, )
    else:
        raise ValueError('Unknown starfile type')

    # write the filtered star file
    star.write(args.o)

    # apply further filtering to the specified tomograms and write corresponding star files
    if args.tomogram:

        # first find the block containing particle data and ensure the specifed column for tomogram ID is present
        tomo_block_name = star.identify_particles_data_block(column_substring=args.tomo_id_col)

        if args.tomogram == 'all':
            # write each tomo's starfile out separately
            tomos_to_write = star.blocks[tomo_block_name][args.tomo_id_col].unique()
        else:
            # alternatively, specify one tomogram to preserve in output star file
            tomos_to_write = [args.tomogram]

        for tomo_name in tomos_to_write:
            # filter a copy of the star file to the requested tomogram name
            star_copy_this_tomo = copy.deepcopy(star)
            star_copy_this_tomo.blocks[tomo_block_name] = star_copy_this_tomo.blocks[tomo_block_name][star_copy_this_tomo.blocks[tomo_block_name][args.tomo_id_col].str.contains(tomo_name)]
            # write the star file
            print(f'{len(star_copy_this_tomo.blocks[tomo_block_name])} rows after filtering by tomogram {tomo_name}')
            if args.o.endswith('.star'):
                outpath = args.o.split('.star')[0]
            else:
                outpath = args.o
            star_copy_this_tomo.write(f'{outpath}_{tomo_name}.star')


if __name__ == '__main__':
    main(add_args().parse_args())
