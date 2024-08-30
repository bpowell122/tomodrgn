"""
Filter a .star file by selected particle or image indices, optionally per-tomogram
"""

import argparse
import numpy as np
import copy
from typing import Literal
from tomodrgn import starfile, utils

log = utils.log


def add_args(_parser):
    _parser.add_argument('input', help='Input .star file')
    _parser.add_argument('--starfile-type', type=str, default='imageseries', choices=('imageseries', 'volumeseries'), help='Type of star file to filter; '
                                                                                                                           'Do rows correspond to particle images or particle volumes')
    _parser.add_argument('--ind', help='selected indices array (.pkl)')
    _parser.add_argument('--ind-type', choices=('particle', 'image'), default='particle', help='use indices to filter by particle (multiple images) or by image (individual images). '
                                                                                               'Only relevant for imageseries star files')
    _parser.add_argument('--action', choices=('keep', 'drop'), default='keep', help='keep or remove particles associated with ind.pkl')
    _parser.add_argument('--tomogram', type=str, help='optionally select by individual tomogram name (if `all` then writes individual star files per tomogram')
    _parser.add_argument('--tomo-id-col', type=str, default='_rlnMicrographName', help='Name of column in input starfile with unique values per tomogram')
    _parser.add_argument('-o', required=True, help='Output .star file (treated as output base name suffixed by tomogram name if specifying `--tomogram`)')
    return _parser


def filter_image_series_starfile(star_path: str,
                                 ind_path: str,
                                 ind_type: Literal['particle', 'image'] = 'particle',
                                 ind_action: Literal['keep', 'drop'] = 'keep') -> starfile.TiltSeriesStarfile:
    """
    Filter an imageseries star file by specified indices in-place.

    :param star_path: path to image series star file on disk
    :param ind_path: path to indices pkl file on disk
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

        # apply filtering
        star.filter(ind_imgs=ind_imgs,
                    ind_ptcls=ind_ptcls)

    log(f'Filtered star file has {len(star.get_ptcl_img_indices())} particles consisting of {len(star.df)} images.')

    return star


def filter_volume_series_starfile(star_path: str,
                                  ind_path: str,
                                  ind_action: Literal['keep', 'drop'] = 'keep') -> starfile.GenericStarfile:
    # load the star file
    star = starfile.GenericStarfile(star_path)
    ptcl_block_name = star.identify_particles_data_block()
    df = star.blocks[ptcl_block_name]
    log(f'Input star file contains {len(df)} particles.')

    # establish indices to drop
    if ind_path is not None:
        ind_ptcls = utils.load_pkl(ind_path)
        if ind_action == 'drop':
            # invert indices on particle level (individual rows)
            ind_ptcls = np.array([i for i in df.index.to_numpy() if i not in ind_ptcls])
        elif ind_action == 'keep':
            pass
        else:
            raise ValueError

        # validate indices
        assert ind_ptcls.max() < len(df), 'A supplied index exceeds the number of unique particles detected'
        assert ind_ptcls.min() >= 0, 'A supplied index is negative (which is not a valid index)'
        assert len(set(ind_ptcls)) == len(ind_ptcls), 'An index was specified multiple times (which is not supported)'

        # apply filtering
        df = df.drop(ind_ptcls).reset_index(drop=True)
        star.blocks[ptcl_block_name] = df

    log(f'Filtered star file contains {len(df)} particles.')

    return star


def main(args):
    # log inputs
    log(args)

    # filter using the appropriate type of star file
    if args.starfile_type == 'imageseries':
        star = filter_image_series_starfile(star_path=args.input,
                                            ind_path=args.ind,
                                            ind_type=args.ind_type,
                                            ind_action=args.action, )
    elif args.starfile_type == 'volumeseries':
        star = filter_volume_series_starfile(star_path=args.input,
                                             ind_path=args.ind,
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
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
