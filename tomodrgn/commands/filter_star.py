'''
Filter a .star file generated by Warp subtomogram export

'''

import argparse
import numpy as np
import copy
from tomodrgn import starfile, utils

def add_args(parser):
    parser.add_argument('input', help='Input .star file')
    parser.add_argument('--tomo-id-col', type=str, default='_rlnMicrographName', help='Name of column in input starfile with unique values per tomogram')
    parser.add_argument('--ptcl-id-col', type=str, default='_rlnGroupName', help='Name of column in input starfile with unique values per particle, '
                                                                                 'if `index` then each row is treated as a unique particle')
    parser.add_argument('--starfile-source', type=str, default='warp', choices=('warp', 'relion31', 'cistem'), help='Software that created starfile; used to identify particles data block')
    parser.add_argument('--ind', help='selected indices array (.pkl)')
    parser.add_argument('--ind-type', choices=('particle', 'image'), default='particle',
                        help='use indices to filter by particle or by individual image')
    parser.add_argument('--tomogram', type=str,
                        help='optionally select by individual tomogram name (if `all` then writes individual star files per tomogram')
    parser.add_argument('--action', choices=('keep', 'drop'), default='keep',
                        help='keep or remove particles associated with ind.pkl')
    parser.add_argument('-o', required=True, help='Output .star file')
    return parser

def check_invert_indices(args, in_ind, all_ind):
    # get the right indices to drop
    if args.action == 'keep':
        # invert selection
        ind_to_drop = np.array([i for i in all_ind if i not in in_ind])
    else:
        # we already have the ind to drop
        ind_to_drop = in_ind
    return ind_to_drop


def main(args):
    # ingest star file
    in_star = starfile.GenericStarfile(args.input)
    print(f'Loaded starfile : {args.input}')
    print(f'Parsing starfile with starfile_source mode: {args.starfile_source}')
    if args.starfile_source == 'warp':
        ptcl_block = 'data_'
    elif args.starfile_source == 'relion31':
        ptcl_block = 'data_particles'
    elif args.starfile_source == 'cistem':
        ptcl_block = 'data_'
    else:
        raise ValueError
    print(f'{len(in_star.blocks[ptcl_block])} rows in input star file')
    print(f'Filtering action is : {args.action}')

    # load filtering selection
    if args.ind:
        print(f'Particles will be filtered by indices : {args.ind}')
        print(f'Indices will be interpreted as uniquely corresponding per: {args.ind_type}')
        in_ind = utils.load_pkl(args.ind)
    if args.tomogram == 'all':
        print('Particles will be filtered separately by all tomograms')
    elif args.tomogram:
        print(f'Particles will be filtered by tomogram : {args.tomogram}')

    # configure filtering for input data type
    if args.tomogram:
        unique_tomo_header = args.tomo_id_col
        assert unique_tomo_header in in_star.blocks[ptcl_block].columns, f'{unique_tomo_header} not found in starfile {ptcl_block} block'
    if args.ptcl_id_col == 'index':
        unique_particle_header = None
    else:
        unique_particle_header = args.ptcl_id_col
        assert unique_particle_header in in_star.blocks[ptcl_block].columns, f'{unique_particle_header} not found in starfile {ptcl_block} block'

    if args.ind:
        # what stride / df column to which to apply the indices
        if args.ind_type == 'particle':
            # indexed per particle
            if unique_particle_header is not None:
                unique_particles = in_star.blocks[ptcl_block][unique_particle_header].unique()
            else:
                unique_particles = in_star.blocks[ptcl_block].index
            all_ind = np.arange(len(unique_particles))
            ind_to_drop = check_invert_indices(args, in_ind, all_ind)

            # validate inputs
            assert ind_to_drop.max() < len(unique_particles), 'A supplied index exceeds the number of unique particles detected'
            assert in_ind.min() >= 0, 'A supplied index is negative which is not valid'

            # execute
            particles_to_drop = unique_particles[ind_to_drop]
            if unique_particle_header is not None:
                in_star.blocks[ptcl_block] = in_star.blocks[ptcl_block][~in_star.blocks[ptcl_block][unique_particle_header].isin(particles_to_drop)]
            else:
                in_star.blocks[ptcl_block].drop(particles_to_drop, inplace=True)
            print(f'{len(in_star.blocks[ptcl_block])} rows after filtering by particle indices')

        elif args.ind_type == 'image':
            # indexed per individual image
            assert args.input_type == 'warp_imageseries', 'A volumeseries starfile cannot filter on a per-image basis because each row represents a volume'
            all_ind = in_star.blocks[ptcl_block].index.to_numpy()
            ind_to_drop = check_invert_indices(args, in_ind, all_ind)

            # validate inputs
            assert in_ind.max() < all_ind.max(), f'A supplied index exceeds maximum index of {ptcl_block} block'
            assert in_ind.min() >= 0, 'A supplied index is negative which is not valid'

            # execute
            in_star.blocks[ptcl_block].drop(ind_to_drop, in_place=True)
            print(f'{len(in_star.blocks[ptcl_block])} rows after filtering by image indices')

        elif args.ind_type == 'tilt':
            # indexed per tilt, range of [0, ntilts], assumes all particles have same number of tilts in same order
            assert args.input_type == 'warp_imageseries', 'A volumeseries starfile cannot filter on a per-image basis because each row represents a volume'
            ntilts = in_star.blocks[ptcl_block]['_rlnGroupName'].value_counts().unique()
            assert len(ntilts) == 1, 'All particles must have the same number of tilt images to filter by tilt index'
            all_ind = np.arange(ntilts)
            ind_to_drop = check_invert_indices(args, in_ind, all_ind)

            # validate inputs
            assert ind_to_drop.max() < ntilts, 'A supplied index exceeds the number of tilts detected'
            assert in_ind.min() >= 0, 'A supplied index is negative which is not valid'

            # execute
            nimages = in_star.blocks[ptcl_block].max()
            tilts_to_drop = [i for i in range(nimages) if i%ntilts in ind_to_drop]
            in_star.blocks[ptcl_block].drop(tilts_to_drop, in_place=True)
            print(f'{len(in_star.blocks[ptcl_block])} rows after filtering by tilt indices')

    if args.tomogram:
        if args.tomogram == 'all':
            # write each tomo's starfile out separately
            tomos_to_write = in_star.blocks[ptcl_block][unique_tomo_header].unique()
            for tomo in tomos_to_write:
                individual_outstar = copy.deepcopy(in_star)
                individual_outstar.blocks[ptcl_block] = individual_outstar.blocks[ptcl_block][individual_outstar.blocks[ptcl_block][unique_tomo_header].str.contains(tomo)]
                print(f'{len(individual_outstar.blocks[ptcl_block])} rows after filtering by tomogram {tomo}')
                if args.o.endswith('.star'):
                    outpath = args.o.split('.star')[0]
                else:
                    outpath = args.o
                individual_outstar.write(f'{outpath}_{tomo}.star')
        else:
            # write out star file for specified tomogram only
            if args.action == 'keep':
                in_star.blocks[ptcl_block] = in_star.blocks[ptcl_block][in_star.blocks[ptcl_block][unique_tomo_header].str.contains(args.tomogram)]
            elif args.action == 'drop':
                in_star.blocks[ptcl_block] = in_star.blocks[ptcl_block][~in_star.blocks[ptcl_block][unique_tomo_header].str.contains(args.tomogram)]
            print(f'{len(in_star.blocks[ptcl_block])} rows after filtering by tomogram')

    in_star.write(args.o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)
