'''
Filter a .star file generated by Warp subtomogram export

'''

import argparse
import numpy as np
import copy
from cryodrgn import starfile, utils

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Input .star file')
    parser.add_argument('--input-type', choices=('warp_particleseries', 'warp_volumeseries', 'm_volumeseries'),
                        default='warp_particleseries', help='input data .star source (subtomos as images vs as volumes')
    parser.add_argument('--ind', help='optionally select by indices array (.pkl)')
    parser.add_argument('--ind-type', choices=('particle', 'image', 'tilt'), default='particle',
                        help='use indices to filter by particle, by individual image, or by tilt index')
    parser.add_argument('--tomogram', type=str,
                        help='optionally select by individual tomogram name (`all` means write individual star files per tomogram')
    parser.add_argument('--action', choices=('keep', 'drop'), default='keep',
                        help='keep or remove particles associated with ind/tomogram selection')
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

def configure_dataframe_filtering_headers(args):
    if args.input_type == 'warp_particleseries':
        unique_particle_header = '_rlnGroupName'
        unique_tomo_header = '_rlnImageName'
    elif args.input_type == 'warp_volumeseries':
        unique_particle_header = None # each row is a unique particle
        unique_tomo_header = '_rlnMicrographName'
    elif args.input_type == 'm_volumeseries':
        unique_particle_header = None # each row is a unique particle
        unique_tomo_header = '_rlnMicrographName'
    return unique_particle_header, unique_tomo_header

def main(args):
    # ingest star file
    in_star = starfile.GenericStarfile(args.input)
    print(f'Loaded starfile : {args.input}')
    print(f'{len(in_star.blocks["data_"])} rows in input star file')
    print(f'Filtering action is : {args.action}')

    # load filtering selection
    if args.ind:
        print(f'Particles will be filtered by indices : {args.ind}')
        in_ind = utils.load_pkl(args.ind)
    if args.tomogram:
        print(f'Particles will be filtered by tomogram : {args.tomogram}')

    # configure filtering for input data type
    unique_particle_header, unique_tomo_header = configure_dataframe_filtering_headers(args)
    assert in_star.blocks['data_'][unique_particle_header], f'Cound not find {unique_particle_header} in star file, please check --input-type'
    assert in_star.blocks['data_'][unique_tomo_header], f'Cound not find {unique_tomo_header} in star file, please check --input-type'

    if args.ind:
        # what stride / df column to which to apply the indices
        if args.ind_type == 'particle':
            # indexed per particle, i.e. all_ind with stride = ntilts
            if unique_particle_header is not None:
                unique_particles = in_star.blocks['data_'][unique_particle_header].unique()
            else:
                unique_particles = in_star.blocks['data_'].index
            all_ind = np.arange(len(unique_particles))
            ind_to_drop = check_invert_indices(args, in_ind, all_ind)

            # validate inputs
            assert ind_to_drop.max() < len(unique_particles), 'A supplied index exceeds the number of unique particles detected'
            assert in_ind.min() >= 0, 'Negative indices are not allowed'

            # execute
            particles_to_drop = unique_particles[ind_to_drop]
            if unique_particle_header is not None:
                in_star.blocks['data_'] = in_star.blocks['data_'][~in_star.blocks['data_'][unique_particle_header].isin(particles_to_drop)]
            else:
                in_star.blocks['data_'].drop(particles_to_drop, inplace=True)
            print(f'{len(in_star.blocks["data_"])} rows after filtering by particle indices')

        elif args.ind_type == 'image':
            # indexed per individual image
            assert args.input_type == 'warp_particleseries', 'A volumeseries starfile cannot filter on a per-image basis'
            all_ind = in_star.blocks['data_'].index.to_numpy()
            ind_to_drop = check_invert_indices(args, in_ind, all_ind)

            # validate inputs
            assert in_ind.max() < all_ind.max(), 'A supplied index exceeds maximum index of data_ block'
            assert in_ind.min() >= 0, 'Negative indices are not allowed'

            # execute
            in_star.blocks['data_'].drop(ind_to_drop, in_place=True)
            print(f'{len(in_star.blocks["data_"])} rows after filtering by image indices')

        elif args.ind_type == 'tilt':
            # indexed per tilt, range of [0, ntilts], assumes all particles have same number of tilts in same order
            assert args.input_type == 'warp_particleseries', 'A volumeseries starfile cannot filter on a per-tilt basis'
            ntilts = in_star.blocks['data_']['_rlnGroupName'].value_counts().unique()
            assert len(ntilts) == 1, 'All particles must have the same number of tilt images'
            all_ind = np.arange(ntilts)
            ind_to_drop = check_invert_indices(args, in_ind, all_ind)

            # validate inputs
            assert ind_to_drop.max() < ntilts, 'A supplied index exceeds the number of tilt angles detected'
            assert in_ind.min() >= 0, 'Negative indices are not allowed'

            # execute
            nimages = in_star.blocks['data_'].max()
            tilts_to_drop = [i for i in range(nimages) if i%ntilts in ind_to_drop]
            in_star.blocks['data_'].drop(tilts_to_drop, in_place=True)
            print(f'{len(in_star.blocks["data_"])} rows after filtering by tilt indices')

    if args.tomogram:
        if args.tomogram == 'all':
            # write each tomo's starfile out separately
            tomos_to_write = in_star.blocks['data_'][unique_tomo_header].unique()
            for tomo in tomos_to_write:
                individual_outstar = copy.deepcopy(in_star)
                individual_outstar.blocks['data_'] = individual_outstar.blocks['data_'][individual_outstar.blocks['data_'][unique_tomo_header].str.contains(tomo)]
                print(f'{len(individual_outstar.blocks["data_"])} rows after filtering by tomogram {tomo}')
                if args.o.endswith('.star'):
                    outpath = args.o.split('.star')[0]
                else:
                    outpath = args.o
                individual_outstar.write(f'{outpath}_{tomo}.star')
        else:
            # write out star file for specified tomogram only
            if args.action == 'keep':
                in_star.blocks['data_'] = in_star.blocks['data_'][in_star.blocks['data_'][unique_tomo_header].str.contains(args.tomogram)]
            elif args.action == 'drop':
                in_star.blocks['data_'] = in_star.blocks['data_'][~in_star.blocks['data_'][unique_tomo_header].str.contains(args.tomogram)]
            print(f'{len(in_star.blocks["data_"])} rows after filtering by tomogram')

    in_star.write(args.o)

if __name__ == '__main__':
    main(parse_args().parse_args())
