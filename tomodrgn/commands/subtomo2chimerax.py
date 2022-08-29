'''
Create a .cxs script for ChimeraX to arrange uniquely generated tomoDRGN volumes into tomogram with optional label-based color mapping
Adapted from relionsubtomo2ChimeraX.py, written by Huy Bui, McGill University, doi: https://doi.org/10.5281/zenodo.6820119
'''

import os
import matplotlib.cm
import numpy as np
import argparse
import matplotlib.cm

from tomodrgn import starfile
from tomodrgn import utils
from tomodrgn import mrc

log = utils.log


def add_args(parser):
    parser.add_argument('starfile', type=os.path.abspath, help='Input particle_volumeseries starfile from Warp subtomogram export')
    parser.add_argument('-o', '--outfile', type=os.path.abspath, required=True, help='Output .cxc script to be opened in ChimeraX')
    parser.add_argument('--tomoname', type=str, help='Name of tomogram in input starfile for which to display tomoDRGN vols in ChimeraX')
    parser.add_argument('--tomo-id-col', type=str, default='_rlnMicrographName', help='Name of column in input starfile to filter by --tomoname')
    parser.add_argument('--star-apix-override', type=float, help='Override pixel size of input particle_volumeseries starfile (A/px)')
    parser.add_argument('--vols-dir', type=os.path.abspath, required=True, help='Path to tomoDRGN reconstructed volumes')
    parser.add_argument('--vols-apix-override', type=float, help='Override pixel size of tomoDRGN-reconstructed particle volumes (A/px)')
    parser.add_argument('--ind', type=os.path.abspath, help='Ind.pkl used in training run that produced volumes in vols-dir (if applicable)')

    group = parser.add_argument_group('ChimeraX rendering options')
    group.add_argument('--vols-render-level', type=float, default=0.7, help='Isosurface level to render all tomoDRGN reconstructed volumes in ChimeraX')
    group.add_argument('--coloring-labels', type=os.path.abspath, help='labels.pkl file used to assign colors to each volume (typically kmeans labels.pkl')
    group.add_argument('--matplotlib-colormap', type=str, default='tab20', help='Name of colormap to sample labels, from https://matplotlib.org/stable/tutorials/colors/colormaps.html')

    return parser


def main(args):
    # load the star file
    log(f'Loading and checking starfile {args.starfile}')
    ptcl_star = starfile.GenericStarfile(args.starfile)
    rots_cols = ['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']
    trans_cols = ['_rlnCoordinateX', '_rlnCoordinateY', '_rlnCoordinateZ']
    assert np.all([col in ptcl_star.blocks['data_'].columns for col in rots_cols]), f'`{rots_cols} columns not found in starfile `data_` block'
    assert np.all([col in ptcl_star.blocks['data_'].columns for col in trans_cols]), f'{trans_cols} columns not found in starfile `data_` block'

    # get the pixel size used in star file for translations
    if args.star_apix_override is None:
        assert '_rlnPixelSize' in ptcl_star.blocks['data_'].columns, 'Could not find `_rlnPixelSize` column in input starfile. Make sure it is present, or use --star-apix-override'
        star_apix = float(ptcl_star.blocks['data_']['_rlnPixelSize'][0])
    else:
        star_apix = args.star_apix_override

    # filter by indices used during training (if used) and by tomogram of interest to get df with n_rows == n_volumes
    if args.ind is not None:
        log(f'Filtering starfile by indices {args.ind}')
        ind = utils.load_pkl(args.ind)
        ptcl_star.blocks['data_'] = ptcl_star.blocks['data_'].iloc[ind]
    log(f'Isolating starfile rows containing particles from {args.tomoname}')
    assert args.tomoname in ptcl_star.blocks['data_'][args.tomo_id_col].values, f'{args.tomoname} not found in {args.tomo_id_col}'
    df_tomo = ptcl_star.blocks['data_'][ptcl_star.blocks['data_'][args.tomo_id_col] == args.tomoname]
    df_tomo.reset_index(inplace=True)

    # load the first tomodrgn vol to get boxsize and pixel size
    log(f'Enumerating volumes in {args.vols_dir}')
    all_vols = os.listdir(os.path.abspath(args.vols_dir))
    all_vols = [os.path.join(args.vols_dir, vol) for vol in all_vols if vol.endswith('.mrc')]
    vol, header = mrc.parse_mrc(all_vols[0])
    vol_box = vol.shape[0]
    vol_apix = header.get_apix() if args.vols_apix_override is None else args.vols_apix_override

    # check number of volumes is consistent between df_tomo and all_vols
    assert len(df_tomo) == len(all_vols), f'Mismatch between number of rows in dataframe ({len(df_tomo)}) and number of volumes in --vols-dir ({len(all_vols)}). ' \
                                          f'Possible causes: --ind not being used; incorrect star file; incorrectly generated tomodrgn volumes directory'

    if args.coloring_labels is not None:
        log(f'Loading labels for volume coloring in chimerax from {args.coloring_labels}')
        # load the labels file and
        labels = utils.load_pkl(args.coloring_labels)

        # normalize labels to [0,1]
        labels_norm = (labels - labels.min()) / (labels.max() - labels.min())

        # filter by ind.pkl + tomoname
        labels = labels[df_tomo['index']]

        # prepare colormap
        log(f'Using colormap {args.matplotlib_colormap}')
        cmap = matplotlib.cm.get_cmap(args.matplotlib_colormap)
        labels_rgba = cmap(labels_norm)

        # save text mapping of label : rgba specification
        labels_set = set(labels)
        labels_outfile = f'{os.path.splitext(args.outfile)[0]}_rgba_labels.txt'
        with open(labels_outfile, 'w') as f:
            f.write('Mapping of unique labels : RGBA specification (normalized 0-1)\n')
            for label in labels_set:
                f.write(f'{label} : {cmap((label - labels.min()) / (labels.max() - labels.min()))}\n')
        log(f'Saved key of labels : RGBA specifications (normalized 0-1) to {labels_outfile}')

    # write commands for each volume
    with open(args.outfile, 'w') as f:
        for i in range(len(df_tomo)):
            # write volume opening command
            f.write(f'open {args.vols_dir}/vol_{i:03d}.mrc\n')

            # prepare rotation matrix
            eulers_relion = df_tomo.loc[i, rots_cols].to_numpy(dtype=np.float32)
            rot = utils.R_from_relion(eulers_relion[0], eulers_relion[1], eulers_relion[2])

            # prepare tomogram-scale translations
            coord_px = df_tomo.loc[i, trans_cols].to_numpy(dtype=np.float32)
            coord_ang = coord_px * star_apix

            # incorporate translations due to refinement
            if '_rlnOriginXAngst' in df_tomo.columns:
                shift_ang = df_tomo.loc[i, ['_rlnOriginXAngst', '_rlnOriginYAngst', '_rlnOriginZAngst']].to_numpy(dtype=np.float32)
                coord_ang -= shift_ang

            # incorporate translations due to box rotation
            vol_radius_ang = (np.array([vol_box,vol_box,vol_box]) - 1) / 2 * vol_apix
            shift_volrot_ang = np.matmul(rot, -vol_radius_ang.transpose())
            coord_ang += shift_volrot_ang

            # write volume positioning command
            chimerax_view_matrix = np.concatenate((rot, coord_ang[:,np.newaxis]), axis=1)
            f.write(f'view matrix mod #{i+1:d}{"".join([f",{i:.2f}" for i in chimerax_view_matrix.flatten()])}\n')

            if args.coloring_labels is not None:
                # write volume coloring command
                f.write(f'color #{i+1:d} rgba{tuple(labels_rgba[i])}\n')

            f.write('\n')

        f.write(f'\nvolume #{1:d}-{len(df_tomo):d} step 1 level {args.vols_render_level:f}\n\n')
        f.write('view orient')
    log(f'Saved chimerax commands to {args.outfile}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)