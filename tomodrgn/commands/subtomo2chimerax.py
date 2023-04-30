'''
Create a .cxs script for ChimeraX to arrange uniquely generated tomoDRGN volumes into tomogram with optional label-based color mapping
Adapted from relionsubtomo2ChimeraX.py, written by Huy Bui, McGill University, doi: https://doi.org/10.5281/zenodo.6820119
'''

import os
import matplotlib.cm
import numpy as np
import argparse
import matplotlib.cm

from tomodrgn import starfile, utils, mrc

log = utils.log


def add_args(parser):
    parser.add_argument('starfile', type=os.path.abspath, help='Input particle_volumeseries starfile from Warp subtomogram export')
    parser.add_argument('-o', '--outfile', type=os.path.abspath, required=True, help='Output .cxc script to be opened in ChimeraX')
    parser.add_argument('--tomoname', type=str, help='Name of tomogram in input starfile for which to display tomoDRGN vols in ChimeraX')
    parser.add_argument('--vols-dir', type=os.path.abspath, help='Path to tomoDRGN reconstructed volumes')
    parser.add_argument('--vol-path', type=os.path.abspath, help='Path to single consensus volume (instead of tomoDRGN volumes)')
    parser.add_argument('--ind', type=os.path.abspath, help='Ind.pkl used in training run that produced volumes in vols-dir (if applicable)')
    parser.add_argument('--tomo-id-col-override', type=str, help='Name of column in input starfile to filter by --tomoname')
    parser.add_argument('--star-apix-override', type=float, help='Override pixel size of input particle_volumeseries starfile (A/px)')
    parser.add_argument('--vols-apix-override', type=float, help='Override pixel size of tomoDRGN-reconstructed particle volumes (A/px)')

    group = parser.add_argument_group('ChimeraX rendering options')
    group.add_argument('--vols-render-level', type=float, default=0.7, help='Isosurface level to render all tomoDRGN reconstructed volumes in ChimeraX')
    group.add_argument('--coloring-labels', type=os.path.abspath, help='labels.pkl file used to assign colors to each volume (typically kmeans labels.pkl')
    group.add_argument('--matplotlib-colormap', type=str, help='Colormap to apply to --coloring-labels (default = ChimeraX color scheme per label value) from https://matplotlib.org/stable/tutorials/colors/colormaps.html (e.g. tab10)')

    return parser


def main(args):
    # load the star file
    log(f'Loading starfile {args.starfile}')
    ptcl_star = starfile.GenericStarfile(args.starfile)

    # confirm required metadata are present in star file
    log('Checking requisite metadata in star file')
    if len([col_name for col_name in ptcl_star.blocks['data_'].columns if '_rlnCoordinate' in col_name]) > 0:
        # column headers use relion naming
        rots_cols = ['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']
        trans_cols = ['_rlnCoordinateX', '_rlnCoordinateY', '_rlnCoordinateZ']
        tomo_id_col = '_rlnMicrographName'
    else:
        m_trans_cols = [col_name for col_name in ptcl_star.blocks['data_'].columns if '_wrpCoordinate' in col_name]
        assert len(m_trans_cols) >= 3  # column headers use m naming
        if len(m_trans_cols) == 3:  # m temporal sampling == 1
            rots_cols = ['_wrpAngleRot', '_wrpAngleTilt', '_wrpAnglePsi']
            trans_cols = ['_wrpCoordinateX', '_wrpCoordinateY', '_wrpCoordinateZ']
        else:  # m temporal sampling > 1
            assert len(m_trans_cols) % 3 == 0
            rots_cols = ['_wrpAngleRot1', '_wrpAngleTilt1', '_wrpAnglePsi1']
            trans_cols = ['_wrpCoordinateX1', '_wrpCoordinateY1', '_wrpCoordinateZ1']
        tomo_id_col = '_wrpSourceName'
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
        ptcl_star.blocks['data_'].reset_index(inplace=True)
    log(f'Isolating starfile rows containing particles from {args.tomoname}')
    if args.tomo_id_col_override is not None:
        tomo_id_col = args.tomo_id_col_override
    assert args.tomoname in ptcl_star.blocks['data_'][tomo_id_col].values, f'{args.tomoname} not found in {tomo_id_col}'
    df_tomo = ptcl_star.blocks['data_'][ptcl_star.blocks['data_'][tomo_id_col] == args.tomoname]

    # load the first tomodrgn vol to get boxsize and pixel size
    if args.vols_dir:
        log(f'Enumerating volumes in {args.vols_dir}')
        all_vols = os.listdir(os.path.abspath(args.vols_dir))
        all_vols = [os.path.join(args.vols_dir, vol) for vol in all_vols if vol.endswith('.mrc')]
    elif args.vol_path:
        assert args.vol_path.endswith('.mrc')
        all_vols = [os.path.abspath(args.vol_path)]
    else:
        raise RuntimeError('One of --vols-dir or --vol-path is required')
    vol, header = mrc.parse_mrc(all_vols[0])
    vol_box = vol.shape[0]
    vol_apix = header.get_apix() if args.vols_apix_override is None else args.vols_apix_override

    # check number of volumes is consistent between df_tomo and all_vols
    if args.vols_dir:
        assert len(df_tomo) == len(all_vols), f'Mismatch between number of rows in dataframe ({len(df_tomo)}) and number of volumes in --vols-dir ({len(all_vols)}). ' \
                                              f'Possible causes: --ind not being used; incorrect star file; incorrectly generated tomodrgn volumes directory'

    if args.coloring_labels is not None:
        log(f'Loading labels for volume coloring in chimerax from {args.coloring_labels}')
        labels = utils.load_pkl(args.coloring_labels)

        # check number of labels matches either df_allparticles (filter by tomogram) or matches df_tomo (no filtering needed)
        if len(labels) == len(ptcl_star.blocks['data_']):
            log('Filtering labels array by tomogram')
            labels = labels[df_tomo.index.to_numpy()]
        elif len(labels) == len(df_tomo):
            pass
        else:
            raise RuntimeError(f'Number of labels ({len(labels)}) does not correspond to length of starfile ({len(ptcl_star.blocks["data_"])}) or number of particles in tomogram ({len(df_tomo)})')

        # organize models by shared label for ease of selection in chimerax
        labels_set = set(labels)
        labels_models_grouped = []
        for label in labels_set:
            models = [str(i + 1) for i, l in enumerate(labels) if l == label]
            models = ",".join(models)
            labels_models_grouped.append(models)

        # prepare colormap
        if args.matplotlib_colormap == None:
            log('Using Chimerax color scheme')
            chimerax_colors = np.array(((192, 192, 192, 2.55),
                                        (255, 255, 178, 2.55),
                                        (178, 255, 255, 2.55),
                                        (178, 178, 255, 2.55),
                                        (255, 178, 255, 2.55),
                                        (255, 178, 178, 2.55),
                                        (178, 255, 178, 2.55),
                                        (229, 191, 153, 2.55),
                                        (153, 191, 229, 2.55),
                                        (204, 204, 153, 2.55)), dtype=float)
            chimerax_colors *= 100 / 255  # normalize 0-100 for chimerax
            chimerax_colors = np.around(chimerax_colors).astype(int)
            labels_rgba = np.ones((len(labels), 4), dtype=int)
            for i, label in enumerate(labels_set):
                labels_rgba[labels == label] = chimerax_colors[i % len(chimerax_colors)]
        else:
            log(f'Using matplotlib color scheme {args.matplotlib_colormap}')
            cmap = matplotlib.cm.get_cmap(args.matplotlib_colormap)
            labels_norm = (labels - labels.min()) / (labels.max() - labels.min())
            labels_rgba = cmap(labels_norm)
            labels_rgba[:,:-1] *= 100
            labels_rgba = np.around(labels_rgba).astype(int)

        # save text mapping of label : rgba specification
        labels_outfile = f'{os.path.splitext(args.outfile)[0]}_rgba_labels.txt'
        with open(labels_outfile, 'w') as f:
            f.write('Mapping of unique labels : RGBA specification : chimerax models \n\n')
            for i, label in enumerate(labels_set):
                f.write(f'Label : {label} \n')
                if args.matplotlib_colormap is None:
                    f.write(f'RGB : {chimerax_colors[i % len(chimerax_colors)]} \n')
                else:
                    label_norm = (label - labels.min()) / (labels.max() - labels.min())
                    label_rgba = np.array(cmap(label_norm))
                    label_rgba[:-1] *= 100
                    label_rgba = np.around(label_rgba).astype(int)
                    f.write(f'RGB : {label_rgba[:-1]}\n')
                f.write(f'models : {labels_models_grouped[i]}\n')
                f.write('\n')
        log(f'Saved key of unique labels : RGBA specification : chimerax models to {labels_outfile}')

    # write commands for each volume
    with open(args.outfile, 'w') as f:
        for i in range(len(df_tomo)):
            # write volume opening command
            if args.vol_path:
                f.write(f'open "{all_vols[0]}"\n')
            else:
                f.write(f'open "{args.vols_dir}/vol_{i:03d}.mrc"\n')

            # prepare rotation matrix
            eulers_relion = df_tomo.iloc[i][rots_cols].to_numpy(dtype=np.float32)
            rot = utils.R_from_relion(eulers_relion[0], eulers_relion[1], eulers_relion[2])

            # prepare tomogram-scale translations
            coord_px = df_tomo.iloc[i][trans_cols].to_numpy(dtype=np.float32)
            coord_ang = coord_px * star_apix

            # incorporate translations due to refinement
            if '_rlnOriginXAngst' in df_tomo.columns:
                shift_ang = df_tomo.iloc[i]['_rlnOriginXAngst', '_rlnOriginYAngst', '_rlnOriginZAngst'].to_numpy(dtype=np.float32)
                coord_ang -= shift_ang

            # incorporate translations due to box rotation
            vol_radius_ang = (np.array([vol_box, vol_box, vol_box]) - 1) / 2 * vol_apix
            shift_volrot_ang = np.matmul(rot, -vol_radius_ang.transpose())
            coord_ang += shift_volrot_ang

            # write volume positioning command
            chimerax_view_matrix = np.concatenate((rot, coord_ang[:, np.newaxis]), axis=1)
            f.write(f'view matrix mod #{i + 1:d}{"".join([f",{i:.2f}" for i in chimerax_view_matrix.flatten()])}\n')

            if args.coloring_labels is not None:
                # write volume coloring command
                f.write(f'color #{i + 1:d} rgba({"%,".join([str(channel) for channel in labels_rgba[i]])})\n')

            f.write('\n')

        f.write(f'\nvolume #{1:d}-{len(df_tomo):d} step 1 level {args.vols_render_level:f}\n\n')
        f.write('view orient')
    log(f'Saved chimerax commands to {args.outfile}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)