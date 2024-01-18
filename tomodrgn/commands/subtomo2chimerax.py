'''
Create a .cxs script for ChimeraX to arrange uniquely generated tomoDRGN volumes into tomogram with optional label-based color mapping
Adapted from relionsubtomo2ChimeraX.py, written by Huy Bui, McGill University, doi: https://doi.org/10.5281/zenodo.6820119

Example usage
    marker mode (places chimerax markers only)
        python /nobackup/users/bmp/software/tomodrgn/tomodrgn/commands/subtomo2chimerax.py ../m_output_starfiles/10499_22k_box64_angpix6_volumeseries.star -o test_subtomo2chimerax --ind ../27_vae_box96_256x3_128_256x3_128_256x3_b1_gaussian/ind_keep.20981_particles.pkl --mode markers --marker-radius-angstrom 150 --coloring-labels analyze.49/kmeans20/labels.pkl

    volumes mode (places unique tomoDRGN volumes per particle)
        python /nobackup/users/bmp/software/tomodrgn/tomodrgn/commands/subtomo2chimerax.py ../m_output_starfiles/10499_22k_box64_angpix6_volumeseries.star -o test_subtomo2chimerax_volumes --ind ../27_vae_box96_256x3_128_256x3_128_256x3_b1_gaussian/ind_keep.20981_particles.pkl --mode volumes -w weights.49.pkl -c config.pkl --zfile z.49.pkl --downsample 64 --vols-apix 11.5625 --vols-render-level 0.008 --coloring-labels analyze.49/kmeans20/labels.pkl

'''

import os
import matplotlib.cm
import numpy as np
import argparse
import matplotlib.cm
import matplotlib.pyplot as plt

from tomodrgn import starfile, utils, mrc, analysis

log = utils.log


def add_args(parser):
    parser.add_argument('starfile', type=os.path.abspath, help='Input particle_volumeseries starfile from Warp subtomogram export')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Path to directory to store ouptut script(s), volumes, etc')
    parser.add_argument('--tomoname', type=str, help='Optionally name single tomogram in input starfile for which to display tomoDRGN vols in ChimeraX')
    parser.add_argument('--tomo-id-col-override', type=str, help='Name of column in input starfile to filter by --tomoname')
    parser.add_argument('--star-apix-override', type=float, help='Override pixel size of input particle_volumeseries starfile (A/px)')
    parser.add_argument('--ind', type=os.path.abspath, help='Ind.pkl used in training run that produced volumes in vols-dir (if applicable)')
    parser.add_argument('--mode', choices=('volumes', 'volume', 'markers'), default='markers', help='Whether to render tomogram scene with unique volumes per particle, a single volume for all particles, or markers for each particle')

    group = parser.add_argument_group('Volume specification options - single pregenerated volume')
    group.add_argument('--vol-path', type=os.path.abspath, help='Path to single consensus volume (instead of tomoDRGN volume ensemble)')

    group = parser.add_argument_group('Volume specification options - generate tomoDRGN volume ensemble')
    group.add_argument('-w', '--weights', help='Model weights from train_vae')
    group.add_argument('-c', '--config', help='config.pkl file from train_vae')
    group.add_argument('--zfile', type=os.path.abspath, help='Text/.pkl file with z-values to evaluate')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volume')
    group.add_argument('--invert', action='store_true', help='Invert contrast of output volume')
    group.add_argument('-d','--downsample', type=int, help='Downsample volumes to this box size (pixels)')
    group.add_argument('--lowpass', type=float, default=None, help='Lowpass filter to this resolution in Å.')

    group = parser.add_argument_group('ChimeraX rendering options')
    group.add_argument('--vols-render-level', type=float, default=0.7, help='Isosurface level to render all volumes in ChimeraX')
    group.add_argument('--vols-apix', type=float, default=None, help='Pixel size of subtomogram volumes. If downsampling, need to manually adjust by ratio of old/new box sizes.')
    group.add_argument('--marker-radius-angstrom', type=float, help='Radius of markers if `mode` is set to `markers`')
    group.add_argument('--coloring-labels', type=os.path.abspath, help='labels.pkl file used to assign colors to each volume (typically kmeans labels.pkl')
    group.add_argument('--matplotlib-colormap', type=str, help='Colormap to apply to --coloring-labels (default = ChimeraX color scheme per label value) from https://matplotlib.org/stable/tutorials/colors/colormaps.html (e.g. tab10)')

    return parser


def validate_starfile(args, ptcl_star):
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

    # confirm tomo_id_col_override in columns
    if args.tomo_id_col_override is not None:
        tomo_id_col = args.tomo_id_col_override
        assert tomo_id_col in ptcl_star.blocks['data_'].columns

    return rots_cols, trans_cols, star_apix, tomo_id_col


def generate_colormap(args, ptcl_star):
    # load labels per particle
    if args.coloring_labels is not None:
        log(f'Loading labels for volume coloring in chimerax from {args.coloring_labels}')
        labels = utils.load_pkl(args.coloring_labels)
        # check number of labels matches total number of particles
        assert len(labels) == len(ptcl_star.blocks['data_'])
    else:
        # unique label per particle
        labels = np.arange(len(ptcl_star.blocks['data_']))
        np.random.shuffle(labels)  # labels should be randomized so that colormap does not form gradient along particle_index

    # prepare colormap of per-particle labels
    labels_set = set(labels)
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
        labels_rgba[:, :-1] *= 100
        labels_rgba = np.around(labels_rgba).astype(int)

    # save colors per particle in df
    ptcl_star.blocks['data_']['labels'] = labels
    ptcl_star.blocks['data_']['labels_colors_rgba'] = labels_rgba.tolist()

    return None


def plot_label_count_distribution(ptcl_star, tomo_id_col, outdir):
    ind_by_tomo = [group.index.to_numpy() for group_name, group in ptcl_star.blocks['data_'].groupby(tomo_id_col)]
    class_labels = ptcl_star.blocks['data_']['labels'].to_numpy()

    if len(set(class_labels)) > 100:
        log('Found more than 100 unique labels; skipping generation of label-vs-tomogram distribution plot')
        return

    label_distribution = np.zeros((len(ind_by_tomo), len(set(class_labels))))
    for i, ind_one_tomo in enumerate(ind_by_tomo):
        # don't use np.unique directly on one tomogram in case that tomogram has zero particles in given class
        counts_one_tomo = np.asarray([np.sum(class_labels[ind_by_tomo[i]] == label) for label in np.unique(class_labels, return_counts=False)])
        label = 'particle count per class'
        label_distribution[i] = counts_one_tomo

    fig, ax = plt.subplots(1, 1, )  #figsize=(np.asarray([1, 1]) * label_distribution.T.shape))

    distribution_plot = ax.imshow(label_distribution.T)
    fig.colorbar(distribution_plot, ax=ax, label=label)

    ax.set_xlabel('tomogram')
    ax.set_ylabel('class label')

    plt.tight_layout()
    plt.savefig(f'{outdir}/labels_distribution_count.png')
    plt.close()


def validate_rendering_mode(args):
    if args.mode =='volumes' or args.mode == 'volume':
        if args.vols_dir is not None:
            log(f'Enumerating volumes in {args.vols_dir}')
            all_vols = os.listdir(os.path.abspath(args.vols_dir))
            all_vols = [os.path.join(args.vols_dir, vol) for vol in all_vols if vol.endswith('.mrc')]
            all_vols.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.mrc')[0]))  # assumes naming format `vol_001.mrc`
        elif args.vol_path is not None:
            assert args.vol_path.endswith('.mrc')
            all_vols = [os.path.abspath(args.vol_path)]
        vol, header = mrc.parse_mrc(all_vols[0])
        vol_box = vol.shape[0]
        vol_apix = header.get_apix() if args.vols_apix is None else args.vols_apix
    elif args.mode == 'markers':
        # required for rotations to be about the origin with rotation approach used below
        vol_box = 1
        vol_apix = 0
        all_vols = []
    else:
        raise RuntimeError(f'Unknown mode specified: {args.mode=}')

    return vol_box, vol_apix, all_vols


def write_labels_rgba_by_model(df_one_tomo, outdir_one_tomo):
    # organize models by shared label for ease of selection in chimerax
    labels_set = set(df_one_tomo['labels'].to_numpy())

    # save text mapping of label : rgba specification
    labels_outfile = f'{outdir_one_tomo}/rgba_labels.txt'
    with open(labels_outfile, 'w') as f:
        f.write('Mapping of unique labels : RGBA specification : chimerax models \n\n')
        for i, label in enumerate(labels_set):
            f.write(f'Label : {label} \n')

            rgba = df_one_tomo.loc[df_one_tomo["labels"] == label, "labels_colors_rgba"].values[0]
            f.write(f'RGBa % : {",".join([str(i) for i in rgba])} \n')

            f.write(f'models : {",".join([str(i + 1) for i, l in enumerate(df_one_tomo["labels"].to_numpy()) if l == label])} \n')
            f.write('\n')


def write_mapback_script(df_one_tomo, outdir_one_tomo, all_vols, rots_cols, trans_cols, args):
    # write commands for each volume
    with open(f'{outdir_one_tomo}/mapback.cxc', 'w') as f:
        for i in range(len(df_one_tomo)):
            # write volume opening command
            if args.vol_path:
                f.write(f'open "{os.path.basename(all_vols[0])}"\n')
            elif args.vols_dir:
                f.write(f'open "{os.path.basename(all_vols[i])}"\n')
            elif args.mode == 'markers':
                f.write(f'marker #{i+1} position 0,0,0 radius {args.marker_radius_angstrom}\n')

            # prepare rotation matrix
            eulers_relion = df_one_tomo.iloc[i][rots_cols].to_numpy(dtype=np.float32)
            rot = utils.R_from_relion(eulers_relion[0], eulers_relion[1], eulers_relion[2])

            # prepare tomogram-scale translations
            coord_px = df_one_tomo.iloc[i][trans_cols].to_numpy(dtype=np.float32)
            coord_ang = coord_px * args.star_apix

            # incorporate translations due to refinement
            if '_rlnOriginXAngst' in df_one_tomo.columns:
                shift_ang = df_one_tomo.iloc[i]['_rlnOriginXAngst', '_rlnOriginYAngst', '_rlnOriginZAngst'].to_numpy(dtype=np.float32)
                coord_ang -= shift_ang

            # incorporate translations due to box rotation
            vol_radius_ang = (np.array([args.vol_box, args.vol_box, args.vol_box]) - 1) / 2 * args.vol_apix
            shift_volrot_ang = np.matmul(rot, -vol_radius_ang.transpose())
            coord_ang += shift_volrot_ang

            # write volume positioning command
            chimerax_view_matrix = np.concatenate((rot, coord_ang[:, np.newaxis]), axis=1)
            f.write(f'view matrix mod #{i + 1:d}{"".join([f",{i:.2f}" for i in chimerax_view_matrix.flatten()])}\n')

            # write volume coloring command
            f.write(f'color #{i + 1:d} rgba({"%,".join([str(channel) for channel in df_one_tomo.iloc[i]["labels_colors_rgba"]])})\n')

            f.write('\n')

        if args.mode == 'volumes' or args.mode == 'volume':
            f.write('\n')
            f.write(f'volume #{1:d}-{len(df_one_tomo):d} step 1 level {args.vols_render_level:f}')
            f.write('\n\n')

        f.write('view orient')


def main(args):
    # load the star file
    log(f'Loading starfile {args.starfile}')
    ptcl_star = starfile.GenericStarfile(args.starfile)

    # validate star file metadata
    rots_cols, trans_cols, args.star_apix, tomo_id_col = validate_starfile(args, ptcl_star)

    # filter by indices used during training (if used)
    if args.ind is not None:
        log(f'Filtering starfile by indices {args.ind}')
        ind = utils.load_pkl(args.ind)
        ptcl_star.blocks['data_'] = ptcl_star.blocks['data_'].iloc[ind]
        ptcl_star.blocks['data_'].reset_index(inplace=True)

    # populate colormap, add labels and colors to ind-filtered df
    log('Generating colormap')
    generate_colormap(args, ptcl_star)

    # plot distribution of label counts per tomogram
    os.makedirs(args.outdir, exist_ok=True)
    plot_label_count_distribution(ptcl_star, tomo_id_col, args.outdir)

    # get list of all tomograms in star file, optionally filtered to tomo of choice
    log('Finding list of all tomograms in star file')
    tomo_names = ptcl_star.blocks['data_'][tomo_id_col].unique()
    if args.tomoname:
        log(f'Filtering list of all tomograms to specified tomogram: {args.tomoname}')
        assert args.tomoname in tomo_names
        tomo_names = [args.tomoname]

    # loop over all tomograms
    for tomo_name in tomo_names:
        log(f'Working on tomogram: {tomo_name}')
        df_one_tomo = ptcl_star.blocks['data_'][ptcl_star.blocks['data_'][tomo_id_col] == tomo_name]

        # create output dir for this tomogram
        outdir_one_tomo = f'{args.outdir}/tomo_{tomo_name}'
        os.makedirs(outdir_one_tomo, exist_ok=True)

        if args.mode == 'volumes':
            log('Generating tomoDRGN volumes')
            # load z file and assert same length as df
            z_all = utils.load_pkl(args.zfile)
            assert len(z_all) == len(ptcl_star.blocks['data_'])

            # filter z to just this tomogram's inds; write to disk
            z_one_tomo = z_all[df_one_tomo.index.to_numpy()]
            utils.save_pkl(z_one_tomo, f'{outdir_one_tomo}/z_values.pkl')

            # generate all volumes in this tomogram by args and z_values.pkl, store in outdir_one_tomo
            analysis.gen_volumes(args.weights, args.config, f'{outdir_one_tomo}/z_values.pkl', outdir_one_tomo, Apix=args.vols_apix,
                                 flip=args.flip, downsample=args.downsample, invert=args.invert, lowpass=args.lowpass)

        # load the first tomodrgn vol to get boxsize and pixel size
        args.vols_dir = outdir_one_tomo if args.mode == 'volumes' else None
        args.vol_box, args.vol_apix, all_vols = validate_rendering_mode(args)

        # write labels/RGBa in this tomogram
        log('Saving key of unique labels : RGBA specification : chimerax models')
        write_labels_rgba_by_model(df_one_tomo, outdir_one_tomo)

        # write mapback cxc for this tomogram
        log('Saving .cxc file to view subtomogram map-back')
        write_mapback_script(df_one_tomo, outdir_one_tomo, all_vols, rots_cols, trans_cols, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)