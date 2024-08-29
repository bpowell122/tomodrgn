"""
Analyze a volume ensemble in real space using a combination of masking, PCA, UMAP, and k-means clustering.
"""

import argparse
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# noinspection PyPackageRequirements
import umap
from sklearn.decomposition import PCA

from tomodrgn import mrc, utils, analysis, starfile

log = utils.log


def add_args(_parser):
    _parser.add_argument('--voldir', type=os.path.abspath, required=True, help='path to directory containing volumes to analyze')
    _parser.add_argument('--config', type=os.path.abspath, required=True, help='path to train_vae config file')
    _parser.add_argument('--outdir', type=os.path.abspath, default=None, help='path to directory to save outputs. Default is same directory and basename as voldir, appended with `analyze_volumes`')
    _parser.add_argument('--num-pcs', type=int, default=128, help='keep this many PCs when saving PCA and running UMAP')
    _parser.add_argument('--ksample', type=int, default=None, help='Number of kmeans samples to generate (clustering voxel-PCA space). '
                                                                   'Note that this is only recommended if all particles in the dataset have had volumes generated in --voldir, '
                                                                   'to avoid confusion of k-means origin in latent space clustering and/or volume space clustering.')

    group = _parser.add_argument_group('Mask generation arguments')
    group.add_argument('--mask-path', type=os.path.abspath, help='Supply a custom real space mask instead of having tomoDRGN calculate a mask.')
    group.add_argument('--mask', type=str, choices=['none', 'sphere', 'tight', 'soft'], help='Type of real space mask to generate for each volume when calculating voxel-PCA.'
                                                                                             'Note that tight and soft masks are calculated uniquely per-volume.')
    group.add_argument('--thresh', type=float, help='Isosurface percentile at which to threshold volume; default is to use 99th percentile. '
                                                    'Only relevant for tight and soft masks.')
    group.add_argument('--dilate', type=int, help='Number of voxels to dilate thresholded isosurface outwards from mask boundary; default is to use 1/30th of box size (px). '
                                                  'Only relevant for soft mask.')
    group.add_argument('--dist', type=int, help='Number of voxels over which to apply a soft cosine falling edge from dilated mask boundary; default is to use 1/30th of box size (px). '
                                                'Only relevant for soft mask.')

    return _parser


def make_plots(outdir: str,
               config_path: str,
               pc: np.ndarray,
               pca: PCA,
               voxel_trajectories: list[np.ndarray],
               num_pcs_to_sample: int,
               umap_emb: np.ndarray,
               umap_reducer: umap.UMAP,
               kmeans_labels: np.ndarray | None = None,
               kmeans_centers_ind: np.ndarray | None = None) -> None:
    num_ksamples = len(set(kmeans_labels))

    # Make some plots using PCA transformation
    log('Generating PCA plots ...')

    # bar plot PCA explained variance ratio
    plt.bar(np.arange(pc.shape[1]) + 1, pca.explained_variance_ratio_)
    plt.xticks(np.arange(pc.shape[1]) + 1)
    plt.xlabel('principal components')
    plt.ylabel('explained variance')
    plt.tight_layout()
    plt.savefig(f'{outdir}/voxel_pca_explainedvariance.png')
    plt.close()

    # scatter plot latent PCA
    g = sns.jointplot(x=pc[:, 0],
                      y=pc[:, 1],
                      alpha=.1,
                      s=2)
    g.set_axis_labels('v-PC1', 'v-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/voxel_pca_scatter.png')
    plt.close()

    # hexbin plot latent PCA
    g = sns.jointplot(x=pc[:, 0],
                      y=pc[:, 1],
                      kind='hex')
    g.set_axis_labels('v-PC1', 'v-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/voxel_pca_hexbin.png')
    plt.close()

    # scatter plot latent PCA with kmeans center annotations
    if kmeans_labels is not None:
        analysis.scatter_annotate(x=pc[:, 0],
                                  y=pc[:, 1],
                                  centers_ind=kmeans_centers_ind,
                                  annotate=True,
                                  labels=[f'k{i}' for i in range(num_ksamples)])
        plt.xlabel('v-PC1')
        plt.ylabel('v-PC2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/voxel_pca_scatter_annotatekmeans.png')
        plt.close()

        # hexbin plot latent PCA with kmeans center annotations
        g = analysis.scatter_annotate_hex(x=pc[:, 0],
                                          y=pc[:, 1],
                                          centers_ind=kmeans_centers_ind,
                                          annotate=True,
                                          labels=[f'k{i}' for i in range(num_ksamples)])
        g.set_axis_labels('v-PC1', 'v-PC2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/voxel_pca_hexbin_annotatekmeans.png')
        plt.close()

    # scatter plot latent PCA with PCA trajectory annotations
    analysis.scatter_annotate(x=pc[:, 0],
                              y=pc[:, 1],
                              centers_xy=np.vstack([pca.transform(voxel_trajectories[0])[:, :2],  # trajectory along pc 1, trajectory is pc-dimensional so just take first two dims for plotting
                                                    pca.transform(voxel_trajectories[1])[:, :2]]),  # trajectory along pc 2, trajectory is pc-dimensional so just take first two dims for plotting
                              annotate=True,
                              labels=[f'v-PC1_{i}' for i in range(len(voxel_trajectories[0]))] + [f'v-PC2_{i}' for i in range(len(voxel_trajectories[1]))])
    plt.xlabel('v-PC1')
    plt.ylabel('v-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/pc1/voxel_pca_scatter_annotatepca.png')
    plt.close()

    # hexbin plot latent PCA with PCA trajectory annotations
    g = analysis.scatter_annotate_hex(x=pc[:, 0],
                                      y=pc[:, 1],
                                      centers_xy=np.vstack([pca.transform(voxel_trajectories[0])[:, :2],  # trajectory along pc 1, trajectory has pc-dims so just take first two dims for plotting
                                                            pca.transform(voxel_trajectories[1])[:, :2]]),  # trajectory along pc 2, trajectory has pc-dims so just take first two dims for plotting
                                      annotate=True,
                                      labels=[f'v-PC1_{i}' for i in range(len(voxel_trajectories[0]))] + [f'v-PC2_{i}' for i in range(len(voxel_trajectories[1]))])
    g.set_axis_labels('v-PC1', 'v-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/pc1/voxel_pca_hexbin_annotatepca.png')
    plt.close()

    if kmeans_labels is not None:
        # scatter plot latent PCA colored by k-means clusters
        analysis.plot_by_cluster(x=pc[:, 0],
                                 y=pc[:, 1],
                                 labels=kmeans_labels,
                                 labels_sel=num_ksamples,
                                 centers_ind=kmeans_centers_ind,
                                 annotate=True)
        plt.xlabel('v-PC1')
        plt.ylabel('v-PC2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/voxel_pca_scatter_colorkmeanslabel.png')
        plt.close()

        # scatter subplots latent PCA colored by k-means clusters
        analysis.plot_by_cluster_subplot(x=pc[:, 0],
                                         y=pc[:, 1],
                                         labels=kmeans_labels,
                                         labels_sel=num_ksamples)
        plt.xlabel('v-PC1')
        plt.ylabel('v-PC2')
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/voxel_pca_scatter_subplotkmeanslabel.png')
        plt.close()

    # UMAP dimensionality reduction
    log('Generating UMAP plots ...')

    # scatter plot latent UMAP
    g = sns.jointplot(x=umap_emb[:, 0],
                      y=umap_emb[:, 1],
                      alpha=.1,
                      s=2)
    g.set_axis_labels('v-UMAP1', 'v-UMAP2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/voxel_umap_scatter.png')
    plt.close()

    # hexbin plot latent UMAP
    g = sns.jointplot(x=umap_emb[:, 0],
                      y=umap_emb[:, 1],
                      kind='hex')
    g.set_axis_labels('v-UMAP1', 'v-UMAP2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/voxel_umap_hexbin.png')
    plt.close()

    if kmeans_labels is not None:
        # scatter plot latent UMAP with kmeans center annotations
        analysis.scatter_annotate(x=umap_emb[:, 0],
                                  y=umap_emb[:, 1],
                                  centers_ind=kmeans_centers_ind,
                                  annotate=True,
                                  labels=[f'k{i}' for i in range(num_ksamples)])
        plt.xlabel('v-UMAP1')
        plt.ylabel('v-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/voxel_umap_scatter_annotatekmeans.png')
        plt.close()

        # hexbin plot latent UMAP with kmeans center annotations
        g = analysis.scatter_annotate_hex(x=umap_emb[:, 0],
                                          y=umap_emb[:, 1],
                                          centers_ind=kmeans_centers_ind,
                                          annotate=True,
                                          labels=[f'k{i}' for i in range(num_ksamples)])
        g.set_axis_labels('v-UMAP1', 'v-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/voxel_umap_hexbin_annotatekmeans.png')
        plt.close()

    # scatter plot latent UMAP with PCA trajectory annotations
    analysis.scatter_annotate(x=umap_emb[:, 0],
                              y=umap_emb[:, 1],
                              centers_xy=np.vstack([umap_reducer.transform(pca.transform(voxel_trajectories[0])),  # trajectory in voxel space along pc 1, transform to PC space then to UMAP space
                                                    umap_reducer.transform(pca.transform(voxel_trajectories[1]))]),  # trajectory in voxel space along pc 2, transform to PC space then to UMAP space
                              annotate=True,
                              labels=[f'v-PC1_{i}' for i in range(len(voxel_trajectories[0]))] + [f'v-PC2_{i}' for i in range(len(voxel_trajectories[1]))])
    plt.xlabel('v-UMAP1')
    plt.ylabel('v-UMAP2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/pc1/voxel_umap_scatter_annotatepca.png')
    plt.close()

    # hexbin plot latent UMAP with PCA trajectory annotations
    g = analysis.scatter_annotate_hex(x=umap_emb[:, 0],
                                      y=umap_emb[:, 1],
                                      centers_xy=np.vstack([umap_reducer.transform(pca.transform(voxel_trajectories[0])),  # trajectory in voxel space along pc 1, transformed to UMAP space
                                                            umap_reducer.transform(pca.transform(voxel_trajectories[0]))]),  # trajectory in voxel space along pc 2, transformed to UMAP space
                                      annotate=True,
                                      labels=[f'v-PC1_{i}' for i in range(len(voxel_trajectories[0]))] + [f'v-PC2_{i}' for i in range(len(voxel_trajectories[1]))])
    g.set_axis_labels('v-UMAP1', 'v-UMAP2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/pc1/voxel_umap_hexbin_annotatepca.png')
    plt.close()

    if kmeans_labels is not None:
        # scatter plot latent UMAP colored by k-means clusters
        analysis.plot_by_cluster(x=umap_emb[:, 0],
                                 y=umap_emb[:, 1],
                                 labels=kmeans_labels,
                                 labels_sel=num_ksamples,
                                 centers_ind=kmeans_centers_ind,
                                 annotate=True)
        plt.xlabel('v-UMAP1')
        plt.ylabel('v-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/voxel_umap_scatter_colorkmeanslabel.png')
        plt.close()

        # scatter subplots latent UMAP colored by k-means clusters
        analysis.plot_by_cluster_subplot(x=umap_emb[:, 0],
                                         y=umap_emb[:, 1],
                                         labels=kmeans_labels,
                                         labels_sel=num_ksamples)
        plt.xlabel('v-UMAP1')
        plt.ylabel('v-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/voxel_umap_scatter_subplotkmeanslabel.png')
        plt.close()

    for i in range(num_pcs_to_sample):
        analysis.scatter_color(x=umap_emb[:, 0],
                               y=umap_emb[:, 1],
                               c=pc[:, i],
                               cbar_label=f'v-PC{i + 1}')
        plt.xlabel('v-UMAP1')
        plt.ylabel('v-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/pc{i + 1}/voxel_umap_colorlatentpca.png')
        plt.close()

    cfg = utils.load_pkl(config_path)
    starfile_path = cfg['starfile_args']['sourcefile_filtered']
    datadir = cfg['dataset_args']['datadir']
    s = starfile.TiltSeriesStarfile(starfile=starfile_path)
    star_df_backup = s.df.copy(deep=True)

    if kmeans_labels is not None:
        # make plots of first 6 images of each kmeans class
        for label in range(num_ksamples):
            # get indices of particles within this kmeans class
            ptcl_inds_this_label = np.nonzero(kmeans_labels == label)[0]
            # randomly select N particles in the class and sort their indices
            ptcl_inds_random_subset = np.sort(np.random.choice(ptcl_inds_this_label, min(len(ptcl_inds_this_label), 6), replace=False))

            s.filter(ind_ptcls=ptcl_inds_random_subset,
                     sort_ptcl_imgs='dose_ascending',
                     use_first_ntilts=1)
            imgs = s.get_particles_stack(datadir=datadir,
                                         lazy=False)

            analysis.plot_projections(images=imgs,
                                      labels=[f'{ptcl_ind}' for ptcl_ind in ptcl_inds_random_subset],
                                      width_between_imgs_px=30,
                                      height_between_imgs_px=50)
            plt.savefig(f'{outdir}/kmeans{num_ksamples}/particle_images_kmeanslabel{label}.png')
            plt.close()

            s.df = star_df_backup.copy(deep=True)

        # make plot of class label distribution versus tomogram / micrograph in star file order
        analysis.plot_label_count_distribution(ptcl_star=s,
                                               class_labels=kmeans_labels)
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/tomogram_label_distribution.png')
        plt.close()

    # make plots of numeric columns in star file (e.g. pose, coordinate, ctf) for correlations with UMAP
    os.mkdir(f'{outdir}/controls')
    ref_array = utils.load_pkl(f'{outdir}/voxel_pc_umap.pkl')
    ref_names = ['v-UMAP1', 'v-UMAP2']
    s.filter(sort_ptcl_imgs='dose_ascending', use_first_ntilts=1)  # only want one value per particle
    for numeric_column in s.df.select_dtypes(include=[np.number]).columns:
        analysis.plot_three_column_correlation(reference_array=ref_array,
                                               query_array=s.df[numeric_column].to_numpy(),
                                               reference_names=ref_names,
                                               query_name=numeric_column)
        plt.tight_layout()
        plt.savefig(f'{outdir}/controls/{numeric_column}.png')
        plt.close()


def main(args):
    # log the arguments
    log(args)

    # SETUP: assert vol-dir exists and is not empty, create outdir
    log('Validating inputs ...')
    assert os.path.isdir(args.voldir)
    if args.outdir is None:
        args.outdir = f'{args.voldir}_analyze_volumes'
    os.makedirs(args.outdir, exist_ok=True)

    # PREPROCESSING: create natural-sorted list of volumes to iterate through
    log('Finding volumes ...')
    vols_list = glob.glob(os.path.join(args.voldir, '*.mrc'))
    vols_list.sort(key=lambda fpath: int(os.path.basename(fpath).split('_')[-1].split('.mrc')[0]))  # assumes naming format `vol_001.mrc`
    box_size = mrc.parse_mrc(vols_list[0])[0].shape[0]

    # PREPROCESSING: prepare mask (if using common mask for all volumes)
    mask = np.ones((box_size, box_size, box_size), dtype=np.float32)
    if args.mask_path is not None:
        log(f'Using common mask for all volumes: {args.mask_path}')
        mask = mrc.parse_mrc(args.mask_path)[0]
        n_mask_voxels = np.sum(mask)
    elif args.mask in ['none', 'sphere']:
        log(f'Using common mask for all volumes: {args.mask}')
        mask = utils.calc_real_space_mask(vol1=np.ones((box_size, box_size, box_size)),
                                          mask=args.mask)
        n_mask_voxels = np.sum(mask)
    else:
        log(f'Using unique mask for each volume: {args.mask_path}')
        n_mask_voxels = box_size ** 3

    # PREPROCESSING: load volumes
    log('Loading and masking volumes ...')
    vols = np.zeros((len(vols_list), n_mask_voxels), dtype=np.float32)
    for i, vol in enumerate(vols_list):
        vol_unmasked = mrc.parse_mrc(vol)[0]
        if args.mask_path is not None or args.mask in ['none', 'sphere']:
            # mask is already calculated
            # can reduce memory usage by only using voxels common to the shared mask
            vols[i] = vol_unmasked[mask].ravel()
        else:
            # calculate mask
            mask = utils.calc_real_space_mask(vol1=vol_unmasked,
                                              mask=args.mask,
                                              thresh=args.thresh,
                                              dilate=args.dilate,
                                              dist=args.dist)
            # because each volume's mask is unique, need to store full masked volume for each
            vols[i] = (vol_unmasked * mask).ravel()

    # PROCESSING: run PCA, keep first num_pcs, save pkl
    log('Running PCA ...')
    assert args.num_pcs <= vols.shape[1]
    pca = PCA(n_components=args.num_pcs, random_state=42, copy=False)
    pc = pca.fit_transform(vols)
    utils.save_pkl(data=pc,
                   out_pkl=os.path.join(args.outdir, 'voxel_pc.pkl'))

    # PROCESSING: run UMAP, save pkl
    log(f'Running UMAP on first {args.num_pcs} PCs ...')
    umap_reducer = umap.UMAP(random_state=42, n_jobs=1)
    umap_emb = umap_reducer.fit_transform(pc)
    utils.save_pkl(data=umap_emb,
                   out_pkl=os.path.join(args.outdir, 'voxel_pc_umap.pkl'))

    # POST-PROCESSING: get summary volumes based on voxel-PCA pc-sampling and voxel-PCA kmeans clustering
    num_pcs_to_sample = min(args.num_pcs, 10)
    log(f'Sampling volumes along first {num_pcs_to_sample} PCs ...')
    voxel_trajectories = []
    for i in range(num_pcs_to_sample):
        os.mkdir(f'{args.outdir}/pc{i + 1}')
        voxel_pc_trajectory = np.percentile(pc[:, i], np.linspace(start=5, stop=95, num=10))
        voxel_trajectory = analysis.get_pc_traj(pca=pca,
                                                dim=i + 1,
                                                sampling_points=voxel_pc_trajectory)
        # do not do on-data because nearest neighbor search is problematic in high dimensionalities
        voxel_trajectories.append(voxel_trajectory)
        for j in range(len(voxel_trajectory)):
            # copying a header from an equivalent boxsize/pixelsize/etc
            header = mrc.parse_header(vols_list[0])
            if args.mask_path is not None or args.mask in ['none', 'sphere']:
                # construct pc trajectory volumes by backfilling unmasked voxels
                pc_vol = np.zeros((box_size, box_size, box_size), dtype=np.float32)
                pc_vol[mask] = voxel_trajectory[j]
            else:
                # construct pc trajectory volumes by reshaping voxel_trajectory voxels
                pc_vol = voxel_trajectory[j].reshape(box_size, box_size, box_size).astype(np.float32)
            mrc.write(fname=f'{args.outdir}/pc{i + 1}/voxel_pc{i + 1}_vol{j}.mrc',
                      array=pc_vol,
                      header=header,
                      is_vol=True)
    if args.ksample is not None:
        log('Performing K-means clustering ...')
        args.ksample = min(args.ksample, len(vols_list))
        os.makedirs(os.path.join(args.outdir, f'kmeans{args.ksample}'), exist_ok=True)
        kmeans_labels, kmeans_centers = analysis.cluster_kmeans(z=vols, n_clusters=args.ksample, on_data=False, reorder=False)
        _, kmeans_centers_ind = analysis.get_nearest_point(data=vols, query=kmeans_centers)
        utils.save_pkl(kmeans_labels, f'{args.outdir}/kmeans{args.ksample}/voxel_kmeans{args.ksample}_labels.pkl')
        np.savetxt(f'{args.outdir}/kmeans{args.ksample}/voxel_kmeans_centers_ind.txt', kmeans_centers_ind, fmt='%d')
        for j in range(len(kmeans_centers_ind)):
            shutil.copy(vols_list[kmeans_centers_ind[j]], f'{args.outdir}/kmeans{args.ksample}/voxel_kmeans{args.ksample}_cluster{j}.mrc')
    else:
        kmeans_labels = None
        kmeans_centers_ind = None
    del vols

    # POST-PROCESSING: generate plots
    make_plots(outdir=args.outdir,
               config_path=args.config,
               pc=pc, pca=pca,
               voxel_trajectories=voxel_trajectories,
               num_pcs_to_sample=num_pcs_to_sample,
               umap_emb=umap_emb,
               umap_reducer=umap_reducer,
               kmeans_labels=kmeans_labels,
               kmeans_centers_ind=kmeans_centers_ind)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main(add_args(parser).parse_args())
