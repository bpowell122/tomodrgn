"""
Visualize latent space and generate volumes
"""

import argparse
import numpy as np
import os
import shutil
from datetime import datetime as dt
from importlib_resources import files

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from tomodrgn import analysis, utils, starfile

log = utils.log


def add_args(_parser):
    _parser.add_argument('workdir', type=os.path.abspath, help='Directory with tomoDRGN results')
    _parser.add_argument('epoch', type=int, help='Epoch number N to analyze (0-based indexing, corresponding to z.N.pkl, weights.N.pkl)')
    _parser.add_argument('--device', type=int, help='Optionally specify CUDA device')
    _parser.add_argument('-o', '--outdir', help='Output directory for analysis results (default: [workdir]/analyze.[epoch])')
    _parser.add_argument('--skip-vol', action='store_true', help='Skip generation of volumes')
    _parser.add_argument('--skip-umap', action='store_true', help='Skip running UMAP')

    group = _parser.add_argument_group('Arguments for latent space analysis')
    group.add_argument('--pc', type=int, default=2, help='Number of principal component traversals to generate (default: %(default)s)')
    group.add_argument('--pc-ondata', action='store_true', help='Find closest on-data latent point to each PC percentile')
    group.add_argument('--ksample', type=int, default=20, help='Number of kmeans samples to generate (default: %(default)s)')

    group = _parser.add_argument_group('Arguments for volume generation')
    group.add_argument('--downsample', type=int, help='Downsample volumes to this box size (pixels)')
    group.add_argument('--lowpass', type=float, default=None, help='Lowpass filter to this resolution in Å')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volumes')
    group.add_argument('--invert', action='store_true', help='Invert contrast of output volumes')

    return _parser


"""
plt.scatter
sns.distplot

gen pc volumes

gen kmeans volumes

sns.jointplot (scatter PC1 vs PC2)
sns.jointplot (hex PC1 vs PC2)
analysis.scatter_annotate (PC1 vs PC2 with kmeans centers)
analysis.scatter_annotate_hex (PC1 vs PC2 with kmeans centers)

if zdim > 2 and not skip_umap:
    sns.jointplot (scatter UMAP1 vs UMAP2)
    sns.jointplot (hex UMAP1 vs UMAP2)
    analysis.scatter_annotate (UMAP1 vs UMAP2 with kmeans centers)
    analysis.scatter_annotate_hex (UMAP1 vs UMAP2 with kmeans centers)
"""


def analyze_z_onedimensional(z: np.ndarray,
                             outdir: str,
                             vg: analysis.VolumeGenerator,
                             skip_vol: bool = False,
                             ondata: bool = False) -> None:
    """
    Plotting and volume generation for 1D z
    :param z: array of 1-D latent embeddings, shape (nptcls, 1)
    :param outdir: directory in which to save all outputs (plots and generated volumes)
    :param vg: VolumeGenerator instance to aid volume generation at specficied z values
    :param skip_vol: whether to skip generation of volumes
    :param ondata: whether to use the closest on-data latent point to each z percentile for plotting and volume generation
    :return: None
    """
    assert z.shape[1] == 1
    z = z.reshape(-1)
    nptcls = len(z)

    # scatter plot of particle index against latent embedding
    plt.scatter(np.arange(nptcls), z, alpha=.1, s=2)
    plt.xlabel('particle index')
    plt.ylabel('z')
    plt.tight_layout()
    plt.savefig(f'{outdir}/z.png')
    plt.close()

    # histogram of latent embeddings with KDE overlay
    sns.displot(z, kde=True)
    plt.xlabel('z')
    plt.tight_layout()
    plt.savefig(f'{outdir}/z_hist.png')
    plt.close()

    if not skip_vol:
        # sample z values at 5th, 15th, ..., 95th percentiles of the latent distribution
        ztraj = np.percentile(z, np.linspace(start=5, stop=95, num=10))
        if ondata:
            ztraj = analysis.get_nearest_point(z, ztraj)

        # histogram of latent embeddings with KDE overlay
        sns.displot(z, kde=True)
        for percentile in ztraj:
            plt.axvline(percentile, color='red', linestyle='-')
        plt.xlabel('z')
        plt.tight_layout()
        plt.savefig(f'{outdir}/z_hist_percentile_volumes.png')
        plt.close()

        # generate corresponding volumes
        vg.gen_volumes(z_values=ztraj,
                       outdir=outdir, )


def analyze_z_multidimensional(z: np.ndarray,
                               outdir: str,
                               vg: analysis.VolumeGenerator,
                               starfile_path: str,
                               datadir: str = None,
                               skip_vol: bool = False,
                               skip_umap: bool = False,
                               num_pcs: int = 2,
                               pc_ondata: int = False,
                               num_ksamples: int = 20, ) -> None:
    """
    Plotting and volume generation for multidimensional z
    :param z: array of 1-D latent embeddings, shape (nptcls, zdim)
    :param outdir: directory in which to save all outputs (plots and generated volumes)
    :param vg: VolumeGenerator instance to aid volume generation at specficied z values
    :param starfile_path: path to star file used during model training through which to load images
    :param datadir: path to particle images on disk, used when plotting images per kmeans class
    :param skip_vol: whether to skip generation of volumes
    :param skip_umap: whether to skip latent embeddings UMAP dimensionality reduction
    :param num_pcs: number of principal components along which to generate volumes. If 0, then no PCA is performed
    :param pc_ondata: whether to use the closest on-data latent point to each PCA axis trajectory for plotting and volume generation
    :param num_ksamples: number of latent clusters to form by k-means clustering for plotting and volume generation
    :return: None
    """
    zdim = z.shape[1]

    # Principal component analysis
    log('Perfoming principal component analysis ...')
    pc, pca = analysis.run_pca(z)
    z_trajectories = []
    for i in range(num_pcs):
        os.mkdir(f'{outdir}/pc{i + 1}')

        z_pc_trajectory = np.percentile(pc[:, i], np.linspace(start=5, stop=95, num=10))
        z_trajectory = analysis.get_pc_traj(pca=pca,
                                            dim=i + 1,
                                            sampling_points=z_pc_trajectory)
        if pc_ondata:
            z_trajectory, z_pc_ind = analysis.get_nearest_point(z, z_trajectory)
            np.savetxt(f'{outdir}/pc{i + 1}/z_percentiles_ind.txt', z_pc_ind, fmt='%d')

        z_trajectories.append(z_trajectory)
        np.savetxt(f'{outdir}/pc{i + 1}/z_percentiles.txt', z_trajectory)

        if not skip_vol:
            vg.gen_volumes(z_values=z_trajectory,
                           outdir=f'{outdir}/pc{i + 1}')

    # K-means clustering
    log('Performing K-means clustering ...')
    kmeans_labels, kmeans_centers = analysis.cluster_kmeans(z, num_ksamples)
    kmeans_centers, kmeans_centers_ind = analysis.get_nearest_point(z, kmeans_centers)
    if not os.path.exists(f'{outdir}/kmeans{num_ksamples}'):
        os.mkdir(f'{outdir}/kmeans{num_ksamples}')
    utils.save_pkl(kmeans_labels, f'{outdir}/kmeans{num_ksamples}/labels.pkl')
    np.savetxt(f'{outdir}/kmeans{num_ksamples}/centers.txt', kmeans_centers)
    np.savetxt(f'{outdir}/kmeans{num_ksamples}/centers_ind.txt', kmeans_centers_ind, fmt='%d')
    if not skip_vol:
        vg.gen_volumes(z_values=kmeans_centers,
                       outdir=f'{outdir}/kmeans{num_ksamples}')

    # Make some plots using PCA transformation
    log('Generating PCA plots ...')

    # bar plot PCA explained variance ratio
    plt.bar(np.arange(z.shape[1]) + 1, pca.explained_variance_ratio_)
    plt.xticks(np.arange(z.shape[1]) + 1)
    plt.xlabel('principal components')
    plt.ylabel('explained variance')
    plt.tight_layout()
    plt.savefig(f'{outdir}/z_pca_explainedvariance.png')
    plt.close()

    # scatter plot latent PCA
    g = sns.jointplot(x=pc[:, 0],
                      y=pc[:, 1],
                      alpha=.1,
                      s=2)
    g.set_axis_labels('l-PC1', 'l-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/z_pca_scatter.png')
    plt.close()

    # hexbin plot latent PCA
    g = sns.jointplot(x=pc[:, 0],
                      y=pc[:, 1],
                      kind='hex')
    g.set_axis_labels('l-PC1', 'l-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/z_pca_hexbin.png')
    plt.close()

    # scatter plot latent PCA with kmeans center annotations
    analysis.scatter_annotate(x=pc[:, 0],
                              y=pc[:, 1],
                              centers_ind=kmeans_centers_ind,
                              annotate=True,
                              labels=[f'k{i}' for i in range(num_ksamples)])
    plt.xlabel('l-PC1')
    plt.ylabel('l-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/kmeans{num_ksamples}/z_pca_scatter_annotatekmeans.png')
    plt.close()

    # hexbin plot latent PCA with kmeans center annotations
    g = analysis.scatter_annotate_hex(x=pc[:, 0],
                                      y=pc[:, 1],
                                      centers_ind=kmeans_centers_ind,
                                      annotate=True,
                                      labels=[f'k{i}' for i in range(num_ksamples)])
    g.set_axis_labels('l-PC1', 'l-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/kmeans{num_ksamples}/z_pca_hexbin_annotatekmeans.png')
    plt.close()

    # scatter plot latent PCA with PCA trajectory annotations
    analysis.scatter_annotate(x=pc[:, 0],
                              y=pc[:, 1],
                              centers_xy=np.vstack([pca.transform(z_trajectories[0])[:, :2],  # trajectory along pc 1, trajectory is pc-dimensional so just take first two dims for plotting
                                                    pca.transform(z_trajectories[1])[:, :2]]),  # trajectory along pc 2, trajectory is pc-dimensional so just take first two dims for plotting
                              annotate=True,
                              labels=[f'l-PC1_{i}' for i in range(len(z_trajectories[0]))] + [f'l-PC2_{i}' for i in range(len(z_trajectories[1]))])
    plt.xlabel('l-PC1')
    plt.ylabel('l-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/pc1/z_pca_scatter_annotatepca.png')
    plt.close()

    # hexbin plot latent PCA with PCA trajectory annotations
    g = analysis.scatter_annotate_hex(x=pc[:, 0],
                                      y=pc[:, 1],
                                      centers_xy=np.vstack([pca.transform(z_trajectories[0])[:, :2],  # trajectory along pc 1, trajectory is pc-dimensional so just take first two dims for plotting
                                                            pca.transform(z_trajectories[1])[:, :2]]),  # trajectory along pc 2, trajectory is pc-dimensional so just take first two dims for plotting
                                      annotate=True,
                                      labels=[f'l-PC1_{i}' for i in range(len(z_trajectories[0]))] + [f'l-PC2_{i}' for i in range(len(z_trajectories[1]))])
    g.set_axis_labels('l-PC1', 'l-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/pc1/z_pca_hexbin_annotatepca.png')
    plt.close()

    # scatter plot latent PCA colored by k-means clusters
    analysis.plot_by_cluster(x=pc[:, 0],
                             y=pc[:, 1],
                             labels=kmeans_labels,
                             labels_sel=num_ksamples,
                             centers_ind=kmeans_centers_ind,
                             annotate=True)
    plt.xlabel('l-PC1')
    plt.ylabel('l-PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/kmeans{num_ksamples}/z_pca_scatter_colorkmeanslabel.png')
    plt.close()

    # scatter subplots latent PCA colored by k-means clusters
    analysis.plot_by_cluster_subplot(x=pc[:, 0],
                                     y=pc[:, 1],
                                     labels=kmeans_labels,
                                     labels_sel=num_ksamples)
    plt.xlabel('l-PC1')
    plt.ylabel('l-PC2')
    plt.savefig(f'{outdir}/kmeans{num_ksamples}/z_pca_scatter_subplotkmeanslabel.png')
    plt.close()

    # UMAP dimensionality reduction
    if zdim > 2 and not skip_umap:
        log('Running UMAP ...')
        umap_emb, umap_reducer = analysis.run_umap(z)
        utils.save_pkl(umap_emb, f'{outdir}/umap.pkl')

        log('Generating UMAP plots ...')

        # scatter plot latent UMAP
        g = sns.jointplot(x=umap_emb[:, 0],
                          y=umap_emb[:, 1],
                          alpha=.1,
                          s=2)
        g.set_axis_labels('l-UMAP1', 'l-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/z_umap_scatter.png')
        plt.close()

        # hexbin plot latent UMAP
        g = sns.jointplot(x=umap_emb[:, 0],
                          y=umap_emb[:, 1],
                          kind='hex')
        g.set_axis_labels('l-UMAP1', 'l-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/z_umap_hexbin.png')
        plt.close()

        # scatter plot latent UMAP with kmeans center annotations
        analysis.scatter_annotate(x=umap_emb[:, 0],
                                  y=umap_emb[:, 1],
                                  centers_ind=kmeans_centers_ind,
                                  annotate=True,
                                  labels=[f'k{i}' for i in range(num_ksamples)])
        plt.xlabel('l-UMAP1')
        plt.ylabel('l-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/z_umap_scatter_annotatekmeans.png')
        plt.close()

        # hexbin plot latent UMAP with kmeans center annotations
        g = analysis.scatter_annotate_hex(x=umap_emb[:, 0],
                                          y=umap_emb[:, 1],
                                          centers_ind=kmeans_centers_ind,
                                          annotate=True,
                                          labels=[f'k{i}' for i in range(num_ksamples)])
        g.set_axis_labels('l-UMAP1', 'l-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/z_umap_hexbin_annotatekmeans.png')
        plt.close()

        # scatter plot latent UMAP with PCA trajectory annotations
        analysis.scatter_annotate(x=umap_emb[:, 0],
                                  y=umap_emb[:, 1],
                                  centers_xy=np.vstack([umap_reducer.transform(z_trajectories[0]),  # trajectory in latent space along pc 1, transformed to UMAP space
                                                        umap_reducer.transform(z_trajectories[1])]),  # trajectory in latent space along pc 2, transformed to UMAP space
                                  annotate=True,
                                  labels=[f'l-PC1_{i}' for i in range(len(z_trajectories[0]))] + [f'l-PC2_{i}' for i in range(len(z_trajectories[1]))])
        plt.xlabel('l-UMAP1')
        plt.ylabel('l-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/pc1/z_umap_scatter_annotatepca.png')
        plt.close()

        # hexbin plot latent UMAP with PCA trajectory annotations
        g = analysis.scatter_annotate_hex(x=umap_emb[:, 0],
                                          y=umap_emb[:, 1],
                                          centers_xy=np.vstack([umap_reducer.transform(z_trajectories[0]),  # trajectory in latent space along pc 1, transformed to UMAP space
                                                                umap_reducer.transform(z_trajectories[1])]),  # trajectory in latent space along pc 2, transformed to UMAP space
                                          annotate=True,
                                          labels=[f'l-PC1_{i}' for i in range(len(z_trajectories[0]))] + [f'l-PC2_{i}' for i in range(len(z_trajectories[1]))])
        g.set_axis_labels('l-UMAP1', 'l-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/pc1/z_umap_hexbin_annotatepca.png')
        plt.close()

        # scatter plot latent UMAP colored by k-means clusters
        analysis.plot_by_cluster(x=umap_emb[:, 0],
                                 y=umap_emb[:, 1],
                                 labels=kmeans_labels,
                                 labels_sel=num_ksamples,
                                 centers_ind=kmeans_centers_ind,
                                 annotate=True)
        plt.xlabel('l-UMAP1')
        plt.ylabel('l-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/z_umap_scatter_colorkmeanslabel.png')
        plt.close()

        # scatter subplots latent UMAP colored by k-means clusters
        analysis.plot_by_cluster_subplot(x=umap_emb[:, 0],
                                         y=umap_emb[:, 1],
                                         labels=kmeans_labels,
                                         labels_sel=num_ksamples)
        plt.xlabel('l-UMAP1')
        plt.ylabel('l-UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{num_ksamples}/z_umap_scatter_subplotkmeanslabel.png')
        plt.close()

        for i in range(num_pcs):
            analysis.scatter_color(x=umap_emb[:, 0],
                                   y=umap_emb[:, 1],
                                   c=pc[:, i],
                                   cbar_label=f'l-PC{i + 1}')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.tight_layout()
            plt.savefig(f'{outdir}/pc{i + 1}/z_umap_colorlatentpca.png')
            plt.close()

    # make plots of first 6 images of each kmeans class
    s = starfile.TiltSeriesStarfile(starfile=starfile_path)
    star_df_backup = s.df.copy(deep=True)
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
    if zdim > 2 and not skip_umap:
        ref_array = utils.load_pkl(f'{outdir}/umap.pkl')
        ref_names = ['l-UMAP1', 'l-UMAP2']
    else:
        ref_array = pc
        ref_names = ['l-PC1', 'l-PC2']
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
    # log arguments
    t1 = dt.now()
    log(args)

    # set files to use as inputs for analysis
    config = f'{args.workdir}/config.pkl'
    cfg = utils.load_pkl(config)
    star_path = cfg['starfile_args']['sourcefile_filtered']
    datadir = cfg['dataset_args']['datadir']
    if args.epoch == -1:
        zfile = f'{args.workdir}/z.train.pkl'
        weights = f'{args.workdir}/weights.pkl'
        outdir = f'{args.workdir}/analyze'
    else:
        zfile = f'{args.workdir}/z.{args.epoch}.train.pkl'
        weights = f'{args.workdir}/weights.{args.epoch}.pkl'
        outdir = f'{args.workdir}/analyze.{args.epoch}'

    # override outdir if provided as input argument
    if args.outdir:
        outdir = args.outdir
    log(f'Saving results to {outdir}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    z = utils.load_pkl(zfile)
    zdim = z.shape[1]

    vg = analysis.VolumeGenerator(weights_path=weights,
                                  config_path=config,
                                  downsample=args.downsample,
                                  lowpass=args.lowpass,
                                  flip=args.flip,
                                  invert=args.invert,
                                  cuda=args.device, )

    # plot the loss curve
    loss = analysis.parse_loss(f'{args.workdir}/run.log')
    plt.plot(loss)
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.savefig(f'{outdir}/model_loss.png')
    plt.close()

    if zdim == 1:
        analyze_z_onedimensional(z=z,
                                 outdir=outdir,
                                 vg=vg,
                                 skip_vol=args.skip_vol,
                                 ondata=args.pc_ondata)
    else:
        analyze_z_multidimensional(z=z,
                                   outdir=outdir,
                                   skip_vol=args.skip_vol,
                                   vg=vg,
                                   num_pcs=args.pc,
                                   pc_ondata=args.pc_ondata,
                                   skip_umap=args.skip_umap,
                                   num_ksamples=args.ksample,
                                   starfile_path=star_path,
                                   datadir=datadir)

    # copy over template if file doesn't exist
    out_ipynb = f'{outdir}/tomoDRGN_viz+filt.ipynb'
    if not os.path.exists(out_ipynb):
        log(f'Creating jupyter notebook...')
        ipynb = str(files('tomodrgn.templates').joinpath('tomoDRGN_viz+filt_template.ipynb'))
        assert os.path.isfile(ipynb)
        shutil.copyfile(ipynb, out_ipynb)
        log(out_ipynb)
    else:
        log(f'{out_ipynb} already exists. Skipping')

    log(f'Finished in {dt.now() - t1}')


if __name__ == '__main__':
    matplotlib.use('Agg')  # non-interactive backend
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
