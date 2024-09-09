"""
Functions for analysis of particle metadata: index, pose, ctf, latent embedding, label, tomogram spatial context, etc.
"""
import glob
import os
from datetime import datetime as dt
import numpy as np
import pandas as pd
import subprocess
from typing import Literal, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.figure
import plotly.callbacks
from matplotlib.colors import Colormap
import matplotlib.gridspec as gridspec
import seaborn as sns
from adjustText import adjust_text
import plotly.graph_objs as go
import plotly.colors as pl_colors
from ipywidgets import interactive, widgets

from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
# noinspection PyPackageRequirements
import umap

from tomodrgn import utils, mrc, starfile

log = utils.log


def parse_loss(run_log: str) -> np.ndarray:
    """
    Parse total loss at each epoch from run.log output.

    :param run_log: Path to run.log output file.
    :return: array of total loss at each epoch.
    """
    with open(run_log, 'r') as f:
        lines = f.readlines()

    # every epoch is reported in log with '=====>' line
    lines = [line for line in lines if '=====>' in line]

    # total loss is 4th-to-last word, and is ended by semicolon
    total_loss = np.asarray([x.split()[-4][:-1] for x in lines], dtype=float)

    return total_loss


def parse_all_losses(run_log: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse MSE, KLD, and total loss at each epoch from run.log output.

    :param run_log: Path to run.log output file.
    :return: tuple of arrays for MSE loss, KLD loss, and total loss, at each epoch.
    """
    with open(run_log) as f:
        lines = f.readlines()
    lines = [x for x in lines if '====>' in x]

    loss_mse = np.asarray([x.split()[10][:-1] for x in lines], dtype=float)
    loss_kld = np.asarray([x.split()[13][:-1] for x in lines], dtype=float)
    loss_total = np.asarray([x.split()[17][:-1] for x in lines], dtype=float)

    return loss_mse, loss_kld, loss_total


###################################
# Latent dimensionality reduction #
###################################


def run_pca(z: np.ndarray,
            verbose: bool = True) -> tuple[np.ndarray, PCA]:
    """
    Run principal component analysis on the latent embeddings.

    :param z: array of latent embeddings, shape (nptcls, zdim)
    :param verbose: if True, log additional information to STDOUT
    :return: tuple of principal-component-transformed latent embeddings and the fit sklearn PCA object
    """
    # run PCA keeping zdim components (i.e., all components)
    pca = PCA(z.shape[1])
    pca.fit(z)
    if verbose:
        log('Explained variance ratio:')
        log(pca.explained_variance_ratio_)
    pc = pca.transform(z)
    return pc, pca


def get_pc_traj(pca: PCA,
                dim: int,
                sampling_points: np.ndarray) -> np.ndarray:
    """
    Sample latent embeddings along specified principal component `dim` at coordininates in PC-space specified by `sampling_points`.
    Note that sampled points are precisely on-pc-axis and are therefore off-data.

    :param pca: pre-fit sklearn PCA object from `analysis.run_pca`
    :param dim: PC dimension for the trajectory (1-based index)
    :param sampling_points: array of points to sample along specified principal component.
    :return: array of latent embedding values along PC, shape (len(sampling_points), zdim).
    """
    # sanity check inputs
    zdim = pca.get_params()['n_components']
    assert dim <= zdim

    # initialize an array of points to inverse transform from PC space to latent space
    traj_pca = np.zeros((len(sampling_points), zdim))
    # only sampling points along `dim` axis, all other points for all other dims are 0
    traj_pca[:, dim - 1] = sampling_points

    # inverse transform the points
    ztraj_pca = pca.inverse_transform(traj_pca)
    return ztraj_pca


def run_tsne(z: np.ndarray,
             n_components: int = 2,
             perplexity: int = 1_000,
             random_state: int | np.random.RandomState | None = None,
             **kwargs: Any) -> np.ndarray:
    """
    Run t-SNE dimensionality reduction on latent embeddings.

    :param z: array of latent embeddings, shape (nptcls, zdim)
    :param n_components: number of dimensions in the embedded t-SNE space, passed to sklearn.manifold.TSNE
    :param perplexity: related to the number of nearest neighbors that is used in other manifold learning algorithms, passed to sklearn.manifold.TSNE
    :param random_state: random state for reproducible runs, passed to sklearn.manifold.TSNE
    :param kwargs: additional key word arguments passed to sklearn.manifold.TSNE
    :return: array of t-SNE embeddings, shape (len(z), n_components)
    """
    # sanity check inputs
    if len(z) > 10_000:
        log(f'WARNING: {len(z)} datapoints > {10_000}. This may take a while...')
    assert perplexity < len(z)

    # perform embedding
    z_embedded = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, **kwargs).fit_transform(z)
    return z_embedded


def run_umap(z: np.ndarray,
             random_state: int | np.random.RandomState | None = None,
             **kwargs: Any) -> tuple[np.ndarray, umap.UMAP]:
    """
    Run UMAP dimensionality reduction on latent embeddings.

    :param z: array of latent embeddings, shape (nptcls, zdim)
    :param random_state: random state for reproducible runs, passed to umap.UMAP
    :param kwargs: additional key word arguments passed to umap.UMAP
    :return: array of UMAP embeddings, shape (len(z), 2), and fit reducer object
    """
    # perform embedding
    reducer = umap.UMAP(random_state=random_state, **kwargs)
    z_embedded = reducer.fit_transform(z)
    return z_embedded, reducer


#####################
# Latent clustering #
#####################


def cluster_kmeans(z: np.ndarray,
                   n_clusters: int,
                   random_state: int | np.random.RandomState | None = None,
                   on_data: bool = True,
                   reorder: bool = True,
                   **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster latent embeddings using k-means clustering.
    If reorder=True, reorders clusters according to agglomerative clustering of cluster centers

    :param z: array of latent embeddings, shape (nptcls, zdim)
    :param n_clusters: number of clusters to form, passed to sklearn.cluster.KMeans
    :param random_state: random state for reproducible runs, passed to sklearn.cluster.KMeans
    :param on_data: adjust cluster centers to nearest point on the data manifold `z`
    :param reorder: reorder clusters according to agglomerative clustering of cluster centers
    :param kwargs: additional key word arguments passed to sklearn.cluster.KMeans
    :return: array of cluster labels shape (len(z)), array of cluster centers shape (n_clusters, zdim)
    """
    # perform clustering
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state,
                    init='k-means++',
                    n_init='auto',
                    **kwargs)
    labels = kmeans.fit_predict(z)
    centers = kmeans.cluster_centers_

    # resample cluster centers to nearest on-data points within input latent embeddings
    if on_data:
        centers, _ = get_nearest_point(data=z,
                                       query=centers)

    # reorder clusters by agglomerative clustering distance
    if reorder:
        # clustermap generation
        g = sns.clustermap(centers)

        # reindexing centers and labels
        reordered = g.dendrogram_row.reordered_ind
        centers = centers[reordered]
        reorder_label_mapping = {old_label: new_label for new_label, old_label in enumerate(reordered)}
        labels = np.array([reorder_label_mapping[old_label] for old_label in labels])

        plt.close()

    return labels, centers


def cluster_gmm(z: np.ndarray,
                n_components: int,
                random_state: int | np.random.RandomState | None = None,
                on_data: bool = True,
                **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster latent embeddings using a K-component full covariance Gaussian mixture model.

    :param z: array of latent embeddings, shape (nptcls, zdim)
    :param n_components: number of components to use in GMM, passed to sklearn.mixture.GaussianMixture
    :param random_state: random state for reproducible runs, passed to sklearn.cluster.KMeans
    :param on_data: adjust cluster centers to nearest point on the data manifold `z`
    :param kwargs: additional key word arguments passed to sklearn.mixture.GaussianMixture
    :return: array of cluster labels shape (len(z)), array of cluster centers shape (n_clusters, zdim)
    """
    # perform clustering
    clf = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=random_state,
                          **kwargs)
    labels = clf.fit_predict(z)
    centers = clf.means_

    # resample cluster centers to nearest on-data points within input latent embeddings
    if on_data:
        centers, _ = get_nearest_point(data=z,
                                       query=centers)
    return labels, centers


def get_nearest_point(data: np.ndarray,
                      query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the closest point in `data` to `query`.

    :param data: reference array to locate "on-data" points within, shape (n_ref_points, d_dimensions)
    :param query: query array of "off-data" points, shape (n_query_points, d_dimensions)
    :return: array of the nearest point within `data` to each point in `query` shape (len(query), d_dimensions),
            array of the corresponding indices within the data array shape (len(query))
    """
    # calculate pairwise distance matrix, shape (n_query_points, n_ref_points)
    dists = cdist(XA=query, XB=data)

    # find the indices of the closest point within data to each query point by minimizing over the reference data axis
    ind = dists.argmin(axis=1)
    return data[ind], ind


#################################################
# Helper functions for index array manipulation #
#################################################


def convert_original_indices(ind_sel: np.ndarray,
                             n_original: int,
                             ind_original_sel: np.ndarray | None = None) -> np.ndarray:
    """
    Convert selected indices relative to a filtered particle stack into indices relative to the unfiltered particle stack.
    Often useful when selecting a subset of indices when analyzing a model itself trained using `--ind` to filter the referenced particle stack.

    :param ind_sel: indices to keep from filtered particle stack
    :param n_original: the number of particles in the original unfiltered particle stack
    :param ind_original_sel: indices used to generate the filtered particle stack from the unfiltered particle stack
    :return: array of ind_sel indices re-indexed to be relative to the unfiltered particle stack
    """
    # sanity check inputs
    assert np.min(ind_sel) >= 0
    assert np.max(ind_sel) < n_original, f'A selected index is out of range given the original unfiltered number of particles: {n_original}'
    if ind_original_sel:
        assert np.min(ind_original_sel) >= 0
        assert np.max(ind_original_sel) < n_original, f'An index is out of range given the original unfiltered number of particles: {n_original}'
        assert np.max(ind_sel) <= np.max(ind_original_sel), f'A selected index is out of range given the originally selected indices'

    # calculate reindexed indices
    unfiltered_indices = np.arange(n_original)
    filtered_indices = unfiltered_indices[ind_original_sel]
    sel_filtered_indices_reindexed = filtered_indices[ind_sel]

    return sel_filtered_indices_reindexed


def combine_ind(selections_list: list[np.ndarray],
                n_ind_total: int,
                kind: Literal['intersection', 'union'] = 'union') -> tuple[np.ndarray, np.ndarray]:
    """
    Combine multiple indices selections by either intersection or union.

    :param selections_list: list of arrays of indices to combine
    :param n_ind_total: number of total indices (both selected and not selected)
    :param kind: type of combination selection to perform, one of 'intersection', 'union'
    :return: array of combined and sorted selected indices, array of sorted unselected indices
    """
    # combine indices
    tmp = set(selections_list[0])
    if kind == 'intersection':
        ind_selected = tmp.intersection(*selections_list)
    elif kind == 'union':
        ind_selected = tmp.union(*selections_list)
    else:
        raise ValueError(f'Ind combination kind not recognized: {kind=}')

    # sort and convert to numpy arrays
    ind_selected_not = np.array(sorted(set(np.arange(n_ind_total)) - ind_selected))
    ind_selected = np.array(sorted(ind_selected))

    return ind_selected, ind_selected_not


def get_ind_for_cluster(labels: np.ndarray,
                        selected_clusters: list[int]) -> np.ndarray:
    """
    Get the indices of particles belonging to the selected clusters.

    :param labels: array of cluster labels for each particle, shape (nptcls)
    :param selected_clusters: list of selected cluster labels
    :return: array of indices of particles belonging to the selected clusters
    """
    # sanity check inputs
    assert all([selected_cluster in labels for selected_cluster in selected_clusters])

    # get selected indices
    ind_selected = np.array([ind for ind, label in enumerate(labels) if label in selected_clusters])
    return ind_selected


############
# Plotting #
############


def get_colors_matplotlib(num_colors: int,
                          cmap: Colormap | str | None = None) -> list[tuple[float, float, float, float]]:
    """
    Sample num_colors colors from the specified color map as RGBA tuples

    :param num_colors: the number of colors to sample from the color map.
            If using a qualitative colormap such as `tab10`, the first `num_colors` are sampled sequentially.
            Otherwise, colors are sampled uniformly and sequentially from one pass over the entire colormap.
    :param cmap: the matplotlib colormap from which to sample colors. See: https://matplotlib.org/stable/users/explain/colors/colormaps.html#classes-of-colormaps
    :return: list of RGBA tuples for each color sampled from the color map.
    """
    # get the colormap object
    cm = plt.get_cmap(cmap)

    # if using a qualitative colormap, num_colors cannot exceed the cycle length without repeating colors
    qualitative_cycle_lenths = {'Pastel1': 9,
                                'Pastel2': 8,
                                'Paired': 12,
                                'Accent': 8,
                                'Dark2': 8,
                                'Set1': 9,
                                'Set2': 8,
                                'Set3': 12,
                                'tab10': 10,
                                'tab20': 20,
                                'tab20b': 20,
                                'tab20c': 20}
    if cmap in qualitative_cycle_lenths.keys():
        cycle_length = qualitative_cycle_lenths[cmap]
        assert num_colors <= cycle_length, f'Too many colors requested ({num_colors}) for qualitative colormap {cmap} with {cycle_length} colors available'
        colors = [cm(i / cycle_length) for i in range(num_colors)]
    else:
        colors = [cm(i / num_colors) for i in range(num_colors)]

    return colors


def get_colors_chimerax(num_colors: int) -> list[tuple[float, float, float, float]]:
    """
    Sample num_colors from the ChimeraX color scheme as RGBA tuples normalized [0,1].

    :param num_colors: the number of colors to sample from the color map.
            The first num_colors colors are sampled sequentially.
            If more colors are requested than exist in the ChimeraX color scheme, colors are repeated in order.
    :return: list of RGBA tuples for each color sampled from the color map.
    """
    chimerax_colors = np.divide(((192, 192, 192, 255),
                                 (255, 255, 178, 255),
                                 (178, 255, 255, 255),
                                 (178, 178, 255, 255),
                                 (255, 178, 255, 255),
                                 (255, 178, 178, 255),
                                 (178, 255, 178, 255),
                                 (229, 191, 153, 255),
                                 (153, 191, 229, 255),
                                 (204, 204, 153, 255)), 255)

    colors = [chimerax_colors[i % len(chimerax_colors)] for i in range(num_colors)]
    return colors


def scatter_annotate(x: np.ndarray,
                     y: np.ndarray,
                     centers_xy: np.ndarray | None = None,
                     centers_ind: np.ndarray | None = None,
                     annotate: bool = False,
                     labels: list | None = None,
                     alpha: float = 0.1,
                     s: int = 1,
                     **kwargs: Any) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Create a scatter plot with optional annotations for each cluster center and corresponding label.

    :param x: array of x coordinates to plot
    :param y: array of y coordinates to plot
    :param centers_xy: optionally an array of x,y coordinates of cluster centers to superimpose, shape (nclusters, 2). Mutually exclusive with specifying `centers_ind`.
    :param centers_ind: optionally an array indices indexing `x` and `y` as the cluster centers to superimpose, shape (nclusters). Mutually exclusive with specifying `centers_xy`.
    :param annotate: whether to annotate the plot with text labels corresponding to each cluster center
    :param labels: list of text labels for each cluster center
    :param alpha: transparency of scatter points, passed to matplotlib.pyplot.scatter
    :param s: size of scatter points, passed to matplotlib.pyplot.scatter
    :param kwargs: additional keyword arguments passed to matplotlib.pyplot.scatter
    :return: matplotlib figure and axis
    """
    # create the base plot
    fig, ax = plt.subplots()
    ax.scatter(x=x,
               y=y,
               alpha=alpha,
               s=s,
               rasterized=True,
               **kwargs)

    # plot cluster centers either by directly passed centers_xy, or by indexing into all data xy
    if centers_ind is not None:
        assert centers_xy is None
        centers_xy = np.array([[x[center_ind], y[center_ind]] for center_ind in centers_ind])
    if centers_xy is not None:
        plt.scatter(centers_xy[:, 0], centers_xy[:, 1], c='k')

    # add annotations for cluter center labels
    if annotate:
        assert centers_xy is not None
        if labels is None:
            labels = range(len(centers_xy))

        # use the adjustTexts library to tweak label placement for improved legibility
        texts = [plt.text(x=float(centers_xy[i, 0]),
                          y=float(centers_xy[i, 1]),
                          s=str(label),
                          ha='center',
                          va='center')
                 for i, label in enumerate(labels)]
        adjust_text(texts=texts,
                    expand=(1.3, 1.3),
                    arrowprops=dict(arrowstyle='->', color='red'))

    return fig, ax


def scatter_annotate_hex(x: np.ndarray,
                         y: np.ndarray,
                         centers_xy: np.ndarray | None = None,
                         centers_ind: np.ndarray | None = None,
                         annotate: bool = False,
                         labels: list | None = None,
                         **kwargs: Any) -> sns.FacetGrid:
    """
    Create a hexbin plot with optional annotations for each cluster center and corresponding label.

    :param x: array of x coordinates to plot
    :param y: array of y coordinates to plot
    :param centers_xy: optionally an array of x,y coordinates of cluster centers to superimpose, shape (nclusters, 2). Mutually exclusive with specifying `centers_ind`.
    :param centers_ind: optionally an array indices indexing `x` and `y` as the cluster centers to superimpose, shape (nclusters). Mutually exclusive with specifying `centers_xy`.
    :param annotate: whether to annotate the plot with text labels corresponding to each cluster center
    :param labels: list of text labels for each cluster center
    :param kwargs: additional keyword arguments passed to seaborn.jointplot
    :return: matplotlib figure
    """
    # create the base plot
    g = sns.jointplot(x=x,
                      y=y,
                      kind='hex',
                      **kwargs)

    # plot cluster centers either by directly passed centers_xy, or by indexing into all data xy
    if centers_ind is not None:
        assert centers_xy is None
        centers_xy = np.array([[x[center_ind], y[center_ind]] for center_ind in centers_ind])
    if centers_xy is not None:
        g.ax_joint.scatter(centers_xy[:, 0], centers_xy[:, 1], color='k', edgecolor='grey')

    # add annotations for cluter center labels
    if annotate:
        assert centers_xy is not None
        if labels is None:
            labels = range(len(centers_xy))

        # use the adjustTexts library to tweak label placement for improved legibility
        texts = [plt.text(x=float(centers_xy[i, 0]),
                          y=float(centers_xy[i, 1]),
                          s=str(label),
                          ha='center',
                          va='center')
                 for i, label in enumerate(labels)]
        adjust_text(texts=texts,
                    expand=(1.3, 1.3),
                    arrowprops=dict(arrowstyle='->', color='red'))

    return g


def scatter_color(x: np.ndarray,
                  y: np.ndarray,
                  c: np.ndarray,
                  cmap: str = 'viridis',
                  s: int = 1,
                  alpha: float = 0.1,
                  cbar_label: str | None = None,
                  **kwargs: Any) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Create a scatter plot colored by auto-mapped values of `c` according to specified `cmap`, and plot a corresponding colorbar.

    :param x: array of x coordinates to plot
    :param y: array of y coordinates to plot
    :param c: array of values by which to map color of each xy point
    :param cmap: the matplotlib colormap from which to sample colors. See: https://matplotlib.org/stable/users/explain/colors/colormaps.html#classes-of-colormaps
    :param s: size of scatter points, passed to matplotlib.pyplot.scatter
    :param alpha: transparency of scatter points, passed to matplotlib.pyplot.scatter
    :param cbar_label: optional text to label the colorbar
    :param kwargs: additional keyword arguments passed to matplotlib.pyplot.scatter
    :return: matplotlib figure and axis
    """
    # sanity check inputs
    assert len(x) == len(y) == len(c)

    # draw base plot
    fig, ax = plt.subplots()
    sc = ax.scatter(x=x,
                    y=y,
                    s=s,
                    alpha=alpha,
                    rasterized=True,
                    cmap=cmap,
                    c=c,
                    **kwargs)

    # add the colorbar
    cbar = plt.colorbar(sc)
    cbar.set_alpha(1)
    fig.draw_without_rendering()
    if cbar_label:
        cbar.set_label(cbar_label)

    return fig, ax


def plot_by_cluster(x: np.ndarray,
                    y: np.ndarray,
                    labels: np.ndarray,
                    labels_sel: int | np.ndarray[int],
                    centers_xy: np.ndarray | None = None,
                    centers_ind: np.ndarray | None = None,
                    annotate: bool = False,
                    s: int = 2,
                    alpha: float = 0.1,
                    cmap=None,
                    **kwargs) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Plot all points `x,y` with colors per class `labels`, with optional annotations for each cluster center and corresponding label.

    :param x: array of x coordinates to plot, shape (nptcls)
    :param y: array of y coordinates to plot, shape (nptcls)
    :param labels: array of cluster labels for each particle, shape (nptcls)
    :param labels_sel: selected cluster labels to plot.
            If int, plot all classes matching label in `range(labels_sel)`.
            If array of int, plot all classes matching label in `labels_sel`.
    :param centers_xy: optionally an array of x,y coordinates of cluster centers to superimpose, shape (n_labels_sel, 2). Mutually exclusive with specifying `centers_ind`.
    :param centers_ind: optionally an array indices indexing `x` and `y` as the cluster centers to superimpose, shape (n_labels_sel). Mutually exclusive with specifying `centers_xy`.
    :param annotate: whether to annotate the plot with text labels corresponding to each cluster center
    :param s: size of scatter points, passed to matplotlib.pyplot.scatter
    :param alpha: transparency of scatter points, passed to matplotlib.pyplot.scatter
    :param cmap: the matplotlib colormap from which to sample colors. See: https://matplotlib.org/stable/users/explain/colors/colormaps.html#classes-of-colormaps
    :param kwargs: additional keyword arguments passed to matplotlib.pyplot.scatter
    :return: matplotlib figure and axis
    """
    # create the base plot
    fig, ax = plt.subplots()

    # get colors for clusters by sampling cmap
    if type(labels_sel) is int:
        labels_sel = list(range(labels_sel))
    colors = get_colors_matplotlib(num_colors=len(labels_sel),
                                   cmap=cmap)

    # scatter by cluster
    for i, label_sel in enumerate(labels_sel):
        label_inds = labels == label_sel
        ax.scatter(x=x[label_inds],
                   y=y[label_inds],
                   s=s,
                   alpha=alpha,
                   label=f'k{label_sel}',
                   color=colors[i],
                   rasterized=True,
                   **kwargs)

    # plot cluster centers
    if centers_ind is not None:
        assert centers_xy is None
        centers_xy = np.array([[x[center_ind], y[center_ind]] for center_ind in centers_ind])
    if centers_xy is not None:
        ax.scatter(x=centers_xy[:, 0], y=centers_xy[:, 1], c='k')

    # add annotations for cluter center labels
    if annotate:
        assert centers_xy is not None

        # use the adjustTexts library to tweak label placement for improved legibility
        texts = [plt.text(x=float(centers_xy[i, 0]),
                          y=float(centers_xy[i, 1]),
                          s=f'cluster {label_sel}',
                          ha='center',
                          va='center')
                 for i, label_sel in enumerate(labels_sel)]
        adjust_text(texts=texts,
                    expand=(1.3, 1.3),
                    arrowprops=dict(arrowstyle='->', color='red'))

    return fig, ax


def plot_by_cluster_subplot(x: np.ndarray,
                            y: np.ndarray,
                            labels_sel: int | np.ndarray[int],
                            labels: np.ndarray,
                            s: int = 2,
                            alpha: float = 0.1,
                            cmap: str | None = None,
                            **kwargs) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Plot all points `x,y` with colors per class `labels` on individual subplots for each of `labels_sel`.

    :param x: array of x coordinates to plot, shape (nptcls)
    :param y: array of y coordinates to plot, shape (nptcls)
    :param labels: array of cluster labels for each particle, shape (nptcls)
    :param labels_sel: selected cluster labels to plot.
            If int, plot all classes matching label in `range(labels_sel)`.
            If array of int, plot all classes matching label in `labels_sel`.
    :param s: size of scatter points, passed to matplotlib.pyplot.scatter
    :param alpha: transparency of scatter points, passed to matplotlib.pyplot.scatter
    :param cmap: the matplotlib colormap from which to sample colors. See: https://matplotlib.org/stable/users/explain/colors/colormaps.html#classes-of-colormaps
    :param kwargs: additional keyword arguments passed to matplotlib.pyplot.scatter
    :return: matplotlib figure and axes
    """
    # get colors for clusters by sampling cmap
    if type(labels_sel) is int:
        labels_sel = list(range(labels_sel))
    colors = get_colors_matplotlib(len(labels_sel), cmap)

    # create base plot
    ncol = int(np.ceil(len(labels_sel) ** .5))
    nrow = int(np.ceil(len(labels_sel) / ncol))
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, figsize=(2 * ncol, 2 * nrow))

    # draw subplots
    for (i, ax) in enumerate(axes.ravel()):
        if i >= len(labels_sel):
            ax.axis('off')
        else:
            label_inds = labels == labels_sel[i]
            ax.scatter(x[label_inds],
                       y[label_inds],
                       s=s,
                       alpha=alpha,
                       rasterized=True,
                       color=colors[i],
                       **kwargs)
            ax.set_title(labels_sel[i])

    return fig, axes


def plot_euler(theta: np.ndarray,
               phi: np.ndarray,
               psi: np.ndarray) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure, plt.Axes]:
    """
    Plot the distribution of Euler angles as a hexbin of `theta` and `phi`, and a histogram of `psi`.

    :param theta: array of euler angles `theta`, shape (nimgs)
    :param phi: array of euler angles `phi`, shape (nimgs)
    :param psi: array of euler angles `psi`, shape (nimgs)
    :return: matplotlib hexbin figure of theta vs phi, matplotlib figure and axes of psi histogram
    """

    # create base plot for theta and phi as jointplot
    g = sns.jointplot(x=theta,
                      y=phi,
                      kind='hex',
                      xlim=(-180, 180),
                      ylim=(0, 180))
    g.set_axis_labels("theta", "phi")

    # create second plot for psi as histogram
    fig, ax = plt.subplots()
    ax.hist(psi)
    ax.set_xlabel('psi')

    return g, fig, ax


def plot_translations(trans) -> matplotlib.figure.Figure:
    """
    Plot the distribution of shifts in x-axis vs shifts in y-axis (units: px)

    :param trans: translations in x and y, shape (nimgs, 2)
    :return: matplotlib hexbin figure of tx vs ty, axes of hexbin
    """
    # set near-zero translations to 0.0 to allow sns.jointplot to work
    trans[np.isclose(trans, 0.0, atol=1e-4)] = 0.0
    g = sns.jointplot(x=trans[:, 0],
                      y=trans[:, 1],
                      kind='hex')
    g.set_axis_labels('tx (px)', 'ty (px)')

    return g


def plot_losses(runlog: str) -> tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Plot the total loss, reconstruction loss, and KLD divergence per epoch.

    :param runlog: the run.log auto-generated by tomodrgn train_vae
    :return: matplotlib figure and axes of line plots for each loss
    """
    losses = parse_all_losses(runlog)
    labels = ['reconstruction loss', 'latent loss', 'total loss']

    n_rows = 1
    n_cols = 3
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * n_cols, 3 * n_rows), sharex='all')

    for (i, ax) in enumerate(axes.ravel()):
        ax.plot(losses[i])
        ax.set_ylabel(labels[i])
        if i == 1:
            ax.set_xlabel('epoch')

    plt.tight_layout()

    return fig, axes


def plot_projections(images: np.ndarray,
                     labels: list[str],
                     width_between_imgs_px: int = 20,
                     height_between_imgs_px: int = 40) -> None:
    """
    Plot a stack of grayscale images.
    The rendered figure has precisely the correct number of pixels for each image, avoiding the default resampling behavior of `plt.imshow`.

    :param images: array of images to plot, shape (nimgs, boxsize, boxsize)
    :param labels: list of text labels to annotate the title of each image, len(nimgs)
    :param width_between_imgs_px:
    :param height_between_imgs_px:
    :return: matplotlib figure and axes
    """
    # derive dimensions of each image
    width_one_img_px = images.shape[-1]
    height_one_img_px = images.shape[-2]

    # derive number of rows and columns
    ncols = int(np.ceil(len(images) ** .5))
    nrows = int(np.ceil(len(images) / ncols))

    # derive overall figure dimensions
    figwidth_px = (width_one_img_px * ncols) + (width_between_imgs_px * (ncols - 1))
    figheight_px = (height_one_img_px + height_between_imgs_px) * nrows
    figwidth_inches = figwidth_px / plt.rcParams['figure.dpi']
    figheight_inches = figheight_px / plt.rcParams['figure.dpi']

    # create figure and associated gridspec to lay out images
    fig = plt.figure(figsize=(figwidth_inches, figheight_inches))
    gs = gridspec.GridSpec(nrows=nrows,
                           ncols=ncols,
                           figure=fig,
                           top=1,
                           bottom=0,
                           right=1,
                           left=0,
                           wspace=width_between_imgs_px / width_one_img_px,
                           hspace=height_between_imgs_px / (height_one_img_px * nrows))

    # plot images
    for i in range(nrows * ncols):
        if i < len(images):
            ax = fig.add_subplot(gs[i])
            ax.imshow(images[i], interpolation=None, vmin=np.min(images), vmax=np.max(images))
            ax.set_aspect(aspect='equal', anchor='S')
            ax.axis('off')
            ax.set_title(labels[i])
        else:
            ax = fig.add_subplot(gs[i])
            ax.axis('off')


def plot_label_count_distribution(ptcl_star: starfile.TiltSeriesStarfile,
                                  class_labels: np.ndarray) -> None:
    """
    Plot the distribution of class labels per tomogram or micrograph as a heatmap.

    :param ptcl_star: image series star file describing the same number of particles as class_labels
    :param class_labels: array of class labels, shape (nptcls, 1)
    :return: None
    """
    # get a df with one row corresponding to each particle
    df_first_img = ptcl_star.df.groupby(ptcl_star.header_ptcl_uid, as_index=False, sort=False).first()
    # group particles by source tomogram
    try:
        ind_ptcls_per_tomo = [group.index.to_numpy() for group_name, group in df_first_img.groupby(ptcl_star.header_ptcl_micrograph, sort=False)]
    except KeyError:
        # for some reason, header_ptcl_micrograph not found in df.columns
        # falling back to trying to split header_ptcl_uid on `_` and taking 0th value as stand-in for tomogram ID, following Warp convention of `XXX_YYYYY`
        df_first_img[['_tempTomogramUID', '_tempParticleUID']] = df_first_img[ptcl_star.header_ptcl_uid].str.split('_', expand=True)
        ind_ptcls_per_tomo = [group.index.to_numpy() for group_name, group in df_first_img.groupby('_tempTomogramUID', sort=False)]

    label_distribution = np.zeros((len(ind_ptcls_per_tomo), len(set(class_labels))))
    for i, ind_one_tomo in enumerate(ind_ptcls_per_tomo):
        # don't use np.unique directly on one tomogram in case that tomogram has zero particles in given class
        counts_one_tomo = np.asarray([np.sum(class_labels[ind_ptcls_per_tomo[i]] == label) for label in np.unique(class_labels, return_counts=False)])
        label_distribution[i] = counts_one_tomo

    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(1 * len(ind_ptcls_per_tomo), 1 * len(set(class_labels))))

    distribution_plot = ax.imshow(label_distribution.T)
    fig.colorbar(distribution_plot, ax=ax, label='particle count per class')

    ax.set_xlabel('tomogram UID')
    ax.set_ylabel('class label')


def plot_three_column_correlation(reference_array: np.ndarray,
                                  query_array: np.ndarray,
                                  reference_names: list[str],
                                  query_name: str) -> None:
    """
    Plot two reference vectors (e.g. l-UMAP1 and l-UMAP2) for potential correlation with a third query vector (e.g. CoordinateX, DefocusU, etc).

    Produces a figure with 1 row and 3 columns of subplots:

    #. hexbin of reference vector 1 vs query vector.
    #. hexbin of reference vector 2 vs query vector.
    #. scatter of reference vector 1 vs reference vector 2 colored by query vector.

    :param reference_array: array of reference vector values, shape (nptcls, 2)
    :param query_array: array of query vector values, shape (nptcls, 1)
    :param reference_names: list naming each reference vector
    :param query_name: name of query vector
    :return: None
    """
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5))

    ax0.hexbin(reference_array[:, 0], query_array)
    ax0.set_xlabel(reference_names[0])
    ax0.set_ylabel(query_name)

    ax1.hexbin(reference_array[:, 1], query_array)
    ax1.set_xlabel(reference_names[1])
    ax1.set_ylabel(query_name)

    scatter1 = ax2.scatter(reference_array[:, 0], reference_array[:, 1], c=query_array, cmap='viridis', s=0.5)
    fig.colorbar(scatter1, ax=ax2, label=query_name)
    ax2.set_xlabel(reference_names[0])
    ax2.set_ylabel(reference_names[1])


def ipy_plot_interactive(df: pd.DataFrame) -> widgets.Box:
    """
    Create and display an interactive plotly scatter plot and associated ipywidgets custom widgets, allowing exploration of numeric columns of a pandas dataframe.
    * The scatter plot plots the selected dataframe columns with optional colormapping based on a third dataframe column. Hovertext indicates the hovered particle's row index in the input dataframe.
    * The widget at bottom left allows control of the scatter plot: column selections for x, y, and colormap; which plotly colormap to use; marker size and opacity.
    * A custom selection can be made through a lasso selection tool on the scatter plot. The row index of selected points in the input dataframe is displayed in the table at bottom center.
    * The widget at bottom right allows lasso-selected points to be saved as a numpy array of df row indices stored in a timestamped `pkl` file.

    Sample usage:
        `ipy_plot_interactive(df)`

    :param df: pandas dataframe to interactively plot, colormap, and select points from.
    :return: ipywidgets.Box containing the interactive figure and widgets
    """

    def initialize_scatterplot(_df: pd.DataFrame) -> go.FigureWidget:
        """
        Initialize the plotly FigureWidget containing the scatter plot drawn with OpenGL backend for performance > 10k points.

        :param _df: pandas dataframe to interactively plot
        :return: plotly FigureWidget containing scatter plot (lacking controls for interactive plotting)
        """
        xaxis, yaxis = _df.columns[0], _df.columns[1]
        _fig1 = go.FigureWidget([go.Scattergl(x=_df[xaxis],
                                              y=_df[yaxis],
                                              mode='markers',
                                              text=[f'index {i}' for i in _df.index],
                                              marker=dict(size=2,
                                                          opacity=0.5)
                                              )],
                                layout_margin=dict(t=20, b=20, r=50, l=50))
        _fig1.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, height=400)
        _fig1.layout.dragmode = 'lasso'
        return _fig1

    def initialize_table(_df: pd.DataFrame) -> go.FigureWidget:
        """
        Initialize the plotly FigureWidget containing the Table of scatter point row indices from the input dataframe.

        :param _df: pandas dataframe to draw scatter point row indices from.
        :return: plotly FigureWidget containing table (lacking controls for saving indices)
        """
        _fig2 = go.FigureWidget([go.Table(header=dict(values=['selected indices']),
                                          cells=dict(values=[_df.index]))],
                                layout_margin=dict(t=0, b=20, r=50, l=50))
        _fig2.update_layout(height=250, width=300)
        return _fig2

    def update_scatterplot(xaxis: str,
                           yaxis: str,
                           color_by: str,
                           colorscale: str,
                           size: float,
                           opacity: float) -> None:
        """
        Callback to update the `fig1` scatter plot.

        :param xaxis: the column from the dataframe to plot as x-axis values
        :param yaxis: the column from the dataframe to plot as y-axis values
        :param color_by: the column from the dataframe to use for color mapping
        :param colorscale: a named colormap recognized by plotly to map `color_by` values
        :param size: size of scatter point markers
        :param opacity: opacity of scatter point markers
        :return: None
        """
        scatter = fig1.data[0]
        with fig1.batch_update():
            scatter.x = df[xaxis]
            scatter.y = df[yaxis]
            scatter.marker.colorscale = colorscale
            if colorscale is None:
                scatter.marker.color = None
            else:
                scatter.marker.color = df[color_by] if color_by != 'index' else df.index
            scatter.marker.size = size
            scatter.marker.opacity = opacity
            fig1.layout.xaxis.title = xaxis
            fig1.layout.yaxis.title = yaxis

    def update_table(trace: go.Scatter,
                     points: plotly.callbacks.Points,
                     selector: plotly.callbacks.LassoSelector | plotly.callbacks.BoxSelector) -> None:
        """
        Callback to update the `fig2` table of selected indices based on the current Lasso or Box selection of points in `fig1` scatter plot.

        :param trace: the plotly trace that this callback is called from, mandatory parameter in signature but not explicitly used
        :param points: the selected points associated with the active selection
        :param selector: the selector associated with the trace that this callback is called from, mandatory parameter in signature but not explicitly used
        :return: None
        """
        # this function must take all three parameters for the callback to function, but we do not currently use either trace or selector
        _ = trace
        _ = selector

        with fig2.batch_update():
            if len(points.point_inds) == 0:
                # if no points are selected, interpret this as resetting the selection so populate the table with all indices
                fig2.data[0].cells.values = [df.index]
            else:
                # populate the table with indices of selected points
                fig2.data[0].cells.values = [df.loc[points.point_inds].index]

    def save_indices_pkl_callback(b: widgets.Button) -> None:
        """
        Callback to save the currently selected indices in `fig2` table as a sorted and non-redundant numpy array to a pickle file.

        :param b: the clicked button widget calling this callback
        :return: None
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = b

        # define the output pickle file contents
        inds_selected = np.array(list(set(fig2.data[0].cells.values[0])))

        # define the output pickle file name
        cwd = os.getcwd()
        current_datetime = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
        selection_count = len(inds_selected)
        out_path_inds = f'{cwd}/selected_indices_{current_datetime}_{selection_count}-particles.pkl'

        # save the indices pickle
        utils.save_pkl(data=inds_selected, out_pkl=out_path_inds)
        with log_output:
            print(f'Saved {out_path_inds} \n')

    def clear_output_callback(b: widgets.Button) -> None:
        """
        Callback to clear text output in the Output widget.

        :param b: the clicked button widget calling this callback
        :return: None
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = b

        # clear the text output
        log_output.clear_output()

    # create figure and table
    fig1 = initialize_scatterplot(df)
    fig2 = initialize_table(df)

    # create widget and define interactivity for scatter plot adjustment
    scatter_display_widget = interactive(update_scatterplot,
                                         xaxis=widgets.Dropdown(options=df.select_dtypes('number').columns, value=df.select_dtypes('number').columns[0]),
                                         yaxis=widgets.Dropdown(options=df.select_dtypes('number').columns, value=df.select_dtypes('number').columns[1]),
                                         color_by=widgets.Dropdown(options=df.columns, value=df.columns[0]),
                                         colorscale=widgets.Dropdown(options=[None] + plotly.colors.named_colorscales(), value=None),
                                         size=widgets.FloatSlider(min=0, max=10, value=1, continuous_update=False),
                                         opacity=widgets.FloatSlider(min=0, max=1, value=0.5, continuous_update=False))

    # create widgets and define interactivity for saving indices, logging output, and clearing logged output
    save_inds_button = widgets.Button(description='save selected indices', layout=widgets.Layout(width='auto'))
    clear_output_button = widgets.Button(description='clear log output', layout=widgets.Layout(width='auto'))
    log_output = widgets.Output(layout=widgets.Layout(border='1px solid black', overflow='auto', height='200px'))
    save_inds_button.on_click(save_indices_pkl_callback)
    clear_output_button.on_click(clear_output_callback)

    # define interactivity for updating table with lasso or box selection
    fig1.data[0].on_selection(update_table)

    # arrange layout of figure, table, and widget objects
    controls_widget = widgets.VBox(children=[widgets.HBox([save_inds_button, clear_output_button]),
                                             log_output],
                                   layout=widgets.Layout(width='40%', display='inline-flex'))
    container = widgets.VBox([fig1,
                              widgets.HBox(children=[scatter_display_widget, fig2, controls_widget],
                                           layout=widgets.Layout(width='100%', display='inline-flex'))])

    # print some helpful instructions
    print('Welcome to the interactive dataframe explorer!')
    print('  1. Scatter plot settings are in the bottom left corner of the widget.')
    print('  2. Hover over a point in the scatter plot to see its current (x,y) coordinate and its row index in the input dataframe.')
    print('  3. Left click and drag on the scatter plot to draw an interactive lasso selection.')
    print('  4. Create a lasso selection around no points and then double left click on the plot to reset the selection')
    print('  5. Click "save selected indices" in the bottom right corner of the widget to save an indices.pkl file to the current working directory.')
    print('  6. Hover over the scatter plot and select "Download plot as a png" to save a snapshot of the plot including lasso selection definition.')

    return container


#############################
# Volume generation helpers #
#############################


class VolumeGenerator:
    """
    Convenience class generate volume ensembles repeatedly.
    Intended for use with variable input latent embeddings and output directories.
    """

    def __init__(self,
                 weights_path: str,
                 config_path: str,
                 downsample: int | None = None,
                 lowpass: float | None = None,
                 flip: bool = False,
                 invert: bool = False,
                 cuda: int | None = None):
        """
        Instantiate a `VolumeGenerator` object.

        :param weights_path: path to trained model `weights.*.pkl` from `train_vae.py`
        :param config_path: path to trained model `config.pkl` from `train_vae.py`
        :param downsample: downsample reconstructed volumes to this box size (units: px) by Fourier cropping, None means to skip downsampling
        :param lowpass: lowpass filter reconstructed volumes to this resolution (units: Å), None means to skip lowpass filtering
        :param flip: flip the chirality of the reconstructed volumes
        :param invert: invert the data sign of the reconstructed volumes (light-on-dark vs dark-on-light)
        :param cuda: specify the CUDA device index to use for reconstructing volumes, None means to let the system auto-detect the first available CUDA device and otherwise fall back to cpu
        """
        self.weights_path = weights_path
        self.config_path = config_path
        self.downsample = downsample
        self.lowpass = lowpass
        self.flip = flip
        self.invert = invert
        self.cuda = cuda

    def gen_volumes(self,
                    z_values: np.ndarray,
                    outdir: str) -> None:
        """
        Generate volumes at specified latent embeddings and save to specified output directory.
        Calls `analysis.gen_volumes` which launches a subprocess call to `eval_vol.py`.

        :param z_values: array of latent embeddings at which to generate volumes, shape (nptcls, zdim)
        :param outdir: path to output directory in which to save volumes
        :return: None
        """
        # create output directory
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # save the z values used to generate the volumes in the output directory for future convenience and to allow `eval_vol.py` to read them
        zfile = f'{outdir}/z_values.txt'
        np.savetxt(zfile, z_values)

        # generate the corresponding volumes
        gen_volumes(weights_path=self.weights_path,
                    config_path=self.config_path,
                    z_path=zfile,
                    outdir=outdir,
                    downsample=self.downsample,
                    lowpass=self.lowpass,
                    flip=self.flip,
                    invert=self.invert,
                    cuda=self.cuda)


def gen_volumes(weights_path: str,
                config_path: str,
                z_path: str,
                outdir: str,
                downsample: int | None = None,
                lowpass: float | None = None,
                flip: bool = False,
                invert: bool = False,
                cuda: int | None = None) -> None:
    """
    Generate volumes from a trained model.
    Launches a subprocess call to `eval_vol.py`.

    :param weights_path: path to trained model `weights.*.pkl` from `train_vae.py`
    :param config_path: path to trained model `config.pkl` from `train_vae.py`
    :param z_path: path to a .txt or .pkl file containing latent embeddings to evaluate, shape (nptcls, zdim)
    :param outdir: path to output directory in which to save volumes
    :param downsample: downsample reconstructed volumes to this box size (units: px) by Fourier cropping, None means to skip downsampling
    :param lowpass: lowpass filter reconstructed volumes to this resolution (units: Å), None means to skip lowpass filtering
    :param flip: flip the chirality of the reconstructed volumes
    :param invert: invert the data sign of the reconstructed volumes (light-on-dark vs dark-on-light)
    :param cuda: specify the CUDA device index to use for reconstructing volumes, None means to let the system auto-detect the first available CUDA device and otherwise fall back to cpu
    :return: None
    """
    # construct the eval_vol command to generate volumes
    cmd = f'tomodrgn eval_vol --weights {weights_path} --config {config_path} --zfile {z_path} -o {outdir}'
    if downsample is not None:
        cmd += f' -d {downsample}'
    if lowpass is not None:
        cmd += f' --lowpass {lowpass}'
    if flip:
        cmd += f' --flip'
    if invert:
        cmd += f' --invert'

    # prepend the command with setting CUDA-visible devices
    if cuda is not None:
        cmd = f'CUDA_VISIBLE_DEVICES={cuda} && {cmd}'

    # run the command
    log(f'Generating volumes with command:\n {cmd}')
    subprocess.check_call(cmd, shell=True)


def load_dataframe(*,
                   z: np.ndarray | None = None,
                   pc: np.ndarray | None = None,
                   tsne_emb: np.ndarray | None = None,
                   umap_emb: np.ndarray | None = None,
                   euler: np.ndarray | None = None,
                   trans: np.ndarray | None = None,
                   labels: np.ndarray | None = None,
                   **kwargs) -> pd.DataFrame:
    """
    Merge known types of numpy arrays into a single pandas dataframe for downstream analysis.
    Only supplied key word arguments will be added as columns to the dataframe.

    :param z: array of latent embeddings, shape (nptcls, zdim)
    :param pc: array of PCA-transformed latent embeddings, shape (nptcls, PC-dim)
    :param tsne_emb: array of t-SNE-transformed latent embeddings, shape (nptcls, tSNE-dim)
    :param umap_emb: array of UMAP-transformed latent embeddings, shape (nptcls, UMAP-dim)
    :param euler: array of Euler angles, shape (nptcls, 3)
    :param trans: array of translation vectors, shape (nptcls, 2)
    :param labels: array of labels, shape (nptcls, )
    :param kwargs: key:value pairs of additional column_name: numpy array of values to add as additional columns to the dataframe
    :return:
    """
    # add supplied parameter arrays to a dictionary
    data = {}
    if umap_emb is not None:
        data['l-UMAP1'] = umap_emb[:, 0]
        data['l-UMAP2'] = umap_emb[:, 1]
    if tsne_emb is not None:
        data['l-TSNE1'] = tsne_emb[:, 0]
        data['l-TSNE2'] = tsne_emb[:, 1]
    if pc is not None:
        zdim = pc.shape[1]
        for i in range(zdim):
            data[f'l-PC{i + 1}'] = pc[:, i]
    if labels is not None:
        data['labels'] = labels
    if euler is not None:
        data['theta'] = euler[:, 0]
        data['phi'] = euler[:, 1]
        data['psi'] = euler[:, 2]
    if trans is not None:
        data['tx'] = trans[:, 0]
        data['ty'] = trans[:, 1]
    if z is not None:
        zdim = z.shape[1]
        for i in range(zdim):
            data[f'z{i}'] = z[:, i]
    for kk, vv in kwargs.items():
        data[kk] = vv

    # assert all items have the same length
    array_lengths = {name: len(array) for name, array in data.items()}
    assert len(set(array_lengths.values())) == 1, f'Not all supplied arrays have the same length: {array_lengths}'

    # create a dataframe from the formatted dictionary
    df = pd.DataFrame(data=data)
    df['index'] = df.index

    return df


def recursive_load_dataframe(volumeseries_star_path: str,
                             tomo_id_column: str) -> pd.DataFrame:
    """
    Create merged dataframe containing:

    #. imageseries star file used to train model (referenced in train_vae ``config.pkl``)
    #. volumeseries star file specified here
    #. any ``*.pkl`` file found recursively within this notebook's directory which contains a numpy array with first axis shape matching the number of particles in the imageseries star file.

    Data are added assuming all indexing matches imageseries star file particle order.

    :param volumeseries_star_path: absolute path to volume series star file, must reference the same set of particles referenced by the starfile used for tomodrgn train_vae
    :param tomo_id_column: full string name of column containing unique values per tomogram in volseries star file
    :return: pandas dataframe containing all described data
    """
    # get the epoch being analyzed from cwd (assumed format: analyze.N)
    cwd = os.getcwd()
    epoch = int(os.path.basename(cwd).split('.')[-1])

    # load the config of train_vae assumed to be 1 directory above
    cfg = utils.load_pkl(f'{cwd}/../config.pkl')

    # load the imageseries filtered star file referenced in config from training, keeping 1st image dose-ascending per particle
    ptcls_star = starfile.TiltSeriesStarfile(cfg['starfile_args']['sourcefile_filtered'])
    ptcls_star.filter(sort_ptcl_imgs='dose_ascending', use_first_ntilts=1)

    # create standalone dataframe for all data to be added
    df_merged = ptcls_star.df.copy(deep=True)

    # calculate the number of particles referenced in imageseries star file
    ptcl_img_inds = ptcls_star.get_ptcl_img_indices()
    n_ptcls = len(ptcl_img_inds)

    # load volumeseries star
    vols_star = starfile.GenericStarfile(volumeseries_star_path)

    # filter volumeseries star by imageseries star filtered particle indices
    vols_star_ptcls_block = vols_star.identify_particles_data_block(column_substring=tomo_id_column)
    df_vols = vols_star.blocks[vols_star_ptcls_block].copy(deep=True)
    df_vols['_UnfilteredParticleInds'] = np.arange(len(df_vols))
    df_vols = df_vols.iloc[df_merged['_UnfilteredParticleInds']]  # internally map particle filtering applied at training time

    # filter out irrelevant columns from volumeseries star
    cols_to_keep = []
    for col in df_vols.columns:
        if any(col_substring in col for col_substring in ['Coordinate', 'Angle', tomo_id_column, '_UnfilteredParticleInds']) and 'Defocus' not in col:
            cols_to_keep.append(col)
    cols_to_drop = list(set(df_vols.columns) - set(cols_to_keep))
    df_vols.drop(cols_to_drop, axis=1, inplace=True)

    # merge into df_merged
    df_merged = df_merged.merge(df_vols, how='right', on='_UnfilteredParticleInds', suffixes=('_img', ''))

    # generate recursive list of *.pkl files within cwd
    paths_to_add = glob.glob(f'{cwd}/**/*.pkl', recursive=True)
    paths_to_add.insert(0, f'{os.path.dirname(cwd)}/z.{epoch}.train.pkl')
    known_column_names = dict()
    data_pkl = dict()
    print()  # newline for visual separation of star file loading output from column name specification output
    for path in paths_to_add:

        # load pkl
        f = utils.load_pkl(path)

        # check if numpy array, skip if not
        if type(f) is not np.ndarray:
            continue

        # check if shape[0] matches length of star file, skip if not
        if len(f) != n_ptcls:
            continue

        # check that number of dimensions is at most 2 (not sure how to deal with ndarrays with 3+ axes), skip if so
        if f.ndim > 2:
            continue

        # log that this pkl file is being added to df_merged and log name format of added columns
        column_name = os.path.splitext(os.path.basename(path))[0]
        if f.ndim == 1:
            column_names = [f'{column_name}']
            f = f.reshape(-1, 1)
        else:
            column_names = [f'{column_name}-{i}' for i in range(f.shape[1])]
        log(f'Mapping {path} to dataframe columns {column_names}')

        # sanity check that the proposed column name has not yet been created (which would cause naming conflict)
        for column_name in column_names:
            if column_name in known_column_names.keys():
                log(f'WARNING: found duplicate column name {column_name} which is caused by two files sharing the same base filename: {path} and {known_column_names[column_name]}. '
                    f'Skipping values at {path}.')
            else:
                known_column_names[column_name] = path

        # add new cols to df_merged corresponding to each column along dim[1] of array (e.g. z dimension 0, 1, 2, ...)
        for i, column_name in enumerate(column_names):
            data_pkl[column_name] = f[:, i]

    # merge with df_merged
    df_pkl = pd.DataFrame.from_dict(data=data_pkl)
    df_merged = pd.concat([df_merged, df_pkl], axis=1)

    return df_merged


##################################################
# Interactive 3-D particle and tomogram plotting #
##################################################


def convert_angstroms_to_voxels(df: pd.DataFrame,
                                tomogram_array_shape: tuple[int, int, int],
                                tomo_pixelsize: float,
                                starfile_pixelsize: float = 1) -> pd.DataFrame:
    """
    Rescale dataframe coordinates from angstroms to unitless voxels corresponding to reconstructed tomograms.
    A starfile (loaded as a dataframe) expresses the 3-D coordinate of each particle typically in either Ångstroms or in pixels (with a pixel size set at particle extraction time).
    In subsequent operations we will wish to plot particle locations superimposed on tomogram voxel data.
    This requires rescaling the particle coordinates from units of Ångstroms or extraction-sized-pixels to tomogram-sized-voxels.

    :param df: dataframe containing particle coordinates to rescale
    :param tomogram_array_shape: shape of reconstructed tomogram in voxels
    :param tomo_pixelsize: pixel size of reconstructed tomogram in Å/px
    :param starfile_pixelsize: pixel size of particles at extraction time when writing the star file from which `df` was loaded.
            Note: star files produced from particle extraction in M always have pixel size 1 Å/px.
    :return: dataframe with rescaled coordinates aligned to the tomogram voxel array
    """
    # unpack the shape of the tomogram 3-D volume array
    tomo_px_x, tomo_px_y, tomo_px_z = tomogram_array_shape

    # rlnCoordinate from starfile is measured in px, so needs to be rescaled to dimensionless tomo voxels
    df.loc[:, '_rlnCoordinateX'] = df['_rlnCoordinateX'].astype('float') * starfile_pixelsize / tomo_pixelsize
    df.loc[:, '_rlnCoordinateY'] = df['_rlnCoordinateY'].astype('float') * starfile_pixelsize / tomo_pixelsize
    df.loc[:, '_rlnCoordinateZ'] = df['_rlnCoordinateZ'].astype('float') * starfile_pixelsize / tomo_pixelsize

    # sanity check that all particle coordinates now fall within the tomogram voxel limits
    assert 0 <= df['_rlnCoordinateX'].all() <= tomo_px_x
    assert 0 <= df['_rlnCoordinateY'].all() <= tomo_px_y
    assert 0 <= df['_rlnCoordinateZ'].all() <= tomo_px_z

    return df


def ipy_tomo_ptcl_viewer(path_to_tomograms: str,
                         tomo_star_mapping: dict[str, str],
                         tomo_id_column: str,
                         df_particles: pd.DataFrame) -> widgets.Box:
    """
    An interactive tomogram and particle viewer using plotly and ipywidgets.
    Allows correlation of tomogram image data with particle locations with particle metadata attributes expressed in ``df_particles``.

    :param path_to_tomograms: path to directory containing tomogram volumes
    :param tomo_star_mapping: dictionary mapping tomogram filenames to the corresponding string identifying each tomogram in the volumeseries star file `tomo_id_column` column
    :param tomo_id_column: name of the column within the volumeseries star file that contains unique values for each tomogram
    :param df_particles: pandas dataframe containing particle information to plot and colormap particles.
            Minimum required columns are [coordinate X, Y, and Z, tomo_id_column, '_UnfilteredParticleInds']
    :return: ipywidgets.Box containing the interactive tomogram and particle viewer widget
    """

    class TomoDataContainer:
        def __init__(self):
            self.tomogram = None
            self.tomo_x_vx = 0
            self.tomo_x_ang = 0
            self.tomo_y_vx = 0
            self.tomo_y_ang = 0
            self.tomo_z_vx = 0
            self.tomo_z_ang = 0
            self.df_particles_sub = None
            self.scatter_x_ang = 0
            self.scatter_y_ang = 0
            self.scatter_z_ang = 0

    def tomo_load_callback(b):
        """
        Load the selected tomogram into memory and render it as image slices along the z axis.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = b

        selected_tomogram = tomo_selection_widget.value
        selected_tomogram = os.path.join(path_to_tomograms, selected_tomogram)
        if selected_tomogram is None:
            with output:
                print('Use the "Tomogram:" dropdown to select the tomogram to load')
            return
        else:
            with output:
                print(f'Loading tomogram slices from {selected_tomogram} ...')
        tomogram, tomogram_header = mrc.parse_mrc(selected_tomogram)
        tomo_angpix = tomogram_header.get_apix()
        with output:
            print(f'Using tomogram shape: {tomogram.shape}')
            print(f'Using tomogram voxel size: {tomo_angpix}')

        # prepare bounds / meshes for tomogram rendering
        tomo_z_vx, tomo_y_vx, tomo_x_vx = tomogram.shape
        # tomo_mesh_z_vx, tomo_mesh_y_vx, tomo_mesh_x_vx = np.mgrid[0:tomo_x_vx, 0:tomo_y_vx, 0:tomo_z_vx]
        tomodatacontainer.tomogram = tomogram
        tomodatacontainer.tomo_x_vx = tomo_x_vx
        tomodatacontainer.tomo_x_ang = tomo_x_vx * tomo_angpix
        tomodatacontainer.tomo_y_vx = tomo_x_vx
        tomodatacontainer.tomo_y_ang = tomo_y_vx * tomo_angpix
        tomodatacontainer.tomo_z_vx = tomo_x_vx
        tomodatacontainer.tomo_z_ang = tomo_z_vx * tomo_angpix

        # graphical data
        # tomo_volume = go.Volume(
        #     name='tomo_volume',
        #     x=tomo_mesh_x_vx.flatten(),
        #     y=tomo_mesh_y_vx.flatten(),
        #     z=tomo_mesh_z_vx.flatten(),
        #     value=tomogram.flatten(),
        #     isomin=np.percentile(tomogram, 90),
        #     isomax=np.percentile(tomogram, 99),
        #     opacity=0.5,
        #     surface=dict(count=5),  # needs to be a large number for good volume rendering
        #     showscale=False,  # no colorbar
        #     # flatshading=False,
        #     # caps= dict(x_show=False, y_show=False, z_show=False),  # no caps
        #     visible=False,
        # )

        tomo_slice = go.Surface(
            name='tomo_slice',
            x=np.linspace(0, tomodatacontainer.tomo_x_ang, tomodatacontainer.tomo_x_vx),
            y=np.linspace(0, tomodatacontainer.tomo_y_ang, tomodatacontainer.tomo_y_vx),
            z=np.ones((tomo_y_vx, tomo_x_vx)) * (tomodatacontainer.tomo_z_ang / 2),
            surfacecolor=tomogram[tomo_z_vx // 2],
            colorscale='gray',
            opacity=1,
            showscale=False,  # no colorbar
        )

        with fig.batch_update():
            fig.add_traces([
                # tomo_volume,
                tomo_slice,
            ])
            fig.update_layout(scene=dict(
                xaxis=dict(range=[0, max(tomodatacontainer.tomo_x_ang, tomodatacontainer.scatter_x_ang)], ),
                yaxis=dict(range=[0, max(tomodatacontainer.tomo_y_ang, tomodatacontainer.scatter_y_ang)], ),
                zaxis=dict(range=[0, max(tomodatacontainer.tomo_z_ang, tomodatacontainer.scatter_z_ang)], ),
                aspectratio=dict(x=1, y=1, z=1),
            ))

        # tomo_volume_iso_widget.min = np.min(tomogram)
        # tomo_volume_iso_widget.value = (np.percentile(tomogram, 90), np.percentile(tomogram, 99))
        # tomo_volume_iso_widget.max = np.max(tomogram)

        tomo_slice_slider_widget.max = tomo_z_vx
        tomo_slice_slider_widget.value = tomo_z_vx // 2
        tomo_slice_thickness_widget.max = tomo_z_vx // 4
        tomo_slice_thickness_widget.value = 1

        with output:
            print('Done!')

    def ptcls_load_callback(b):
        """
        Load two copies of the particles into the figure as a main scatter plot and a copy scatter plot for subset indication.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = b

        selected_tomogram = tomo_selection_widget.value
        selected_tomogram = tomo_star_mapping[selected_tomogram]  # tomo_selection_widget displays keys of a dict mapping tomogram file name on disk to tomogram name in star file / dataframe
        if selected_tomogram is None:
            with output:
                print('Use the "Tomogram:" dropdown to select the tomogram to load')
            return
        else:
            with output:
                print(f'Loading particles from {selected_tomogram} ...')

        df_particles_sub = df_particles[df_particles[tomo_id_column] == selected_tomogram]

        # identify Coordinate column names
        if any(df_particles_sub.columns.str.contains(pat='_rlnCoordinate')):
            # column headers use relion/nextpyp naming
            coords_cols = ['_rlnCoordinateX', '_rlnCoordinateY', '_rlnCoordinateZ']
        elif any(df_particles_sub.columns.str.contains(pat='_wrpCoordinate')):
            # columns use warp/m naming
            m_coords_cols = [col_name for col_name in df_particles_sub.columns if '_wrpCoordinate' in col_name]
            if len(m_coords_cols) == 3:
                # m temporal sampling == 1
                coords_cols = ['_wrpCoordinateX', '_wrpCoordinateY', '_wrpCoordinateZ']
            else:
                # m temporal sampling > 1, have 3*temporal_sampling pose columns
                assert len(m_coords_cols) % 3 == 0
                coords_cols = ['_wrpCoordinateX1', '_wrpCoordinateY1', '_wrpCoordinateZ1']
        else:
            raise RuntimeError(f'Could not identify star file columns containing particle tomogram-level XYZ coordinates from columns: {df_particles.columns}')
        with output:
            print(f'Using coordinate columns: {coords_cols}')

        # identify Coordinate pixel size in angstroms
        # possibility that multiple columns contain the substring Pixel, therefore we take the largest pixel size value
        # assumption is that star file might contain original micrograph pixel size + extracted particle (binned) pixel size; we want the latter to correctly position particles
        potential_pixel_columns = [col for col in df_particles_sub.columns if 'Pixel' in col]
        if len(potential_pixel_columns) == 1:
            star_angpix = df_particles_sub.loc[0, potential_pixel_columns[0]]
        else:
            star_angpix = max(df_particles_sub.loc[0, potential_pixel_columns])
        with output:
            print(f'Using starfile pixel size: {star_angpix}')

        df_particles_sub[coords_cols] = df_particles_sub[coords_cols] * star_angpix
        x, y, z, color = df_particles_sub[coords_cols + ['_UnfilteredParticleInds']].to_numpy().T

        tomodatacontainer.scatter_x_ang = max(x)
        tomodatacontainer.scatter_y_ang = max(y)
        tomodatacontainer.scatter_z_ang = max(z)
        tomodatacontainer.df_particles_sub = df_particles_sub

        ptcls_scatter = go.Scatter3d(
            name='ptcls_scatter',
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                color=color,
                opacity=1,
                size=ptcl_marker_size_widget.value,
            ),
            showlegend=False,
            hoverinfo='text',
            hovertext=[f'index {index}' for index in df_particles_sub.index.to_numpy()]
        )

        ptcls_subset_scatter = go.Scatter3d(
            name='subset_scatter',
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                color='red',
                opacity=0.5,
                size=ptcl_subset_size_widget.value,
            ),
            visible=ptcl_subset_display_widget.value,
            showlegend=False,
            hoverinfo='text',
            hovertext=[f'index {index}' for index in df_particles_sub.index.to_numpy()]
        )

        with fig.batch_update():
            fig.add_traces([
                ptcls_scatter,
                ptcls_subset_scatter
            ])
            fig.update_layout(scene=dict(
                xaxis=dict(range=[0, max(tomodatacontainer.tomo_x_ang, tomodatacontainer.scatter_x_ang)], ),
                yaxis=dict(range=[0, max(tomodatacontainer.tomo_y_ang, tomodatacontainer.scatter_y_ang)], ),
                zaxis=dict(range=[0, max(tomodatacontainer.tomo_z_ang, tomodatacontainer.scatter_z_ang)], ),
                aspectratio=dict(x=1, y=1, z=1),
            ))

        with output:
            print('Done!')

    def continuous_update_callback(event):
        """
        Toggle continuous updates for all continously interactive widgets (sliders)
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        for widget in [
            # tomo_volume_iso_widget,
            # tomo_volume_opacity_widget,
            tomo_slice_slider_widget,
            tomo_slice_thickness_widget,
            ptcl_marker_size_widget,
            ptcl_subset_size_widget,
            ptcl_subset_thresh_widget,
        ]:
            widget.continuous_update = continuous_update_widget.value

    def reset_callback(event):
        """
        Reset the widget state by deleting tomogram from memory and removing figure traces.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        tomodatacontainer.__init__()
        with fig.batch_update():
            fig.data = []
        output.clear_output()
        import gc
        gc.collect()

    # def tomo_volume_display_callback(event):
    #     """
    #     Update the figure to toggle visibility of the tomogram volume rendering
    #     """
    #     if tomodatacontainer.tomogram.size > 50**3:
    #         with output:
    #             print(f'Caution: rendering volume with > 50**3 elements can be very slow with plotly! Please wait ...')
    #     with fig.batch_update():
    #         fig.update_traces(visible=tomo_volume_display_widget.value, selector = ({'name':'tomo_volume'}))

    # def tomo_volume_iso_callback(event):
    #     """
    #     Update the figure to render the tomogram volume with the selected minimum and maximum isosurface thresholds.
    #     """
    #     with fig.batch_update():
    #         fig.update_traces(isomin=tomo_volume_iso_widget.value[0],
    #                            isomax=tomo_volume_iso_widget.value[1],
    #                           selector = ({'name':'tomo_volume'}))

    # def tomo_volume_opacity_callback(event):
    #     with fig.batch_update():
    #         fig.update_traces(opacity=tomo_volume_opacity_widget.value, selector = ({'name':'tomo_volume'}))

    def tomo_slice_display_callback(event):
        """
        Update the figure to toggle visibility of the tomogram slice rendering.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        with fig.batch_update():
            fig.update_traces(visible=tomo_slice_display_widget.value, selector=({'name': 'tomo_slice'}))

    def tomo_slice_slider_callback(event):
        """
        Update the figure to render the selected tomogram slice based on slice position, slice thickness, and active axis.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        # noinspection PyTypeChecker
        axis = str(tomo_slice_axis_widget.value)
        # noinspection PyTypeChecker
        plane = int(tomo_slice_slider_widget.value)
        # noinspection PyTypeChecker
        thickness = int(tomo_slice_thickness_widget.value)

        if thickness == 1:
            plane_min = plane
            plane_max = plane + 1
        else:
            plane_min = plane - thickness // 2
            plane_max = plane + thickness // 2

        if axis == 'z':
            new_texture = np.mean(tomodatacontainer.tomogram[plane_min:plane_max, :, :], axis=0)
            new_x = np.linspace(0, tomodatacontainer.tomo_x_ang, tomodatacontainer.tomo_x_vx, endpoint=False)
            new_y = np.linspace(0, tomodatacontainer.tomo_y_ang, tomodatacontainer.tomo_y_vx, endpoint=False)
            new_z = np.full((tomodatacontainer.tomo_y_vx, tomodatacontainer.tomo_x_vx), plane * tomodatacontainer.tomo_z_ang / tomodatacontainer.tomo_z_vx)
            with fig.batch_update():
                fig.update_traces(x=new_x,
                                  y=new_y,
                                  z=new_z,
                                  surfacecolor=new_texture,
                                  selector=({'name': 'tomo_slice'}))
        elif axis == 'y':
            new_texture = np.mean(tomodatacontainer.tomogram[:, plane_min:plane_max, :], axis=1)
            new_x = np.linspace(0, tomodatacontainer.tomo_x_ang, tomodatacontainer.tomo_x_vx, endpoint=False)
            new_y = np.full((tomodatacontainer.tomo_y_vx,), plane * tomodatacontainer.tomo_y_ang / tomodatacontainer.tomo_y_vx)
            new_z = np.linspace(0, tomodatacontainer.tomo_z_ang, tomodatacontainer.tomo_z_vx, endpoint=False)
            with fig.batch_update():
                fig.update_traces(x=new_x,
                                  y=new_y,
                                  z=np.array([new_z] * len(new_x)).T,
                                  surfacecolor=new_texture,
                                  selector=({'name': 'tomo_slice'}))
        elif axis == 'x':
            new_texture = np.mean(tomodatacontainer.tomogram[:, :, plane_min:plane_max], axis=2).T
            new_x = np.full(tomodatacontainer.tomo_x_vx, plane * tomodatacontainer.tomo_x_ang / tomodatacontainer.tomo_x_vx)
            new_y = np.linspace(0, tomodatacontainer.tomo_y_ang, tomodatacontainer.tomo_y_vx, endpoint=False)
            new_z = np.linspace(0, tomodatacontainer.tomo_z_ang, tomodatacontainer.tomo_z_vx, endpoint=False)
            with fig.batch_update():
                fig.update_traces(x=new_x,
                                  y=new_y,
                                  z=np.array([new_z] * len(new_y)),
                                  surfacecolor=new_texture,
                                  selector=({'name': 'tomo_slice'}))
        else:
            raise ValueError

    def tomo_slice_thickness_callback(event):
        """
        Update the tomogram slice slider min and max to constrain allowable slices within tomogram volume bounds with new thickness.
        Then update the figure via `tomo_slice_slider_callback`.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        # noinspection PyTypeChecker
        slice_thickness = int(tomo_slice_thickness_widget.value)
        tomo_slice_slider_widget.min = slice_thickness // 2
        if tomo_slice_axis_widget.value == 'x':
            tomo_slice_slider_widget.max = tomodatacontainer.tomo_x_vx - slice_thickness // 2
        elif tomo_slice_axis_widget.value == 'y':
            tomo_slice_slider_widget.max = tomodatacontainer.tomo_y_vx - slice_thickness // 2
        elif tomo_slice_axis_widget.value == 'z':
            tomo_slice_slider_widget.max = tomodatacontainer.tomo_z_vx - slice_thickness // 2
        else:
            raise ValueError
        tomo_slice_slider_callback('')

    def tomo_slice_axis_callback(event):
        """
        Update the tomogram slice slider max to constrain allowable slices within tomogram volume bounds with new axis.
        Update the tomogram thickness slider to constrain allowable thickness as less than the tomogram thickness along the new axis.
        Then update the figure via `tomo_slice_thickness_callback` which will further call `tomo_slice_slider_callback`.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        if tomo_slice_axis_widget.value == 'x':
            tomo_slice_slider_widget.max = tomodatacontainer.tomo_x_vx
            tomo_slice_thickness_widget.max = tomodatacontainer.tomo_x_vx // 4
        elif tomo_slice_axis_widget.value == 'y':
            tomo_slice_slider_widget.max = tomodatacontainer.tomo_y_vx
            tomo_slice_thickness_widget.max = tomodatacontainer.tomo_y_vx // 4
        elif tomo_slice_axis_widget.value == 'z':
            tomo_slice_slider_widget.max = tomodatacontainer.tomo_z_vx
            tomo_slice_thickness_widget.max = tomodatacontainer.tomo_y_vx // 4
        else:
            raise ValueError
        tomo_slice_thickness_callback('')

    def ptcl_marker_display_callback(event):
        """
        Update the figure to toggle visibility of the particles scatterplot rendering.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        with fig.batch_update():
            fig.update_traces(visible=ptcl_marker_display_widget.value, selector=({'name': 'ptcls_scatter'}))

    def ptcl_marker_size_callback(event):
        """
        Update the particles scatter plot marker size
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        with fig.batch_update():
            fig.update_traces(marker=dict(size=ptcl_marker_size_widget.value, ), selector=({'name': 'ptcls_scatter'}))

    def ptcl_colorby_column_callback(event):
        """
        Update the particles scatter plot colors by passing the selected column's values to the selected colormap.
        Also updates the hovertext of each point to report the associated column value for that particle.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        if ptcl_colorby_column_widget.value is None:
            fig.update_traces(marker=dict(color='mediumturquoise'),
                              hovertext=[f'index {index}' for index in
                                         tomodatacontainer.df_particles_sub.index.to_numpy()],
                              selector=({'name': 'ptcls_scatter'}))
            return

        if not pd.api.types.is_numeric_dtype(tomodatacontainer.df_particles_sub[ptcl_colorby_column_widget.value]):
            with output:
                print('Must choose a dataframe column with a numeric dtype.')
            return

        color_mapping = tomodatacontainer.df_particles_sub[ptcl_colorby_column_widget.value].to_numpy()
        color_map = ptcl_colorby_colormap_widget.value
        with fig.batch_update():
            # set hovertext for ptcls_scatter
            fig.update_traces(marker=dict(color=color_mapping,
                                          colorscale=color_map),
                              hovertext=[f'index {index}, color by value {value:.6f}' for index, value in
                                         zip(tomodatacontainer.df_particles_sub.index.to_numpy(),
                                             tomodatacontainer.df_particles_sub[
                                                 ptcl_colorby_column_widget.value].to_numpy())],
                              selector=({'name': 'ptcls_scatter'}))
            # duplicate hovertext for subset_scatter (otherwise hovering when subset_scatter shown overrides ptcls_scatter
            fig.update_traces(marker=dict(color=color_mapping,
                                          colorscale=color_map),
                              hovertext=[f'index {index}, color by value {value:.6f}' for index, value in
                                         zip(tomodatacontainer.df_particles_sub.index.to_numpy(),
                                             tomodatacontainer.df_particles_sub[
                                                 ptcl_colorby_column_widget.value].to_numpy())],
                              selector=({'name': 'subset_scatter'}))

    def ptcl_subset_display_callback(event):
        """
        Update the figure to toggle visibility of the particles scatterplot subset rendering.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        with fig.batch_update():
            fig.update_traces(visible=ptcl_subset_display_widget.value, selector=({'name': 'subset_scatter'}))

    def ptcl_subset_size_callback(event):
        """
        Update the particles scatter plot subset marker size.
        If existing subset selection (via setting marker size per-particle) is in place, ensure that it is restored.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        with fig.batch_update():
            fig.update_traces(marker=dict(size=ptcl_subset_size_widget.value, ), selector=({'name': 'subset_scatter'}))
            ptcl_subset_thresh_callback('')

    def ptcl_subset_column_callback(event):
        """
        Set the particles scatter plot subset column by which to filter the subset selection.
        Adjust the extent of `ptcl_subset_thresh_widget` to min and max of selected column.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        selected_column = ptcl_subset_column_widget.value

        if not pd.api.types.is_numeric_dtype(tomodatacontainer.df_particles_sub[selected_column]):
            with output:
                print('Must choose a dataframe column with a numeric dtype.')
            return

        old_min = ptcl_subset_thresh_widget.min
        old_max = ptcl_subset_thresh_widget.max
        new_min = tomodatacontainer.df_particles_sub[selected_column].min()
        new_max = tomodatacontainer.df_particles_sub[selected_column].max()

        if (old_min == new_min) and (old_max == new_max):
            return

        ptcl_subset_thresh_widget.min = min(old_min,
                                            new_min)  # update lower bound to temp value to avoid "setting max < min" error
        ptcl_subset_thresh_widget.min = new_min
        ptcl_subset_thresh_widget.max = max(old_max,
                                            new_max)  # update upper bound to temp value to avoid "setting min > max" error
        ptcl_subset_thresh_widget.max = new_max
        ptcl_subset_thresh_widget.value = [new_min, new_max]

    def ptcl_subset_thresh_callback(event):
        """
        Update the particles scatter plot subset markers to only show markers meeting thresholding criteria from `ptcl_subset_thresh_widget`.
        """
        # this function must take a parameter for the callback to function, but we do not currently use it
        _ = event

        # get mask 0,1 of particles meeting threshold, depending on toggle of invert_thresh
        subset_mask = np.array((tomodatacontainer.df_particles_sub[ptcl_subset_column_widget.value] >=
                                ptcl_subset_thresh_widget.value[0]) & (
                                       tomodatacontainer.df_particles_sub[ptcl_subset_column_widget.value] <=
                                       ptcl_subset_thresh_widget.value[1]))
        if ptcl_subset_invert_thresh_widget.value:
            subset_mask = np.logical_not(subset_mask)
        with fig.batch_update():
            fig.update_traces(marker=dict(size=ptcl_subset_size_widget.value * subset_mask * 2),
                              selector=({'name': 'subset_scatter'}))

    # TOMOGRAM / PARTICLES FIGURE LAYOUT
    fig = go.FigureWidget(layout=go.Layout(
        autosize=True,
        height=600,
        width=600,
        scene=dict(
            xaxis=dict(range=[0, 1], ),
            yaxis=dict(range=[0, 1], ),
            zaxis=dict(range=[0, 1], ),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        xaxis=go.layout.XAxis(linecolor="black", linewidth=1, mirror=False),
        yaxis=go.layout.YAxis(linecolor="black", linewidth=1, mirror=False),
        margin=go.layout.Margin(l=20, r=20, b=20, t=20),
        dragmode='orbit',
    ))

    style = {'description_width': 'initial'}

    # FIGURE INITIALIZATION CONTROLS
    tomo_selection_widget = widgets.Dropdown(
        options=tomo_star_mapping.keys(),
        value=None,
        description='Tomogram: ',
        style=style,
    )
    tomo_load_widget = widgets.Button(
        description='Load tomogram',
    )
    tomo_load_widget.on_click(tomo_load_callback)
    ptcls_load_widget = widgets.Button(
        description='Load particles',
    )
    ptcls_load_widget.on_click(ptcls_load_callback)
    continuous_update_widget = widgets.Checkbox(
        value=False,
        description='Continuous figure updates',
        style=style,
    )
    continuous_update_widget.observe(continuous_update_callback, names=['value'])
    reset_widget = widgets.Button(
        description='Reset widget',
    )
    reset_widget.on_click(reset_callback)

    # # TOMOGRAM ISOSURFACE CONTROLS
    # tomo_volume_display_widget = widgets.Checkbox(
    #     value=False,
    #     style=style,
    # )
    # tomo_volume_display_widget.observe(tomo_volume_display_callback, names=['value'])
    # tomo_volume_iso_widget = widgets.FloatRangeSlider(
    #     value=[0, 1],
    #     min=0,
    #     max=1,
    #     step=0.01,
    #     continuous_update=False,
    #     orientation='horizontal',
    #     readout=True,
    #     readout_format='.2f',
    #     style=style,
    # )
    # tomo_volume_iso_widget.observe(tomo_volume_iso_callback, names=['value'])
    # tomo_volume_opacity_widget = widgets.FloatSlider(
    #     value=0.5,
    #     min=0,
    #     max=1,
    #     step=0.01,
    #     continuous_update=False,
    #     orientation='horizontal',
    #     readout=True,
    #     readout_format='.2f',
    #     style=style,
    # )
    # tomo_volume_opacity_widget.observe(tomo_volume_opacity_callback, names=['value'])

    # TOMOGRAM SLICE CONTROLS
    tomo_slice_display_widget = widgets.Checkbox(
        value=True,
        style=style,
    )
    tomo_slice_display_widget.observe(tomo_slice_display_callback, names=['value'])
    tomo_slice_slider_widget = widgets.IntSlider(
        min=0,
        max=1,
        value=1,
        continuous_update=False,
        style=style,
    )
    tomo_slice_slider_widget.observe(tomo_slice_slider_callback, names=['value'])
    tomo_slice_thickness_widget = widgets.IntSlider(
        min=1,
        max=1,
        value=1,
        continuous_update=False,
        style=style,
    )
    tomo_slice_thickness_widget.observe(tomo_slice_thickness_callback, names=['value'])
    tomo_slice_axis_widget = widgets.Dropdown(
        options=['x', 'y', 'z'],
        value='z',
        style=style,
    )
    tomo_slice_axis_widget.observe(tomo_slice_axis_callback, names=['value'])

    # PARTICLE MARKER CONTROLS
    ptcl_marker_display_widget = widgets.Checkbox(
        value=True,
        style=style,
    )
    ptcl_marker_display_widget.observe(ptcl_marker_display_callback, names=['value'])
    ptcl_marker_size_widget = widgets.IntSlider(
        value=7,
        min=0,
        max=25,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style,
    )
    ptcl_marker_size_widget.observe(ptcl_marker_size_callback, names=['value'])
    ptcl_colorby_column_widget = widgets.Dropdown(
        options=[None, ] + df_particles.columns.to_list(),
        value=None,
        style=style,
    )
    ptcl_colorby_column_widget.observe(ptcl_colorby_column_callback, names=['value'])
    ptcl_colorby_colormap_widget = widgets.Dropdown(
        options=pl_colors.named_colorscales(),
        value=None,
        style=style,
    )
    ptcl_colorby_colormap_widget.observe(ptcl_colorby_column_callback, names=[
        'value'])  # shares the callback with ptcl_colorby_column_widget because intertwined functionality
    ptcl_subset_display_widget = widgets.Checkbox(
        value=False,
        style=style,
    )
    ptcl_subset_display_widget.observe(ptcl_subset_display_callback, names=['value'])
    ptcl_subset_size_widget = widgets.IntSlider(
        value=15,
        min=0,
        max=25,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style,
    )
    ptcl_subset_size_widget.observe(ptcl_subset_size_callback, names=['value'])
    ptcl_subset_column_widget = widgets.Dropdown(
        options=df_particles.columns.to_list(),
        value=None,
        style=style,
    )
    ptcl_subset_column_widget.observe(ptcl_subset_column_callback, names=['value'])
    ptcl_subset_thresh_widget = widgets.FloatRangeSlider(
        value=[0, 1],
        min=0,
        max=1,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style,
    )
    ptcl_subset_thresh_widget.observe(ptcl_subset_thresh_callback, names=['value'])
    ptcl_subset_invert_thresh_widget = widgets.Checkbox(
        value=False,
        style=style,
    )
    ptcl_subset_invert_thresh_widget.observe(ptcl_subset_thresh_callback, names=['value'])

    # LOGGED OUTPUT
    output = widgets.Output()

    # WIDGET LAYOUT
    controls_layout = widgets.Layout(justify_content='space-around')
    initialization_controls_container = widgets.HBox(children=[tomo_selection_widget,
                                                               tomo_load_widget,
                                                               ptcls_load_widget,
                                                               continuous_update_widget,
                                                               reset_widget, ],
                                                     layout=widgets.Layout(width='100%', display='inline-flex',
                                                                           justify_content='space-around',
                                                                           border='solid 1px'))
    # tomo_volume_controls_container = widgets.VBox(children=[
    #     widgets.HBox(children=[widgets.Label('Display tomogram volume'), tomo_volume_display_widget],
    #                  layout=controls_layout),
    #     widgets.HBox(children=[widgets.Label('tomo isosurface cutoffs'), tomo_volume_iso_widget],
    #                  layout=controls_layout),
    #     widgets.HBox(children=[widgets.Label('tomo opacity'), tomo_volume_opacity_widget], layout=controls_layout), ],
    #                                               layout=widgets.Layout(border='solid 1px'))
    tomo_slice_controls_container = widgets.VBox(children=[
        widgets.HBox(children=[widgets.Label('Display tomogram slice'), tomo_slice_display_widget],
                     layout=controls_layout),
        widgets.HBox(children=[widgets.Label('tomo slice position'), tomo_slice_slider_widget], layout=controls_layout),
        widgets.HBox(children=[widgets.Label('tomo slice thickness'), tomo_slice_thickness_widget],
                     layout=controls_layout),
        widgets.HBox(children=[widgets.Label('tomo slice axis'), tomo_slice_axis_widget], layout=controls_layout, )],
        layout=widgets.Layout(border='solid 1px'))
    output_container = widgets.VBox(children=[output],
                                    layout=widgets.Layout(border='solid 1px', height='150px'))
    ptcls_controls_container = widgets.VBox(children=[
        widgets.HBox(children=[widgets.Label('Display particles'), ptcl_marker_display_widget], layout=controls_layout),
        widgets.HBox(children=[widgets.Label('particle size'), ptcl_marker_size_widget], layout=controls_layout),
        widgets.HBox(children=[widgets.Label('particle color by'), ptcl_colorby_column_widget], layout=controls_layout),
        widgets.HBox(children=[widgets.Label('particle color map'), ptcl_colorby_colormap_widget],
                     layout=controls_layout), ],
        layout=widgets.Layout(border='solid 1px'))
    subset_controls_container = widgets.VBox(children=[
        widgets.HBox(children=[widgets.Label('Display subset'), ptcl_subset_display_widget], layout=controls_layout),
        widgets.HBox(children=[widgets.Label('subset size'), ptcl_subset_size_widget], layout=controls_layout),
        widgets.HBox(children=[widgets.Label('subset select by'), ptcl_subset_column_widget], layout=controls_layout),
        widgets.HBox(children=[widgets.Label('subset select cutoffs'), ptcl_subset_thresh_widget],
                     layout=controls_layout),
        widgets.HBox(children=[widgets.Label('invert subset selection'), ptcl_subset_invert_thresh_widget],
                     layout=controls_layout), ],
        layout=widgets.Layout(border='solid 1px'))
    runtime_controls_container = widgets.VBox(children=[  # tomo_volume_controls_container,
        tomo_slice_controls_container,
        ptcls_controls_container,
        subset_controls_container,
        output_container],
        layout=widgets.Layout(width='500px'))
    overall_widget = widgets.VBox(children=[initialization_controls_container,
                                            widgets.HBox(children=[runtime_controls_container, fig],
                                                         layout=widgets.Layout(width='100%',
                                                                               justify_content='space-between'))])

    # create holder for data
    tomodatacontainer = TomoDataContainer()

    # DISPLAY FINAL WIDGET
    return overall_widget
