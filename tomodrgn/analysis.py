"""
Functions for analysis of particle metadata: index, pose, ctf, latent embedding, label, tomogram spatial context, etc.
"""
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

from tomodrgn import utils, mrc

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
    total_loss = np.asarray([x.split()[-4][:-1] for x in lines])

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

    loss_mse = np.asarray([x.split()[10][:-1] for x in lines])
    loss_kld = np.asarray([x.split()[13][:-1] for x in lines])
    loss_total = np.asarray([x.split()[17][:-1] for x in lines])

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
             **kwargs: Any) -> np.ndarray:
    """
    Run UMAP dimensionality reduction on latent embeddings.
    :param z: array of latent embeddings, shape (nptcls, zdim)
    :param random_state: random state for reproducible runs, passed to umap.UMAP
    :param kwargs: additional key word arguments passed to umap.UMAP
    :return: array of UMAP embeddings, shape (len(z), 2)
    """
    # noinspection PyPackageRequirements
    import umap

    # perform embedding
    reducer = umap.UMAP(random_state=random_state, **kwargs)
    z_embedded = reducer.fit_transform(z)
    return z_embedded


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
        # temporarily suppress rendering figures since we use seaborn to get reordered row indices
        plt.ioff()

        # clustermap generation
        g = sns.clustermap(centers)

        # reindexing centers and labels
        reordered = g.dendrogram_row.reordered_ind
        centers = centers[reordered]
        reorder_label_mapping = {old_label: new_label for new_label, old_label in enumerate(reordered)}
        labels = np.array([reorder_label_mapping[old_label] for old_label in labels])

        # restore figure rendering
        plt.ion()
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


def _get_colors(num_colors: int,
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
    fig, ax = plt.subplot()
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
                         **kwargs: Any) -> matplotlib.figure.Figure:
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
    colors = _get_colors(num_colors=len(labels_sel),
                         cmap=cmap)

    # scatter by cluster
    for i, label_sel in enumerate(labels_sel):
        label_inds = labels == label_sel
        ax.scatter(x=x[label_inds],
                   y=y[label_inds],
                   s=s,
                   alpha=alpha,
                   label=f'cluster {label_sel}',
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
    colors = _get_colors(len(labels_sel), cmap)

    # create base plot
    ncol = int(np.ceil(len(labels_sel) ** .5))
    nrow = int(np.ceil(len(labels_sel) / ncol))
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, figsize=(10, 10))

    # draw subplots
    for ax, (i, label_sel) in zip(axes.ravel(), enumerate(labels_sel)):
        label_inds = labels == label_sel
        ax.scatter(x[label_inds],
                   y[label_inds],
                   s=s,
                   alpha=alpha,
                   rasterized=True,
                   color=colors[i],
                   **kwargs)
        ax.set_title(label_sel)

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
    fig, ax = plt.figure()
    ax.hist(psi)
    ax.set_xlabel('psi')

    return g, fig, ax


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
                   tsne: np.ndarray | None = None,
                   umap: np.ndarray | None = None,
                   euler: np.ndarray | None = None,
                   trans: np.ndarray | None = None,
                   labels: np.ndarray | None = None,
                   **kwargs) -> pd.DataFrame:
    """
    # TODO make a version of this function that auto-scans an analysis.EPOCH dir and loads all standard detected files
       as part of simplifying the jupyter analysis / visualization notebook.
    Merge known types of numpy arrays into a single pandas dataframe for downstream analysis.
    Only supplied key word arguments will be added as columns to the dataframe.
    :param z: array of latent embeddings, shape (nptcls, zdim)
    :param pc: array of PCA-transformed latent embeddings, shape (nptcls, PC-dim)
    :param tsne: array of t-SNE-transformed latent embeddings, shape (nptcls, tSNE-dim)
    :param umap: array of UMAP-transformed latent embeddings, shape (nptcls, UMAP-dim)
    :param euler: array of Euler angles, shape (nptcls, 3)
    :param trans: array of translation vectors, shape (nptcls, 2)
    :param labels: array of labels, shape (nptcls, )
    :param kwargs: key:value pairs of additional column_name: numpy array of values to add as additional columns to the dataframe
    :return:
    """
    # add supplied parameter arrays to a dictionary
    data = {}
    if umap is not None:
        data['UMAP1'] = umap[:, 0]
        data['UMAP2'] = umap[:, 1]
    if tsne is not None:
        data['TSNE1'] = tsne[:, 0]
        data['TSNE2'] = tsne[:, 1]
    if pc is not None:
        zdim = pc.shape[1]
        for i in range(zdim):
            data[f'PC{i + 1}'] = pc[:, i]
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


### TOMOGRAM RENDERING FUNCTIONS ###

def render_tomogram_volume(tomogram):
    # note: manually decrease brightness slider to make greyscale
    # note: manually decrease opacity slider to increase interpretability of volume
    ipv.gcf()
    tomoplot = ipv.volshow(tomogram,
                           level=[0.00, 0.68, 1.00],
                           opacity=[0.00, 0.68, 1.00],
                           downscale=2,
                           lighting=True,
                           controls=True,
                           max_opacity=1.0)


def render_tomogram_xyzslices(tomogram):
    zmax, ymax, xmax = tomogram.shape
    # these tomograms are (510,680,680)

    triangles = [(0, 1, 2), (0, 2, 3)]
    u = [0., 1., 1., 0.]
    v = [1., 1., 0., 0.]

    def calculate_tomogram_slice(slice_middle, thickness=10, axis='z'):
        if thickness == 1:
            slice_min = slice_middle
            slice_max = slice_middle + 1
        else:
            slice_min = slice_middle - thickness // 2
            slice_max = slice_min + thickness // 2

        if axis == 'z':
            texture = np.sum(tomogram[slice_min:slice_max, :, :], axis=0)
            x = [0.0, xmax, xmax, 0.0]
            y = [0.0, 0.0, ymax, ymax]
            z = [slice_middle, slice_middle, slice_middle, slice_middle]
        elif axis == 'y':
            texture = np.sum(tomogram[:, slice_min:slice_max, :], axis=1)
            x = [0.0, xmax, xmax, 0.0]
            y = [slice_middle, slice_middle, slice_middle, slice_middle]
            z = [0.0, 0.0, zmax, zmax]
        elif axis == 'x':
            texture = np.sum(tomogram[:, :, slice_min:slice_max], axis=2)
            x = [slice_middle, slice_middle, slice_middle, slice_middle]
            y = [0, ymax, ymax, 0]
            z = [0.0, 0, zmax, zmax]

        texture = (texture - texture.min()) / (texture.max() - texture.min()) * 255
        PIL_texture = Image.fromarray(np.uint8(texture), mode="L")

        return PIL_texture, x, y, z

    # TODO: if quiverplot is not None, update quiverplot by displaying points within slice thickness only
    def update_slice(*args):
        slice_middle_new = slice_slider.value
        thickness = thickness_slider.value
        axis = axis_dropdown.value

        PIL_texture_new, x_new, y_new, z_new = calculate_tomogram_slice(slice_middle_new,
                                                                        thickness=thickness,
                                                                        axis=axis)

        tomoplot.x = x_new
        tomoplot.y = y_new
        tomoplot.z = z_new
        tomoplot.texture = PIL_texture_new

    def update_slider_range(*args):
        thickness = thickness_slider.value
        axis = axis_dropdown.value

        axis_max = {'x': xmax, 'y': ymax, 'z': zmax}[axis]
        slice_slider.max = axis_max - thickness_slider.value // 2
        slice_slider.min = thickness_slider.value // 2

    def update_slice_and_slider(*args):
        update_slider_range()
        update_slice()

    # create interactive widgets
    slice_slider = widgets.IntSlider(min=0,
                                     max=zmax,
                                     value=255,
                                     continuous_update=True,
                                     description='Slice')
    thickness_slider = widgets.IntSlider(min=1,
                                         max=50,
                                         value=5,
                                         continuous_update=True,
                                         description='Thickness')
    axis_dropdown = widgets.Dropdown(options=['x', 'y', 'z'],
                                     value='z',
                                     description='Axis to slice',
                                     disabled=False)
    continuous_update_check = widgets.Checkbox(value=True,
                                               description='Continuous updates',
                                               disabled=False)

    # define widget behavior
    slice_slider.observe(update_slice, 'value')
    thickness_slider.observe(update_slice_and_slider, 'value')
    axis_dropdown.observe(update_slice_and_slider, 'value')
    widgets.jslink((continuous_update_check, 'value'), (slice_slider, 'continuous_update'))
    widgets.jslink((continuous_update_check, 'value'), (thickness_slider, 'continuous_update'))

    # create plot
    PIL_texture, x, y, z = calculate_tomogram_slice(250, thickness=20, axis='z')
    ipv.gcf()
    tomoplot = ipv.plot_trisurf(x, y, z, color='grey', triangles=triangles, texture=PIL_texture, u=u, v=v)
    ipv.xlim(0, xmax)
    ipv.ylim(0, ymax)
    ipv.zlim(0, zmax)

    # define UI layout
    ui = widgets.VBox([widgets.HBox(
        [slice_slider, thickness_slider],
        layout=widgets.Layout(width='100%', display='inline-flex', flex_flow='row wrap')
    ),
        widgets.HBox(
            [axis_dropdown, continuous_update_check],
            layout=widgets.Layout(width='100%', display='inline-flex', flex_flow='row wrap')
        )])
    display(ui)


def render_tomogram_isosurface(tomogram):
    ipv.gcf()
    tomoplot = ipv.plot_isosurface(tomogram,
                                   level=0.2,
                                   wireframe=True,
                                   surface=True,
                                   controls=True,
                                   extent=None)


### PLOTTING PARTICLES ASSOCIATED WITH ONE TOMOGRAM ###

def rescale_df_coordinates(df, tomo_max_xyz_nm=(680, 680, 510), tomo_pixelsize=10, starfile_pixelsize=1):
    # reconstructed tomos encompass some volume measured in nm
    # pixel sizes should be supplied as A/px
    tomo_max_x_nm, tomo_max_y_nm, tomo_max_z_nm = tomo_max_xyz_nm  # units = nm

    # rlnCoordinate from starfile is measured in px, so needs to be rescaled to dimensionless tomo voxels
    df.loc[:, '_rlnCoordinateX'] = df['_rlnCoordinateX'].astype('float') * starfile_pixelsize / tomo_pixelsize
    df.loc[:, '_rlnCoordinateY'] = df['_rlnCoordinateY'].astype('float') * starfile_pixelsize / tomo_pixelsize
    df.loc[:, '_rlnCoordinateZ'] = df['_rlnCoordinateZ'].astype('float') * starfile_pixelsize / tomo_pixelsize

    assert 0 <= df['_rlnCoordinateX'].all() <= tomo_max_x_nm
    assert 0 <= df['_rlnCoordinateY'].all() <= tomo_max_y_nm
    assert 0 <= df['_rlnCoordinateZ'].all() <= tomo_max_z_nm

    return df


def subset_ptcls_and_render(df, tomo_name_for_df):
    def subset_df(df, tomo_name_for_df):
        # subset selected tomogram from dropdown
        df_subset = df[df['_rlnMicrographName'] == tomo_name_for_df]

        # reset index
        df_subset = df_subset.reset_index(drop=True)
        df['index_this_tomo_particles'] = df.index

        return df_subset

    def pose_to_vector(df):
        # TODO: implement symmetry parsing
        # right now only poses from subtomos refined in C1 display correctly
        # Rot and Psi must cover [-180,180] or [0,360], and Tilt must cover [0,180]
        # otherwise particles with "C1 poses" outside "sym-restricted rot/tilt/psi" will appear misaligned
        # extract scaled image coordinates
        x = df['_rlnCoordinateX'].to_numpy(dtype=float)
        y = df['_rlnCoordinateY'].to_numpy(dtype=float)
        z = df['_rlnCoordinateZ'].to_numpy(dtype=float)

        # extract euler angles and convert to radians
        rot = df['_rlnAngleRot'].to_numpy(dtype=float) * np.pi / 180
        tilt = df['_rlnAngleTilt'].to_numpy(dtype=float) * np.pi / 180
        psi = df['_rlnAnglePsi'].to_numpy(dtype=float) * np.pi / 180

        # convert euler angles to xyz view vector
        # following formulation of doi: 10.1016/j.jsb.2005.06.001 section 2.2.5
        xv = np.cos(rot) * np.sin(tilt)
        yv = np.sin(rot) * np.sin(tilt)
        zv = np.cos(tilt)

        return x, y, z, xv, yv, zv

    class ParticleLabelPopup(ipv.ui.Popup):
        # index_all_ptcls is a list corresponding to the unfiltered particle indices for all tomograms
        # value is the index of the hovered point among all points in the plot

        index_all_ptcls = traitlets.List().tag(sync=True)
        sel_column = traitlets.Unicode().tag(sync=True)
        sel_column_values = traitlets.List().tag(sync=True)

        template_file = None  # disable the loading from file
        @traitlets.default("template")
        def _default_template(self):
            return """
                    <template>
                        <div :style="{padding: '4px', 'background-color': 'black', color: 'white'}">
                            Particle {{index_all_ptcls[value]}}
                            <br>
                            {{sel_column}}: {{sel_column_values[value]}}
                        </div>
                    </template>
                    """

    def generate_colormap_from_column(df_subset_selected_column, colormap=None):
        # derive colormap from selected column
        cmap = plt.get_cmap(colormap)

        metric_min = df[df_subset_selected_column].min()
        metric_max = df[df_subset_selected_column].max()
        metric_subset = df_subset[df_subset_selected_column]

        normalized_metric = (metric_subset - metric_min) / (metric_max - metric_min)
        return cmap(normalized_metric)

    def selcolumn_dropdown_eventhandler(change):
        df_subset_selected_column = column_selection.value

        # update popup extra data
        popup.sel_column = df_subset_selected_column
        popup.sel_column_values = df_subset.loc[:, df_subset_selected_column].to_list()

        # update the colormap
        colormap = colormap_selection.value
        if colormap is not None:
            colors = generate_colormap_from_column(df_subset_selected_column, colormap)
        else:
            colors = 'dodgerblue'
        quiverplot.color = colors

        # update the subset selection slider values
        if show_selection_checkbox.value:
            old_min = subset_value_range.min
            old_max = subset_value_range.max
            new_min = df[column_selection.value].min()
            new_max = df[column_selection.value].max()

            if (old_min == new_min) and (old_max == new_max):
                return

            subset_value_range.min = min(old_min, new_min)  # update lower bound to temp value to avoid "setting max < min" error
            subset_value_range.max = max(old_max, new_max)  # update upper bound to temp value to avoid "setting min > max" error

            subset_value_range.min = new_min
            subset_value_range.max = new_max
            subset_value_range.value = [new_min, new_max]

        update_subset_quiver()

    # subset dataframe by selected tomo
    df = guess_dtypes(df)
    df_subset = subset_df(df, tomo_name_for_df)

    # create interactive widgets for main plot
    column_selection = widgets.Dropdown(options=df_subset,
                                        description='Color by:',
                                        disabled=False)
    colormaps = [('None', None),
                 ('viridis | Perceptually Uniform Sequential', 'viridis'),
                 ('coolwarm | Diverging', 'coolwarm'),
                 ('hsv | Cyclic', 'hsv'),
                 ('tab10 | Categorical 10', 'tab10'),
                 ('tab20 | Categorical 20', 'tab20'),
                 ('CMRmap | Miscellaneous', 'CMRmap'),
                 ('turbo | Miscellaneous', 'turbo')]
    colormap_selection = widgets.Dropdown(options=colormaps,
                                          description='Colormap:',
                                          disabled=False)
    size = widgets.FloatSlider(value=4, min=0, max=10, description='Marker size')

    # create interactive widgets for subset overlay plot
    # TODO: replace FloatRangeSlider with textbox for pd.query custom selection by user
    show_selection_checkbox = widgets.Checkbox(value=False,
                                               description='Overlay subset plot')
    subset_value_range = widgets.FloatRangeSlider(value=[df_subset[column_selection.value].min(),
                                                         df_subset[column_selection.value].max()],
                                                  description='Subset values',
                                                  min=df_subset[column_selection.value].min(),
                                                  max=df_subset[column_selection.value].max(),
                                                  continuous_update=False,
                                                  orientation='horizontal',
                                                  readout=True,
                                                  readout_format='.1f',
                                                  disabled=True)
    subset_invert_selection = widgets.Checkbox(value=False, description='Invert slider range', disabled=True)
    subset_size = widgets.FloatSlider(value=0, min=0, max=4, description='Subset marker size', disabled=True)

    #     output = widgets.Output()

    def toggle_subset_overlay(*args):
        show_selection = show_selection_checkbox.value
        if show_selection:
            subset_value_range.disabled = False
            subset_invert_selection.disabled = False
            subset_size.disabled = False
            quiverplot_selected.size = 2
        else:
            subset_value_range.disabled = True
            subset_invert_selection.disabled = True
            subset_size.disabled = True
            quiverplot_selected.size = 0

    def update_subset_quiver(*args):
        subset_min, subset_max = subset_value_range.value
        invert_selection = subset_invert_selection.value
        selected_col = column_selection.value

        if not invert_selection:
            selected_ptcls = df_subset[(df_subset[selected_col] >= subset_min) & (df_subset[selected_col] <= subset_max)]
        else:
            selected_ptcls = df_subset[(df_subset[selected_col] <= subset_min) | (df_subset[selected_col] >= subset_max)]

        x_new, y_new, z_new, xv_new, yv_new, zv_new = pose_to_vector(selected_ptcls)

        quiverplot_selected.x = x_new
        quiverplot_selected.y = y_new
        quiverplot_selected.z = z_new
        quiverplot_selected.xv = xv_new
        quiverplot_selected.yv = yv_new
        quiverplot_selected.zv = zv_new

    # define widget behavior
    column_selection.observe(selcolumn_dropdown_eventhandler, names='value')
    colormap_selection.observe(selcolumn_dropdown_eventhandler, names='value')

    show_selection_checkbox.observe(toggle_subset_overlay, names='value')
    subset_value_range.observe(update_subset_quiver, names='value')
    subset_invert_selection.observe(update_subset_quiver, names='value')

    # create plot
    # markers = ['arrow', 'box', 'diamond', 'sphere', 'point_2d', 'square_2d', 'triangle_2d']
    x, y, z, xv, yv, zv = pose_to_vector(df_subset)
    ipv.gcf()
    quiverplot = ipv.quiver(x, y, z, xv, yv, zv, size=4, color='dodgerblue')
    quiverplot_selected = ipv.quiver(x, y, z, xv, yv, zv, size=0, color='red', marker='sphere', alpha=0.5)
    widgets.jslink((size, 'value'), (quiverplot, 'size'))
    widgets.jslink((subset_size, 'value'), (quiverplot_selected, 'size'))

    popup = ParticleLabelPopup()
    popup.index_all_ptcls = df_subset.loc[:, 'index_all_tomos_particles'].to_list()  # get index of hovered particle relative to all particles
    popup.sel_column = column_selection.value
    popup.sel_column_values = df_subset.loc[:, popup.sel_column].to_list()
    quiverplot.popup = popup

    # define UI layout
    quiverplot_widgets = widgets.HBox(
        [column_selection, colormap_selection, size],
        layout=widgets.Layout(width='100%', display='inline-flex', flex_flow='row wrap'))
    selection_widgets = widgets.HBox(
        [show_selection_checkbox, subset_value_range, subset_invert_selection, subset_size],
        layout=widgets.Layout(width='100%', display='inline-flex', flex_flow='row wrap'))
    ui = widgets.VBox([quiverplot_widgets, selection_widgets])  # , output])
    display(ui)


### TOMOGRAM PARTICLE INTERACTIVE WIDGET ###

def interactive_tomo_ptcls(df_merged, tomo_list, tomo_star_mappings):
    def load_tomogram(*args):
        selected_tomogram_path = tomo_selection.value
        selected_rendering_style = rendering_selection.value

        rendering_styles = {'volume': render_tomogram_volume,
                            'xyz slices': render_tomogram_xyzslices,
                            'isosurface': render_tomogram_isosurface}

        with output:
            print(f'Loading {selected_tomogram_path} as {selected_rendering_style}')
            tomogram = mrc.parse_mrc(selected_tomogram_path)[0]
            rendering_styles[selected_rendering_style](tomogram)

    def load_particles(*args):
        selected_tomogram_name = tomo_selection.value.split('/')[-1]
        tomo_name_for_df = tomo_star_mappings[selected_tomogram_name]
        with output:
            subset_ptcls_and_render(df_merged, tomo_name_for_df)

    def toggle_axes(*args):
        axes_on = displayaxes_checkbox.value
        if axes_on:
            ipv.style.axes_on()
        else:
            ipv.style.axes_off()

    def toggle_box(*args):
        box_on = displaybox_checkbox.value
        if box_on:
            ipv.style.box_on()
        else:
            ipv.style.box_off()

    def clear_output(*args):
        output.clear_output()
        with output:
            ipv.clear()
            fig_base = ipv.figure(width=800, height=600)
            ipv.show()

    # tomogram widget initialization
    tomo_selection = widgets.Dropdown(options=tomo_list,
                                      description='Tomogram:',
                                      disabled=False)
    rendering_selection = widgets.Dropdown(options=['xyz slices', 'volume', 'isosurface'],
                                           description='Rendering:',
                                           disabled=False)
    load_tomo_button = widgets.Button(description='Load tomogram',
                                      disabled=False,
                                      button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                      tooltip='Load selected tomogram into memory')
    load_particles_button = widgets.Button(description='Load particles', tooltip='Load particles into display')

    # meta widget initialization
    displayaxes_checkbox = widgets.Checkbox(value=True, description='Display axes')
    displaybox_checkbox = widgets.Checkbox(value=True, description='Display bounding box')
    reset_button = widgets.Button(description='Clear all output', tooltip='Clears all figures, widgets, and text')

    # widget layout
    tomogram_widgets = widgets.HBox([tomo_selection, rendering_selection, load_tomo_button, load_particles_button],
                                    layout=widgets.Layout(border='2px solid'))
    meta_widgets = widgets.HBox([displayaxes_checkbox, displaybox_checkbox, reset_button],
                                layout=widgets.Layout(border='2px solid'))
    output = widgets.Output()
    load_widget = widgets.VBox([tomogram_widgets, meta_widgets, output])

    # widget interactvity
    load_tomo_button.on_click(load_tomogram)
    load_particles_button.on_click(load_particles)
    displayaxes_checkbox.observe(toggle_axes, names='value')
    displaybox_checkbox.observe(toggle_box, names='value')
    reset_button.on_click(clear_output)

    # display and run the widget
    display(load_widget)
    with output:
        ipv.clear()
        fig_base = ipv.figure(width=800, height=600)
        ipv.show()