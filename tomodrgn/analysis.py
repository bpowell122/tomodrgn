import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess

from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from tomodrgn import utils, mrc

import ipyvolume as ipv
import ipywidgets as widgets
from PIL import Image
import traitlets


log = utils.log

def parse_loss(f):
    '''Parse loss from run.log'''
    lines = open(f).readlines()
    lines = [x for x in lines if '====' in x]
    try:
        loss = [x.strip().split()[-1] for x in lines]
        loss = np.asarray(loss).astype(np.float32)
    except:
        loss = [x.split()[-4][:-1] for x in lines]
        loss = np.asarray(loss).astype(np.float32)
    return loss

def parse_all_losses(file_name):
    '''Parse MSE + KLD + Total losses from run.log'''
    with open(file_name) as f:
        lines = f.readlines()
    lines = [x for x in lines if '====' in x]

    loss_mse = [x.split()[10][:-1] for x in lines]
    loss_mse = np.asarray(loss_mse).astype(np.float32)

    loss_kld = [x.split()[13][:-1] for x in lines]
    loss_kld = np.asarray(loss_kld).astype(np.float32)

    loss_total = [x.split()[17][:-1] for x in lines]
    loss_total = np.asarray(loss_total).astype(np.float32)

    return loss_mse, loss_kld, loss_total

### Dimensionality reduction ###

def run_pca(z):
    pca = PCA(z.shape[1])
    pca.fit(z)
    log('Explained variance ratio:')
    log(pca.explained_variance_ratio_)
    pc = pca.transform(z)
    return pc, pca

def get_pc_traj(pca, zdim, numpoints, dim, start, end, percentiles=None):
    '''
    Create trajectory along specified principle component
    
    Inputs:
        pca: sklearn PCA object from run_pca
        zdim (int)
        numpoints (int): number of points between @start and @end
        dim (int): PC dimension for the trajectory (1-based index)
        start (float): Value of PC{dim} to start trajectory
        end (float): Value of PC{dim} to stop trajectory
        percentiles (np.array or None): Define percentile array instead of np.linspace(start,stop,numpoints)
    
    Returns:
        np.array (numpoints x zdim) of z values along PC
    '''
    if percentiles is not None:
        assert len(percentiles) == numpoints
    traj_pca = np.zeros((numpoints,zdim))
    traj_pca[:,dim-1] = np.linspace(start, end, numpoints) if percentiles is None else percentiles
    ztraj_pca = pca.inverse_transform(traj_pca)
    return ztraj_pca

def run_tsne(z, n_components=2, perplexity=1000):
    if len(z) > 10000:
        log('WARNING: {} datapoints > {}. This may take awhile.'.format(len(z), 10000))
    z_embedded = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(z)
    return z_embedded

def run_umap(z, **kwargs):
    import umap # CAN GET STUCK IN INFINITE IMPORT LOOP
    reducer = umap.UMAP(**kwargs)
    z_embedded = reducer.fit_transform(z)
    return z_embedded

### Clustering ###

def cluster_kmeans(z, K, on_data=True, reorder=True):
    '''
    Cluster z by K means clustering
    Returns cluster labels, cluster centers
    If reorder=True, reorders clusters according to agglomerative clustering of cluster centers
    '''
    kmeans = KMeans(n_clusters=K,
                    random_state=0,
                    max_iter=10)
    labels = kmeans.fit_predict(z)
    centers = kmeans.cluster_centers_

    if on_data:
        centers, centers_ind = get_nearest_point(z, centers)

    if reorder:
        g = sns.clustermap(centers)
        reordered = g.dendrogram_row.reordered_ind
        centers = centers[reordered]
        if on_data: centers_ind = centers_ind[reordered]
        tmp = {k:i for i,k in enumerate(reordered)}
        labels = np.array([tmp[k] for k in labels])
    return labels, centers

def cluster_gmm(z, K, on_data=True, random_state=None, **kwargs):
    '''
    Cluster z by a K-component full covariance Gaussian mixture model
    
    Inputs:
        z (Ndata x zdim np.array): Latent encodings
        K (int): Number of clusters
        on_data (bool): Compute cluster center as nearest point on the data manifold
        random_state (int or None): Random seed used for GMM clustering
        **kwargs: Additional keyword arguments passed to sklearn.mixture.GaussianMixture

    Returns: 
        np.array (Ndata,) of cluster labels
        np.array (K x zdim) of cluster centers
    '''
    clf = GaussianMixture(n_components=K, covariance_type='full', random_state=None, **kwargs)
    labels = clf.fit_predict(z)
    centers = clf.means_
    if on_data:
        centers, centers_ind = get_nearest_point(z, centers)
    return labels, centers

def get_nearest_point(data, query):
    '''
    Find closest point in @data to @query
    Return datapoint, index
    '''
    ind = cdist(query, data).argmin(axis=1)
    return data[ind], ind

### HELPER FUNCTIONS FOR INDEX ARRAY MANIPULATION

def convert_original_indices(ind, N_orig, orig_ind):
    '''
    Convert index array into indices into the original particle stack
    ''' # todo -- finish docstring
    return np.arange(N_orig)[orig_ind][ind]

def combine_ind(N, sel1, sel2, kind='intersection'):
    # todo -- docstring
    if kind == 'intersection':
        ind_selected = set(sel1) & set(sel2)
    elif kind == 'union':
        ind_selected = set(sel1) | set(sel2)
    else:
        raise RuntimeError(f"Mode {kind} not recognized. Choose either 'intersection' or 'union'")
    ind_selected_not = np.array(sorted(set(np.arange(N)) - ind_selected))
    ind_selected = np.array(sorted(ind_selected))
    return ind_selected, ind_selected_not

def get_ind_for_cluster(labels, selected_clusters):
    '''Return index array of the selected clusters
    
    Inputs:
        labels: np.array of cluster labels for each particle
        selected_clusters: list of cluster labels to select

    Return:
        ind_selected: np.array of particle indices with the desired cluster labels

    Example usage:
        ind_keep = get_ind_for_cluster(kmeans_labels, [0,4,6,14])
    '''
    ind_selected = np.array([i for i,label in enumerate(labels) if label in selected_clusters])
    return ind_selected


### PLOTTING ###

def _get_colors(K, cmap=None):
    if cmap is not None:
        cm = plt.get_cmap(cmap)
        colors = [cm(i/float(K)) for i in range(K)]
    else:
        colors = ['C{}'.format(i) for i in range(10)]
        colors = [colors[i%len(colors)] for i in range(K)]
    return colors
   
def scatter_annotate(x, y, centers=None, centers_ind=None, annotate=True, labels=None, alpha=.1, s=1):
    fig, ax = plt.subplots()
    plt.scatter(x, y, alpha=alpha, s=s, rasterized=True)

    # plot cluster centers
    if centers_ind is not None:
        assert centers is None
        centers = np.array([[x[i],y[i]] for i in centers_ind])
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], c='k')
    if annotate:
        assert centers is not None
        if labels is None:
            labels = range(len(centers))
        for i in labels:
            ax.annotate(str(i), centers[i,0:2]+np.array([.1,.1]))
    return fig, ax

def scatter_annotate_hex(x, y, centers=None, centers_ind=None, annotate=True, labels=None):
    g = sns.jointplot(x=x, y=y, kind='hex')

    # plot cluster centers
    if centers_ind is not None:
        assert centers is None
        centers = np.array([[x[i],y[i]] for i in centers_ind])
    if centers is not None:
        g.ax_joint.scatter(centers[:,0], centers[:,1], color='k', edgecolor='grey')
    if annotate:
        assert centers is not None
        if labels is None:
            labels = range(len(centers))
        for i in labels:
            g.ax_joint.annotate(str(i), centers[i,0:2]+np.array([.1,.1]), color='black',
                                bbox=dict(boxstyle='square,pad=.1', ec='None', fc='1', alpha=.5))
    return g

def scatter_color(x, y, c, cmap='viridis', s=1, alpha=.1, label=None, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    assert len(x) == len(y) == len(c)
    sc = plt.scatter(x, y, s=s, alpha=alpha, rasterized=True, cmap=cmap, c=c)
    cbar = plt.colorbar(sc)
    cbar.set_alpha(1)
    cbar.draw_all()
    if label:
        cbar.set_label(label)
    return fig, ax

def plot_by_cluster(x, y, K, labels, centers=None, centers_ind=None, annotate=False, 
                    s=2, alpha=0.1, colors=None, cmap=None, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    if type(K) is int:
        K = list(range(K))

    if colors is None:
        colors = _get_colors(len(K), cmap)

    # scatter by cluster
    for i in K:
        ii = labels == i
        x_sub = x[ii]
        y_sub = y[ii]
        plt.scatter(x_sub, y_sub, s=s, alpha=alpha, label='cluster {}'.format(i), color=colors[i], rasterized=True)

    # plot cluster centers
    if centers_ind is not None:
        assert centers is None
        centers = np.array([[x[i],y[i]] for i in centers_ind])
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], c='k')
    if annotate:
        assert centers is not None
        for i in K:
            ax.annotate(str(i), centers[i,0:2])
    return fig, ax

def plot_by_cluster_subplot(x, y, K, labels, 
                            s=2, alpha=.1, colors=None, cmap=None, figsize=None):
    if type(K) is int:
        K = list(range(K))
    ncol = int(np.ceil(len(K)**.5))
    nrow = int(np.ceil(len(K)/ncol))
    fig, ax = plt.subplots(ncol, nrow, sharex=True, sharey=True, figsize=(10,10))
    if colors is None:
        colors = _get_colors(len(K), cmap)
    for i in K:
        ii = labels == i
        x_sub = x[ii]
        y_sub = y[ii]
        a = ax.ravel()[i]
        a.scatter(x_sub, y_sub, s=s, alpha=alpha, rasterized=True, color=colors[i])
        a.set_title(i)
    return fig, ax

def plot_euler(theta,phi,psi,plot_psi=True):
    sns.jointplot(x=theta,y=phi,kind='hex',
              xlim=(-180,180),
              ylim=(0,180)).set_axis_labels("theta", "phi")
    if plot_psi:
        plt.figure()
        plt.hist(psi)
        plt.xlabel('psi')

def ipy_plot_interactive_annotate(df, ind, opacity=.3):
    '''Interactive plotly widget for a cryoDRGN pandas dataframe with annotated points'''
    import plotly.graph_objs as go
    from ipywidgets import interactive
    if 'labels' in df.columns:
        text = [f'Class {k}: index {i}' for i,k in zip(df.index, df.labels)] # hovertext
    else:
        text = [f'index {i}' for i in df.index] # hovertext
    xaxis, yaxis = df.columns[0], df.columns[1]
    scatter = go.Scattergl(x=df[xaxis], 
                           y=df[yaxis], 
                           mode='markers',
                           text=text,
                           marker=dict(size=2,
                                       opacity=opacity,
                                       ))
    sub = df.loc[ind]
    text = [f'{k}){i}' for i,k in zip(sub.index, sub.labels)]
    scatter2 = go.Scatter(x=sub[xaxis],
                            y=sub[yaxis],
                            mode='markers+text',
                            text=text,
                            textposition="top center",
                            textfont=dict(size=9,color='black'),
                            marker=dict(size=5,color='black'))
    f = go.FigureWidget([scatter,scatter2])
    f.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    
    def update_axes(xaxis, yaxis, color_by, colorscale):
        scatter = f.data[0]
        scatter.x = df[xaxis]
        scatter.y = df[yaxis]
        
        scatter.marker.colorscale = colorscale
        if colorscale is None:
            scatter.marker.color = None
        else:
            scatter.marker.color = df[color_by] if color_by != 'index' else df.index
    
        scatter2 = f.data[1]
        scatter2.x = sub[xaxis]
        scatter2.y = sub[yaxis]
        with f.batch_update(): # what is this for??
            f.layout.xaxis.title = xaxis
            f.layout.yaxis.title = yaxis
        
    widget = interactive(update_axes, 
                    yaxis = df.select_dtypes('number').columns, 
                    xaxis = df.select_dtypes('number').columns,
                    color_by = df.columns,
                    colorscale = [None,'hsv','plotly3','deep','portland','picnic','armyrose'])
    return widget, f

def ipy_plot_interactive(df, opacity=.3):
    '''Interactive plotly widget for a cryoDRGN pandas dataframe'''
    import plotly.graph_objs as go
    from ipywidgets import interactive
    if 'labels' in df.columns:
        text = [f'Class {k}: index {i}' for i,k in zip(df.index, df.labels)] # hovertext
    else:
        text = [f'index {i}' for i in df.index] # hovertext
    
    xaxis, yaxis = df.columns[0], df.columns[1]
    f = go.FigureWidget([go.Scattergl(x=df[xaxis],
                                  y=df[yaxis],
                                  mode='markers',
                                  text=text,
                                  marker=dict(size=2,
                                              opacity=opacity,
                                              color=np.arange(len(df)),
                                              colorscale='hsv'
                                             ))])
    scatter = f.data[0]
    N = len(df)
    f.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    f.layout.dragmode = 'lasso'

    def update_axes(xaxis, yaxis, color_by, colorscale):
        scatter = f.data[0]
        scatter.x = df[xaxis]
        scatter.y = df[yaxis]
        
        scatter.marker.colorscale = colorscale
        if colorscale is None:
            scatter.marker.color = None
        else:
            scatter.marker.color = df[color_by] if color_by != 'index' else df.index
        with f.batch_update(): # what is this for??
            f.layout.xaxis.title = xaxis
            f.layout.yaxis.title = yaxis
 
    widget = interactive(update_axes, 
                         yaxis=df.select_dtypes('number').columns, 
                         xaxis=df.select_dtypes('number').columns,
                         color_by = df.columns,
                         colorscale = [None,'hsv','plotly3','deep','portland','picnic','armyrose'])

    t = go.FigureWidget([go.Table(
                        header=dict(values=['index']),
                        cells=dict(values=[df.index]),
                        )])

    def selection_fn(trace, points, selector):
        t.data[0].cells.values = [df.loc[points.point_inds].index]

    scatter.on_selection(selection_fn)
    return widget, f, t

def plot_projections(imgs, labels=None):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    axes = axes.ravel()
    for i in range(min(len(imgs),9)):
        axes[i].imshow(imgs[i], cmap='Greys_r')
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(labels[i])
    return fig, axes

def gen_volumes(weights, config, zfile, outdir, cuda=None,
                Apix=None, flip=False, downsample=None, invert=None,
                lowpass=None):
    '''Call cryodrgn eval_vol to generate volumes at specified z values
    Input:
        weights (str): Path to model weights .pkl
        config (str): Path to config.pkl
        zfile (str): Path to .txt file of z values
        outdir (str): Path to output directory for volumes,
        cuda (int or None): Specify cuda device
        Apix (float or None): Apix of output volume
        flip (bool): Flag to flip chirality of output volumes
        downsample (int or None): Generate volumes at this box size
        invert (bool): Invert contrast of output volumes
        lowpass (float or None): Lowpass filter to this resolution in Å
    '''
    cmd = f'tomodrgn eval_vol --weights {weights} --config {config} --zfile {zfile} -o {outdir}'
    if Apix is not None:
        cmd += f' --Apix {Apix}'
    if flip:
        cmd += f' --flip'
    if downsample is not None:
        cmd += f' -d {downsample}'
    if invert:
        cmd += f' --invert'
    if lowpass is not None:
        cmd += f' --lowpass {lowpass}'
    if cuda is not None:
        cmd = f'CUDA_VISIBLE_DEVICES={cuda} {cmd}'
    log(f'Running command:\n{cmd}')
    return subprocess.check_call(cmd, shell=True)

def load_dataframe(z=None, pc=None, euler=None, trans=None, labels=None, tsne=None, umap=None, **kwargs):
    '''Load results into a pandas dataframe for downstream analysis'''
    data = {}
    if umap is not None:
        data['UMAP1'] = umap[:,0]
        data['UMAP2'] = umap[:,1]
    if tsne is not None:
        data['TSNE1'] = tsne[:,0]
        data['TSNE2'] = tsne[:,1]
    if pc is not None:
        zD = pc.shape[1]
        for i in range(zD):
            data[f'PC{i+1}'] = pc[:,i]
    if labels is not None:
        data['labels'] = labels
    if euler is not None:
        data['theta'] = euler[:,0]
        data['phi'] = euler[:,1]
        data['psi'] = euler[:,2]
    if trans is not None:
        data['tx'] = trans[:,0]
        data['ty'] = trans[:,1]
    if z is not None:
        zD = z.shape[1]
        for i in range(zD):
            data[f'z{i}'] = z[:,i]
    for kk,vv in kwargs.items():
        data[kk] = vv
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