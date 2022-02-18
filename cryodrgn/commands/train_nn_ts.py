'''
Train a NN to model a 3D density map given 2D images from a tilt series with consensus pose assignments
'''
import numpy as np
import sys, os
import argparse
import pickle
from datetime import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import cryodrgn
from cryodrgn import mrc
from cryodrgn import utils
from cryodrgn import dataset
from cryodrgn import ctf
from cryodrgn import models

from cryodrgn.pose import PoseTracker
from cryodrgn.lattice import Lattice

log = utils.log
vlog = utils.vlog
# TODO fix logging: everything to flog instead of log, and optional vlog where appropriate


def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    parser.add_argument('--poses', type=os.path.abspath, required=True, help='Image poses (.pkl)')
    parser.add_argument('--ctf', metavar='pkl', type=os.path.abspath, help='CTF parameters (.pkl)')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=1,
                        help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='Logging interval in N_PTCLS (default: %(default)s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increaes verbosity')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 100000), help='Random seed')

    group = parser.add_argument_group('Dataset loading')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.85, help='Windowing radius (default: %(default)s)')
    group.add_argument('--ind', type=os.path.abspath, help='Filter particle stack by these indices')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    parser.add_argument('--relion31', action='store_true', help='Flag if relion3.1 star format')
    parser.add_argument('--do-dose-weighting', action='store_true', help='Flag to calculate losses per tilt per pixel with dose weighting ')
    parser.add_argument('--dose-override', type=float, default=None, help='Manually specify dose in e- / A2 / tilt')
    parser.add_argument('--do-tilt-weighting', action='store_true', help='Flag to calculate losses per tilt with cosine(tilt_angle) weighting')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b', '--batch-size', type=int, default=8, help='Minibatch size (default: %(default)s)')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer (default: %(default)s)')
    group.add_argument('--lr', type=float, default=1e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: mean, std of dataset)')
    group.add_argument('--amp', action='store_true', help='Use mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')
    group.add_argument('--skip-zeros', action='store_true', help='Reduce CUDA memory usage by ignoring 0-weight fourier components')

    group = parser.add_argument_group('Pose SGD')
    group.add_argument('--do-pose-sgd', action='store_true', help='Refine poses')
    group.add_argument('--pretrain', type=int, default=5, help='Number of epochs with fixed poses before pose SGD (default: %(default)s)')
    group.add_argument('--emb-type', choices=('s2s2', 'quat'), default='quat', help='SO(3) embedding type for pose SGD (default: %(default)s)')
    group.add_argument('--pose-lr', type=float, default=1e-4, help='Learning rate in Adam optimizer (default: %(default)s)')

    group = parser.add_argument_group('Network Architecture')
    group.add_argument('--layers', type=int, default=3, help='Number of hidden layers (default: %(default)s)')
    group.add_argument('--dim', type=int, default=256, help='Number of nodes in hidden layers (default: %(default)s)')
    group.add_argument('--l-extent', type=float, default=0.5, help='Coordinate lattice size (if not using positional encoding) (default: %(default)s)')
    group.add_argument('--pe-type',
                       choices=('geom_ft', 'geom_full', 'geom_lowf', 'geom_nohighf', 'linear_lowf', 'none'),
                       default='geom_lowf', help='Type of positional encoding (default: %(default)s)')
    group.add_argument('--pe-dim', type=int, help='Num sinusoid features in positional encoding (default: D/2)')
    group.add_argument('--domain', choices=('hartley', 'fourier'), default='fourier', help='Volume decoder representation (default: %(default)s)')
    group.add_argument('--activation', choices=('relu', 'leaky_relu'), default='relu', help='Activation (default: %(default)s)')
    return parser


def save_checkpoint(model, lattice, optim, epoch, norm, Apix, out_mrc, out_weights):
    model.eval()
    if isinstance(model, nn.DataParallel):
        model = model.module
    vol = model.eval_volume(lattice.coords, lattice.D, lattice.extent, norm)
    mrc.write(out_mrc, vol.astype(np.float32), Apix=Apix)
    torch.save({
        'norm': norm,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, out_weights)


def train(scaler, model, lattice, optim, y, cumulative_weights, rot, trans=None, ctf_params=None, use_amp=False, skip_zeros=False):
    model.train()
    optim.zero_grad()
    B = y.size(0)
    ntilts=y.size(1)
    D = lattice.D

    # create masks to avoid passing DxD frequencies through model
    lattice_mask = lattice.get_circular_mask(D // 2) # reconstruct circle of pixels instead of whole image
    if skip_zeros: # skip zero_weighted components
        weights_mask = (cumulative_weights != 0).int()
        final_mask = lattice_mask.view(1,D,D) * weights_mask # mask is currently ntilts x D x D
        assert np.all(final_mask.reshape(ntilts,-1).cpu().numpy().sum(axis=1)) > 0, 'Masking error caused index error. Please repeat without --skip-zeros'
        raise NotImplementedError
        # TODO divine scheme to skip model(coord) where cumulative_weights[coord]==0
    else:
        final_mask = lattice_mask

    # process real data at masked fourier components
    y = y.view(B*ntilts, -1)[:, final_mask]
    if trans is not None:
        y = lattice.translate_ht(y, trans.unsqueeze(1), final_mask).view(B*ntilts, -1)
    y = y.view(B, ntilts, -1)

    # align and mask cumulative_weights with y and yhat shape
    cumulative_weights = cumulative_weights.view(ntilts, -1)[:, final_mask]
    cumulative_weights = cumulative_weights.view(1, ntilts, -1)

    with autocast(enabled=use_amp):
        # evaluate model at masked fourier components
        yhat = model(lattice.coords[final_mask] @ rot).view(B*ntilts, -1)
        if ctf_params is not None:
            freqs = lattice.freqs2d[final_mask]
            freqs = freqs.unsqueeze(0).expand(B*ntilts, *freqs.shape) / ctf_params[:, 0].view(B*ntilts, 1, 1)
            yhat *= ctf.compute_ctf(freqs, *torch.split(ctf_params[:, 1:], 1, 1))
        yhat = yhat.view(B, ntilts, -1)

        # calculate weighted loss
        loss = ((yhat - y * cumulative_weights) ** 2).mean() # reconstruction loss vs dose+tilt weighted stack

    # backpropogate loss, update model
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    return loss.item()


def save_config(args, dataset, lattice, model, out_config):
    dataset_args = dict(particles=args.particles,
                        norm=dataset.norm,
                        ntilts=dataset.ntilts,
                        nptcls=dataset.nptcls,
                        nimg=dataset.ntilts*dataset.nptcls,
                        cumulative_weights=dataset.cumulative_weights,
                        invert_data=args.invert_data,
                        ind=args.ind,
                        expanded_ind=dataset.expanded_ind,
                        window=args.window,
                        window_r=args.window_r,
                        datadir=args.datadir,
                        ctf=args.ctf,
                        poses=args.poses,
                        do_pose_sgd=args.do_pose_sgd)
    lattice_args = dict(D=lattice.D,
                        extent=lattice.extent,
                        ignore_DC=lattice.ignore_DC)
    model_args = dict(layers=args.layers,
                      dim=args.dim,
                      pe_type=args.pe_type,
                      pe_dim=args.pe_dim,
                      domain=args.domain,
                      activation=args.activation)
    config = dict(dataset_args=dataset_args,
                  lattice_args=lattice_args,
                  model_args=model_args)
    config['seed'] = args.seed
    with open(out_config, 'wb') as f:
        pickle.dump(config, f)
        meta = dict(time=dt.now(),
                    cmd=sys.argv,
                    version=cryodrgn.__version__)
        pickle.dump(meta, f)


def get_latest(args, flog):
    # assumes args.num_epochs > latest checkpoint
    flog('Detecting latest checkpoint...')
    weights = [f'{args.outdir}/weights.{i}.pkl' for i in range(args.num_epochs)]
    weights = [f for f in weights if os.path.exists(f)]
    args.load = weights[-1]
    flog(f'Loading {args.load}')
    if args.do_pose_sgd:
        i = args.load.split('.')[-2]
        args.poses = f'{args.outdir}/pose.{i}.pkl'
        assert os.path.exists(args.poses)
        flog(f'Loading {args.poses}')
    return args


def plot_weight_distribution(cumulative_weights, spatial_frequencies, args):
    # plot distribution of dose weights across tilts in the spirit of https://doi.org/10.1038/s41467-021-22251-8
    # TODO incorporate xlim matching lattice limit representing frequencies actually used for training?
    import matplotlib.pyplot as plt

    ntilts = cumulative_weights.shape[0]
    sorted_frequency_list = sorted(set(spatial_frequencies.reshape(-1)))
    weights_plot = np.empty((len(sorted_frequency_list), ntilts))

    for i, frequency in enumerate(sorted_frequency_list):
        x, y = np.where(spatial_frequencies == frequency)
        sum_of_weights_at_frequency = cumulative_weights[:, y, x].sum()
        weights_plot[i, :] = (cumulative_weights[:, y, x] / sum_of_weights_at_frequency).sum(axis=1) # sum across multiple pixels at same frequency

    colormap = plt.cm.get_cmap('coolwarm').reversed()
    tilt_colors = colormap(np.linspace(0, 1, ntilts))

    fig, ax = plt.subplots()
    ax.stackplot(sorted_frequency_list, weights_plot.T, colors=tilt_colors)
    ax.set_ylabel('cumulative weights')
    ax.set_xlabel('spatial frequency (1/Å)')
    ax.set_xlim((0, sorted_frequency_list[-1]))
    ax.set_ylim((0, 1))
    plt.savefig(f'{args.outdir}/cumulative_weights_across_frequencies_by_tilt.png', dpi=300)


def main(args):
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    LOG = f'{args.outdir}/run.log'

    def flog(msg):  # HACK: switch to logging module
        return utils.flog(msg, LOG)

    if args.load == 'latest':
        args = get_latest(args, flog)
    flog(' '.join(sys.argv))
    flog(args)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## set the device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    flog('Use cuda {}'.format(use_cuda))
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        flog('WARNING: No GPUs detected')

    # load the particle indices
    if args.ind is not None:
        flog('Filtering image dataset with {}'.format(args.ind))
        ind = pickle.load(open(args.ind, 'rb'))
    else:
        ind = None

    # load the particles
    if args.lazy:
        data = dataset.LazyTiltSeriesMRCData(args.particles, norm=args.norm, invert_data=args.invert_data, ind=ind,
                                             window=args.window, datadir=args.datadir, window_r=args.window_r,
                                             do_dose_weighting=args.do_dose_weighting, dose_override=args.dose_override,
                                             do_tilt_weighting = args.do_tilt_weighting)

    else:
        data = dataset.TiltSeriesMRCData(args.particles, norm=args.norm, invert_data=args.invert_data, ind=ind,
                                         window=args.window, datadir=args.datadir, window_r=args.window_r,
                                         do_dose_weighting=args.do_dose_weighting, dose_override=args.dose_override,
                                         do_tilt_weighting = args.do_tilt_weighting)
    D = data.D
    Ntilts = data.ntilts
    Nptcls = data.nptcls
    Nimg = Ntilts*Nptcls
    expanded_ind = data.expanded_ind #this is already filtered by args.ind, np.array of shape (1,)
    cumulative_weights = data.cumulative_weights
    spatial_frequencies = data.spatial_frequencies
    plot_weight_distribution(cumulative_weights, spatial_frequencies, args)

    # instantiate model
    # if args.pe_type != 'none': assert args.l_extent == 0.5
    lattice = Lattice(D, extent=args.l_extent)
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    model = models.get_decoder(3, D, args.layers, args.dim, args.domain, args.pe_type, enc_dim=args.pe_dim,
                               activation=activation)
    flog(model)
    flog('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd) #BMP CHANGED TO AdamW

    # load weights
    if args.load:
        flog('Loading model weights from {}'.format(args.load))
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        assert start_epoch < args.num_epochs
    else:
        start_epoch = 0

    # load poses
    if args.do_pose_sgd:
        assert args.domain == 'hartley', "Need to use --domain hartley if doing pose SGD"
        posetracker = PoseTracker.load(args.poses, Nimg, D, args.emb_type, expanded_ind)
        pose_optimizer = torch.optim.SparseAdam(list(posetracker.parameters()), lr=args.pose_lr)
    else:
        posetracker = PoseTracker.load(args.poses, Nimg, D, None, expanded_ind)

    # load CTF
    if args.ctf is not None:
        flog('Loading ctf params from {}'.format(args.ctf))
        ctf_params = ctf.load_ctf_for_training(D - 1, args.ctf)
        if args.ind is not None: ctf_params = ctf_params[expanded_ind]
        ctf_params = torch.tensor(ctf_params)
    else:
        ctf_params = None
    Apix = ctf_params[0, 0] if ctf_params is not None else 1

    # save configuration
    out_config = f'{args.outdir}/config.pkl'
    save_config(args, data, lattice, model, out_config)

    # Mixed precision training with AMP
    use_amp = args.amp
    flog(f'AMP acceleration enabled (autocast + gradscaler) : {use_amp}')
    # TODO: add warnings if various layers / batchsize / etc not multiples of 8, not optimized

    # parallelize
    if args.multigpu and torch.cuda.device_count() > 1:
        flog(f'Using {torch.cuda.device_count()} GPUs!')
        args.batch_size *= torch.cuda.device_count()
        flog(f'Increasing batch size to {args.batch_size}')
        model = nn.DataParallel(model)
    elif args.multigpu:
        flog(f'WARNING: --multigpu selected, but {torch.cuda.device_count()} GPUs detected')

    # train
    data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    expanded_ind_rebased = torch.tensor([np.arange(i * Ntilts, (i + 1) * Ntilts) for i in range(Nptcls)]).to(device) # redefine training inds to remove gaps created by filtering dataset when loading
    cumulative_weights = torch.tensor(cumulative_weights).to(device)
    scaler = GradScaler(enabled=use_amp)
    for epoch in range(start_epoch, args.num_epochs):
        t2 = dt.now()
        loss_accum = 0
        batch_it = 0
        for batch, ind in data_generator:
            batch_ind = expanded_ind_rebased[ind].view(-1) # need to use expanded indexing for ctf and poses
            batch_it += len(batch_ind) # total number of images seen (+= batchsize*ntilts)
            y = batch.to(device)
            batch_ind = batch_ind.to(device)
            if args.do_pose_sgd:
                pose_optimizer.zero_grad()
            r, t = posetracker.get_pose(batch_ind)
            c = ctf_params[batch_ind] if ctf_params is not None else None
            loss_item = train(scaler, model, lattice, optim, y, cumulative_weights, r, trans=t, ctf_params=c, use_amp=use_amp, skip_zeros=args.skip_zeros)
            if args.do_pose_sgd and epoch >= args.pretrain:
                pose_optimizer.step()
            loss_accum += loss_item * len(batch_ind)
            if batch_it % args.log_interval == 0:
                flog('# [Train Epoch: {}/{}] [{}/{} subtomos] loss={:.6f}'.format(epoch + 1,
                                                                                  args.num_epochs,
                                                                                  int(batch_it / Ntilts),
                                                                                  Nptcls,
                                                                                  loss_item))
        flog('# =====> Epoch: {} Average loss = {:.6}; Finished in {}'.format(epoch + 1,
                                                                              loss_accum / Nimg,
                                                                              dt.now() - t2))
        if args.checkpoint and epoch % args.checkpoint == 0:
            out_mrc = '{}/reconstruct.{}.mrc'.format(args.outdir, epoch)
            out_weights = '{}/weights.{}.pkl'.format(args.outdir, epoch)
            save_checkpoint(model, lattice, optim, epoch, data.norm, Apix, out_mrc, out_weights)
            if args.do_pose_sgd and epoch >= args.pretrain:
                out_pose = '{}/pose.{}.pkl'.format(args.outdir, epoch)
                posetracker.save(out_pose)

    ## save model weights and evaluate the model on 3D lattice
    out_mrc = '{}/reconstruct.mrc'.format(args.outdir)
    out_weights = '{}/weights.pkl'.format(args.outdir)
    save_checkpoint(model, lattice, optim, epoch, data.norm, Apix, out_mrc, out_weights)
    if args.do_pose_sgd and epoch >= args.pretrain:
        out_pose = '{}/pose.pkl'.format(args.outdir)
        posetracker.save(out_pose)

    td = dt.now() - t1
    flog('Finished in {} ({} per epoch)'.format(td, td / (args.num_epochs - start_epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)
