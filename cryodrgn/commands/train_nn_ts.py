'''
Train a NN to model a 3D density map given 2D images from a tilt series with pose assignments
'''
import numpy as np
import sys, os
import argparse
import pickle
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import apex.amp as amp
except:
    pass

import cryodrgn
from cryodrgn import mrc
from cryodrgn import utils
from cryodrgn import fft
from cryodrgn import dataset
from cryodrgn import ctf
from cryodrgn import models

from cryodrgn.pose import PoseTracker
from cryodrgn.lattice import Lattice

log = utils.log
vlog = utils.vlog


def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    parser.add_argument('--poses', type=os.path.abspath, required=True, help='Image poses (.pkl)')
    parser.add_argument('--ctf', metavar='pkl', type=os.path.abspath, help='CTF parameters (.pkl)')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=1,
                        help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='Logging interval in N_IMGS (default: %(default)s)')
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

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b', '--batch-size', type=int, default=8, help='Minibatch size (default: %(default)s)')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer (default: %(default)s)')
    group.add_argument('--lr', type=float, default=1e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: mean, std of dataset)')
    group.add_argument('--amp', action='store_true', help='Use mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')

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


def train(model, lattice, optim, y, rot, trans=None, ctf_params=None, use_amp=False):
    model.train()
    optim.zero_grad()
    B = y.size(0)
    ntilts=y.size(1)
    log(f'B = {B}')
    log(f'ntilts={ntilts}')
    D = lattice.D
    # reconstruct circle of pixels instead of whole image
    mask = lattice.get_circular_mask(D // 2)
    yhat = model(lattice.coords[mask] @ rot).view(B*ntilts, -1)
    if ctf_params is not None:
        freqs = lattice.freqs2d[mask]
        freqs = freqs.unsqueeze(0).expand(B*ntilts, *freqs.shape) / ctf_params[:, 0].view(B*ntilts, 1, 1)
        yhat *= ctf.compute_ctf(freqs, *torch.split(ctf_params[:, 1:], 1, 1))
    y = y.view(B*ntilts, -1)[:, mask]
    if trans is not None:
        y = lattice.translate_ht(y, trans.unsqueeze(1), mask).view(B*ntilts, -1)
    y = y.view(B, ntilts, -1)
    yhat = yhat.view(B, ntilts, -1)
    loss = F.mse_loss(yhat, y)
    if use_amp:
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optim.step()
    return loss.item()


def save_config(args, dataset, lattice, model, out_config):
    dataset_args = dict(particles=args.particles,
                        norm=dataset.norm,
                        invert_data=args.invert_data,
                        ind=args.ind,
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


def get_latest(args):
    # assumes args.num_epochs > latest checkpoint
    log('Detecting latest checkpoint...')
    weights = [f'{args.outdir}/weights.{i}.pkl' for i in range(args.num_epochs)]
    weights = [f for f in weights if os.path.exists(f)]
    args.load = weights[-1]
    log(f'Loading {args.load}')
    if args.do_pose_sgd:
        i = args.load.split('.')[-2]
        args.poses = f'{args.outdir}/pose.{i}.pkl'
        assert os.path.exists(args.poses)
        log(f'Loading {args.poses}')
    return args

def indptcl_ntilt_handler(ind, expanded_ind):
    # Helper function to broadcast ind filtering (per unique particle) to all tilt images of that particle
    # expanded_ind = np.array([])
    # for i in ind:
    #     expanded_ind = np.append(expanded_ind, np.arange(i * ntilts, (i + 1) * ntilts)).astype('int')
    # return expanded_ind

    # Alternative implementation: given n_imgs and n_tilts, return dictionary of {ind_particle : ind_array_particle_imgs}
    # expanded_ind = {i : np.arange(i * ntilts, (i + 1) * ntilts) for i in np.arange(nptcls)}
    # return expanded_ind

    # Alternative: create dictionary of int ind mappings once, then parse dict for torch.tensor(ind)
    batch_ind = [expanded_ind[i] for i in ind]
    return torch.tensor(batch_ind)

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

    # load the particles
    #TODO separate filtering for ind_particles.pkl (_rlnGroupName) and ind_tilts.pkl (_tiltangle)
    if args.ind is not None:
        flog('Filtering image dataset with {}'.format(args.ind))
        ind = pickle.load(open(args.ind, 'rb'))
    else:
        ind = None

    if args.lazy:
        raise NotImplementedError

    else:
        data = dataset.TiltSeriesMRCDataset(args.particles, norm=args.norm, invert_data=args.invert_data, ind=ind,
                               window=args.window, datadir=args.datadir, window_r=args.window_r)
    D = data.D
    Ntilts = data.ntilts
    Nptcls = data.nptcls
    Nimg = Ntilts*Nptcls
    # expanded_ind = {i: np.arange(i * Ntilts, (i + 1) * Ntilts) for i in np.arange(Nptcls)} # dictionary {ind_ptcl : inds_imgs}
    expanded_ind = torch.tensor([np.arange(i * Ntilts, (i + 1) * Ntilts) for i in np.arange(Nptcls)]) # 2D numpy array [(ind_ptcl, inds_imgs)]

    # instantiate model
    # if args.pe_type != 'none': assert args.l_extent == 0.5
    lattice = Lattice(D, extent=args.l_extent)

    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    model = models.get_decoder(3, D, args.layers, args.dim, args.domain, args.pe_type, enc_dim=args.pe_dim,
                               activation=activation)
    flog(model)
    flog('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

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
        posetracker = PoseTracker.load(args.poses, Nimg, D, args.emb_type, ind)
        pose_optimizer = torch.optim.SparseAdam(list(posetracker.parameters()), lr=args.pose_lr)
    else:
        posetracker = PoseTracker.load(args.poses, Nimg, D, None, ind)

    # load CTF
    if args.ctf is not None:
        flog('Loading ctf params from {}'.format(args.ctf))
        ctf_params = ctf.load_ctf_for_training(D - 1, args.ctf)
        if args.ind is not None: ctf_params = ctf_params[ind]
        ctf_params = torch.tensor(ctf_params)
    else:
        ctf_params = None
    Apix = ctf_params[0, 0] if ctf_params is not None else 1

    # save configuration
    out_config = f'{args.outdir}/config.pkl'
    save_config(args, data, lattice, model, out_config)

    # Mixed precision training with AMP
    if args.amp:
        assert args.batch_size % 8 == 0
        assert (D - 1) % 8 == 0
        assert args.dim % 8 == 0
        # Also check zdim, enc_mask dim?
        model, optim = amp.initialize(model, optim, opt_level='O1')

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
    expanded_ind = expanded_ind.to(device)
    for epoch in range(start_epoch, args.num_epochs):
        t2 = dt.now()
        loss_accum = 0
        batch_it = 0
        for batch, ind in data_generator:
            batch_ind = expanded_ind[ind] #need to use expanded indexing for ctf and poses
            batch_it += len(batch_ind)
            y = batch.to(device)
            batch_ind = batch_ind.to(device)
            if args.do_pose_sgd:
                pose_optimizer.zero_grad()
            r, t = posetracker.get_pose(batch_ind)
            c = ctf_params[batch_ind] if ctf_params is not None else None
            log(f'batch_ind.shape: {batch_ind.shape}')
            log(f'rot.shape: {r.shape}')
            log(f'trans.shape: {t.shape}')
            log(f'ctf.shape: {c.shape}')
            loss_item = train(model, lattice, optim, y, r, trans=t, ctf_params=c, use_amp=args.amp)
            if args.do_pose_sgd and epoch >= args.pretrain:
                pose_optimizer.step()
            loss_accum += loss_item * len(batch_ind)
            if batch_it % args.log_interval == 0:
                flog('# [Train Epoch: {}/{}] [{}/{} subtomos] loss={:.6f}'.format(epoch + 1, args.num_epochs, batch_it,
                                                                                Nptcls, loss_item))
        flog('# =====> Epoch: {} Average loss = {:.6}; Finished in {}'.format(epoch + 1, loss_accum / Nimg,
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
