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

import tomodrgn
from tomodrgn import mrc
from tomodrgn import utils
from tomodrgn import dataset
from tomodrgn import ctf
from tomodrgn import models
from tomodrgn import dose

from tomodrgn.lattice import Lattice

log = utils.log
vlog = utils.vlog


def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=200, help='Logging interval in N_PTCLS (default: %(default)s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increaes verbosity')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 100000), help='Random seed')

    group = parser.add_argument_group('Dataset loading')
    group.add_argument('--ind', type=os.path.abspath, help='Filter particle stack by these indices')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.85, help='Windowing radius (default: %(default)s)')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory')

    group.add_argument_group('Tilt series')
    group.add_argument('--do-dose-weighting', action='store_true', help='Flag to calculate losses per tilt per pixel with dose weighting ')
    group.add_argument('--dose-override', type=float, default=None, help='Manually specify dose in e- / A2 / tilt')
    group.add_argument('--do-tilt-weighting', action='store_true', help='Flag to calculate losses per tilt with cosine(tilt_angle) weighting')
    group.add_argument('--enable-trans', action='store_true', help='Apply translations in starfile. Not recommended if using centered + re-extracted particles')
    group.add_argument('--sample-ntilts', type=int, default=None, help='Number of tilts to sample from each particle per epoch. Default: min(ntilts) from dataset')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b', '--batch-size', type=int, default=8, help='Minibatch size (default: %(default)s)')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer (default: %(default)s)')
    group.add_argument('--lr', type=float, default=1e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: mean, std of dataset)')
    group.add_argument('--amp', action='store_true', help='Use mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')

    group = parser.add_argument_group('Network Architecture')
    group.add_argument('--layers', type=int, default=3, help='Number of hidden layers (default: %(default)s)')
    group.add_argument('--dim', type=int, default=256, help='Number of nodes in hidden layers (default: %(default)s)')
    group.add_argument('--pe-type', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf', 'gaussian', 'none'), default='geom_lowf', help='Type of positional encoding')
    group.add_argument('--feat-sigma', type=float, default=0.5, help="Scale for random Gaussian features")
    group.add_argument('--pe-dim', type=int, help='Num sinusoid features in positional encoding (default: D/2)')
    group.add_argument('--activation', choices=('relu', 'leaky_relu'), default='relu', help='Activation (default: %(default)s)')
    group.add_argument('--skip-zeros-decoder', action='store_true', help='Ignore fourier pixels exposed to > 2.5x critical dose')
    group.add_argument('--use-decoder-symmetry', action='store_true', help='Exploit fourier symmetry to only decode half of each image. Reduces GPU memory usage at cost of longer epoch times')
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


def train(scaler, model, lattice, y, rot, tran, weights, dec_mask, optim, ctf_params=None, use_amp=False):
    optim.zero_grad()
    model.train()

    B = y.size(0)
    ntilts=y.size(1)
    D = lattice.D
    y = y[dec_mask].view(1, -1)

    with autocast(enabled=use_amp):
        # evaluate model at masked fourier components
        input_coords = (lattice.coords @ rot).view(B,ntilts,D,D,3)[dec_mask,:].unsqueeze(0)  # shape 1 x dec_mask.flat() x 3, because dec_mask can have variable # elements per particle
        yhat = model(input_coords).view(1,-1) # 1 x B*ntilts*N[final_mask]

        if ctf_params is not None:
            freqs = lattice.freqs2d.unsqueeze(0).expand(B*ntilts,-1,-1) / ctf_params[:,:,1].view(B*ntilts,1,1)
            ctf_weights = ctf.compute_ctf(freqs, *torch.split(ctf_params.view(B*ntilts,-1)[:, 2:], 1, 1)).view(B,ntilts,D,D)
            yhat *= ctf_weights[dec_mask].view(1,-1)

        # calculate weighted loss
        loss = (weights[dec_mask].view(1,-1)*((yhat - y) ** 2)).mean() # reconstruction loss vs input stack weighted by dose+tilt

    # backpropogate loss, update model
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    return loss.item()


def save_config(args, dataset, lattice, model, out_config):
    dataset_args = dict(particles=args.particles,
                        norm=dataset.norm,
                        ntilts=dataset.ntilts_training,
                        invert_data=args.invert_data,
                        ind=args.ind,
                        window=args.window,
                        window_r=args.window_r,
                        datadir=args.datadir)
    lattice_args = dict(D=lattice.D,
                        extent=lattice.extent,
                        ignore_DC=lattice.ignore_DC)
    model_args = dict(layers=args.layers,
                      dim=args.dim,
                      pe_type=args.pe_type,
                      feat_sigma=args.feat_sigma,
                      pe_dim=args.pe_dim,
                      domain='fourier',
                      activation=args.activation,
                      skip_zeros_decoder=args.skip_zeros_decoder,
                      use_decoder_symmetry=args.use_decoder_symmetry)
    training_args = dict(n=args.num_epochs,
                         B=args.batch_size,
                         wd=args.wd,
                         lr=args.lr,
                         amp=args.amp,
                         multigpu=args.multigpu,
                         lazy=args.lazy,
                         do_dose_weighting=args.do_dose_weighting,
                         dose_override=args.dose_override,
                         do_tilt_weighting=args.do_tilt_weighting,
                         verbose=args.verbose,
                         log_interval=args.log_interval,
                         checkpoint=args.checkpoint,
                         outdir=args.outdir)
    config = dict(dataset_args=dataset_args,
                  lattice_args=lattice_args,
                  model_args=model_args,
                  training_args=training_args)
    config['seed'] = args.seed
    with open(out_config, 'wb') as f:
        pickle.dump(config, f)
        meta = dict(time=dt.now(),
                    cmd=sys.argv,
                    version=tomodrgn.__version__)
        pickle.dump(meta, f)


def get_latest(args, flog):
    # assumes args.num_epochs > latest checkpoint
    flog('Detecting latest checkpoint...')
    weights = [f'{args.outdir}/weights.{i}.pkl' for i in range(args.num_epochs)]
    weights = [f for f in weights if os.path.exists(f)]
    args.load = weights[-1]
    flog(f'Loading {args.load}')
    return args



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
    # TODO move this to sbatch
    flog(f'Git revision hash: {utils.check_git_revision_hash("/nobackup/users/bmp/software/tomodrgn/.git")}')

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## set the device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    flog('Use cuda {}'.format(use_cuda))
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        flog('WARNING: No GPUs detected')

    # load the particle indices
    if args.ind is not None:
        flog('Filtering supplied particle indices with {}'.format(args.ind))
        ind = pickle.load(open(args.ind, 'rb'))
    else:
        ind = None

    # load the particles
    if args.lazy:
        raise NotImplementedError
        # data = dataset.LazyTiltSeriesMRCData(args.particles, norm=args.norm, invert_data=args.invert_data, ind=ind,
        #                                      window=args.window, datadir=args.datadir, window_r=args.window_r,
        #                                      do_dose_weighting=args.do_dose_weighting, dose_override=args.dose_override,
        #                                      do_tilt_weighting = args.do_tilt_weighting)

    else:
        data = dataset.TiltSeriesDatasetAllParticles(args.particles, norm=args.norm, invert_data=args.invert_data,
                                                     ind_ptcl=ind, window=args.window, datadir=args.datadir,
                                                     window_r=args.window_r, do_dose_weighting=args.do_dose_weighting,
                                                     dose_override=args.dose_override, do_tilt_weighting=args.do_tilt_weighting)
    D = data.D
    nptcls = data.nptcls
    Apix = data.ptcls[data.ptcls_list[0]].ctf[0,1]

    # instantiate model
    lattice = Lattice(D, extent=0.5)
    if args.lazy:
        raise NotImplementedError
        # flog(f'Pixels per particle (raw data):  {np.prod(data.particles[0][0].get().shape)*Ntilts} ')
    else:
        flog(f'Pixels per particle (first particle): {np.prod(data.ptcls[data.ptcls_list[0]].images.shape)} ')

    # determine which pixels to decode (cache by related weights_dict keys)
    flog(f'Constructing baseline circular lattice for decoder with radius {lattice.D // 2}')
    lattice_mask = lattice.get_circular_mask(lattice.D // 2)  # reconstruct box-inscribed circle of pixels
    for weights_key in data.weights_dict.keys():
        ntilts = data.weights_dict[weights_key].shape[0]
        if args.skip_zeros_decoder:
            # decode pixels with nonzero weights only (can vary by tilt)
            dec_mask = data.weights_dict[weights_key] != 0  # mask is currently ntilts x D x D
            flog(f'Pixels decoded per particle (+ skip-zeros-decoder mask):  {dec_mask.sum()}')
        else:
            # decode pixels within defined circular radius in fourier space (constant for all tilts)
            dec_mask = lattice_mask.view(1, D, D).repeat(ntilts, 1, 1).cpu().numpy()
            flog(f'Pixels decoded per particle (+ fourier circular mask):  {dec_mask.sum()}')
        assert dec_mask.shape == (ntilts, D, D)
        data.dec_mask_dict[weights_key] = dec_mask

    # save plots of all weighting schemes
    for i, weights_key in enumerate(data.weights_dict.keys()):
        weights = data.weights_dict[weights_key]
        dec_mask = data.dec_mask_dict[weights_key]
        spatial_frequencies = data.spatial_frequencies
        dose.plot_weight_distribution(weights * dec_mask, spatial_frequencies, args.outdir, weight_distribution_index=i)

    if args.sample_ntilts:
        assert args.sample_ntilts <= data.ntilts_range[0], \
            f'The number of tilts requested to be sampled per particle ({args.sample_ntilts}) ' \
            f'exceeds the number of tilt images for at least one particle ({data.ntilts_range[0]})'
        data.ntilts_training = args.sample_ntilts
        flog(f'Sampling {args.sample_ntilts} tilts per particle')
    else:
        data.ntilts_training = data.ntilts_range[0]

    # instantiate model
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    model = models.FTPositionalDecoder(3, lattice.D, args.layers, args.dim, activation, args.pe_type, args.pe_dim, args.amp, args.feat_sigma, args.use_decoder_symmetry)
    flog(model)
    flog('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # save configuration
    out_config = f'{args.outdir}/config.pkl'
    save_config(args, data, lattice, model, out_config)

    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Mixed precision training with AMP
    use_amp = args.amp
    flog(f'AMP acceleration enabled (autocast + gradscaler) : {use_amp}')
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        if not args.batch_size % 8 == 0: flog('Warning: recommended to have batch size divisible by 8 for AMP training')
        if not (D - 1) % 8 == 0: flog('Warning: recommended to have image size divisible by 8 for AMP training')
        assert args.dim % 8 == 0, "Decoder hidden layer dimension must be divisible by 8 for AMP training"

    # load weights if restarting from checkpoint
    if args.load:
        flog('Loading model weights from {}'.format(args.load))
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        assert start_epoch < args.num_epochs
        try:
            scaler.load_state_dict(checkpoint['scaler'])
        except:
            flog('No GradScaler instance found in specified checkpoint; creating new GradScaler')
        model.train()
    else:
        start_epoch = 0

    # parallelize
    if args.multigpu and torch.cuda.device_count() > 1:
        flog(f'Using {torch.cuda.device_count()} GPUs!')
        args.batch_size *= torch.cuda.device_count()
        flog(f'Increasing batch size to {args.batch_size}')
        model = nn.DataParallel(model)
    elif args.multigpu:
        flog(f'WARNING: --multigpu selected, but {torch.cuda.device_count()} GPUs detected')
    model.to(device)

    # apply translations as final preprocessing step, since we aren't optimizing poses
    if args.enable_trans:
        flog('Applying translations as final preprocessing step')
        for ptcl_id in data.ptcls_list:
            images = torch.tensor(data.ptcls[ptcl_id].images, device='cpu')
            ntilts = data.ptcls[ptcl_id].ntilts
            trans = torch.tensor(data.ptcls[ptcl_id].trans, device='cpu')
            data.ptcls[ptcl_id].images = lattice.translate_ht(images.view(ntilts, -1), trans.unsqueeze(1))
    else:
        flog('Not applying translations')

    # train
    data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    for epoch in range(start_epoch, args.num_epochs):
        t2 = dt.now()
        loss_accum = 0
        batch_it = 0
        for batch_images, batch_rot, batch_trans, batch_ctf, batch_weights, batch_dec_mask, batch_ptcl_ids in data_generator:
            # impression counting
            batch_it += len(batch_ptcl_ids)

            # training minibatch
            loss_item = train(scaler, model, lattice, batch_images, batch_rot, batch_trans, batch_weights, batch_dec_mask, optim, ctf_params=batch_ctf, use_amp=use_amp)
            loss_accum += loss_item * len(batch_ptcl_ids)
            if batch_it % args.log_interval == 0:
                flog(f'# [Train Epoch: {epoch+1}/{args.num_epochs}] [{batch_it}/{nptcls} subtomos] Average loss={loss_accum / nptcls:.6f}')

        flog(f'# =====> Epoch: {epoch+1} Average loss = {loss_accum/nptcls:.6}; Finished in {dt.now() - t2}')
        if args.checkpoint and epoch % args.checkpoint == 0:
            flog(f'GPU memory usage: {utils.check_memory_usage()}')
            out_mrc = '{}/reconstruct.{}.mrc'.format(args.outdir, epoch)
            out_weights = '{}/weights.{}.pkl'.format(args.outdir, epoch)
            save_checkpoint(model, lattice, optim, epoch, data.norm, Apix, out_mrc, out_weights)

    ## save model weights and evaluate the model on 3D lattice
    out_mrc = '{}/reconstruct.mrc'.format(args.outdir)
    out_weights = '{}/weights.pkl'.format(args.outdir)
    save_checkpoint(model, lattice, optim, epoch, data.norm, Apix, out_mrc, out_weights)

    td = dt.now() - t1
    flog('Finished in {} ({} per epoch)'.format(td, td / (args.num_epochs - start_epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)
