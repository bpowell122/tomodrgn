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
from tomodrgn import mrc, utils, dataset, ctf, models, starfile
from tomodrgn.lattice import Lattice

log = utils.log
vlog = utils.vlog


def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles_imageseries.star')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS')
    parser.add_argument('--log-interval', type=int, default=200, help='Logging interval in N_PTCLS')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increases verbosity')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 100000), help='Random seed')

    group = parser.add_argument_group('Particle starfile loading and train/test split')
    group.add_argument('--ind', type=os.path.abspath, metavar='PKL', help='Filter starfile by particles (unique rlnGroupName values) using np array pkl')
    group.add_argument('--sequential-tilt-sampling', action='store_true', help='Supply particle images of one particle to encoder in starfile order')
    group.add_argument('--starfile-source', type=str, choices=('warp', 'cistem'), default='warp', help='Software used to extract particles and write star file (sets expected column headers)')

    group = parser.add_argument_group('Dataset loading and preprocessing')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.85, help='Real space inner windowing radius for cosine falloff to radius 1')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--stack-path', type=os.path.abspath, help='For cisTEM image stack only, path to stack.mrc due to file name not present in star file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD)')
    group.add_argument('--Apix', type=float, default=1.0, help='Override A/px from input starfile; useful if starfile does not have _rlnDetectorPixelSize col')

    group.add_argument_group('Weighting and masking')
    group.add_argument('--recon-tilt-weight', action='store_true', help='Weight reconstruction loss by cosine(tilt_angle)')
    group.add_argument('--recon-dose-weight', action='store_true', help='Weight reconstruction loss per tilt per pixel by dose dependent amplitude attenuation')
    group.add_argument('--dose-override', type=float, default=None, help='Manually specify dose in e- / A2 / tilt')
    group.add_argument('--l-dose-mask', action='store_true', help='Do not train on frequencies exposed to > 2.5x critical dose. Training lattice is intersection of this with --l-extent')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs')
    group.add_argument('-b', '--batch-size', type=int, default=1, help='Minibatch size')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer')
    group.add_argument('--lr', type=float, default=0.0002, help='Learning rate in Adam optimizer. Should co-scale linearly with batch size.')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: mean, std of dataset)')
    group.add_argument('--no-amp', action='store_true', help='Disable use of mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs. Specify GPUs i,j via `export CUDA_VISIBLE_DEVICES=i,j` before tomodrgn train_vae')

    group = parser.add_argument_group('Network Architecture')
    group.add_argument('--layers', type=int, default=3, help='Number of hidden layers')
    group.add_argument('--dim', type=int, default=512, help='Number of nodes in hidden layers')
    group.add_argument('--l-extent', type=float, default=0.5, help='Coordinate lattice size (if not using positional encoding)')
    group.add_argument('--pe-type', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf', 'gaussian', 'none'), default='gaussian', help='Type of positional encoding')
    group.add_argument('--pe-dim', type=int, help='Num sinusoid features in positional encoding (default: D/2)')
    group.add_argument('--activation', choices=('relu', 'leaky_relu'), default='relu', help='Activation')
    group.add_argument('--feat-sigma', type=float, default=0.5, help="Scale for random Gaussian features")

    group = parser.add_argument_group('Dataloader arguments')
    group.add_argument('--num-workers', type=int, default=0, help='Number of workers to use when batching particles for training. Has moderate impact on epoch time')
    group.add_argument('--prefetch-factor', type=int, default=2, help='Number of particles to prefetch per worker for training. Has moderate impact on epoch time')
    group.add_argument('--persistent-workers', action='store_true', help='Whether to persist workers after dataset has been fully consumed. Has minimal impact on run time')
    group.add_argument('--pin-memory', action='store_true', help='Whether to use pinned memory for dataloader. Has large impact on epoch time. Recommended.')
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

    B, ntilts, D, D = y.shape

    with autocast(enabled=use_amp):
        # translate input images if translations provided
        if not torch.all(tran == 0):
            y = lattice.translate_ht(y.view(B * ntilts, -1), tran.view(B * ntilts, 1, 2))
            y = y.view(B, ntilts, D, D)
        y = y[dec_mask].view(1, -1)

        # prepare lattice at masked fourier components
        input_coords = lattice.coords @ rot  # shape B x ntilts x D*D x 3
        input_coords = input_coords[dec_mask.view(B, ntilts, D*D), :]  # shape np.sum(dec_mask) x 3
        input_coords = input_coords.unsqueeze(0)  # singleton batch dimension for model to run (dec_mask can have variable # elements per particle, therefore cannot reshape with b > 1)

        # internally reshape such that batch dimension has length > 1, allowing splitting along batch dimension for DataParallel
        pseudo_batchsize = utils.first_n_factors(input_coords.shape[-2], lower_bound=8)[0]
        input_coords = input_coords.view(pseudo_batchsize, -1, 3)

        # evaluate model
        yhat = model(input_coords).view(1,-1) # 1 x B*ntilts*N[final_mask]

        if not torch.all(ctf_params == 0):
            freqs = lattice.freqs2d.unsqueeze(0).expand(B * ntilts, -1, -1) / ctf_params[:,:,1].view(B*ntilts,1,1)  # convert freqs to 1/A
            ctf_weights = ctf.compute_ctf(freqs, *torch.split(ctf_params.view(B*ntilts,-1)[:, 2:], 1, 1)).view(B,ntilts,D,D)
            yhat *= ctf_weights[dec_mask].view(1,-1)

        # calculate weighted loss
        loss = (weights[dec_mask].view(1,-1) * ((yhat - y) ** 2)).mean()  # input loss vs reconstruction differences weighted by ctf and tilt + dose-dependent amplitude attenuation

    # backpropagate loss, update model
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
                        datadir=args.datadir,
                        sequential_tilt_sampling=args.sequential_tilt_sampling)
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
                      l_dose_mask=args.l_dose_mask)
    training_args = dict(n=args.num_epochs,
                         B=args.batch_size,
                         wd=args.wd,
                         lr=args.lr,
                         amp=not args.no_amp,
                         multigpu=args.multigpu,
                         lazy=args.lazy,
                         recon_dose_weight=args.recon_dose_weight,
                         dose_override=args.dose_override,
                         recon_tilt_weight=args.recon_tilt_weight,
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

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## set the device
    device = utils.get_default_device()
    if device == torch.device('cpu'):
        args.no_amp = True
        flog('Warning: pytorch AMP does not support non-CUDA (e.g. cpu) devices. Automatically disabling AMP and continuing')
        # https://github.com/pytorch/pytorch/issues/55374

    # load star file
    flog(f'Reading star file with format: {args.starfile_source}')
    if args.starfile_source == 'warp':
        ptcls_star = starfile.TiltSeriesStarfile.load(args.particles)
    elif args.starfile_source == 'cistem':
        ptcls_star = starfile.TiltSeriesStarfileCisTEM.load(args.particles, args.stack_path)
        ptcls_star.convert_to_relion_conventions()
    else:
        raise NotImplementedError
    ptcls_to_imgs_ind = ptcls_star.get_ptcl_img_indices()
    ptcls_star.plot_particle_uid_ntilt_distribution(args.outdir)

    # load the particle indices
    if args.ind is not None:
        flog('Filtering supplied particle indices with {}'.format(args.ind))
        ind = pickle.load(open(args.ind, 'rb'))
        assert len(ptcls_to_imgs_ind) >= len(ind), 'More particles specified than particles found in star file'
        assert len(ptcls_to_imgs_ind) >= max(ind), 'Specified particles exceed the number of particles found in star file'
    else:
        ind = None

    # load the particles
    data = dataset.TiltSeriesMRCData(ptcls_star,
                                     norm=args.norm,
                                     invert_data=args.invert_data,
                                     ind_ptcl=ind,
                                     ind_img=None,
                                     window=args.window,
                                     datadir=args.datadir,
                                     window_r=args.window_r,
                                     recon_dose_weight=args.recon_dose_weight,
                                     dose_override=args.dose_override,
                                     recon_tilt_weight=args.recon_tilt_weight,
                                     l_dose_mask=args.l_dose_mask,
                                     lazy=args.lazy,
                                     sequential_tilt_sampling=args.sequential_tilt_sampling)
    D = data.D
    nptcls = data.nptcls
    Apix = data.ctf_params[0,1] if data.ctf_params is not None else args.Apix
    flog(f'Pixels per particle: {data.D ** 2 * data.ntilts_training} ')

    # instantiate lattice
    lattice = Lattice(D, extent=args.l_extent, device=device)

    # instantiate model
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    model = models.FTPositionalDecoder(3, lattice.D, args.layers, args.dim, activation,
                                       enc_type=args.pe_type, enc_dim=args.pe_dim, feat_sigma=args.feat_sigma)
    model.to(device)
    flog(model)
    flog('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # save configuration
    out_config = f'{args.outdir}/config.pkl'
    save_config(args, data, lattice, model, out_config)

    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Mixed precision training with AMP
    use_amp = not args.no_amp
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

    # train
    flog('Done all preprocessing; starting training now!')
    data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
    for epoch in range(start_epoch, args.num_epochs):
        t2 = dt.now()
        loss_accum = 0
        batch_it = 0
        for batch_images, batch_rot, batch_trans, batch_ctf, batch_decoder_weights, batch_decoder_mask, batch_indices in data_generator:
            # impression counting
            batch_it += len(batch_images)

            # transfer to GPU
            batch_images = batch_images.to(device)
            batch_rot = batch_rot.to(device)
            batch_trans = batch_trans.to(device)
            batch_ctf = batch_ctf.to(device)
            batch_decoder_weights = batch_decoder_weights.to(device)
            batch_decoder_mask = batch_decoder_mask.to(device)

            # training minibatch
            loss_item = train(scaler, model, lattice, batch_images, batch_rot, batch_trans,
                              batch_decoder_weights, batch_decoder_mask, optim,
                              ctf_params=batch_ctf, use_amp=use_amp)
            loss_accum += loss_item * len(batch_images)
            if batch_it % args.log_interval == 0:
                flog(f'# [Train Epoch: {epoch+1}/{args.num_epochs}] [{batch_it}/{nptcls} particles]  loss={loss_item:.6f}')

        flog(f'# =====> Epoch: {epoch+1} Average loss = {loss_accum/batch_it:.6}; Finished in {dt.now() - t2}')
        if args.checkpoint and epoch % args.checkpoint == 0:
            if device.type != 'cpu':
                flog(f'GPU memory usage: {utils.check_memory_usage()}')
            out_mrc = f'{args.outdir}/reconstruct.{epoch}.mrc'
            out_weights = f'{args.outdir}/weights.{epoch}.pkl'
            save_checkpoint(model, lattice, optim, epoch, data.norm, Apix, out_mrc, out_weights)

    ## save model weights and evaluate the model on 3D lattice
    out_mrc = f'{args.outdir}/reconstruct.mrc'
    out_weights = f'{args.outdir}/weights.pkl'
    save_checkpoint(model, lattice, optim, epoch, data.norm, Apix, out_mrc, out_weights)

    td = dt.now() - t1
    flog(f'Finished in {td} ({td / (args.num_epochs - start_epoch)} per epoch)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)
