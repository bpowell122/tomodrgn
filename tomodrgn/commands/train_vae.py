'''
Train a VAE for heterogeneous reconstruction with known pose for tomography data
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
from tomodrgn import utils, dataset, ctf, dose
from tomodrgn.models import TiltSeriesHetOnlyVAE
from tomodrgn.lattice import Lattice
from tomodrgn.beta_schedule import get_beta_schedule

log = utils.log
vlog = utils.vlog
# TODO fix logging: everything to flog instead of log, and optional vlog where appropriate

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    parser.add_argument('--zdim', type=int, required=True, help='Dimension of latent variable')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=200, help='Logging interval in N_PTCLS (default: %(default)s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increases verbosity')
    parser.add_argument('--seed', type=int, default=np.random.randint(0,100000), help='Random seed')
    parser.add_argument('--pose', type=os.path.abspath, help='Optionally override star file poses with cryodrgn-format pose.pkl')
    parser.add_argument('--ctf', type=os.path.abspath, help='Optionally override star file CTF with cryodrgn-format ctf.pkl')

    group = parser.add_argument_group('Dataset loading')
    group.add_argument('--ind', type=os.path.abspath, metavar='PKL', help='Filter particle stack by these indices')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.85,  help='Windowing radius')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD)')
    group.add_argument('--Apix', type=float, default=1.0, help='Override A/px from input starfile; useful if starfile does not have _rlnDetectorPixelSize col')

    group.add_argument_group('Tilt series')
    group.add_argument('--recon-tilt-weight', action='store_true', help='Weight reconstruction loss by cosine(tilt_angle)')
    group.add_argument('--recon-dose-weight', action='store_true', help='Weight reconstruction loss per tilt per pixel by dose dependent amplitude attenuation')
    group.add_argument('--dose-override', type=float, default=None, help='Manually specify dose in e- / A2 / tilt')
    group.add_argument('--l-dose-mask', action='store_true', help='Do not train on frequencies exposed to > 2.5x critical dose. Training lattice is intersection of this with --l-extent')
    group.add_argument('--sample-ntilts', type=int, default=None, help='Number of tilts to sample from each particle per epoch. Default: min(ntilts) from dataset')
    group.add_argument('--sequential-tilt-sampling', action='store_true', help='Supply particle images of one particle to encoder in starfile order')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs')
    group.add_argument('-b','--batch-size', type=int, default=1, help='Minibatch size')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer')
    group.add_argument('--lr', type=float, default=0.0002, help='Learning rate in Adam optimizer. Should co-scale linearly with batch size.')
    group.add_argument('--beta', default=None, help='Choice of beta schedule or a constant for KLD weight')
    group.add_argument('--beta-control', type=float, help='KL-Controlled VAE gamma. Beta is KL target')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: 0, std of dataset)')
    group.add_argument('--no-amp', action='store_true', help='Disable use of mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')

    group = parser.add_argument_group('Encoder Network')
    group.add_argument('--enc-layers-A', dest='qlayersA', type=int, default=3, help='Number of hidden layers for each tilt')
    group.add_argument('--enc-dim-A', dest='qdimA', type=int, default=256, help='Number of nodes in hidden layers for each tilt')
    group.add_argument('--out-dim-A', type=int, default=128, help='Number of nodes in output layer of encA == ntilts * number of nodes input to encB')
    group.add_argument('--enc-layers-B', dest='qlayersB', type=int, default=3, help='Number of hidden layers encoding merged tilts')
    group.add_argument('--enc-dim-B', dest='qdimB', type=int, default=512, help='Number of nodes in hidden layers encoding merged tilts')
    group.add_argument('--enc-mask', type=int, help='Circular mask of image for encoder (default: D/2; -1 for no mask)')
    group.add_argument('--pooling-function', type=str, choices=('concatenate', 'max', 'mean', 'median', 'set_encoder'), default='concatenate', help='Function used to pool features along ntilts dimension after encA')

    group = parser.add_argument_group('Set transformer args')
    group.add_argument('--num-seeds', type=int, default=1, help='number of seeds for PMA')
    group.add_argument('--num-heads', type=int, default=4, help='number of heads for multi head attention blocks')
    group.add_argument('--layer-norm', action='store_true', help='whether to apply layer normalization in the set transformer block')

    group = parser.add_argument_group('Decoder Network')
    group.add_argument('--dec-layers', dest='players', type=int, default=3, help='Number of hidden layers')
    group.add_argument('--dec-dim', dest='pdim', type=int, default=256, help='Number of nodes in hidden layers')
    group.add_argument('--l-extent', type=float, default=0.5, help='Coordinate lattice size (if not using positional encoding) (default: %(default)s)')
    group.add_argument('--pe-type', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf', 'gaussian', 'none'), default='gaussian', help='Type of positional encoding')
    group.add_argument('--feat-sigma', type=float, default=0.5, help="Scale for random Gaussian features")
    group.add_argument('--pe-dim', type=int, help='Num features in positional encoding')
    group.add_argument('--activation', choices=('relu','leaky_relu'), default='relu', help='Activation')

    group = parser.add_argument_group('Dataloader arguments')
    group.add_argument('--num-workers', type=int, default=0, help='Number of workers to use when batching particles for training. Has moderate impact on epoch time')
    group.add_argument('--prefetch-factor', type=int, default=2, help='Number of particles to prefetch per worker for training. Has moderate impact on epoch time')
    group.add_argument('--persistent-workers', action='store_true', help='Whether to persist workers after dataset has been fully consumed. Has minimal impact on run time')
    group.add_argument('--pin-memory', action='store_true', help='Whether to use pinned memory for dataloader. Has large impact on epoch time. Recommended.')
    return parser


def get_latest(args, flog):
    # assumes args.num_epochs > latest checkpoint
    flog('Detecting latest checkpoint...')
    weights = [f'{args.outdir}/weights.{i}.pkl' for i in range(args.num_epochs)]
    weights = [f for f in weights if os.path.exists(f)]
    args.load = weights[-1]
    flog(f'Loading {args.load}')
    return args


def save_config(args, dataset, lattice, model, out_config):
    dataset_args = dict(particles=args.particles,
                        norm=dataset.norm,
                        ntilts=dataset.ntilts_training,
                        invert_data=args.invert_data,
                        ind=args.ind,
                        window=args.window,
                        window_r=args.window_r,
                        datadir=args.datadir,
                        sequential_tilt_sampling=args.sequential_tilt_sampling,
                        pose=args.pose,
                        ctf=args.ctf)
    lattice_args = dict(D=lattice.D,
                        extent=lattice.extent,
                        ignore_DC=lattice.ignore_DC)
    model_args = dict(in_dim=model.in_dim.item(),  # .item() converts tensor to int for unpickling on non-gpu systems
                      qlayersA=args.qlayersA,
                      qdimA=args.qdimA,
                      qlayersB=args.qlayersB,
                      qdimB=args.qdimB,
                      out_dimA=args.out_dim_A,
                      players=args.players,
                      pdim=args.pdim,
                      zdim=args.zdim,
                      enc_mask=args.enc_mask,
                      pe_type=args.pe_type,
                      feat_sigma=args.feat_sigma,
                      pe_dim=args.pe_dim,
                      domain='fourier',
                      activation=args.activation,
                      l_dose_mask=args.l_dose_mask,
                      pooling_function=args.pooling_function,
                      num_seeds=args.num_seeds,
                      num_heads=args.num_heads,
                      layer_norm=args.layer_norm)
    training_args = dict(n=args.num_epochs,
                         B=args.batch_size,
                         wd=args.wd,
                         lr=args.lr,
                         beta=args.beta,
                         beta_control=args.beta_control,
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


def train_batch(scaler, model, lattice, y, rot, tran, cumulative_weights, dec_mask, optim, beta, beta_control=None, ctf_params=None, use_amp=False):
    optim.zero_grad()
    model.train()

    B, ntilts, D, D = y.shape

    with autocast(enabled=use_amp):
        if not torch.all(tran == 0):
            y = lattice.translate_ht(y.view(B * ntilts, -1), tran.view(B * ntilts, 1, 2))
        y = y.view(B, ntilts, D, D)

        z_mu, z_logvar, z, y_recon, ctf_weights = run_batch(model, lattice, y, rot, dec_mask, B, ntilts, D, ctf_params)
        loss, gen_loss, kld_loss = loss_function(z_mu, z_logvar, y, y_recon, cumulative_weights, dec_mask, beta, beta_control)

    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    return torch.tensor((loss, gen_loss, kld_loss)).detach().cpu().numpy()


def run_batch(model, lattice, y, rot, dec_mask, B, ntilts, D, ctf_params=None):
    # encode
    input = y.clone()

    if not torch.all(ctf_params == 0):
        # phase flip the CTF-corrupted image
        freqs = lattice.freqs2d.unsqueeze(0).expand(B * ntilts, *lattice.freqs2d.shape) / ctf_params[:,:,1].view(B * ntilts, 1, 1)
        ctf_weights = ctf.compute_ctf(freqs, *torch.split(ctf_params.view(B*ntilts,-1)[:, 2:], 1, 1)).view(B, ntilts, D, D)
        input *= ctf_weights.sign()  # phase flip by the ctf to be all positive amplitudes
    else:
        ctf_weights = None

    z_mu, z_logvar = _unparallelize(model).encode(input, B, ntilts) # ouput is B x zdim, i.e. one value per ptcl (not per img)
    z = _unparallelize(model).reparameterize(z_mu, z_logvar)

    # prepare lattice at masked fourier components
    # TODO reimplement decoding more cleanly and efficiently with NestedTensors when autograd available: https://github.com/pytorch/nestedtensor
    input_coords = lattice.coords @ rot  # shape B x ntilts x D*D x 3 from [D*D x 3] @ [B x ntilts x 3 x 3]
    input_coords = input_coords[dec_mask.view(B, ntilts, D * D), :]  # shape np.sum(dec_mask) x 3
    input_coords = input_coords.unsqueeze(0)  # singleton batch dimension for model to run (dec_mask can have variable # elements per particle, therefore cannot reshape with b > 1)

    # slicing by #pixels per particle after dec_mask is applied to indirectly get correct tensors subset for concatenating z with matching particle coords
    ptcl_pixel_counts = [int(i) for i in torch.sum(dec_mask.view(B, -1), dim=1)]
    input_coords = torch.cat([_unparallelize(model).cat_z(input_coords_ptcl, z[i].unsqueeze(0)) for i, input_coords_ptcl in enumerate(input_coords.split(ptcl_pixel_counts, dim=1))], dim=1)

    # internally reshape such that batch dimension has length > 1, allowing splitting along batch dimension for DataParallel
    pseudo_batchsize = utils.first_n_factors(input_coords.shape[1], lower_bound=8)[0]
    input_coords = input_coords.view(pseudo_batchsize, input_coords.shape[1]//pseudo_batchsize, input_coords.shape[-1])

    # decode
    y_recon = model(input_coords).view(1, -1)  # 1 x B*ntilts*N[final_mask]

    if ctf_weights is not None:
        y_recon *= ctf_weights[dec_mask].view(1,-1)

    return z_mu, z_logvar, z, y_recon, ctf_weights


def _unparallelize(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def loss_function(z_mu, z_logvar, y, y_recon, cumulative_weights, dec_mask, beta, beta_control=None):
    # reconstruction error
    y = y[dec_mask].view(1,-1)
    gen_loss = torch.mean((cumulative_weights[dec_mask].view(1,-1) * ((y_recon - y) ** 2)))

    # latent loss
    kld = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0)
    if beta_control is None:
        kld_loss = beta * kld / dec_mask.sum().float()
    else:
        kld_loss = beta_control * (beta - kld)**2 / dec_mask.sum().float()

    # total loss
    loss = gen_loss + kld_loss
    return loss, gen_loss, kld_loss


def eval_z(model, lattice, data, args, device, use_amp=False):
    model.eval()
    assert not model.training
    with torch.no_grad():
        with autocast(enabled=use_amp):
            z_mu_all = torch.zeros((data.nptcls, model.zdim), device=device, dtype=torch.half if use_amp else torch.float)
            z_logvar_all = torch.zeros((data.nptcls, model.zdim), device=device, dtype=torch.half if use_amp else torch.float)
            data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                        persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
            for batch_images, _, batch_tran, batch_ctf, _, _, batch_indices in data_generator:
                B, ntilts, D, D = batch_images.shape

                # transfer to GPU
                batch_images = batch_images.to(device)
                batch_ctf = batch_ctf.to(device)
                batch_tran = batch_tran.to(device)

                # correct for translations
                if not torch.all(batch_tran == 0):
                    batch_images = lattice.translate_ht(batch_images.view(B * ntilts, -1), batch_tran.view(B * ntilts, 1, 2))
                batch_images = batch_images.view(B, ntilts, D, D)

                # correct for CTF via phase flipping
                if not torch.all(batch_ctf == 0):
                    freqs = lattice.freqs2d.unsqueeze(0).expand(B * ntilts, -1, -1) / batch_ctf[:,:,1].view(B * ntilts, 1, 1)
                    ctf_weights = ctf.compute_ctf(freqs, *torch.split(batch_ctf.view(B*ntilts,-1)[:, 2:], 1, 1)).view(B, ntilts, D, D)
                    batch_images *= ctf_weights.sign()

                z_mu, z_logvar = _unparallelize(model).encode(batch_images, B, ntilts)
                z_mu_all[batch_indices] = z_mu
                z_logvar_all[batch_indices] = z_logvar
            z_mu_all = z_mu_all.cpu().numpy()
            z_logvar_all = z_logvar_all.cpu().numpy()

            if np.any(z_mu_all == np.nan) or np.any(z_mu_all == np.inf):
                nan_count = np.sum(np.isnan(z_mu_all))
                inf_count = np.sum(np.isinf(z_mu_all))
                sys.exit(f'Latent evaluation at end of epoch failed: z.pkl would contain {nan_count} NaN and {inf_count} Inf')

            return z_mu_all, z_logvar_all


def save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z, scaler):
    '''
    Save model weights and latent encoding z
    '''
    # save model weights
    torch.save({
        'epoch':epoch,
        'model_state_dict':_unparallelize(model).state_dict(),
        'optimizer_state_dict':optim.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None
        }, out_weights)
    # save z
    with open(out_z,'wb') as f:
        pickle.dump(z_mu.astype(np.float32), f)
        pickle.dump(z_logvar.astype(np.float32), f)


def main(args):
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    LOG = f'{args.outdir}/run.log'

    def flog(msg): # HACK: switch to logging module
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

    # load the particle indices
    if args.ind is not None:
        flog(f'Reading supplied particle indices {args.ind}')
        ind = pickle.load(open(args.ind, 'rb'))
    else:
        ind = None

    # load the particles + poses + ctf from input starfile
    flog(f'Loading dataset from {args.particles}')
    data = dataset.TiltSeriesMRCData(args.particles, norm=args.norm, invert_data=args.invert_data,
                                     ind_ptcl=ind, window=args.window, datadir=args.datadir,
                                     window_r=args.window_r, recon_dose_weight=args.recon_dose_weight,
                                     dose_override=args.dose_override, recon_tilt_weight=args.recon_tilt_weight,
                                     l_dose_mask=args.l_dose_mask, lazy=args.lazy, sequential_tilt_sampling=args.sequential_tilt_sampling)
    D = data.D
    nptcls = data.nptcls

    if args.pose is not None:
        rot, trans = utils.load_pkl(args.pose)
        rot = rot[data.ptcls_to_imgs_ind.flatten().astype(int)]
        assert rot.shape == (data.nimgs,3,3)
        if trans is not None:
            trans = trans[data.ptcls_to_imgs_ind.flatten().astype(int)]
            assert trans.shape == (data.nimgs,2)
        data.trans = np.asarray(trans, dtype=np.float32)
        data.rot = np.asarray(rot, dtype=np.float32)
    if args.ctf is not None:
        ctf_params = utils.load_pkl(args.ctf)
        ctf_params = ctf_params[data.ptcls_to_imgs_ind.flatten().astype(int)]
        assert ctf_params.shape == (data.nimgs,9)
        data.ctf_params = np.asarray(ctf_params, dtype=np.float32)

    # instantiate lattice
    lattice = Lattice(D, extent=args.l_extent, device=device)

    # determine which pixels to encode (equivalently applicable to all particles)
    if args.enc_mask is None:
        args.enc_mask = D // 2
    if args.enc_mask > 0:
        # encode pixels within defined circular radius in fourier space
        assert args.enc_mask <= D // 2
        enc_mask = lattice.get_circular_mask(args.enc_mask)
        in_dim = enc_mask.sum()
    elif args.enc_mask == -1:
        # encode all pixels in fourier space
        enc_mask = None
        in_dim = lattice.D ** 2 if not args.use_real else (lattice.D - 1) ** 2
    else:
        raise RuntimeError("Invalid argument for encoder mask radius {}".format(args.enc_mask))
    flog(f'Pixels encoded per tilt (+ enc-mask):  {in_dim.sum()}')

    # determine which pixels to decode (cache by related weights_dict keys)
    flog(f'Constructing circular coordinate lattice for decoder with radius {lattice.D // 2} px')
    lattice_mask = lattice.get_circular_mask(lattice.D // 2) # reconstruct box-inscribed circle of pixels
    for i, weights_key in enumerate(data.weights_dict.keys()):
        dec_mask = (data.dec_mask_dict[weights_key] * lattice_mask.view(1, D, D).cpu().numpy()).astype(dtype=bool)  # lattice mask excludes DC and > nyquist frequencies; weights mask excludes > optimal dose
        data.dec_mask_dict[weights_key] = dec_mask
        flog(f'Pixels decoded per particle (--l-dose-mask: {args.l_dose_mask} and --l-extent: {args.l_extent}, mask scheme: {i}):  {dec_mask.sum()}')

        # save plots of all weighting schemes
        weights = data.weights_dict[weights_key]
        spatial_frequencies = data.spatial_frequencies
        dose.plot_weight_distribution(weights * dec_mask, spatial_frequencies, args.outdir, weight_distribution_index=i)

    if args.sample_ntilts:
        assert args.sample_ntilts <= data.ntilts_range[0], \
            f'The number of tilts requested to be sampled per particle ({args.sample_ntilts}) ' \
            f'exceeds the number of tilt images for at least one particle ({data.ntilts_range[0]})'
        data.ntilts_training = args.sample_ntilts
    else:
        data.ntilts_training = data.ntilts_range[0]
    flog(f'Sampling {data.ntilts_training} tilts per particle')

    # instantiate model
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    flog(f'Pooling function prior to encoder B: {args.pooling_function}')
    model = TiltSeriesHetOnlyVAE(lattice, args.qlayersA, args.qdimA, args.out_dim_A, data.ntilts_training,
                                 args.qlayersB, args.qdimB, args.players, args.pdim, in_dim, args.zdim,
                                 enc_mask=enc_mask, enc_type=args.pe_type, enc_dim=args.pe_dim,
                                 domain='fourier', activation=activation, l_dose_mask=args.l_dose_mask,
                                 feat_sigma=args.feat_sigma, pooling_function=args.pooling_function,
                                 num_seeds=args.num_seeds, num_heads=args.num_heads, layer_norm=args.layer_norm)
    # model.to(device)
    flog(model)
    flog('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    flog('{} parameters in encoder'.format(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)))
    flog('{} parameters in decoder'.format(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)))

    # save configuration
    out_config = f'{args.outdir}/config.pkl'
    save_config(args, data, lattice, model, out_config)

    # set beta schedule
    if args.beta is None:
        args.beta = 1./args.zdim
    # try:
    #     args.beta = float(args.beta)
    # except ValueError:
    #     assert args.beta_control, "Need to set beta control weight for schedule {}".format(args.beta)
    beta_schedule = get_beta_schedule(args.beta, n_iterations = args.num_epochs * nptcls + args.batch_size)

    # instantiate optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-4)  # https://github.com/pytorch/pytorch/issues/40497#issuecomment-1084807134

    # Mixed precision training with AMP
    use_amp = not args.no_amp
    flog(f'AMP acceleration enabled (autocast + gradscaler) : {use_amp}')
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        if not args.batch_size % 8 == 0: flog('Warning: recommended to have batch size divisible by 8 for AMP training')
        if not (D-1) % 8 == 0: flog('Warning: recommended to have image size divisible by 8 for AMP training')
        if in_dim % 8 != 0: flog('Warning: recommended to have masked image dimensions divisible by 8 for AMP training')
        assert args.qdimA % 8 == 0, "Encoder hidden layer dimension must be divisible by 8 for AMP training"
        assert args.qdimB % 8 == 0, "Encoder hidden layer dimension must be divisible by 8 for AMP training"
        if args.zdim % 8 != 0: flog('Warning: recommended to have z dimension divisible by 8 for AMP training')
        assert args.pdim % 8 == 0, "Decoder hidden layer dimension must be divisible by 8 for AMP training"

    # restart from checkpoint
    if args.load:
        flog('Loading checkpoint from {}'.format(args.load))
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        assert start_epoch < args.num_epochs
        try:
            scaler.load_state_dict(checkpoint['scaler'])
        except:
            flog('No GradScaler instance found in specified checkpoint; creating new GradScaler')
    else:
        start_epoch = 0
        model.to(device)

    # parallelize
    if args.multigpu and torch.cuda.device_count() > 1:
        log(f'Using {torch.cuda.device_count()} GPUs!')
        args.batch_size *= torch.cuda.device_count()
        log(f'Increasing batch size to {args.batch_size}')
        model = nn.DataParallel(model)
    elif args.multigpu:
        log(f'WARNING: --multigpu selected, but {torch.cuda.device_count()} GPUs detected')

    # train
    flog('Done all preprocessing; starting training now!')
    data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
    for epoch in range(start_epoch, args.num_epochs):
        t2 = dt.now()
        losses_accum = np.zeros((3), dtype=np.float32)
        batch_it = 0

        for batch_images, batch_rot, batch_trans, batch_ctf, batch_weights, batch_dec_mask, batch_indices in data_generator:
            # impression counting
            batch_it += len(batch_indices) # total number of ptcls seen
            global_it = nptcls*epoch + batch_it
            beta = beta_schedule(global_it)

            # transfer to GPU
            batch_images = batch_images.to(device)
            batch_rot = batch_rot.to(device)
            batch_trans = batch_trans.to(device)
            batch_ctf = batch_ctf.to(device)
            batch_weights = batch_weights.to(device)
            batch_dec_mask = batch_dec_mask.to(device)

            # training minibatch
            losses_batch = train_batch(scaler, model, lattice, batch_images, batch_rot, batch_trans,
                                       batch_weights, batch_dec_mask, optim, beta, args.beta_control,
                                       ctf_params=batch_ctf, use_amp=use_amp)

            # logging
            if batch_it % args.log_interval == 0:
                log(f'# [Train Epoch: {epoch+1}/{args.num_epochs}] [{batch_it}/{nptcls} subtomos] gen loss={losses_batch[1]:.6f}, kld={losses_batch[2]:.6f}, beta={beta:.6f}, loss={losses_batch[0]:.6f}')
            losses_accum += losses_batch * len(batch_images)
        flog(f'# =====> Epoch: {epoch+1} Average gen loss = {losses_accum[1]/batch_it:.6f}, KLD = {losses_accum[2]/batch_it:.6f}, total loss = {losses_accum[0]/batch_it:.6f}; Finished in {dt.now()-t2}')
        if args.checkpoint and (epoch+1) % args.checkpoint == 0:
            if device.type != 'cpu':
                flog(f'GPU memory usage: {utils.check_memory_usage()}')
            out_weights = f'{args.outdir}/weights.{epoch}.pkl'
            out_z = f'{args.outdir}/z.{epoch}.pkl'
            z_mu, z_logvar = eval_z(model, lattice, data, args, device, use_amp=use_amp)
            save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z, scaler)

    # save model weights, latent encoding, and evaluate the model on 3D lattice
    out_weights = f'{args.outdir}/weights.pkl'
    out_z = f'{args.outdir}/z.pkl'
    z_mu, z_logvar = eval_z(model, lattice, data, args, device, use_amp=use_amp)
    save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z, scaler)

    td = dt.now() - t1
    flog(f'Finished in {td} ({td / (args.num_epochs - start_epoch)} per epoch)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)



