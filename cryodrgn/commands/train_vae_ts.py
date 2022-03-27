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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import cryodrgn
from cryodrgn import utils
from cryodrgn import dataset
from cryodrgn import ctf

from cryodrgn.pose import PoseTracker
from cryodrgn.models import TiltSeriesHetOnlyVAE
from cryodrgn.lattice import Lattice
from cryodrgn.beta_schedule import get_beta_schedule
try:
    import apex.amp as amp
except:
    pass

log = utils.log
vlog = utils.vlog
# TODO fix logging: everything to flog instead of log, and optional vlog where appropriate

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    parser.add_argument('--zdim', type=int, required=True, help='Dimension of latent variable')
    parser.add_argument('--poses', type=os.path.abspath, required=True, help='Image poses (.pkl)')
    parser.add_argument('--ctf', metavar='pkl', type=os.path.abspath, help='CTF parameters (.pkl)')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='Logging interval in N_PTCLS (default: %(default)s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increaes verbosity')
    parser.add_argument('--seed', type=int, default=np.random.randint(0,100000), help='Random seed')

    group = parser.add_argument_group('Dataset loading')
    group.add_argument('--ind', type=os.path.abspath, metavar='PKL', help='Filter particle stack by these indices')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.85,  help='Windowing radius (default: %(default)s)')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--relion31', action='store_true', help='Flag if relion3.1 star format')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD)')

    group = parser.add_argument_group('Tilt series')
    group.add_argument('--do-dose-weighting', action='store_true', help='Flag to calculate losses per tilt per pixel with dose weighting ')
    group.add_argument('--dose-override', type=float, default=None, help='Manually specify dose in e- / A2 / tilt')
    group.add_argument('--do-tilt-weighting', action='store_true', help='Flag to calculate losses per tilt with cosine(tilt_angle) weighting')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b','--batch-size', type=int, default=8, help='Minibatch size (default: %(default)s)')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer (default: %(default)s)')
    group.add_argument('--lr', type=float, default=1e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--beta', default=None, help='Choice of beta schedule or a constant for KLD weight (default: 1/zdim)')
    group.add_argument('--beta-control', type=float, help='KL-Controlled VAE gamma. Beta is KL target. (default: %(default)s)')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: 0, std of dataset)')
    group.add_argument('--amp', action='store_true', help='Use mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')

    group = parser.add_argument_group('Encoder Network')
    group.add_argument('--enc-layers-A', dest='qlayersA', type=int, default=3, help='Number of hidden layers for each tilt(default: %(default)s)')
    group.add_argument('--enc-dim-A', dest='qdimA', type=int, default=256, help='Number of nodes in hidden layers for each tilt (default: %(default)s)')
    group.add_argument('--enc-layers-B', dest='qlayersB', type=int, default=1, help='Number of hidden layers encoding merged tilts (default: %(default)s)')
    group.add_argument('--enc-dim-B', dest='qdimB', type=int, default=256, help='Number of nodes in hidden layers encoding merged tilts (default: %(default)s)')
    # group.add_argument('--encode-mode', default='tiltseries', choices=('resid','mlp', 'tiltseries'), help='Type of encoder network (default: %(default)s)')
    # group.add_argument('--enc-mask', type=int, help='Circular mask of image for encoder (default: D/2; -1 for no mask)')
    # group.add_argument('--use-real', action='store_true', help='Use real space image for encoder (for convolutional encoder)')
    group.add_argument('--skip-zeros-encoder', action='store_true', help='Ignore fourier pixels exposed to > 2.5x critical dose when encoding latent space')
    group.add_argument('--weight-encoder', action='store_true', help='Apply dose/tilt pixel weighting scheme when encoding images to latent space')

    group = parser.add_argument_group('Decoder Network')
    group.add_argument('--dec-layers', dest='players', type=int, default=3, help='Number of hidden layers (default: %(default)s)')
    group.add_argument('--dec-dim', dest='pdim', type=int, default=256, help='Number of nodes in hidden layers (default: %(default)s)')
    group.add_argument('--pe-type', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf','none'), default='geom_lowf', help='Type of positional encoding (default: %(default)s)')
    group.add_argument('--pe-dim', type=int, help='Num features in positional encoding (default: image D)')
    group.add_argument('--domain', choices=('hartley','fourier'), default='fourier', help='Decoder representation domain (default: %(default)s)')
    group.add_argument('--activation', choices=('relu','leaky_relu'), default='relu', help='Activation (default: %(default)s)')
    group.add_argument('--skip-zeros-decoder', action='store_true', help='Ignore fourier pixels exposed to > 2.5x critical dose when reconstructing image for MSE loss')
    return parser


def plot_weight_distribution(cumulative_weights, spatial_frequencies, args):
    # plot distribution of dose weights across tilts in the spirit of https://doi.org/10.1038/s41467-021-22251-8
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    ntilts = cumulative_weights.shape[0]
    max_frequency_box_edge = spatial_frequencies[0,spatial_frequencies.shape[0]//2]
    sorted_frequency_list = sorted(set(spatial_frequencies[spatial_frequencies < max_frequency_box_edge].reshape(-1)))
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

    # ax.xaxis.set_major_locator(mticker.MaxNLocator())
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels([f'1/{1/xtick:.1f}' if xtick != 0.0 else 0 for xtick in ticks_loc])

    plt.savefig(f'{args.outdir}/cumulative_weights_across_frequencies_by_tilt.png', dpi=300)


def get_latest(args):
    # assumes args.num_epochs > latest checkpoint
    log('Detecting latest checkpoint...')
    weights = [f'{args.outdir}/weights.{i}.pkl' for i in range(args.num_epochs)]
    weights = [f for f in weights if os.path.exists(f)]
    args.load = weights[-1]
    log(f'Loading {args.load}')
    return args


def save_config(args, dataset, lattice, model, out_config):
    dataset_args = dict(particles=args.particles,
                        norm=dataset.norm,
                        ntilts=dataset.ntilts,
                        # nptcls=dataset.nptcls,
                        # nimg=dataset.ntilts*dataset.nptcls,
                        # cumulative_weights=dataset.cumulative_weights,
                        invert_data=args.invert_data,
                        ind=args.ind,
                        expanded_ind=dataset.expanded_ind,
                        # keepreal=args.use_real,
                        window=args.window,
                        window_r=args.window_r,
                        datadir=args.datadir,
                        ctf=args.ctf,
                        poses=args.poses,
                        relion31=args.relion31)
    lattice_args = dict(D=lattice.D,
                        extent=lattice.extent,
                        ignore_DC=lattice.ignore_DC)
    model_args = dict(in_dim=model.in_dim,
                      qlayersA=args.qlayersA,
                      qdimA=args.qdimA,
                      qlayersB=args.qlayersB,
                      qdimB=args.qdimB,
                      players=args.players,
                      pdim=args.pdim,
                      zdim=args.zdim,
                      # encode_mode=args.encode_mode,
                      # enc_mask=args.enc_mask,
                      pe_type=args.pe_type,
                      pe_dim=args.pe_dim,
                      domain=args.domain,
                      activation=args.activation,
                      skip_zeros_encoder=args.skip_zeros_encoder,
                      skip_zeros_decoder=args.skip_zeros_decoder)
    training_args = dict(n=args.num_epochs,
                         B=args.batch_size,
                         wd=args.wd,
                         lr=args.lr,
                         beta=args.beta,
                         beta_control=args.beta_control,
                         amp=args.amp,
                         multigpu=args.multigpu,
                         lazy=args.lazy,
                         do_dose_weighting=args.do_dose_weighting,
                         dose_override=args.dose_override,
                         do_tilt_weighting=args.do_tilt_weighting,
                         weight_encoder=args.weight_encoder,
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
                    version=cryodrgn.__version__)
        pickle.dump(meta, f)


def train_batch(scaler, model, lattice, y, cumulative_weights_encoder, cumulative_weights_decoder, final_mask_encoder, final_mask_decoder, rot, trans, optim, beta, beta_control=None, ctf_params=None, use_amp=False, weight_encoder=False):
    optim.zero_grad()
    model.train()

    B = y.size(0)
    ntilts = y.size(1)
    D = lattice.D

    with autocast(enabled=use_amp):
        y = preprocess_input(y, lattice, trans, final_mask_encoder, B, ntilts, D, ctf_params)
        z_mu, z_logvar, z, y_recon, ctf_weights = run_batch(model, lattice, y, rot, cumulative_weights_encoder, final_mask_encoder, final_mask_decoder, B, ntilts, D, ctf_params, weight_encoder=weight_encoder)
        loss, gen_loss, kld = loss_function(z_mu, z_logvar, y, ctf_weights, y_recon, cumulative_weights_decoder, final_mask_decoder, beta, beta_control)

    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    return loss.item(), gen_loss.item(), kld.item()


def preprocess_input(y, lattice, trans, final_mask_encoder, B, ntilts, D, ctf_params=None, weight_encoder = False):
    if trans is not None:
        # center the image
        y = lattice.translate_ht(y.view(B*ntilts,-1), trans.unsqueeze(1))
    y = y.view(B, ntilts, D, D)
    # return translated   ####, CTF-phase-flipped, dose+tilt-weighted, masked### particle stack for encoder+decoder+loss calculation
    return y ###, ctf_weights


def run_batch(model, lattice, y, rot, cumulative_weights_encoder, final_mask_encoder, final_mask_decoder, B, ntilts, D, ctf_params=None, weight_encoder=False):
    # encode
    # TODO use final_mask_encoder if skip_zeros_encoder to replace 0-weight pixels with 0-value for encoder so that encoder model can be shared for all tilts - or would this be inaccurate to model?
    input = y[:, final_mask_encoder]
    if ctf_params is not None:
        # phase flip the CTF-corrupted image
        # freqs = lattice.freqs2d.unsqueeze(0).expand(B*ntilts, *lattice.freqs2d.shape) / ctf_params[:, 0].view(B*ntilts, 1, 1)
        # c = ctf.compute_ctf(freqs, *torch.split(ctf_params[:, 1:], 1, 1)).view(B, ntilts, D, D)
        freqs = lattice.freqs2d.unsqueeze(0).expand(B * ntilts, -1, -1) / ctf_params[:, 0].view(B * ntilts, 1, 1)
        ctf_weights = ctf.compute_ctf(freqs, *torch.split(ctf_params[:, 1:], 1, 1)).view(B, ntilts, D, D)
        input *= ctf_weights[:, final_mask_encoder].sign() # phase flip by the ctf to be all positive amplitudes
    else:
        ctf_weights = None
    if weight_encoder:
        # apply cumulative_weights when encoding the image
        input *= cumulative_weights_encoder
    z_mu, z_logvar = _unparallelize(model).encode(input, B, ntilts) # ouput is B x zdim, i.e. one value per ptcl (not per img)
    z = _unparallelize(model).reparameterize(z_mu, z_logvar)

    # decode
    # mask = lattice.get_circular_mask(D // 2)  # restrict to circular mask
    # y_recon = model(lattice.coords[mask] / lattice.extent / 2 @ rot, z).view(B*ntilts, -1)
    input_coords = (lattice.coords @ rot).view(B, ntilts, D, D, 3)[:, final_mask_decoder, :]
    y_recon = model(input_coords, z).view(B, -1)  # B x ntilts*N[final_mask]
    # if c is not None:
    #     y_recon *= c.view(B*ntilts, -1)[:, mask]
    if ctf_weights is not None:
        y_recon *= ctf_weights[:,final_mask_decoder]

    return z_mu, z_logvar, z, y_recon, ctf_weights


def _unparallelize(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def loss_function(z_mu, z_logvar, y, ctf_weights, y_recon, cumulative_weights_decoder, final_mask_decoder, beta, beta_control=None):
    # reconstruction error
    # B = y.size(0)
    # ntilts = y.size(1)
    # y *= ctf_weights.sign() # undo phase flipping from minibatch preprocessing
    y = y[:,final_mask_decoder]
    # gen_loss = F.mse_loss(y_recon, y.view(B*ntilts,-1)[:, mask])
    gen_loss = (cumulative_weights_decoder.view(1, -1) * ((y_recon - y) ** 2)).mean()

    # latent loss
    kld = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0)

    # total loss
    if beta_control is None:
        loss = gen_loss + beta*kld/final_mask_decoder.sum().float()
    else:
        loss = gen_loss + beta_control*(beta-kld)**2/final_mask_decoder.sum().float()
    return loss, gen_loss, kld


def eval_z(model, lattice, data, cumulative_weights_encoder, final_mask_encoder, expanded_ind_rebased, batch_size, device, trans=None, ctf_params=None, weight_encoder=False):
    assert not model.training
    z_mu_all = []
    z_logvar_all = []
    data_generator = DataLoader(data, batch_size=batch_size, shuffle=False)
    for batch, ind in data_generator:
        batch_ind = expanded_ind_rebased[ind].view(-1)  # need to use expanded indexing for ctf and poses
        y = batch.to(device)
        B = y.size(0)
        ntilts = y.size(1)
        D = lattice.D
        if trans is not None:
            y = lattice.translate_ht(y.view(B*ntilts,-1), trans[batch_ind].unsqueeze(1)).view(B,ntilts,D,D)
        y = y.view(B, ntilts, D, D)[:, final_mask_encoder]
        if ctf_params is not None:
            freqs = lattice.freqs2d.unsqueeze(0).expand(B * ntilts, -1, -1) / ctf_params[batch_ind, 0].view(B * ntilts, 1, 1)
            ctf_weights = ctf.compute_ctf(freqs, *torch.split(ctf_params[batch_ind, 1:], 1, 1)).view(B, ntilts, D, D)[:,final_mask_encoder]
            y *= ctf_weights.sign() # phase flip by the ctf
        if weight_encoder:
            y *= cumulative_weights_encoder
        z_mu, z_logvar = _unparallelize(model).encode(y, B, ntilts)
        z_mu_all.append(z_mu.detach().cpu().numpy())
        z_logvar_all.append(z_logvar.detach().cpu().numpy())
    z_mu_all = np.vstack(z_mu_all)
    z_logvar_all = np.vstack(z_logvar_all)
    return z_mu_all, z_logvar_all


def save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z):
    '''
    Save model weights and latent encoding z
    '''
    # save model weights
    torch.save({
        'epoch':epoch,
        'model_state_dict':_unparallelize(model).state_dict(),
        'optimizer_state_dict':optim.state_dict(),
        }, out_weights)
    # save z
    with open(out_z,'wb') as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)


def check_memory_usage(flog):
    import subprocess
    gpu_memory_usage = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8')\
        .strip().split('\n')
    flog(f'GPU memory usage (MB): {[device for device in gpu_memory_usage]}')


def main(args):
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    LOG = f'{args.outdir}/run.log'

    def flog(msg): # HACK: switch to logging module
        return utils.flog(msg, LOG)

    if args.load == 'latest':
        args = get_latest(args)
    flog(' '.join(sys.argv))
    flog(args)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set the device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    flog('Use cuda {}'.format(use_cuda))
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        log('WARNING: No GPUs detected')

    # set beta schedule
    if args.beta is None:
        args.beta = 1./args.zdim
    try:
        args.beta = float(args.beta)
    except ValueError:
        assert args.beta_control, "Need to set beta control weight for schedule {}".format(args.beta)
    beta_schedule = get_beta_schedule(args.beta)

    # load the particle indices
    if args.ind is not None:
        flog('Filtering image dataset with {}'.format(args.ind))
        ind = pickle.load(open(args.ind, 'rb'))
    else:
        ind = None

    # load the particles
    flog(f'Loading dataset from {args.particles}')
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

    # load poses
    posetracker = PoseTracker.load(args.poses, Nimg, D, None, expanded_ind)

    # load CTF
    if args.ctf is not None:
        flog('Loading ctf params from {}'.format(args.ctf))
        ctf_params = ctf.load_ctf_for_training(D - 1, args.ctf)
        if args.ind is not None: ctf_params = ctf_params[expanded_ind]
        assert ctf_params.shape == (Nimg, 8)
        ctf_params = torch.tensor(ctf_params)
    else:
        ctf_params = None
    Apix = ctf_params[0, 0] if ctf_params is not None else 1

    # instantiate lattice
    lattice = Lattice(D, extent=0.5)

    # create dataset mask for which pixels will be evaluated
    flog(f'Pixels per particle (input):  {np.prod(data.particles[0].shape)} ')
    lattice_mask = lattice.get_circular_mask(lattice.D // 2) # reconstruct box-inscribed circle of pixels
    flog(f'Pixels per particle (+ fourier circular mask):  {Ntilts*lattice_mask.sum()}')
    if args.skip_zeros_encoder: # skip zero_weighted components when encoding particles
        weights_mask = cumulative_weights != 0
        final_mask_encoder = lattice_mask.view(1,D,D).cpu().numpy() * weights_mask # mask is currently ntilts x D x D
        flog(f'Pixels per particle (+ skip-zeros-encoder mask):  {final_mask_encoder.sum()}')
    else:
        final_mask_encoder = lattice_mask.view(1,D,D).repeat(Ntilts,1,1).cpu().numpy()
    assert final_mask_encoder.shape == (Ntilts, D, D)
    # mask cumulative weights encoder by final_mask
    cumulative_weights_encoder = cumulative_weights[final_mask_encoder]

    if args.skip_zeros_decoder: # skip zero_weighted components when encoding particles
        weights_mask = cumulative_weights != 0
        final_mask_decoder = lattice_mask.view(1,D,D).cpu().numpy() * weights_mask # mask is currently ntilts x D x D
        flog(f'Pixels per particle (+ skip-zeros-decoder mask):  {final_mask_decoder.sum()}')
    else:
        final_mask_decoder = lattice_mask.view(1,D,D).repeat(Ntilts,1,1).cpu().numpy()
    assert final_mask_decoder.shape == (Ntilts, D, D)
    # mask cumulative weights by final_mask
    cumulative_weights_decoder = cumulative_weights[final_mask_decoder]

    # instantiate model
    weight_encoder = True if args.weight_encoder else False
    skip_zeros_encoder = True if args.skip_zeros_encoder else False
    skip_zeros_decoder = True if args.skip_zeros_decoder else False
    if skip_zeros_encoder:
        enc_mask = final_mask_encoder
        flog('Skip_zeros_encoder WAS used, so critical-dose-masked tilt images will be encoded by simultaneously by encA')
        flog('Arguments relating to encB will be ignored')
    else:
        enc_mask = lattice.get_circular_mask(D//2)
        flog('Skip_zeros_encoder WAS NOT used, so uniformly-masked tilt images will be encoded by encA and encB')
    in_dim = enc_mask.sum()
    # # instantiate model
    #
    # if args.enc_mask is None:
    #     args.enc_mask = D // 2
    # if args.enc_mask > 0:
    #     assert args.enc_mask <= D // 2
    #     enc_mask = lattice.get_circular_mask(args.enc_mask)
    #     in_dim = enc_mask.sum()
    # elif args.enc_mask == -1:
    #     enc_mask = None
    #     in_dim = lattice.D ** 2 if not args.use_real else (lattice.D - 1) ** 2
    # else:
    #     raise RuntimeError("Invalid argument for encoder mask radius {}".format(args.enc_mask))
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    model = TiltSeriesHetOnlyVAE(lattice, args.qlayersA, args.qdimA, Ntilts, args.qlayersB, args.qdimB,
                                 args.players, args.pdim, in_dim, args.zdim,
                                 # enc_mask=enc_mask,
                                 enc_type=args.pe_type, enc_dim=args.pe_dim,
                                 domain=args.domain, activation=activation, use_amp=args.amp,
                                 skip_zeros_encoder=skip_zeros_encoder, skip_zeros_decoder=skip_zeros_decoder)
    flog(model)
    flog('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    flog('{} parameters in encoder'.format(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)))
    flog('{} parameters in decoder'.format(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)))

    # save configuration
    out_config = f'{args.outdir}/config.pkl'
    save_config(args, data, lattice, model, out_config)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Mixed precision training with AMP
    use_amp = args.amp
    flog(f'AMP acceleration enabled (autocast + gradscaler) : {use_amp}')
    # TODO: add warnings if various layers / batchsize / etc not multiples of 8, not optimized

    # restart from checkpoint
    if args.load:
        flog('Loading checkpoint from {}'.format(args.load))
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        model.train()
    else:
        start_epoch = 0

    # parallelize
    if args.multigpu and torch.cuda.device_count() > 1:
        log(f'Using {torch.cuda.device_count()} GPUs!')
        args.batch_size *= torch.cuda.device_count()
        log(f'Increasing batch size to {args.batch_size}')
        model = nn.DataParallel(model)
    elif args.multigpu:
        log(f'WARNING: --multigpu selected, but {torch.cuda.device_count()} GPUs detected')

    # train
    data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    expanded_ind_rebased = torch.tensor([np.arange(i * Ntilts, (i + 1) * Ntilts) for i in range(Nptcls)]).to(device) # redefine training inds to remove gaps created by filtering dataset when loading
    cumulative_weights_encoder = torch.tensor(cumulative_weights_encoder)
    cumulative_weights_decoder = torch.tensor(cumulative_weights_decoder)
    final_mask_encoder = torch.tensor(final_mask_encoder)
    final_mask_decoder = torch.tensor(final_mask_decoder)
    num_epochs = args.num_epochs
    scaler = GradScaler(enabled=use_amp)
    for epoch in range(start_epoch, num_epochs):
        t2 = dt.now()
        gen_loss_accum = 0
        loss_accum = 0
        kld_accum = 0
        eq_loss_accum = 0
        batch_it = 0
        for batch, ind in data_generator:
            # impression counting
            batch_ind = expanded_ind_rebased[ind].view(-1) # need to use expanded indexing for ctf and poses
            batch_it += len(batch_ind) # total number of images seen (+= batchsize*ntilts)
            global_it = Nimg*epoch + batch_it
            beta = beta_schedule(global_it)

            # training minibatch
            y = batch.to(device)
            batch_ind = batch_ind.to(device)
            rot, tran = posetracker.get_pose(batch_ind)
            ctf_param = ctf_params[batch_ind] if ctf_params is not None else None
            loss, gen_loss, kld = train_batch(scaler, model, lattice, y, cumulative_weights_encoder,
                                              cumulative_weights_decoder, final_mask_encoder, final_mask_decoder, rot,
                                              tran, optim, beta, args.beta_control, ctf_params=ctf_param,
                                              use_amp=use_amp, weight_encoder=weight_encoder)

            # logging
            gen_loss_accum += gen_loss*len(batch_ind)
            kld_accum += kld*len(batch_ind)
            loss_accum += loss*len(batch_ind)

            if batch_it % args.log_interval == 0:
                log('# [Train Epoch: {}/{}] [{}/{} subtomos] gen loss={:.6f}, kld={:.6f}, beta={:.6f}, loss={:.6f}'.format(epoch+1, num_epochs, int(batch_it/Ntilts), Nptcls, gen_loss, kld, beta, loss))

        flog('# =====> Epoch: {} Average gen loss = {:.6}, KLD = {:.6f}, total loss = {:.6f}; Finished in {}'.format(epoch+1, gen_loss_accum/Nimg, kld_accum/Nimg, loss_accum/Nimg, dt.now()-t2))
        if args.checkpoint and epoch % args.checkpoint == 0:
            check_memory_usage(flog)
            out_weights = '{}/weights.{}.pkl'.format(args.outdir,epoch)
            out_z = '{}/z.{}.pkl'.format(args.outdir, epoch)
            model.eval()
            with torch.no_grad():
                z_mu, z_logvar = eval_z(model, lattice, data, cumulative_weights_encoder, final_mask_encoder, expanded_ind_rebased, args.batch_size, device, posetracker.trans, ctf_params, weight_encoder=weight_encoder)
                save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z)

    # save model weights, latent encoding, and evaluate the model on 3D lattice
    out_weights = '{}/weights.pkl'.format(args.outdir)
    out_z = '{}/z.pkl'.format(args.outdir)
    model.eval()
    with torch.no_grad():
        z_mu, z_logvar = eval_z(model, lattice, data, cumulative_weights_encoder, final_mask_encoder,
                                expanded_ind_rebased, args.batch_size, device, posetracker.trans, ctf_params,
                                weight_encoder=weight_encoder)
        save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z)

    td = dt.now() - t1
    flog('Finished in {} ({} per epoch)'.format(td, td / (num_epochs - start_epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)



