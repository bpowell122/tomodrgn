'''
Evaluate z for a stack of images using the config.pkl and weights.pkl of a pretrained model
'''
import numpy as np
import os
import argparse
import pickle
from datetime import datetime as dt
import pprint

import torch
from torch.utils.data import DataLoader

from tomodrgn import utils, dataset, ctf
from tomodrgn.models import TiltSeriesHetOnlyVAE

log = utils.log
vlog = utils.vlog

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('-w', '--weights', help='Model weights')
    parser.add_argument('-c', '--config', required=True, help='config.pkl file from train_vae')
    parser.add_argument('--out-z', metavar='PKL', type=os.path.abspath, required=True, help='Output pickle for z')
    parser.add_argument('--log-interval', type=int, default=1000, help='Logging interval in N_IMGS (default: %(default)s)')
    parser.add_argument('-b','--batch-size', type=int, default=64, help='Minibatch size (default: %(default)s)')
    parser.add_argument('-v','--verbose',action='store_true',help='Increases verbosity')

    group = parser.add_argument_group('Dataset loading')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--ind', type=os.path.abspath, help='Filter particle stack by these indices')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--sequential-tilt-sampling', action='store_true', help='Supply particle images of one particle to encoder in starfile order')

    return parser


def eval_batch(model, lattice, y, trans, ctf_params=None):
    B, ntilts, D, D = y.shape

    # correct for translations
    if not torch.all(trans == 0):
        y = lattice.translate_ht(y.view(B * ntilts, -1), trans.view(B * ntilts, 1, 2))
    y = y.view(B, ntilts, D, D)

    # correct for CTF via phase flipping
    if not torch.all(ctf_params == 0):
        freqs = lattice.freqs2d.unsqueeze(0).expand(B * ntilts, -1, -1) / ctf_params[:, :, 1].view(B * ntilts, 1, 1)
        ctf_weights = ctf.compute_ctf(freqs, *torch.split(ctf_params.view(B * ntilts, -1)[:, 2:], 1, 1)).view(B, ntilts, D, D)
        y *= ctf_weights.sign()

    z_mu, z_logvar = model.encode(y, B, ntilts)

    return z_mu.detach().cpu().numpy(), z_logvar.detach().cpu().numpy()

def main(args):
    t1 = dt.now()

    # make output directory
    if not os.path.exists(os.path.dirname(args.out_z)):
        os.makedirs(os.path.dirname(args.out_z))

    ## set the device
    device = utils.get_default_device()
    torch.set_grad_enabled(False)

    log(args)
    cfg = utils.load_pkl(args.config)
    log('Loaded configuration:')
    pprint.pprint(cfg)

    zdim = cfg['model_args']['zdim']

    # load the particle indices
    if args.ind is not None:
        log(f'Reading supplied particle indices {args.ind}')
        ind = pickle.load(open(args.ind, 'rb'))
    else:
        ind = None

    # load the particles
    data = dataset.TiltSeriesMRCData(args.particles,
                                     datadir=args.datadir,
                                     ind_ptcl=ind,
                                     lazy=args.lazy,
                                     invert_data=args.invert_data,
                                     norm=cfg['dataset_args']['norm'],
                                     window=cfg['dataset_args']['window'],
                                     window_r=cfg['dataset_args']['window_r'],
                                     recon_dose_weight=cfg['training_args']['recon_dose_weight'],
                                     dose_override=cfg['training_args']['dose_override'],
                                     recon_tilt_weight=cfg['training_args']['recon_tilt_weight'],
                                     l_dose_mask=cfg['model_args']['l_dose_mask'],
                                     sequential_tilt_sampling=args.sequential_tilt_sampling)

    nptcls = data.nptcls
    data.ntilts_training = cfg['dataset_args']['ntilts']

    # instantiate model and lattice
    model, lattice = TiltSeriesHetOnlyVAE.load(cfg, args.weights, device=device)

    # evaluation loop
    model.eval()
    z_mu_all = []
    z_logvar_all = []
    batch_it = 0
    data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    for batch_images, _, batch_trans, batch_ctf, _, _, batch_ptcl_ids in data_generator:
        # impression counting
        batch_it += len(batch_ptcl_ids)

        # transfer to GPU
        batch_images = batch_images.to(device)
        batch_trans = batch_trans.to(device)
        batch_ctf = batch_ctf.to(device)

        z_mu, z_logvar = eval_batch(model, lattice, batch_images, batch_trans, ctf_params=batch_ctf)
        z_mu_all.append(z_mu)
        z_logvar_all.append(z_logvar)

        if batch_it % args.log_interval == 0:
            log(f'# [{batch_it}/{nptcls} particles]')

    z_mu_all = np.vstack(z_mu_all)
    z_logvar_all = np.vstack(z_logvar_all)

    with open(args.out_z,'wb') as f:
        pickle.dump(z_mu_all, f)
        pickle.dump(z_logvar_all, f)

    log(f'Finished in {dt.now() - t1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)

