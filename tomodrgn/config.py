"""
Functions to create, edit, and view tomoDRGN config.pkl files
"""
import argparse
import sys
from datetime import datetime as dt
from pprint import pprint

import tomodrgn
from tomodrgn import utils
from tomodrgn.dataset import TiltSeriesMRCData
from tomodrgn.lattice import Lattice
from tomodrgn.models import TiltSeriesHetOnlyVAE, FTPositionalDecoder, DataParallelPassthrough
from tomodrgn.starfile import TiltSeriesStarfile


def save_config(args: argparse.Namespace,
                star: TiltSeriesStarfile,
                data: TiltSeriesMRCData,
                lat: Lattice,
                model: TiltSeriesHetOnlyVAE | FTPositionalDecoder | DataParallelPassthrough,
                out_config: str) -> None:
    """
    Save input arguments and precalculated data, lattice, and model metadata.
    Called automatically during `train_vae` and `train_nn`.

    :param args: argparse namespace from parse_args
    :param star: TiltSeriesStarfile object describing input data
    :param data: TiltSeriesMRCData object containing input data
    :param lat: Lattice object created using input data
    :param model: TiltSeriesHetOnlyVAE or FTPositionalDecoder object created using input data (no weights included)
    :param out_config: name of output `.pkl` file in whcih to save configuration
    :return: None
    """
    starfile_args = dict(sourcefile=star.sourcefile,
                         source_software=star.source_software,
                         ind_imgs=star.ind_imgs,
                         ind_ptcls=star.ind_ptcls,
                         sort_ptcl_imgs=star.sort_ptcl_imgs,
                         use_first_ntilts=star.use_first_ntilts,
                         use_first_nptcls=star.use_first_nptcls,
                         sourcefile_filtered=star.sourcefile_filtered,
                         type=type(star))

    dataset_args = dict(star_random_subset=data.star_random_subset,
                        datadir=data.datadir,
                        lazy=data.lazy,
                        norm=data.norm,
                        invert_data=data.invert_data,
                        window=data.window,
                        window_r=data.window_r,
                        window_r_outer=data.window_r_outer,
                        recon_dose_weight=data.recon_dose_weight,
                        recon_tilt_weight=data.recon_tilt_weight,
                        l_dose_mask=data.l_dose_mask,
                        constant_mintilt_sampling=data.constant_mintilt_sampling,
                        sequential_tilt_sampling=data.sequential_tilt_sampling,
                        type=type(data))

    lattice_args = dict(boxsize=lat.boxsize,
                        extent=lat.extent,
                        ignore_DC=lat.ignore_dc,
                        type=type(lat))

    if type(model) is TiltSeriesHetOnlyVAE:
        model_args = dict(in_dim=model.encoder.in_dim,
                          qlayersA=args.qlayersA,
                          qdimA=args.qdimA,
                          out_dimA=args.out_dim_A,
                          ntilts=data.ntilts_training,
                          qlayersB=args.qlayersB,
                          qdimB=args.qdimB,
                          zdim=args.zdim,
                          players=args.players,
                          pdim=args.pdim,
                          activation=args.activation,
                          enc_mask=model.enc_mask,
                          pooling_function=args.pooling_function,
                          feat_sigma=args.feat_sigma,
                          num_seeds=args.num_seeds,
                          num_heads=args.num_heads,
                          layer_norm=args.layer_norm,
                          pe_type=args.pe_type,
                          pe_dim=args.pe_dim,
                          type=type(model))
        training_args = dict(n=args.num_epochs,
                             B=args.batch_size,
                             wd=args.wd,
                             lr=args.lr,
                             beta=args.beta,
                             beta_control=args.beta_control,
                             amp=not args.no_amp,
                             multigpu=args.multigpu,
                             lazy=args.lazy,
                             verbose=args.verbose,
                             log_interval=args.log_interval,
                             checkpoint=args.checkpoint,
                             outdir=args.outdir)
    elif type(model) is FTPositionalDecoder:
        model_args = dict(in_dim=3,
                          hidden_layers=args.layers,
                          hidden_dim=args.dim,
                          activation=args.activation,
                          pe_type=args.pe_type,
                          pe_dim=args.pe_dim,
                          feat_sigma=args.feat_sigma,
                          type=type(model))
        training_args = dict(n=args.num_epochs,
                             B=args.batch_size,
                             wd=args.wd,
                             lr=args.lr,
                             amp=not args.no_amp,
                             multigpu=args.multigpu,
                             lazy=args.lazy,
                             verbose=args.verbose,
                             log_interval=args.log_interval,
                             checkpoint=args.checkpoint,
                             outdir=args.outdir)
    else:
        raise TypeError(f'Unsupported model type: {type(model)}')

    meta = dict(time=dt.now(),
                cmd=' '.join(sys.argv),
                version=tomodrgn.__version__)  # TODO replace with something like https://discuss.python.org/t/usual-pythonic-way-to-add-inject-the-git-hash-to-the-version-string/45169/10

    config = dict(starfile_args=starfile_args,
                  dataset_args=dataset_args,
                  lattice_args=lattice_args,
                  model_args=model_args,
                  training_args=training_args,
                  meta=meta)
    config['seed'] = args.seed
    config['angpix'] = star.get_tiltseries_pixelsize()
    utils.save_pkl(data=config, out_pkl=out_config)


def print_config(config: str | dict) -> None:
    """
    Print the contents of a tomoDRGN config.pkl file.

    :param config: path to config.pkl file, or preloaded config dict
    :return: None
    """
    # check inputs
    if type(config) is str:
        config = utils.load_pkl(config)
    assert type(config) is dict, f'Unrecognized config type: {type(config)}'

    # display the config contents
    pprint(config)
