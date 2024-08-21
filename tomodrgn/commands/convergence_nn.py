"""
Assess convergence of a decoder-only network relative to an external volume by FSC
"""

import argparse
import fnmatch
import os

import matplotlib.pyplot as plt
import numpy as np

from tomodrgn import utils, convergence, analysis


def add_args(parser):
    parser.add_argument('training_directory', type=os.path.abspath, help='train_nn directory containing reconstruct.N.mrc')
    parser.add_argument('reference_volume', type=os.path.abspath, help='volume against which to calculate FSC')
    parser.add_argument('--max-epoch', type=int, help='Maximum epoch for which to calculate FSCs')
    parser.add_argument('--include-dc', action='store_true', help='Include FSC calculation for DC component, default False because DC component default excluded during training')
    parser.add_argument('--fsc-mask', choices=('none', 'sphere', 'tight', 'soft'), default='soft', help='Type of mask applied to volumes before calculating FSC')
    return parser


def make_plots(resolution: np.ndarray,
               fscs: np.ndarray,
               fsc_metrics: np.ndarray,
               epochs: np.ndarray | list,
               outdir: str) -> None:
    """
    Save an array of standard plots characterizing homogeneous volume model training convergence.
    :param resolution: array of resolution values in units 1/px and shape (boxsize // 2)
    :param fscs: array of FSC curves with shape (len(vol_paths), boxsize // 2)
    :param fsc_metrics: array of FSC metrics with shape (3, len(vol_paths)): resolution crossing FSC 0.143, resolution crossing FSC 0.5, integral under FSC before 0.143 crossing
    :param epochs: list of epochs being evaluated and saved
    :param outdir: output directory in which to save plots
    :return: None
    """
    def plot_and_save(x_values: np.ndarray | list,
                      y_values: np.ndarray | list,
                      x_label: str,
                      y_label: str,
                      outpath: str,
                      **kwargs) -> None:
        """
        Helper function to produce a line plot and save it as an image to the specified path.
        :param x_values: list or array of values to plot along x axis
        :param y_values: list or array of values to plot along y axis
        :param x_label: label for x axis
        :param y_label: label for y axis
        :param outpath: name of output image to save plot as
        :param kwargs: additional key word arguments passed to matplotlib.pyplot.plot
        :return: None
        """
        plt.plot(x_values, y_values, **kwargs)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(outpath)

    plot_and_save(x_values=epochs,
                  y_values=fsc_metrics[0],
                  x_label='epoch',
                  y_label='0.143 fsc frequency (1/px)',
                  outpath=f'{outdir}/convergence_nn_fsc0.143resolution.png',
                  linestyle='-', marker='.', mfc='red', mec='red', markersize=2)
    plt.close()
    plot_and_save(x_values=epochs,
                  y_values=fsc_metrics[1],
                  x_label='epoch',
                  y_label='0.5 fsc frequency (1/px)',
                  outpath=f'{outdir}/convergence_nn_fsc0.5resolution.png',
                  linestyle='-', marker='.', mfc='red', mec='red', markersize=2)
    plt.close()
    plot_and_save(x_values=epochs,
                  y_values=fsc_metrics[2],
                  x_label='epoch',
                  y_label='fsc-frequency integral (1/px)',
                  outpath=f'{outdir}/convergence_nn_fsc0.143integral.png',
                  linestyle='-', marker='.', mfc='red', mec='red', markersize=2)
    plt.close()
    epoch_colors = analysis.get_colors(len(epochs), 'coolwarm')
    for epoch in epochs:
        plot_and_save(x_values=resolution,
                      y_values=fscs.T[:, epoch],
                      x_label='spatial frequency (1/px)',
                      y_label='fsc',
                      outpath=f'{outdir}/convergence_nn_fullfsc_all.png',
                      color=epoch_colors[epoch])
    plt.close()
    plot_and_save(x_values=resolution,
                  y_values=fscs.T[:, -1],
                  x_label='spatial frequency (1/px)',
                  y_label='fsc',
                  outpath=f'{outdir}/convergence_nn_fullfsc_final.png')
    plt.close()


def main(args):
    # make output log file
    logfile = f'{args.training_directory}/convergence_nn.log'

    def flog(msg):  # HACK: switch to logging module
        return utils.flog(msg, logfile)

    # log arguments
    flog(args)

    # identify volumes generated at each epoch
    reconstruct_vols_paths = []
    for file in os.listdir(args.training_directory):
        if fnmatch.fnmatch(file, 'reconstruct.*.mrc'):
            reconstruct_vols_paths.append(os.path.abspath(os.path.join(args.training_directory, file)))
    reconstruct_vols_paths = sorted(reconstruct_vols_paths, key=lambda path: int(os.path.splitext(path)[0].split('.')[-1]))
    epochs = [int(os.path.splitext(path)[0].split('.')[-1]) for path in reconstruct_vols_paths]
    assert epochs, f'No reconstruct.*.mrc files detected in {args.training_directory}; exiting...'
    flog(f'Found the following epochs at which to compute an FSC: {epochs}')

    # optionally limit convergence analysis to a certain maximum epoch
    if args.max_epoch:
        epochs = [epoch for epoch in epochs if epoch < args.max_epoch]
        reconstruct_vols_paths = reconstruct_vols_paths[:len(epochs)]
        flog(f'Using first {len(epochs)} epochs to compute FSCs')

    # make output directory
    outdir = f'{args.training_directory}/convergence.{len(epochs)}'
    os.makedirs(outdir, exist_ok=True)

    # calculate convergence metrics
    resolution, fscs, fsc_metrics = convergence.fsc_referencevol_to_manyvols(reference_vol=args.reference_volume,
                                                                             vol_paths=reconstruct_vols_paths,
                                                                             fsc_mask=args.fsc_mask,
                                                                             include_dc=args.include_dc)
    flog(f'Maximum FSC=0.143 resolution of {np.array(fsc_metrics[0]).max()} was first reached at epoch {np.array(fsc_metrics[0]).argmax()}')
    flog(f'Maximum FSC=0.5 resolution of {np.array(fsc_metrics[1]).max()} was first reached at epoch {np.array(fsc_metrics[1]).argmax()}')
    flog(f'Maximum FSC integral prior to crossing FSC=0.143 of {np.array(fsc_metrics[2]).max()} was first reached at epoch {np.array(fsc_metrics[2]).argmax()}')
    flog(f'Final FSC=0.143 resolution: {np.array(fsc_metrics[0])[-1]}')
    flog(f'Final FSC=0.5 resolution: {np.array(fsc_metrics[1])[-1]}')

    # save standard plots
    flog(f'Saving plots to {outdir}')
    make_plots(resolution=resolution,
               fscs=fscs,
               fsc_metrics=fsc_metrics,
               epochs=epochs,
               outdir=outdir)

    # save all other outputs
    utils.save_pkl((resolution, fscs), f'{outdir}/freqs_fscs.pkl')
    flog('Done!')
    os.rename(logfile, f'{outdir}/convergence_nn.log')


if __name__ == '__main__':
    main(add_args(argparse.ArgumentParser(description=__doc__)).parse_args())
