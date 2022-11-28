'''
Assess convergence of a decoder-only network relative to an external volume by FSC
'''

import numpy as np
import matplotlib.pyplot as plt
import argparse, os, fnmatch
from tomodrgn import utils

def add_args(parser):
    parser.add_argument('training_directory', type=os.path.abspath, help='train_nn directory containing reconstruct.N.mrc')
    parser.add_argument('reference_volume', type=os.path.abspath, help='volume against which to calculate FSC')
    parser.add_argument('--max-epoch', type=int, help='Maximum epoch for which to calculate FSCs')
    parser.add_argument('--include-dc', action='store_true', help='Include FSC calculation for DC component, default False because DC component default excluded during training')
    parser.add_argument('--fsc-mask', choices=('none', 'sphere', 'tight', 'soft'), default='soft', help='Type of mask applied to volumes before calculating FSC')
    return parser


def main(args):
    LOG = f'{args.training_directory}/convergence_nn.log'
    def flog(msg):  # HACK: switch to logging module
        return utils.flog(msg, LOG)

    reconstruction_vols = []
    for file in os.listdir(args.training_directory):
        if fnmatch.fnmatch(file, 'reconstruct.*.mrc'):
            reconstruction_vols.append(os.path.abspath(os.path.join(args.training_directory, file)))
    reconstruction_vols = sorted(reconstruction_vols, key=lambda x: int(os.path.splitext(x)[0].split('.')[-1]))
    epochs = [i for i in range(len(reconstruction_vols))]
    assert epochs, f'No reconstruct.*.mrc files detected in {args.training_directory}; exiting...'
    flog(f'Found {len(epochs)} epochs to compute FSC')

    if args.max_epoch:
        epochs = epochs[:args.max_epoch]
        reconstruction_vols = reconstruction_vols[:args.max_epoch]
        flog(f'Using first {len(epochs)} epochs to compute FSC')

    outdir = f'{args.training_directory}/convergence.{len(epochs)}'
    os.makedirs(outdir, exist_ok=True)

    flog(f'Using FSC reference volume: {args.reference_volume}')

    resolutions_point143 = []
    resolutions_point5 = []
    integrals = []
    fscs = []
    for vol in reconstruction_vols:
        flog(f'Processing volume: {vol}')
        x, fsc = utils.calc_fsc(args.reference_volume, vol, mask=args.fsc_mask)
        if not args.include_dc:
            x = x[1:]
            fsc = fsc[1:]
        fscs.append(fsc)

        # find the resolution at which FSC crosses 0.5 correlation"
        if np.all(fsc >= 0.5):
            threshold_resolution = x[-1]
            resolutions_point5.append(threshold_resolution)
        else:
            threshold_resolution = x[np.argmax(fsc < 0.5)]
            resolutions_point5.append(threshold_resolution)

        # find the resolution at which FSC crosses 0.143 correlation"
        if np.all(fsc >= 0.143):
            threshold_resolution = x[-1]
            threshold_index = x.shape[0]
            resolutions_point143.append(threshold_resolution)
        else:
            threshold_resolution = x[np.argmax(fsc < 0.143)]
            threshold_index = np.argmax(fsc < 0.143)
            resolutions_point143.append(threshold_resolution)

        # calculate the integral of correlation against resolution
        integral = np.trapz(fsc[:threshold_index], x[:threshold_index])
        integrals.append(integral)

    fscs = np.array(fscs)

    flog(f'Maximum 0.143 resolution of {np.array(resolutions_point143).max()} was first reached at epoch {np.array(resolutions_point143).argmax()}')
    flog(f'Maximum 0.5 resolution of {np.array(resolutions_point5).max()} was first reached at epoch {np.array(resolutions_point5).argmax()}')
    flog(f'Maximum FSC integral over 0.143 of {np.array(integrals).max()} was first reached at epoch {np.array(integrals).argmax()}')
    flog(f'Final FSC 0.143 resolution: {np.array(resolutions_point143)[-1]}')
    flog(f'Final FSC 0.5 resolution: {np.array(resolutions_point5)[-1]}')

    def plot_and_save(x_axis, y_values, xlabel, ylabel, outpath, **kwargs):
        plt.plot(x_axis, y_values, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(args.training_directory, outpath))

    flog(f'Saving plots to {outdir}')
    plot_and_save(epochs, resolutions_point143, 'epoch', '0.143 fsc frequency (1/px)', f'{outdir}/convergence_nn_fsc0.143resolution.png', linestyle='-', marker='.', mfc='red', mec='red', markersize=2)
    plt.clf()
    plot_and_save(epochs, resolutions_point5, 'epoch', '0.5 fsc frequency (1/px)', f'{outdir}/convergence_nn_fsc0.5resolution.png', linestyle='-', marker='.', mfc='red', mec='red', markersize=2)
    plt.clf()
    plot_and_save(epochs, integrals, 'epoch', 'fsc-frequency integral (1/px)', f'{outdir}/convergence_nn_fsc0.143integral.png', linestyle='-', marker='.', mfc='red', mec='red', markersize=2)
    plt.clf()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(epochs)))
    for epoch in epochs:
        plot_and_save(x, fscs.T[:,epoch], 'spatial frequency (1/px)', 'fsc', f'{outdir}/convergence_nn_fullfsc_all.png', color=colors[epoch])
    plt.clf()
    plot_and_save(x, fscs.T[:,-1], 'spatial frequency (1/px)', 'fsc', f'{outdir}/convergence_nn_fullfsc_final.png')

    flog('Done!')
    os.rename(LOG, f'{outdir}/convergence_nn.log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)