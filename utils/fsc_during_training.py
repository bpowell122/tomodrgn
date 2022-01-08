import numpy as np
import matplotlib.pyplot as plt
import argparse, os, fnmatch
from cryodrgn import utils

def add_args(parser):
    parser.add_argument('--training-dir', type=os.path.abspath, help='cryodrgn train_nn directory containing reconstruct.N.mrc')
    parser.add_argument('--reference-volume', type=os.path.abspath, help='volume against which to calculate FSC')
    return parser


def main(args):
    reconstruction_vols = []
    for file in os.listdir(args.training_dir):
        if fnmatch.fnmatch(file, 'reconstruct.*.mrc'):
            reconstruction_vols.append(os.path.abspath(os.path.join(args.training_dir, file)))
    reconstruction_vols = sorted(reconstruction_vols, key=lambda x: int(os.path.splitext(x)[0].split('.')[-1]))
    epochs = [i for i in range(len(reconstruction_vols))]

    resolutions = []
    integrals = []
    for vol in reconstruction_vols:
        print(f'Processing volume: {vol}')
        x, fsc = utils.calc_fsc(args.reference_volume, vol)

        # find the resolution at which FSC crosses the specified threshold\n",
        if np.all(fsc >= 0.143):
            threshold_resolution = x[-1]
            threshold_index = x.shape[0]
            resolutions.append(threshold_resolution)
        else:
            threshold_resolution = x[np.argmax(fsc < 0.143)]
            threshold_index = np.argmax(fsc < 0.143)
            resolutions.append(threshold_resolution)

        # calculate the integral of correlation against resolution
        integral = np.trapz(fsc[:threshold_index], x[:threshold_index])
        integrals.append(integral)

    print(f'Maximum resolution of {np.array(resolutions).max()} was first reached at epoch {np.array(resolutions).argmax()}')
    print(f'Maximum FSC integral of {np.array(integrals).max()} was first reached at epoch {np.array(integrals).argmax()}')

    utils.save_pkl(np.array(integrals), os.path.join(args.training_dir, 'fsc_during_training_integral.pkl'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    ax1.plot(epochs, resolutions, linestyle='-', marker='.', mfc='red', mec='red', markersize=2)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('0.143 fsc frequency (1/px)')
    ax2.plot(epochs, integrals, linestyle='-', marker='.', mfc='red', mec='red', markersize=2)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('fsc-frequency integral (1/px)')
    fig.suptitle(args.reference_volume)
    plt.tight_layout()
    plt.savefig(os.path.join(args.training_dir, 'fsc_during_training'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)