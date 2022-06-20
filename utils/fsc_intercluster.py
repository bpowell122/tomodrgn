import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, os, fnmatch
from cryodrgn import utils, mrc
from datetime import datetime as dt

def add_args(parser):
    parser.add_argument('volumes_directory', type=os.path.abspath, help='cryodrgn train_nn directory containing class.*.mrc')
    parser.add_argument('--correlation', type=str, choices=('CC','FSC'), help='CC or FSC as correlation metric')
    parser.add_argument('--fsc-threshold', type=float, default=0.143, help='FSC threshold at which to report resolution')
    parser.add_argument('--ignore-dc', action='store_true', help='Skip calculating FSC for DC component')
    return parser

def calc_cc(vol1, vol2):
    '''
    Helper function to calculate the zero-mean correlation coefficient as defined in eq 2 in https://journals.iucr.org/d/issues/2018/09/00/kw5139/index.html
    vol1 and vol2 should be maps of the same box size, structured as numpy arrays with ndim=3, i.e. by loading with cryodrgn.mrc.parse_mrc
    '''
    zmean1 = (vol1 - np.mean(vol1))
    zmean2 = (vol2 - np.mean(vol2))
    cc = (np.sum(zmean1 ** 2) ** -0.5) * (np.sum(zmean2 ** 2) ** -0.5) * np.sum(zmean1 * zmean2)
    return cc

def main(args):
    cluster_vols = []
    for file in os.listdir(args.volumes_directory):
        if fnmatch.fnmatch(file, '*.mrc'):
            cluster_vols.append(os.path.abspath(os.path.join(args.volumes_directory, file)))
    cluster_vols = sorted(cluster_vols, key=lambda x: int(os.path.splitext(x)[0].split('vol_')[-1]))
    cluster_ids = [i for i in range(len(cluster_vols))]


    threshold = args.fsc_threshold
    correlations = np.ones((len(cluster_ids), len(cluster_ids)))

    for i,vol1 in enumerate(cluster_vols):
        t1 = dt.now()
        print(f'Calculating all correlations against cluster: {vol1}')
        for j,vol2 in enumerate(cluster_vols):
            if i == j:
                # on diagonal
                correlations[i,j] = 0.5 if args.correlation == 'FSC' else 1.0
            elif i > j:
                # already calculated correlation for this volume pair
                correlations[i,j] = correlations[j,i]
            else:
                # correlation not yet calculated for this volume pair
                if args.correlation == 'FSC':
                    x, fsc = utils.calc_fsc(vol1, vol2)
                    if args.ignore_dc:
                        x = x[1:]
                        fsc = fsc[1:]
                    correlations[i,j] = x[-1] if np.all(fsc >= threshold) else x[np.argmax(fsc < threshold)]
                elif args.correlation == 'CC':
                    a, _ = mrc.parse_mrc(vol1)
                    b, _ = mrc.parse_mrc(vol2)
                    correlations[i,j] = calc_cc(a, b)

        t2 = dt.now()
        print(t2-t1)

    sns.heatmap(correlations, linewidths=0.5)
    if args.correlation == 'FSC':
        plt.xlabel(f'inter-cluster FSC={threshold} cutoff (1/px)')
    elif args.correlation == 'CC':
        plt.xlabel(f'inter-cluster CC')
    plt.title(args.volumes_directory)
    plt.tight_layout()
    plt.savefig(os.path.join(args.volumes_directory, f'intercluster_{args.correlation}.png'), dpi=300)

    print(f'Saved {args.volumes_directory}/intercluster_{args.correlation}.png')
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)