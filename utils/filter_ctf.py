'''
Filter a ctf.pkl by defocus values
'''

import argparse
import os
from tomodrgn import utils
import seaborn as sns
import numpy as np

def add_args(parser):
    parser.add_argument('ctf', type=os.path.abspath, help='ctf.pkl file to filter')
    parser.add_argument('--low', type=int, help='lower bound of defocus range to keep')
    parser.add_argument('--high', type=int, help='upper bound of defocus range to keep')
    parser.add_argument('--plot', action='store_true', help='optionally save a histogram.png of the defocus range')
    parser.add_argument('-o', default='ind_defocus-filt.pkl', help='name of filtered output indices.pkl file')
    return parser

def main(args):
    ctf = utils.load_pkl(args.ctf)
    defocusU = ctf[:,2]
    defocusV = ctf[:,3]

    if not args.low:
        low = defocusU.min()
        print('--low not provided; using min(defocusU)')
    else:
        low = args.low

    if not args.high:
        high = defocusU.max()
        print('--high not provided; using max(defocusU)')
    else:
        high = args.high

    ind = np.where((defocusU >= low) & (defocusU <= high))[0]
    utils.save_pkl(ind, args.o)
    print(f'Filtered down to {ind.shape} particles')
    print(ind)

    if args.plot:
        jointplot = sns.jointplot(x=defocusU, y=defocusV, kind='reg')
        jointplot.set_axis_labels('Defocus U (Å)', 'Defocus V (Å)')
        jointplot.ax_joint.axvline(x=low, c='r')
        jointplot.ax_joint.axvline(x=high, c='r')
        jointplot.savefig('defocus_jointplot.png')

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog='Example usage: $ python filter_ctf.py ctf.pkl ind.pkl --low 20000 --high 23000',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    main(parser.parse_args())