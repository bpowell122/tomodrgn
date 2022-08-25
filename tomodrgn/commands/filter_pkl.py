'''Filter cryodrgn data stored in a .pkl file'''

import argparse
import numpy as np
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Input data .pkl')
    parser.add_argument('--ind', help='Selected indices array (.pkl)')
    parser.add_argument('--first', type=int, help='First N datapoints')
    parser.add_argument('-o', type=os.path.abspath, help='Output data .pkl')
    return parser

def load_pkl(x):
    return pickle.load(open(x,'rb'))

def main(args):
    x = load_pkl(args.input)

    assert args.ind or args.first, '--ind and/or --first must be specified'

    if args.ind and args.first:
        print(f'Loading indices: {args.ind}')
        ind = load_pkl(args.ind)
        print(f'Further filtering by first: {args.first}')
        ind_first = np.arange(args.first)
        ind = ind[ind_first]
    elif args.first:
        print(f'Filtering by first: {args.first}')
        ind = np.arange(args.first)

    print(f'Old shape: {x.shape}')
    x = x[ind]
    print(f'New shape: {x.shape}')
    print(f'Saving {args.o}')
    pickle.dump(x, open(args.o,'wb'))

if __name__ == '__main__':
    main(parse_args().parse_args())
