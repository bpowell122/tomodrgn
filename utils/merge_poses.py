'''
Concatenate poses.pkl from many substacks into one stack
'''

import argparse

import numpy as np

from cryodrgn import utils

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('poses_list', help='.txt listing poses.pkl files to be concatenated')
    parser.add_argument('--o', default='poses_concatenated.pkl', help='name of output.pkl file')

    return parser

def main(args):
    with open(args.poses_list) as file:
        files = file.readlines()
        files = [line.rstrip() for line in files]

    output_rots = np.empty([])
    output_trans = np.empty([])

    for file in files:
         temp = utils.load_pkl(file)
         assert len(temp) == 2 # rot, trans
         temp_rots, temp_trans = temp
         output_rots = np.append(temp_rots)
         output_trans = np.append(temp_trans)

    outfile = (output_rots, output_trans)

    print(f'Concatenated {output_rots.shape[0]} particles')
    print(f'Saving {args.o}')

    utils.save_pkl(outfile, args.o)

if __name__ == '__main__':
    args = parse_args().parse_args()
    main(args)

