'''
Concatenate poses.pkl from many substacks into one stack
'''

import argparse

import numpy as np

from tomodrgn import utils

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('poses_list', help='.txt listing poses.pkl files to be concatenated')
    parser.add_argument('-o', default='poses_concatenated.pkl', help='name of output.pkl file')

    return parser

def main(args):
    with open(args.poses_list) as file:
        files = file.readlines()
        files = [line.rstrip() for line in files]

    output_rots = np.empty([1,3,3])
    output_trans = np.empty([1,2])

    for file in files:
        temp = utils.load_pkl(file)
        if type(temp) == tuple:
            assert len(temp) == 2, len(temp)
            temp_rots, temp_trans = temp
            output_rots = np.append(output_rots, temp_rots, axis=0)
            output_trans = np.append(output_trans, temp_trans, axis=0)
        else:
            temp_rots = temp
            output_rots = np.append(output_rots, temp_rots, axis=0)

    if output_rots.shape[0] == output_trans.shape[0]:
        outfile = (output_rots[1:,:,:], output_trans[1:,:])
    else:
        outfile = (output_rots[1:,:,:], np.zeros((output_rots.shape[0]-1,2)))

    print(f'Concatenated {len(outfile[0])} particles')
    print(f'Saving {args.o}')

    utils.save_pkl(outfile, args.o)

if __name__ == '__main__':
    args = parse_args().parse_args()
    main(args)

