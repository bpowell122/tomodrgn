"""
Display config information of a tomoDRGN job
"""

import argparse
import os

from tomodrgn import config


def add_args(_parser):
    _parser.add_argument('workdir', type=os.path.abspath, help='Directory with tomoDRGN results')
    return _parser


def main(args):
    config.print_config(f'{args.workdir}/config.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
