"""
Display config information of a tomoDRGN job
"""

import argparse
import os

from tomodrgn import config


def add_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('workdir', type=os.path.abspath, help='Directory with tomoDRGN results')

    return parser


def main(args):
    config.print_config(f'{args.workdir}/config.pkl')


if __name__ == '__main__':
    main(add_args().parse_args())
