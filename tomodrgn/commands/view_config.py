"""
Display config information of a tomoDRGN job
"""

import argparse
import os

from tomodrgn import config


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        # this script is called from tomodrgn.__main__ entry point, in which case a parser is already created
        pass

    parser.add_argument('workdir', type=os.path.abspath, help='Directory with tomoDRGN results')

    return parser


def main(args):
    config.print_config(f'{args.workdir}/config.pkl')


if __name__ == '__main__':
    main(add_args().parse_args())
