"""
Evaluate a trained FTPositionalDecoder model, optionally conditioned on values of latent embedding z.
"""
import argparse
import os
from datetime import datetime as dt

from tomodrgn import utils, models

log = utils.log
vlog = utils.vlog


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        # this script is called from tomodrgn.__main__ entry point, in which case a parser is already created
        pass

    group = parser.add_argument_group('Core arguments')
    group.add_argument('-w', '--weights', help='Model weights from train_vae')
    group.add_argument('-c', '--config', required=True, help='config.pkl file from train_vae')
    group.add_argument('-o', type=os.path.abspath, required=True, help='Output .mrc or directory')
    group.add_argument('--prefix', default='vol_', help='Prefix when writing out multiple .mrc files')

    group = parser.add_argument_group('Specify z values')
    group.add_argument('--zfile', type=os.path.abspath, help='Text/.pkl file with z-values to evaluate')

    group = parser.add_argument_group('Volume arguments')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volume')
    group.add_argument('--invert', action='store_true', help='Invert contrast of output volume')
    group.add_argument('--downsample', type=int, help='Downsample volumes to this box size (pixels)')
    group.add_argument('--lowpass', type=float, default=None, help='Lowpass filter to this resolution in Å. Requires settings --Apix.')

    group = parser.add_argument_group('Compute arguments')
    group.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size to parallelize volume generation (32-64 works well for box64 volumes)')
    group.add_argument('--no-amp', action='store_true', help='Disable use of mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs. Specify GPUs i,j via `export CUDA_VISIBLE_DEVICES=i,j`')
    return parser


def main(args):
    t1 = dt.now()

    # instantiate the volume generator
    vg = models.VolumeGenerator(config=args.config,
                                weights_path=args.weights,
                                model=None,
                                lat=None,
                                amp=not args.no_amp)

    # generate volumes
    vg.generate_volumes(z=args.zfile,
                        out_dir=args.o,
                        out_name='vol',
                        downsample=args.downsample,
                        lowpass=args.lowpass,
                        flip=args.flip,
                        invert=args.invert,
                        batch_size=args.batch_size, )

    td = dt.now() - t1
    log(f'Finished in {td}')


if __name__ == '__main__':
    main(add_args().parse_args())
