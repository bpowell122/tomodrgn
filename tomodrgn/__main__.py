"""
TomoDRGN neural network reconstruction
"""


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    import tomodrgn
    parser.add_argument('--version', action='version', version='tomoDRGN ' + tomodrgn.__version__)

    import tomodrgn.commands.analyze
    import tomodrgn.commands.analyze_volumes
    import tomodrgn.commands.backproject_voxel
    import tomodrgn.commands.cleanup
    import tomodrgn.commands.convergence_nn
    import tomodrgn.commands.convergence_vae
    import tomodrgn.commands.downsample
    import tomodrgn.commands.eval_images
    import tomodrgn.commands.eval_vol
    import tomodrgn.commands.filter_star
    import tomodrgn.commands.graph_traversal
    import tomodrgn.commands.pc_traversal
    import tomodrgn.commands.subtomo2chimerax
    import tomodrgn.commands.train_nn
    import tomodrgn.commands.train_vae
    import tomodrgn.commands.view_config

    modules = [tomodrgn.commands.analyze,
               tomodrgn.commands.analyze_volumes,
               tomodrgn.commands.backproject_voxel,
               tomodrgn.commands.cleanup,
               tomodrgn.commands.convergence_nn,
               tomodrgn.commands.convergence_vae,
               tomodrgn.commands.downsample,
               tomodrgn.commands.eval_images,
               tomodrgn.commands.eval_vol,
               tomodrgn.commands.filter_star,
               tomodrgn.commands.graph_traversal,
               tomodrgn.commands.pc_traversal,
               tomodrgn.commands.subtomo2chimerax,
               tomodrgn.commands.train_nn,
               tomodrgn.commands.train_vae,
               tomodrgn.commands.view_config, ]

    subparsers = parser.add_subparsers(title='Choose a command')
    subparsers.required = 'True'

    def get_str_name(_module):
        return os.path.splitext(os.path.basename(_module.__file__))[0]

    for _module in modules:
        this_parser = subparsers.add_parser(get_str_name(_module), description=_module.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        _module.add_args(this_parser)
        this_parser.set_defaults(func=_module.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
