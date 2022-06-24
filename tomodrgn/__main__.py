'''TomoDRGN neural network reconstruction'''

def main():
    import argparse, os
    parser = argparse.ArgumentParser(description=__doc__)
    import tomodrgn
    parser.add_argument('--version', action='version', version='tomoDRGN '+tomodrgn.__version__)

    import tomodrgn.commands.downsample
    import tomodrgn.commands.parse_pose_star
    import tomodrgn.commands.parse_ctf_star
    import tomodrgn.commands.backproject_voxel
    import tomodrgn.commands.train_nn
    import tomodrgn.commands.train_vae
    import tomodrgn.commands.eval_vol
    import tomodrgn.commands.eval_images
    import tomodrgn.commands.analyze
    import tomodrgn.commands.pc_traversal
    import tomodrgn.commands.graph_traversal
    import tomodrgn.commands.view_config
    import tomodrgn.commands.convergence_vae

    modules = [tomodrgn.commands.downsample,
        tomodrgn.commands.parse_pose_star,
        tomodrgn.commands.parse_ctf_star,
        tomodrgn.commands.train_nn_ts,
        tomodrgn.commands.backproject_voxel,
        tomodrgn.commands.train_vae_ts,
        tomodrgn.commands.eval_vol,
        tomodrgn.commands.eval_images,
        tomodrgn.commands.analyze,
        tomodrgn.commands.pc_traversal,
        tomodrgn.commands.graph_traversal,
        tomodrgn.commands.view_config,
	    tomodrgn.commands.analyze_convergence,
        ]

    subparsers = parser.add_subparsers(title='Choose a command')
    subparsers.required = 'True'

    def get_str_name(module):
        return os.path.splitext(os.path.basename(module.__file__))[0]

    for module in modules:
        this_parser = subparsers.add_parser(get_str_name(module), description=module.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        module.add_args(this_parser)
        this_parser.set_defaults(func=module.main)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()

