'''CryoDRGN neural network reconstruction'''

def main():
    import argparse, os
    parser = argparse.ArgumentParser(description=__doc__)
    import cryodrgn
    parser.add_argument('--version', action='version', version='cryoDRGN '+cryodrgn.__version__)

    import cryodrgn.commands.downsample
    import cryodrgn.commands.preprocess
    import cryodrgn.commands.parse_pose_star
    import cryodrgn.commands.parse_pose_csparc
    import cryodrgn.commands.parse_ctf_star
    import cryodrgn.commands.parse_ctf_csparc
    import cryodrgn.commands.backproject_voxel
    import cryodrgn.commands.train_nn
    import cryodrgn.commands.train_vae
    import cryodrgn.commands.eval_vol
    import cryodrgn.commands.eval_images
    import cryodrgn.commands.analyze
    import cryodrgn.commands.pc_traversal
    import cryodrgn.commands.graph_traversal
    import cryodrgn.commands.view_config
    import cryodrgn.commands.analyze_convergence
    import cryodrgn.commands.train_nn_ts
    import cryodrgn.commands.train_vae_ts

    modules = [cryodrgn.commands.downsample,
        cryodrgn.commands.preprocess,
        cryodrgn.commands.parse_pose_csparc,
        cryodrgn.commands.parse_pose_star,
        cryodrgn.commands.parse_ctf_csparc,
        cryodrgn.commands.parse_ctf_star,
        cryodrgn.commands.train_nn,
        cryodrgn.commands.backproject_voxel,
        cryodrgn.commands.train_vae,
        cryodrgn.commands.eval_vol,
        cryodrgn.commands.eval_images,
        cryodrgn.commands.analyze,
        cryodrgn.commands.pc_traversal,
        cryodrgn.commands.graph_traversal,
        cryodrgn.commands.view_config,
	    cryodrgn.commands.analyze_convergence,
        cryodrgn.commands.train_nn_ts,
        cryodrgn.commands.train_vae_ts
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

