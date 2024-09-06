"""
Quickly test train_vae and analyze with default parameters (most common commands)
"""

import os
import shutil

from testing_module import CommandTester


def main():
    # remove pre-existing output and create new output folder
    workdir = 'output'
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    # instantiate the tester
    tester = CommandTester(workdir, verbose=True)

    # add the tests
    tester.commands.append(
        'tomodrgn train_vae '
        'data/10076_both_32_sim.star '
        '-o output/01_vae_both_sim '
        '--zdim 8 '
        '--uninvert-data '
        '--seed 42 '
        '--log-interval 100 '
        '--enc-dim-A 64 '
        '--enc-layers-A 2 '
        '--out-dim-A 64 '
        '--enc-dim-B 32 '
        '--enc-layers-B 4 '
        '--dec-dim 16 '
        '--dec-layers 3 '
        '-n 5')
    tester.commands.append(
        'tomodrgn analyze '
        'output/01_vae_both_sim '
        '--ksample 20')

    # run the tests
    tester.run()

    # report the results
    tester.report_run_summary()


if __name__ == '__main__':
    main()
