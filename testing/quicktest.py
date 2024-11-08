"""
Quickly test train_vae and analyze with default parameters (most common commands)
"""

from testing_module import run_assert_no_error


def test_version(script_runner, output_dir):
    run_assert_no_error(script_runner, command='tomodrgn --version')


def test_train_vae(script_runner, output_dir):
    run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --source-software cryosrpnt -o {output_dir}/01_vae_both_sim --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 16 --dec-layers 3 -n 5')


def test_analyze(script_runner, output_dir):
    run_assert_no_error(script_runner, command=f'tomodrgn analyze {output_dir}/01_vae_both_sim --ksample 20')
