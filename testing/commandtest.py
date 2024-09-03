"""
Test all primary tomoDRGN commands and common argument permutations for error-free functionality.
Takes about 50 minutes on a 2020 MacBook Pro or 10 minutes on an Ubuntu workstation, and creates about 900 MB of outputs.
Note that outputs of some command tests are used as inputs to others.
"""

import os
import shutil
from testing_module import CommandTester


def add_tests_downsample(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('tomodrgn '
                           'downsample '
                           'data/10076_classE_32_sim.star '
                           '--downsample 16 '
                           '--batch-size 50 '
                           '--output output/10076_classE_16_sim.mrcs '
                           '--write-tiltseries-starfile')
    # lazy loading
    tester.commands.append('tomodrgn '
                           'downsample '
                           'data/10076_classE_32_sim.star '
                           '--downsample 16 '
                           '--batch-size 50 '
                           '--output output/10076_classE_16_sim.mrcs '
                           '--write-tiltseries-starfile '
                           '--lazy')
    # chunk
    tester.commands.append('tomodrgn '
                           'downsample '
                           'data/10076_classE_32_sim.star '
                           '--downsample 16 '
                           '--batch-size 50 '
                           '--output output/10076_classE_16_sim.mrcs '
                           '--write-tiltseries-starfile '
                           '--chunk 100')
    return tester


def add_tests_backproject_voxel(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('tomodrgn '
                           'backproject_voxel '
                           'data/10076_classE_32_sim.star '
                           '--output output/00_backproject/classE_sim.mrc '
                           '--uninvert-data')
    # dose weight
    tester.commands.append('tomodrgn '
                           'backproject_voxel '
                           'data/10076_classE_32_sim.star '
                           '--output output/00_backproject/classE_sim_doseweight.mrc '
                           '--uninvert-data '
                           '--recon-dose-weight')
    # tilt weight
    tester.commands.append('tomodrgn '
                           'backproject_voxel '
                           'data/10076_classE_32_sim.star '
                           '--output output/00_backproject/classE_sim_tiltweight.mrc '
                           '--uninvert-data '
                           '--recon-tilt-weight')
    # lazy
    tester.commands.append('tomodrgn '
                           'backproject_voxel '
                           'data/10076_classE_32_sim.star '
                           '--output output/00_backproject/classE_sim_lazy.mrc '
                           '--uninvert-data '
                           '--lazy')
    # manual lowpass filter
    tester.commands.append('tomodrgn '
                           'backproject_voxel '
                           'data/10076_classE_32_sim.star '
                           '--output output/00_backproject/classE_sim_lowpass60.mrc '
                           '--uninvert-data '
                           '--lowpass 60')
    return tester


def add_tests_train_nn(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('tomodrgn '
                           'train_nn data/10076_classE_32_sim.star '
                           '--outdir output/nn_classE_sim '
                           '--uninvert-data '
                           '--seed 42 '
                           '--layers 3 '
                           '--dim 256 '
                           '--num-epochs 40')
    # lazy
    tester.commands.append('tomodrgn '
                           'train_nn '
                           'data/10076_classE_32_sim.star '
                           '--outdir output/nn_classE_sim_lazy '
                           '--uninvert-data '
                           '--seed 42 '
                           '--layers 3 '
                           '--dim 256 '
                           '--num-epochs 1 '
                           '--lazy')
    # dose weight, tilt weight, recon dose mask
    tester.commands.append('tomodrgn '
                           'train_nn data/10076_classE_32_sim.star '
                           '--outdir output/nn_classE_sim_dosetiltweightmask '
                           '--uninvert-data '
                           '--seed 42 '
                           '--layers 3 '
                           '--dim 256 '
                           '--num-epochs 40 '
                           '--l-dose-mask '
                           '--recon-dose-weight '
                           '--recon-tilt-weight')
    # batch size 8
    tester.commands.append('tomodrgn '
                           'train_nn data/10076_classE_32_sim.star '
                           '--outdir output/nn_classE_sim_dosetiltweightmask_batchsize8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--layers 3 '
                           '--dim 256 '
                           '--num-epochs 1 '
                           '--l-dose-mask '
                           '--recon-dose-weight '
                           '--recon-tilt-weight '
                           '--batch-size 8')

    return tester


def add_tests_convergence_nn(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('tomodrgn '
                           'convergence_nn '
                           'output/nn_classE_sim '
                           'data/10076_classE_32.mrc '
                           '--fsc-mask none')
    # dose weight tilt weight recon dose mask, no fsc mask
    tester.commands.append('tomodrgn '
                           'convergence_nn '
                           'output/nn_classE_sim_dosetiltweightmask '
                           'data/10076_classE_32.mrc '
                           '--fsc-mask none')
    # dose weight tilt weight recon dose mask, spherical fsc mask
    tester.commands.append('tomodrgn '
                           'convergence_nn '
                           'output/nn_classE_sim_dosetiltweightmask '
                           'data/10076_classE_32.mrc '
                           '--fsc-mask sphere')
    # dose weight tilt weight recon dose mask, tight fsc mask
    tester.commands.append('tomodrgn '
                           'convergence_nn '
                           'output/nn_classE_sim_dosetiltweightmask '
                           'data/10076_classE_32.mrc '
                           '--fsc-mask tight')
    # dose weight tilt weight recon dose mask, soft fsc mask
    tester.commands.append('tomodrgn '
                           'convergence_nn '
                           'output/nn_classE_sim_dosetiltweightmask '
                           'data/10076_classE_32.mrc '
                           '--fsc-mask soft')

    return tester


def add_tests_train_vae(tester: CommandTester) -> CommandTester:
    # baseline, 2 class heterogeneity, zdim 2
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '--outdir output/vae_both_sim_zdim2 '
                           '--zdim 2 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '--num-epochs 40')
    # baseline, 2 class heterogeneity, zdim 8
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '--outdir output/vae_both_sim_zdim8 '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '-n 1')
    # 2 class heterogeneity, zdim 8, dose weight, tilt wieght, recon dose mask
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '--outdir output/vae_both_sim_zdim8_dosetiltweightmask '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '-n 1 '
                           '--l-dose-mask '
                           '--recon-dose-weight '
                           '--recon-tilt-weight ')
    # 2 class heterogeneity, zdim 8, dose weight, tilt wieght, recon dose mask, lazy
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '--outdir output/vae_both_sim_zdim8_dosetiltweightmask_lazy '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '-n 1 '
                           '--lazy '
                           '--l-dose-mask '
                           '--recon-dose-weight '
                           '--recon-tilt-weight ')
    # 2 class heterogeneity, zdim 8, dose weight, tilt wieght, recon dose mask, batchsize 8
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '--outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '--num-epochs 40 '
                           '--l-dose-mask '
                           '--recon-dose-weight '
                           '--recon-tilt-weight '
                           '--batch-size 8')
    # 1 class heterogeneity, zdim 8
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_classE_32_sim.star '
                           '--outdir output/vae_classE_sim_zdim8 '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '--num-epochs 40')
    # 2 class heterogeneity, zdim 8, pooling method concatenate
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '--outdir output/vae_both_sim_zdim8_concatenate '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--pooling-function concatenate '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '-n 1 ')
    # 2 class heterogeneity, zdim 8, pooling method max
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '--outdir output/vae_both_sim_zdim8_max '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--pooling-function max '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '-n 1 ')
    # 2 class heterogeneity, zdim 8, pooling method mean
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '-o output/vae_both_sim_zdim8_mean '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--pooling-function mean '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '-n 1 ')
    # 2 class heterogeneity, zdim 8, pooling method median
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '--outdir output/vae_both_sim_zdim8_median '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--pooling-function median '
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '-n 1 ')
    # 2 class heterogeneity, zdim 8, pooling method setencoder
    tester.commands.append('tomodrgn '
                           'train_vae '
                           'data/10076_both_32_sim.star '
                           '--outdir output/vae_both_sim_zdim8_setencoder '
                           '--zdim 8 '
                           '--uninvert-data '
                           '--seed 42 '
                           '--log-interval 100 '
                           '--enc-dim-A 64 '
                           '--enc-layers-A 2 '
                           '--out-dim-A 64 '
                           '--pooling-function set_encoder '  # may require --no-amp
                           '--enc-dim-B 32 '
                           '--enc-layers-B 4 '
                           '--dec-dim 256 '
                           '--dec-layers 3 '
                           '--num-seeds 1 '
                           '--num-heads 2 '
                           '--layer-norm '
                           '-n 1 ')

    return tester


def add_tests_convergence_vae(tester: CommandTester) -> CommandTester:
    # baseline, 2 class heterogeneity, zdim 2
    tester.commands.append('tomodrgn '
                           'convergence_vae '
                           'output/vae_both_sim_zdim2 '
                           'latest '
                           '--random-seed 42 '
                           '--final-maxima 2 '
                           '--ground-truth data/10076_class*_32.mrc')
    # 2 class heterogeneity, zdim 8, dose weight, tilt wieght, recon dose mask, batchsize 8
    tester.commands.append('tomodrgn '
                           'convergence_vae '
                           'output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 '
                           'latest '
                           '--random-seed 42 '
                           '--final-maxima 2 '
                           '--ground-truth data/10076_class*_32.mrc')

    return tester


def add_tests_analyze(tester: CommandTester) -> CommandTester:
    # baseline, 2 class heterogeneity, zdim 2
    tester.commands.append('tomodrgn '
                           'analyze '
                           'output/vae_both_sim_zdim2 39 '
                           '--ksample 20')
    # 2 class heterogeneity, zdim 8, dose weight, tilt wieght, recon dose mask, batchsize 8
    tester.commands.append('tomodrgn '
                           'analyze '
                           'output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 39 '
                           '--ksample 20')
    # 1 class heterogeneity, zdim 8
    tester.commands.append('tomodrgn '
                           'analyze '
                           'output/vae_classE_sim_zdim8 39 '
                           '--ksample 20')

    return tester


def add_tests_eval_vol(tester: CommandTester) -> CommandTester:
    # baseline, 2 class heterogeneity, zdim 2 -> eval batch size 1
    tester.commands.append('tomodrgn '
                           'eval_vol '
                           '--weights output/vae_both_sim_zdim2/weights.pkl '
                           '-c output/vae_both_sim_zdim2/config.pkl '
                           '-o output/vae_both_sim_zdim2/eval_vol_allz '
                           '--zfile output/vae_both_sim_zdim2/z.train.pkl ')
    # baseline, 2 class heterogeneity, zdim 2 -> eval batch size 32
    tester.commands.append('tomodrgn '
                           'eval_vol '
                           '--weights output/vae_both_sim_zdim2/weights.pkl '
                           '-c output/vae_both_sim_zdim2/config.pkl '
                           '-o output/vae_both_sim_zdim2/eval_vol_allz '
                           '--zfile output/vae_both_sim_zdim2/z.train.pkl '
                           '-b 32')

    return tester


def add_tests_analyze_volumes(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('tomodrgn '
                           'analyze_volumes '
                           '--voldir output/vae_both_sim_zdim2/eval_vol_allz '
                           '--config output/vae_both_sim_zdim2/config.pkl '
                           '--outdir output/vae_both_sim_zdim2/eval_vol_allz_analyze_volumes_mask_sphere '
                           '--ksample 20 '
                           '--mask sphere ')
    # soft mask (unique per vol)
    tester.commands.append('tomodrgn '
                           'analyze_volumes '
                           '--voldir output/vae_both_sim_zdim2/eval_vol_allz '
                           '--config output/vae_both_sim_zdim2/config.pkl '
                           '--outdir output/vae_both_sim_zdim2/eval_vol_allz_analyze_volumes_mask_soft '
                           '--ksample 20 '
                           '--mask soft ')
    return tester


def add_tests_subtomo2chimerax(tester: CommandTester) -> CommandTester:
    # mode markers
    tester.commands.append('tomodrgn '
                           'subtomo2chimerax '
                           'data/10076_both_32_sim_vols.star '
                           '--mode markers '
                           '--outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_markers '
                           '--tomoname both.tomostar '
                           '--star-angpix-override 10 '
                           '--coloring-labels data/ptcl_labels_D0_E1.pkl')
    # mode volume
    tester.commands.append('tomodrgn '
                           'subtomo2chimerax '
                           'data/10076_both_32_sim_vols.star '
                           '--mode volume '
                           '--outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volume '
                           '--tomoname both.tomostar '
                           '--star-angpix-override 10 '
                           '--vol-path output/00_backproject/classE_sim_doseweight.mrc '
                           '--vol-render-level 0.7 '
                           '--coloring-labels data/ptcl_labels_D0_E1.pkl')
    # mode volumes
    tester.commands.append('tomodrgn '
                           'subtomo2chimerax '
                           'data/10076_both_32_sim_vols.star '
                           '--mode volumes '
                           '--outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volumes '
                           '--tomoname both.tomostar '
                           '--star-angpix-override 10 '
                           '--weights output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/weights.39.pkl '
                           '--config output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/config.pkl '
                           '--zfile output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl '
                           '--vol-render-level 0.7 '
                           '--coloring-labels data/ptcl_labels_D0_E1.pkl')
    # mode volumes cmap tab10
    tester.commands.append('tomodrgn '
                           'subtomo2chimerax '
                           'data/10076_both_32_sim_vols.star '
                           '--mode volumes '
                           '--outdir output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volumes_cmaptab10 '
                           '--tomoname both.tomostar '
                           '--star-angpix-override 10 '
                           '--weights output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/weights.39.pkl '
                           '--config output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/config.pkl '
                           '--zfile output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl '
                           '--vol-render-level 0.7 '
                           '--coloring-labels data/ptcl_labels_D0_E1.pkl '
                           '--colormap tab10 ')

    return tester


def add_tests_filter_star(tester: CommandTester) -> CommandTester:
    # baseline TiltSeriesStarfile
    tester.commands.append('tomodrgn '
                           'filter_star data/10076_both_32_sim.star '
                           '--starfile-type imageseries '
                           '--tomo-id-col _rlnImageName '
                           '--ind data/ind_ptcl_first10last10.pkl '
                           '-o output/10076_both_32_sim_filtered.star')
    # baseline volumeseries star file
    tester.commands.append('tomodrgn '
                           'filter_star '
                           'data/10076_both_32_sim_vols.star '
                           '--starfile-type volumeseries '
                           '--tomo-id-col _rlnImageName '
                           '--ind data/ind_ptcl_first10last10.pkl '
                           '-o output/10076_both_32_sim_vols_filtered.star')

    return tester


def add_tests_eval_images(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('tomodrgn '
                           'eval_images '
                           'data/10076_classE_32_sim.star '
                           '--weights output/vae_classE_sim_zdim8/weights.pkl '
                           '-c output/vae_classE_sim_zdim8/config.pkl '
                           '--out-z output/vae_classE_sim_zdim8/eval_images/z_all.pkl')

    return tester


def add_tests_graph_traversal(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('tomodrgn '
                           'graph_traversal '
                           'output/vae_both_sim_zdim2/z.39.train.pkl '
                           '--anchors 5 10 15 20 '
                           '-o output/vae_both_sim_zdim2/graph_traversal '
                           '--max-neighbors 20 '
                           '--avg-neighbors 20')
    # zdim8
    tester.commands.append('tomodrgn '
                           'graph_traversal '
                           'output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl '
                           '--anchors 5 10 15 20 '
                           '-o output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/graph_traversal '
                           '--max-neighbors 20 '
                           '--avg-neighbors 10')

    return tester


def add_tests_pc_traversal(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('tomodrgn '
                           'pc_traversal '
                           'output/vae_both_sim_zdim2/z.train.pkl '
                           '-o output/vae_both_sim_zdim2/pc_traversal')
    # percentile spacing
    tester.commands.append('tomodrgn '
                           'pc_traversal '
                           'output/vae_both_sim_zdim2/z.train.pkl '
                           '-o output/vae_both_sim_zdim2/pc_traversal '
                           '--use-percentile-spacing')
    # zdim8
    tester.commands.append('tomodrgn '
                           'pc_traversal '
                           'output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.train.pkl '
                           '-o output/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/pc_traversal '
                           '--use-percentile-spacing')

    return tester


def add_tests_view_config(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('tomodrgn '
                           'view_config '
                           'output/vae_both_sim_zdim2 ')
    return tester


def add_tests_cleanup(tester: CommandTester) -> CommandTester:
    # baseline
    tester.commands.append('cp -R output/vae_both_sim_zdim2 output/vae_both_sim_zdim2_copy_cleaned; '
                           'tomodrgn '
                           'cleanup '
                           'output/vae_both_sim_zdim2_copy_cleaned '
                           '--weights '
                           '--zfiles '
                           '--volumes '
                           '--test ')
    tester.commands.append('tomodrgn '
                           'cleanup '
                           'output/vae_both_sim_zdim2_copy_cleaned '
                           '--weights '
                           '--zfiles '
                           '--volumes ')
    return tester


def main():

    # remove pre-existing output and create new output folder
    workdir = 'output'
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)

    # instantiate the tester
    tester = CommandTester(workdir, verbose=False)

    # add the tests
    add_tests_downsample(tester)
    add_tests_backproject_voxel(tester)
    add_tests_train_nn(tester)
    add_tests_convergence_nn(tester)
    add_tests_train_vae(tester)
    add_tests_convergence_vae(tester)
    add_tests_analyze(tester)
    add_tests_eval_vol(tester)
    add_tests_analyze_volumes(tester)
    add_tests_subtomo2chimerax(tester)
    add_tests_filter_star(tester)
    add_tests_eval_images(tester)
    add_tests_graph_traversal(tester)
    add_tests_pc_traversal(tester)
    add_tests_view_config(tester)
    add_tests_cleanup(tester)

    # run the tests
    tester.run()

    # report the results
    tester.report_run_summary()


if __name__ == '__main__':
    main()
