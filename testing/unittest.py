'''
Test all primary tomoDRGN commands and common argument permutations for error-free functionality
'''

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
    tester = CommandTester(workdir)

    # add the tests

    # Test downsampling for clean particles --> should get particle stack at box16 from box32 input
    tester.commands.append('tomodrgn downsample data/10076_classE_32_clean.star -D 16 -b 50 -o output/10076_classE_16_clean.mrcs')

    # Test backproject_voxel for clean_particles --> should get *very* low res 50S volume
    tester.commands.append('tomodrgn backproject_voxel data/10076_classE_32_clean.star -o output/00_backproject/classE_clean.mrc --uninvert-data')

    # Test train_nn for clean particles (testing homogeneous reconstruction) --> should get 50S ribosome volume and loss ~0.1
    tester.commands.append('tomodrgn train_nn data/10076_classE_32_clean.star -o output/01_nn_classE_clean --uninvert-data --seed 42 --Apix 13.1')
    # Test train_nn for simulated particles (testing CTF correction and noisy data learning) --> should get noisier 50S ribosome volume and loss ~0.88
    tester.commands.append('tomodrgn train_nn data/10076_classE_32_sim.star -o output/02_nn_classE_sim --uninvert-data --seed 42')
    # Test train_nn for simulated particles (testing lazy loading) --> should get noisier 50S ribosome volume and loss ~0.88
    tester.commands.append('tomodrgn train_nn data/10076_classE_32_sim.star -o output/03_nn_classE_sim_lazy --uninvert-data --seed 42 --lazy')
    # Test train_nn for simulated particles (testing dose/tilt masking/weighting with dose override) --> should get similar ribosome volume, loss ~0.91, 3068 pixels decoded per particle down from 7960 without --l-dose-mask
    tester.commands.append('tomodrgn train_nn data/10076_classE_32_sim.star -o output/04_nn_classE_sim_dosetiltweightmask --uninvert-data --seed 42 --l-dose-mask --recon-dose-weight --recon-tilt-weight --dose-override 100')

    # Test convergence_nn for simulated particles (testing fsc calculation and convergence with no mask) --> should get fsc 0.5 around 0.32, fsc integral around 0.2707
    tester.commands.append('tomodrgn convergence_nn output/02_nn_classE_sim data/10076_classE_32.mrc --fsc-mask none')
    # Test convergence_nn for simulated particles (testing fsc calculation and convergence with sphere mask) --> should get fsc 0.5 around 0.32, fsc integral around 0.2702
    tester.commands.append('tomodrgn convergence_nn output/02_nn_classE_sim data/10076_classE_32.mrc --fsc-mask sphere')
    # Test convergence_nn for simulated particles (testing fsc calculation and convergence with tight mask) --> should get fsc 0.5 around 0.47, fsc integral around 0.3834
    tester.commands.append('tomodrgn convergence_nn output/02_nn_classE_sim data/10076_classE_32.mrc --fsc-mask tight')
    # Test convergence_nn for simulated particles (testing fsc calculation and convergence with soft mask) --> should get fsc 0.5 around 0.32, fsc integral around 0.2721
    tester.commands.append('tomodrgn convergence_nn output/02_nn_classE_sim data/10076_classE_32.mrc --fsc-mask soft')

    # Test train_vae for clean particles (testing heterogeneity learning and working with different n_tilts per dataset) --> should get loss around 0.12/30.46/0.12 using 8 tilts per particle
    tester.commands.append('tomodrgn train_vae data/10076_both_32_clean.star -o output/05_vae_both_clean --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3')
    # Test train_vae for simulated particles (testing heterogeneity learning in noisy context) --> should get loss around 0.90/16.52/0.90 using 8 tilts per particle
    tester.commands.append('tomodrgn train_vae data/10076_both_32_sim.star -o output/06_vae_both_sim --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 50')
    # Test train_vae for simulated particles (testing multiple dose/tilt masking/weighting schemes) --> should get loss around 0.90/15.98/0.90 using 8 tilts per particle
    tester.commands.append('tomodrgn train_vae data/10076_both_32_sim.star -o output/07_vae_both_sim_dosetiltweightmask --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 --l-dose-mask --recon-dose-weight --recon-tilt-weight --dose-override 100')
    # Test train_vae for simulated particles (testing lazy loading) --> should get loss around 0.90/16.52/0.90 using 8 tilts per particle
    tester.commands.append('tomodrgn train_vae data/10076_both_32_sim.star -o output/07_vae_both_sim_dosetiltweightmask --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 --lazy --l-dose-mask --recon-dose-weight --recon-tilt-weight --dose-override 100')
    # Test train_vae for simulated particles with no heterogeneity --> should get featureless continuous latent space
    tester.commands.append('tomodrgn train_vae data/10076_classE_32_sim.star -o output/08_vae_classE_sim --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 50')

    # Test train_vae for non-error functionality, correct encoderB dimensions and pooling, and basic heterogeneity learning
    tester.commands.append('tomodrgn train_vae data/10076_both_32_clean.star -o output/09_vae_both_clean_concatenate --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 -n 5 --enc-layers-A 2 --out-dim-A 64 --pooling-function concatenate --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3')
    tester.commands.append('tomodrgn train_vae data/10076_both_32_clean.star -o output/10_vae_both_clean_max --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 -n 5 --enc-layers-A 2 --out-dim-A 64 --pooling-function max --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3')
    tester.commands.append('tomodrgn train_vae data/10076_both_32_clean.star -o output/11_vae_both_clean_mean --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 -n 5 --enc-layers-A 2 --out-dim-A 64 --pooling-function mean --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3')
    tester.commands.append('tomodrgn train_vae data/10076_both_32_clean.star -o output/12_vae_both_clean_median --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 -n 5 --enc-layers-A 2 --out-dim-A 64 --pooling-function median --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3')
    tester.commands.append('tomodrgn train_vae data/10076_both_32_clean.star -o output/13_vae_both_clean_setencoder --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 -n 5 --enc-layers-A 2 --out-dim-A 64 --pooling-function set_encoder --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 --num-seeds 1 --num-heads 2 --layer-norm  # may require --no-amp')

    # Test convergence_vae for simulated particles
    tester.commands.append('tomodrgn convergence_vae output/06_vae_both_sim 49 --Apix 13.1 --random-seed 42 --final-maxima 2 --ground-truth data/10076_class*_32.mrc')

    # Test analyze for simulated particles (testing latent learning) --> should get noisy/continuous UMAP, kmeans2 with two distinct ribosome volumes, and pc1 showing dynamics in 50S ribosome belly occupancy
    tester.commands.append('tomodrgn analyze output/06_vae_both_sim 49 --Apix 13.1 --ksample 2')
    # Test analyze for simulated particles (testing latent learning baseline) --> should get continuous UMAP, kmeans2 with two similar ribosome volumes
    tester.commands.append('tomodrgn analyze output/08_vae_classE_sim 49 --Apix 13.1 --ksample 2')

    # Test eval_vol for all particles --> should get director with 144 unique volumes
    tester.commands.append('tomodrgn eval_vol --weights output/06_vae_both_sim/weights.pkl -c output/06_vae_both_sim/config.pkl -o output/06_vae_both_sim/eval_vol_allz --zfile output/06_vae_both_sim/z.pkl --Apix 13.1')
    # Test parallelized eval_vol for faster volume evaluation of many (>100) volumes
    tester.commands.append('tomodrgn eval_vol --weights output/06_vae_both_sim/weights.pkl -c output/06_vae_both_sim/config.pkl -o output/06_vae_both_sim/eval_vol_allz --zfile output/06_vae_both_sim/z.pkl --Apix 13.1 -b 32')

    # Test subtomo2chimerax for classD/classE volumes separately --> in ChimeraX should show all classE volumes to left (small X coord, label 1, light blue) and all classD volumes to right (large X coord, label 0, dark blue)
    tester.commands.append('tomodrgn subtomo2chimerax data/10076_both_32_sim_vols.star -o output/06_vae_both_sim/vol_allz_chimerax.cxc --tomoname both.tomostar --star-apix-override 10 --vols-dir output/06_vae_both_sim/eval_vol_allz --coloring-labels data/ptcl_labels_D0_E1.pkl')

    # Test filter_star isolating first and last 10 particles on imageseries star file --> should get starfile with 180 data rows
    tester.commands.append('tomodrgn filter_star data/10076_both_32_sim.star --tomo-id-col _rlnImageName --ptcl-id-col _rlnGroupName --ind data/ind_ptcl_first10last10.pkl -o output/10076_both_32_sim_filtered.star')
    # Test filter_star isolating first and last 10 particles on volumeseries star file --> should get starfile with 20 data rows
    tester.commands.append('tomodrgn filter_star data/10076_both_32_sim_vols.star --tomo-id-col _rlnImageName --ptcl-id-col index --ind data/ind_ptcl_first10last10.pkl -o output/10076_both_32_sim_vols_filtered.star')

    # Test eval_images with same images + weights as 08_vae_classE_sim --> z_all.pkl should be very close to z.pkl even with undertrained network 07 and different random image ordering
    tester.commands.append('tomodrgn eval_images data/10076_classE_32_sim.star --weights output/08_vae_classE_sim/weights.pkl -c output/08_vae_classE_sim/config.pkl --out-z output/08_vae_classE_sim/eval_images/z_all.pkl')
    # Test eval_images, should eventually compare this output (z_all.pkl) to that of 06_vae_both_sim/z.pkl for similarity given random sampled ntilts
    tester.commands.append('tomodrgn eval_images data/10076_both_32_sim.star --weights output/06_vae_both_sim/weights.pkl -c output/06_vae_both_sim/config.pkl --out-z output/06_vae_both_sim/eval_images/z_all.pkl')

    # Test graph_traversal --> should either report path indices and distance between neighbors or find no path
    tester.commands.append('tomodrgn graph_traversal output/06_vae_both_sim/z.0.pkl --anchors 137 10 20 -o output/06_vae_both_sim/graph_traversal/path.txt --out-z output/06_vae_both_sim/graph_traversal/z.path.txt')

    # Test pc_traversal
    tester.commands.append('tomodrgn pc_traversal output/06_vae_both_sim/z.pkl -o output/06_vae_both_sim/pc_traversal')
    tester.commands.append('tomodrgn pc_traversal output/06_vae_both_sim/z.pkl -o output/06_vae_both_sim/pc_traversal --use-percentile-spacing')

    # run the tests
    tester.run()

    # report the results
    tester.report_run_summary()


if __name__ == '__main__':
    main()