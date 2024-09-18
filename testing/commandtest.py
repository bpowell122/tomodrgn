"""
Test all primary tomoDRGN commands and common argument permutations for error-free functionality.
Takes about 50 minutes on a 2020 MacBook Pro or 10 minutes on an Ubuntu workstation, and creates about 900 MB of outputs.
Note that outputs of some command tests are used as inputs to others.
"""
from testing_module import run_assert_no_error


# tomodrgn version
def test_version(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command='tomodrgn --version')


# tomodrgn downsample
def test_downsample(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn downsample data/10076_classE_32_sim.star --downsample 16 --batch-size 50 --output {output_dir}/10076_classE_16_sim.mrcs --write-tiltseries-starfile')

def test_downsample_lazy(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn downsample data/10076_classE_32_sim.star --downsample 16 --batch-size 50 --output {output_dir}/10076_classE_16_sim.mrcs --write-tiltseries-starfile --lazy')

def test_downsample_chunk(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn downsample data/10076_classE_32_sim.star --downsample 16 --batch-size 50 --output {output_dir}/10076_classE_16_sim.mrcs --write-tiltseries-starfile --chunk 100')


# tomodrgn backproject_voxel
def test_backproject(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn backproject_voxel data/10076_classE_32_sim.star --output {output_dir}/backproject/classE_sim.mrc --uninvert-data')

def test_backproject_dose_weight(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn backproject_voxel data/10076_classE_32_sim.star --output {output_dir}/backproject/classE_sim_doseweight.mrc --uninvert-data --recon-dose-weight')

def test_backproject_tilt_weight(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn backproject_voxel data/10076_classE_32_sim.star --output {output_dir}/backproject/classE_sim_tiltweight.mrc --uninvert-data --recon-tilt-weight')

def test_backproject_lazy(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn backproject_voxel data/10076_classE_32_sim.star --output {output_dir}/backproject/classE_sim_lazy.mrc --uninvert-data --lazy')

def test_backproject_lowpass(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn backproject_voxel data/10076_classE_32_sim.star --output {output_dir}/backproject/classE_sim_lowpass60.mrc --uninvert-data --lowpass 60')


# tomodrgn train_nn
def test_train_nn(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_nn data/10076_classE_32_sim.star --outdir {output_dir}/nn_classE_sim --uninvert-data --seed 42 --layers 3 --dim 256 --num-epochs 40')

def test_train_nn_lazy(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_nn data/10076_classE_32_sim.star --outdir {output_dir}/nn_classE_sim_lazy --uninvert-data --seed 42 --layers 3 --dim 256 --num-epochs 1 --lazy')

def test_train_nn_dose_tilt_weight_dose_mask(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_nn data/10076_classE_32_sim.star --outdir {output_dir}/nn_classE_sim_dosetiltweightmask --uninvert-data --seed 42 --layers 3 --dim 256 --num-epochs 40 --l-dose-mask --recon-dose-weight --recon-tilt-weight')

def test_train_nn_dose_tilt_weight_dose_mask_batchsize8(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_nn data/10076_classE_32_sim.star --outdir {output_dir}/nn_classE_sim_dosetiltweightmask_batchsize8 --uninvert-data --seed 42 --layers 3 --dim 256 --num-epochs 1 --l-dose-mask --recon-dose-weight --recon-tilt-weight --batch-size 8'
)

# tomodrgn train_nn
def test_convergence_nn(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn convergence_nn {output_dir}/nn_classE_sim data/10076_classE_32.mrc --fsc-mask none')

def test_convergence_nn_dose_tilt_weight_dose_mask(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn convergence_nn {output_dir}/nn_classE_sim_dosetiltweightmask data/10076_classE_32.mrc --fsc-mask none')

def test_convergence_nn_dose_tilt_weight_dose_mask_mask_sphere(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn convergence_nn {output_dir}/nn_classE_sim_dosetiltweightmask data/10076_classE_32.mrc --fsc-mask sphere')

def test_convergence_nn_dose_tilt_weight_dose_mask_mask_tight(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn convergence_nn {output_dir}/nn_classE_sim_dosetiltweightmask data/10076_classE_32.mrc --fsc-mask tight')

def test_convergence_nn_dose_tilt_weight_dose_mask_mask_soft(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn convergence_nn {output_dir}/nn_classE_sim_dosetiltweightmask data/10076_classE_32.mrc --fsc-mask soft')


# tomodrgn train_vae
def test_train_vae(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim2 --zdim 2 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 --num-epochs 40')

def test_train_vae_zdim8(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim8 --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 1')

def test_train_vae_dose_tilt_weight_dose_mask(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim8_dosetiltweightmask --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 1 --l-dose-mask --recon-dose-weight --recon-tilt-weight')

def test_train_vae_dose_tilt_weight_dose_mask_lazy(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_lazy --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 1 --lazy --l-dose-mask --recon-dose-weight --recon-tilt-weight')

def test_train_vae_dose_tilt_weight_dose_mask_batchsize8(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 --num-epochs 40 --l-dose-mask --recon-dose-weight --recon-tilt-weight --batch-size 8')

def test_train_vae_nohet(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_classE_32_sim.star --outdir {output_dir}/vae_classE_sim_zdim8 --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 --num-epochs 40')

def test_train_vae_pooling_concatenate(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim8_concatenate --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --pooling-function concatenate --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 1')

def test_train_vae_pooling_max(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim8_max --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --pooling-function max --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 1')

def test_train_vae_pooling_mean(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim8_mean --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --pooling-function mean --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 1')

def test_train_vae_pooling_median(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim8_median --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --pooling-function median --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 1')

def test_train_vae_pooling_setencoder(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn train_vae data/10076_both_32_sim.star --outdir {output_dir}/vae_both_sim_zdim8_setencoder --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --pooling-function set_encoder --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 --num-seeds 1 --num-heads 2 --layer-norm -n 1')


# tomodrgn convergence_vae
def test_convergence_vae_zdim2(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn convergence_vae {output_dir}/vae_both_sim_zdim2 --random-seed 42 --final-maxima 2 --ground-truth data/10076_class*_32.mrc')

def test_convergence_vae_dose_tilt_weight_dose_mask_batchsize8(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn convergence_vae {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 --random-seed 42 --final-maxima 2 --ground-truth data/10076_class*_32.mrc')


# tomodrgn analyze
def test_analyze_zdim2(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn analyze {output_dir}/vae_both_sim_zdim2 --ksample 20')

def test_analyze_dose_tilt_weight_dose_mask_batchsize8(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn analyze {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8 --ksample 20')

def test_analyze_zdim8(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn analyze {output_dir}/vae_classE_sim_zdim8 --ksample 20')


# tomodrgn eval_vol
def test_eval_vol(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn eval_vol --weights {output_dir}/vae_both_sim_zdim2/weights.pkl -c {output_dir}/vae_both_sim_zdim2/config.pkl -o {output_dir}/vae_both_sim_zdim2/eval_vol_allz --zfile {output_dir}/vae_both_sim_zdim2/z.train.pkl')

def test_eval_vol_batchsize32(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn eval_vol --weights {output_dir}/vae_both_sim_zdim2/weights.pkl -c {output_dir}/vae_both_sim_zdim2/config.pkl -o {output_dir}/vae_both_sim_zdim2/eval_vol_allz --zfile {output_dir}/vae_both_sim_zdim2/z.train.pkl -b 32')


# tomodrgn analyze_volumes
def test_analyze_volumes(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn analyze_volumes --voldir {output_dir}/vae_both_sim_zdim2/eval_vol_allz --config {output_dir}/vae_both_sim_zdim2/config.pkl --outdir {output_dir}/vae_both_sim_zdim2/eval_vol_allz_analyze_volumes_mask_sphere --ksample 20 --mask sphere')

def test_analyze_volumes_mask_soft(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn analyze_volumes --voldir {output_dir}/vae_both_sim_zdim2/eval_vol_allz --config {output_dir}/vae_both_sim_zdim2/config.pkl --outdir {output_dir}/vae_both_sim_zdim2/eval_vol_allz_analyze_volumes_mask_soft --ksample 20 --mask soft')


# tomodrgn subtomo2chimerax
def test_subtomo2chimerax_mode_markers(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn subtomo2chimerax data/10076_both_32_sim_vols.star --mode markers --outdir {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_markers --tomoname both.tomostar --star-angpix-override 10 --coloring-labels data/ptcl_labels_D0_E1.pkl')

def test_subtomo2chimerax_mode_volume(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn subtomo2chimerax data/10076_both_32_sim_vols.star --mode volume --outdir {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volume --tomoname both.tomostar --star-angpix-override 10 --vol-path {output_dir}/00_backproject/classE_sim_doseweight.mrc --vol-render-level 0.7 --coloring-labels data/ptcl_labels_D0_E1.pkl')

def test_subtomo2chimerax_mode_volumes(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn subtomo2chimerax data/10076_both_32_sim_vols.star --mode volumes --outdir {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volumes --tomoname both.tomostar --star-angpix-override 10 --weights {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/weights.39.pkl --config {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/config.pkl --zfile {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl --vol-render-level 0.7 --coloring-labels data/ptcl_labels_D0_E1.pkl')

def test_subtomo2chimerax_cmap_tab10(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn subtomo2chimerax data/10076_both_32_sim_vols.star --mode volumes --outdir {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/mapback_volumes_cmaptab10 --tomoname both.tomostar --star-angpix-override 10 --weights {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/weights.39.pkl --config {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/config.pkl --zfile {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl --vol-render-level 0.7 --coloring-labels data/ptcl_labels_D0_E1.pkl --colormap tab10')


# tomodrgn filter_star
def test_filter_star_tiltseriesstarfile(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn filter_star data/10076_both_32_sim.star --starfile-type imageseries --tomo-id-col _rlnImageName --ind data/ind_ptcl_first10last10.pkl -o {output_dir}/10076_both_32_sim_filtered.star')

def test_filter_star_volumeseriesstarfile(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn filter_star data/10076_both_32_sim_vols.star --starfile-type volumeseries --tomo-id-col _rlnImageName --ind data/ind_ptcl_first10last10.pkl -o {output_dir}/10076_both_32_sim_vols_filtered.star')


# tomodrgn eval_images
def test_eval_images(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn eval_images data/10076_classE_32_sim.star --weights {output_dir}/vae_classE_sim_zdim8/weights.pkl -c {output_dir}/vae_classE_sim_zdim8/config.pkl --out-z {output_dir}/vae_classE_sim_zdim8/eval_images/z_all.pkl')


# tomodrgn graph_traversal
def test_graph_traversal(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn graph_traversal {output_dir}/vae_both_sim_zdim2/z.39.train.pkl --anchors 5 10 15 20 -o {output_dir}/vae_both_sim_zdim2/graph_traversal --max-neighbors 20 --avg-neighbors 20')

def test_graph_traversal_zdim8(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn graph_traversal {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.39.train.pkl --anchors 5 10 15 20 -o {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/graph_traversal --max-neighbors 20 --avg-neighbors 10')


# tomodrgn pc_traversal
def test_pc_traversal(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn pc_traversal {output_dir}/vae_both_sim_zdim2/z.train.pkl -o {output_dir}/vae_both_sim_zdim2/pc_traversal')

def test_pc_traversal_percentile_spacing(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn pc_traversal {output_dir}/vae_both_sim_zdim2/z.train.pkl -o {output_dir}/vae_both_sim_zdim2/pc_traversal --use-percentile-spacing')

def test_pc_traversal_zdim8(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn pc_traversal {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/z.train.pkl -o {output_dir}/vae_both_sim_zdim8_dosetiltweightmask_batchsize8/pc_traversal --use-percentile-spacing')


# tomodrgn view_config
def test_view_config(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn view_config {output_dir}/vae_both_sim_zdim2')


# tomodrgn cleanup
def test_cleanup_dryrun(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'cp -R {output_dir}/vae_both_sim_zdim2 {output_dir}/vae_both_sim_zdim2_copy_cleaned')
    result = run_assert_no_error(script_runner, command=f'tomodrgn cleanup {output_dir}/vae_both_sim_zdim2_copy_cleaned --weights --zfiles --volumes --test')

def test_cleanup_delete(script_runner, output_dir):
    result = run_assert_no_error(script_runner, command=f'tomodrgn cleanup {output_dir}/vae_both_sim_zdim2_copy_cleaned --weights --zfiles --volumes')
