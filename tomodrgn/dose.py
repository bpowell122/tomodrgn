import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from . import utils
log = utils.log


def calculate_dose_weights(particles_starfile_obj, dose_override, ntilts, ny_ht, nx_ht, nx, ny):
    '''
    code adapted from Grigorieff lab summovie_1.0.2/src/core/electron_dose.f90
    see also Grant and Grigorieff, eLife (2015) DOI: 10.7554/eLife.06980
    '''
    voltage = particles_starfile_obj.get_tiltseries_voltage()  # kV
    voltage_scaling_factor = 1.0 if voltage == 300 else 0.8  # 1.0 for 300kV, 0.8 for 200kV microscopes

    if dose_override is None:
        dose_per_A2_per_tilt = particles_starfile_obj.get_tiltseries_dose_per_A2_per_tilt(ntilts)  # electrons per square angstrom per tilt micrograph
    else:
        # increment scalar dose_override across ntilts
        dose_per_A2_per_tilt = dose_override * np.arange(1, ntilts+1)

    spatial_frequencies = get_spatial_frequencies(particles_starfile_obj.df, ny_ht, nx_ht, nx, ny)
    spatial_frequencies[ny // 2, nx // 2] = 1  # setting DC component to full weight to avoid divide by zero error in eq 3
    spatial_frequency_critical_dose = (0.24499 * np.power(spatial_frequencies, -1.6649) + 2.8141) * voltage_scaling_factor  # eq 3 from DOI: 10.7554/eLife.06980
    spatial_frequency_optimal_dose = 2.51284 * spatial_frequency_critical_dose

    dose_weights = np.zeros((ntilts, ny_ht, nx_ht))
    for k, dose_at_end_of_tilt in enumerate(dose_per_A2_per_tilt):
        dose_at_start_of_tilt = 0 if k == 0 else dose_per_A2_per_tilt[k-1]
        dose_weights[k] = np.where(abs(dose_at_end_of_tilt - spatial_frequency_optimal_dose) < abs(dose_at_start_of_tilt - spatial_frequency_optimal_dose),
                                   np.exp((-0.5 * dose_at_end_of_tilt) / spatial_frequency_critical_dose),
                                   0.0)
    dose_weights[:, ny // 2, nx // 2] = 1.0  # setting DC component to full weight

    assert dose_weights.min() >= 0.0
    assert dose_weights.max() <= 1.0

    return dose_weights


def get_spatial_frequencies(particles_df, ny_ht, nx_ht, nx, ny):
    # return spatial frequencies of ht_sym in 1/A
    pixel_size = float(particles_df['_rlnDetectorPixelSize'].iloc[0])  # angstroms per pixel
    spatial_frequencies = np.zeros((ny_ht, nx_ht))
    fourier_pixel_sizes = 1.0 / (np.array([nx, ny]))  # in units of 1/px
    box_center_indices = (np.array([nx, ny]) / 2).astype(int)  # this might break if nx, ny not even, or nx!=ny

    for j in range(ny_ht):
        y = ((j - box_center_indices[1]) * fourier_pixel_sizes[1])

        for i in range(nx_ht):
            x = ((i - box_center_indices[0]) * fourier_pixel_sizes[0])

            spatial_frequency = np.sqrt(x ** 2 + y ** 2) / pixel_size  # units of 1/A
            spatial_frequencies[j, i] = spatial_frequency

    return spatial_frequencies


def plot_weight_distribution(cumulative_weights, spatial_frequencies, outdir, weight_distribution_index = None):
    # plot distribution of dose weights across tilts in the spirit of https://doi.org/10.1038/s41467-021-22251-8

    ntilts = cumulative_weights.shape[0]
    max_frequency_box_edge = spatial_frequencies[0,spatial_frequencies.shape[0]//2]
    sorted_frequency_list = sorted(set(spatial_frequencies[spatial_frequencies <= max_frequency_box_edge].reshape(-1)))
    weights_plot = np.zeros((len(sorted_frequency_list), ntilts))

    for i, frequency in enumerate(sorted_frequency_list):
        x, y = np.where(spatial_frequencies == frequency)
        sum_of_weights_at_frequency = cumulative_weights[:, y, x].sum()
        if not sum_of_weights_at_frequency == 0:
            weights_plot[i, :] = (cumulative_weights[:, y, x] / sum_of_weights_at_frequency).sum(axis=1) # sum across multiple pixels at same frequency

    colormap = plt.cm.get_cmap('coolwarm').reversed()
    tilt_colors = colormap(np.linspace(0, 1, ntilts))

    fig, ax = plt.subplots()
    ax.stackplot(sorted_frequency_list, weights_plot.T, colors=tilt_colors)
    ax.set_ylabel('cumulative weights')
    ax.set_xlabel('spatial frequency (1/Å)')
    ax.set_xlim((0, sorted_frequency_list[-1]))
    ax.set_ylim((0, 1))

    # ax.xaxis.set_major_locator(mticker.MaxNLocator())
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels([f'1/{1/xtick:.1f}' if xtick != 0.0 else 0 for xtick in ticks_loc])

    plt.savefig(f'{outdir}/weighting_scheme_{weight_distribution_index}.png', dpi=300)
    plt.close()

