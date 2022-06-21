import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from . import utils
log = utils.log


def calculate_dose_weights(particles_df, dose_override, ntilts, ny_ht, nx_ht, nx, ny):
    pixel_size = particles_df.get_tiltseries_pixelsize()  # angstroms per pixel
    voltage = particles_df.get_tiltseries_voltage()  # kV
    if dose_override is None:
        dose_per_A2_per_tilt = particles_df.get_tiltseries_dose_per_A2_per_tilt(ntilts)  # electrons per square angstrom per tilt micrograph
        log(f'Dose/A2/tilt series extracted from star file: {dose_per_A2_per_tilt}')
    else:
        # increment scalar dose_override across ntilts
        dose_per_A2_per_tilt = dose_override * np.arange(1, ntilts+1)
        log(f'Dose/A2/tilt override series supplied by user: {dose_per_A2_per_tilt}')

    # code adapted from Grigorieff lab summovie_1.0.2/src/core/electron_dose.f90
    # see also Grant and Grigorieff, eLife (2015) DOI: 10.7554/eLife.06980

    dose_weights = np.zeros((ntilts, ny_ht, nx_ht))
    fourier_pixel_sizes = 1.0 / (np.array([nx, ny]))  # in units of 1/px
    box_center_indices = (np.array([nx, ny]) / 2).astype(int)
    critical_dose_at_dc = 2 ** 31  # shorthand way to ensure dc component is always weighted ~1
    voltage_scaling_factor = 1.0 if voltage == 300 else 0.8  # 1.0 for 300kV, 0.8 for 200kV microscopes

    for k, dose_at_end_of_tilt in enumerate(dose_per_A2_per_tilt):
        if k == 0:
            dose_at_start_of_tilt = 0
        else:
            dose_at_start_of_tilt = dose_per_A2_per_tilt[k-1]

        for j in range(ny_ht):
            y = ((j - box_center_indices[1]) * fourier_pixel_sizes[1])

            for i in range(nx_ht):
                x = ((i - box_center_indices[0]) * fourier_pixel_sizes[0])

                if ((i, j) == box_center_indices).all():
                    spatial_frequency_critical_dose = critical_dose_at_dc
                else:
                    spatial_frequency = np.sqrt(x ** 2 + y ** 2) / pixel_size  # units of 1/A
                    spatial_frequency_critical_dose = (0.24499 * spatial_frequency ** (
                        -1.6649) + 2.8141) * voltage_scaling_factor  # eq 3 from DOI: 10.7554/eLife.06980

                # from electron_dose.f90:
                    # There is actually an analytical solution, found by Wolfram Alpha:
                    # optimal_dose = -critical_dose - 2*critical_dose*W(-1)(-1/(2*sqrt(e)))
                    # where W(k) is the analytic continuation of the product log function
                    # http://mathworld.wolfram.com/LambertW-Function.html
                    # However, there is an acceptable numerical approximation, which I
                    # checked using a spreadsheet and the above formula.
                    # Here, we use the numerical approximation:
                spatial_frequency_optimal_dose = 2.51284 * spatial_frequency_critical_dose

                if (abs(dose_at_end_of_tilt - spatial_frequency_optimal_dose) < abs(
                        dose_at_start_of_tilt - spatial_frequency_optimal_dose)):
                    dose_weights[k, j, i] = np.exp(
                        (-0.5 * dose_at_end_of_tilt) / spatial_frequency_critical_dose)  # eq 5 from DOI: 10.7554/eLife.06980
                else:
                    dose_weights[k, j, i] = 0.0

    assert dose_weights.min() >= 0.0
    assert dose_weights.max() <= 1.0
    return dose_weights


def get_spatial_frequencies(particles_df, ny_ht, nx_ht, nx, ny):
    # return spatial frequencies of ht_sym in 1/A
    pixel_size = particles_df.get_tiltseries_pixelsize()  # angstroms per pixel
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


def plot_weight_distribution(cumulative_weights, spatial_frequencies, outdir):
    # plot distribution of dose weights across tilts in the spirit of https://doi.org/10.1038/s41467-021-22251-8

    ntilts = cumulative_weights.shape[0]
    max_frequency_box_edge = spatial_frequencies[0,spatial_frequencies.shape[0]//2]
    sorted_frequency_list = sorted(set(spatial_frequencies[spatial_frequencies < max_frequency_box_edge].reshape(-1)))
    weights_plot = np.empty((len(sorted_frequency_list), ntilts))

    for i, frequency in enumerate(sorted_frequency_list):
        x, y = np.where(spatial_frequencies == frequency)
        sum_of_weights_at_frequency = cumulative_weights[:, y, x].sum()
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

    plt.savefig(f'{outdir}/cumulative_weights_across_frequencies_by_tilt.png', dpi=300)

