import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from . import utils
log = utils.log




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


def calculate_spatial_frequencies(pixel_size, D):
    '''
    Calculate spatial frequencies for image of width D-1 pixels with sampling of pixel_size Å/px
    Parameters
        pixel_size: scalar of pixel size in Å/px, dtype float
        D: scalar of fourier-symmetrized box width in pixels (typically odd, 1px larger than input image)
    Returns
        spatial_frequencies: numpy array of spatial frequency at each pixel (units 1/Å), shape (D, D), dtype float
    '''
    center = (D // 2, D // 2)

    Y, X = np.ogrid[:D, :D]
    spatial_frequencies = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (pixel_size * (D - 1))

    return spatial_frequencies


def calculate_critical_dose_per_frequency(spatial_frequencies, voltage):
    '''
    Calculate critical dose for spatial frequencies following Grant and Grigorieff 2015
    Params
        spatial_frequencies: numpy array of spatial frequency at each pixel (units 1/Å), shape (D, D), dtype float
        voltage: scalar of voltage of electron source in kv, dtype int
    Returns
        spatial_frequencies_critical_dose: numpy array of critical dose for each spatial frequency, shape (D, D), dtype float
    '''
    voltage_scaling_factor = 1.0 if voltage == 300 else 0.8  # 1.0 for 300kV, 0.8 for 200kV microscopes
    D = spatial_frequencies.shape[0]

    spatial_frequencies[D // 2, D // 2] = 1  # setting DC frequency to avoid divide by zero error
    spatial_frequencies_critical_dose = (0.24499 * np.power(spatial_frequencies, -1.6649) + 2.8141) * voltage_scaling_factor  # eq 3 from DOI: 10.7554/eLife.06980
    spatial_frequencies[D // 2, D // 2] = 0  # restoring DC frequency to zero

    return spatial_frequencies_critical_dose


def calculate_dose_weights(spatial_frequencies_critical_dose, cumulative_doses):
    '''
    Calculate weight associated with spatial frequencies given cumulative dose and critical dose
    Params
        spatial_frequencies_critical_dose: numpy array of critical dose for each spatial frequency, shape (D, D), dtype float
        cumulative_doses: numpy array of cumulative dose in e-/A2 at each tilt, shape (ntilts), dtype float
    Returns
        dose_weights: numpy array of weight for each spatial frequency at each cumulative dose, shape (ntilts, D, D), dtype float
    '''

    ntilts = cumulative_doses.shape[0]
    D = spatial_frequencies_critical_dose.shape[0]

    spatial_frequencies_optimal_dose = 2.51284 * spatial_frequencies_critical_dose

    # caution: not precisely equivalent to original summovie implementation due to potential random ordering of cumulative doses
    # thus taking a slightly more conservative approach that masks spatial frequencies at slightly lower doses than original
    cumulative_doses = cumulative_doses.reshape(ntilts, 1, 1)
    spatial_frequencies_critical_dose = spatial_frequencies_critical_dose.reshape(1, D, D)
    dose_weights = np.where(cumulative_doses <= spatial_frequencies_optimal_dose,
                            np.exp((-0.5 * cumulative_doses) / spatial_frequencies_critical_dose),
                            0.0)

    # DC component fully weighted
    dose_weights[:, D // 2, D // 2] = 1.

    return dose_weights


def calculate_tilt_weights(tilts):
    '''
    calculate weight per tilt image associated with higher tilt increasing optical path length and decreasing SNR
    Params
        tilts: numpy array of tilt angles in degrees, shape (ntilts,), dtype float
    Returns
        tilt_weights: numpy array of weight per tilt image, shape (ntilts,) dtype float
    '''
    tilt_weights = np.cos(tilts * np.pi / 180.)
    return tilt_weights


def combine_dose_tilt_weights(dose_weights, tilt_weights):
    '''
    Merge dose weights (per frequency and per tilt) with tilt weights (per tilt) for single output array
    Params
        dose_weights: numpy array of weight for each spatial frequency at each cumulative dose, shape (ntilts, D, D), dtype float
        tilt_weights: numpy array of weight per tilt image, shape (ntilts,) dtype float
    Returns
        weights: merged weights per frequency and per tilt, shape (ntilts, D, D), dtype float
    '''
    weights = dose_weights * tilt_weights.reshape(-1, 1, 1)
    return weights


def calculate_circular_mask(D):
    '''
    Produces binary mask of circle inscribed in box of side length D, with central pixel set to False (DC component)
    Params
        D: scalar of box size in pixels, dtype int
    Returns:
        circular_mask: numpy array of circular mask of pixels, shape (D, D), dtype bool
    '''
    center = (D // 2, D // 2)
    radius = D // 2

    Y, X = np.ogrid[:D, :D]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    circular_mask = dist_from_center <= radius
    circular_mask[center] = False

    return circular_mask


def calculate_dose_mask(dose_weights, circular_mask):
    '''
    Create mask as intersection of non-zero dose weights and circular mask
    Params
        dose_weights: numpy array of weight for each spatial frequency at each cumulative dose, shape (ntilts, D, D), dtype float
        circular_mask: numpy array of circle inscribed in image box, shape (D, D), dtype bool
    Returns
        dose_mask: numpy array of pixels contributing positively to SNR, shape (ntilts, nx, nx), dtype bool
    '''
    dose_mask = (dose_weights != 0.0) & circular_mask
    return dose_mask