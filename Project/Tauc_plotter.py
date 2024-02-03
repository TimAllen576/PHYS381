#!/usr/bin/env python

"""
Reads spectroscopy data for varying temps and wavelengths and uses a
Tauc-like plot method to determine energy of optical band-gap changes
due to temperature.

To customise the code check all points marked with # -------------------

Created Sept 2023
@author: Timothy Allen 66522411
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress
from matplotlib.widgets import Slider

NM_TO_EV = 1240
LOW_T_FILE = "20230906 REAL554 PHYS381 LowT.csv"  # ------------------
HIGH_T_FILE = "20230912 REAL554 PHYS381 HighT.csv"
INI_WAVELENGTH = 1000
FINAL_WAVELENGTH = 200
WAVELENGTH_STEP = 0.25
GLASS_INTERFERENCE_ENERGY = 5.2
SAMPLE_THICKNESS = 44 * 10 ** - 7  # Units of cm
R = 1 / 2  # r-value denoting the transition (1/2 is direct allowed)
FILTER_WINDOW = 99
FILTER_ORDER = 7
PLOT_X_LIMS = (2.5, 4)
PLOT_Y_LIMS = 0, 1.5 * 10 ** 12
DERIVATIVE_BUMP_WIDTH = 75
DERIVATIVE_LINEARITY_REQUIREMENT = 3 * 10 ** 11
REGRESSION_WINDOW_WIDTH = 30
STANDARD_ERROR_BAND_GAP = 0.011 / 2.920 + 0.005124807507007388


def get_data(filename):
    """
    Reads data from a spectroscopy file and returns numpy arrays for
    varying temps and wavelengths.

    Input: filename (string)
    CSV filename.
    Returns: formatted_data (DataFrame)
    Dataframe with columns of wavelengths, baselines and abs data.
    """
    rows = int((INI_WAVELENGTH - FINAL_WAVELENGTH) / WAVELENGTH_STEP + 1)
    raw_data = pd.read_csv(filename, header=[0, 1], nrows=rows)
    # Assume all samples are over same wavelengths
    wavelengths_column = raw_data.loc[:, (slice(None),
                                          "Wavelength (nm)")].iloc[:, 0]
    wavelengths = pd.DataFrame(wavelengths_column.values,
                               columns=["Wavelength (nm)"])
    # Creates new DataFrame, does not preserve relative order of
    # %T and Abs measurements, could use a loop / be sorted if necessary
    measurements_column = raw_data.loc[:, (slice(None), ["%T", "Abs"])]
    aligned_columns = index_aligner(raw_data)
    measurements = pd.DataFrame(measurements_column.values,
                                columns=aligned_columns)
    formatted_data = wavelengths.join(measurements)
    return formatted_data


def index_aligner(raw_data):  # ----------------------------------
    """
    Specialised helper function for reindexing particular dataframes.
    Shifts the index so filenames align with unit then merges the
    levels of the MultiIndex.

    Input: raw_data (Dataframe)
    Cary6000i csv data with header as MultiIndex.
    Returns: new_cols (List)
    Formatted list of proper column names for data.
    """
    top_labels = raw_data.columns.get_level_values(0)
    # Shift right 1
    new_top_labels = pd.Index(['']).append(top_labels[:-1])
    shifted_cols = pd.MultiIndex.from_arrays([
        new_top_labels, raw_data.columns.get_level_values(1)])
    new_cols = []
    for old_cols_tuple in shifted_cols[1::2]:
        new_cols.append(old_cols_tuple[0] + " (" + old_cols_tuple[1] + ")")
    return new_cols


def data_cleaner(low_t_data, high_t_data):  # ----------------------
    """
    Specialised function which removes known bad data points and
    separates baseline data.

    Input: lowTdata (dataframe), highTdata (dataframe)
    Specific low and high temp data with wavelengths, baselines and Abs.
    Returns: all_t_data (Dataframe)
    Combined abs data with data after detector malfunction removed.
    all_baselines (Dataframe)
    Combined baseline data.
    """
    low_t_baseline = low_t_data.iloc[:, 1:3]
    high_t_baseline = high_t_data.iloc[:, 1:3]
    all_baselines = pd.concat([low_t_baseline, high_t_baseline], axis=1)
    # Slice test and mask data where the detector broke from heat?
    sliced_high_t_data = high_t_data.iloc[:, 3:-1]
    malfunction_mask = np.zeros(sliced_high_t_data.shape, dtype=bool)
    malfunction_mask[:801, -10:] = True
    masked_high_t_data = sliced_high_t_data.mask(malfunction_mask,
                                                 other=np.nan)
    # Janky little line to stop errors from duplicate columns labels
    masked_high_t_data.rename(columns={"T325 (Abs)": "T325NoVac (Abs)"},
                              inplace=True)
    all_t_data = pd.concat([low_t_data["Wavelength (nm)"],
                            low_t_data.iloc[:, 3:],
                            masked_high_t_data], axis=1)
    # Cut off data where glass interferes
    glass_interference_wavelength = NM_TO_EV / GLASS_INTERFERENCE_ENERGY
    glass_int_index = (
            all_t_data["Wavelength (nm)"] <
            glass_interference_wavelength).idxmax()
    clean_data = all_t_data[:glass_int_index]
    return clean_data, all_baselines


def scale_n_differentiate(data):
    """
    Calculates optical band gaps for spectroscopic data using a
    Tauc-like method.

    Input : data (DataFrame)
    Clean data to be used for analysis with first column as wavelengths
    and other columns as sample data.
    Returns : sorted_scaled_df (DataFrame)
    Data with columns of wavelengths scaled to photon energy,
    absorbance scaled to ahv^2,
    derivative after data smoothing.
    """
    energy_df = pd.DataFrame({"eV": NM_TO_EV / data["Wavelength (nm)"]})
    scaled_df = pd.concat({"Energy": energy_df}, axis=1)
    abs_factor = 1 / (np.log(np.e) * SAMPLE_THICKNESS)
    for sample_abs in data.columns[1:]:
        sample_ahv = sample_abs.replace("Abs", "ahv")
        scaled_df["ahv_data", sample_ahv] = (
                (abs_factor * data[sample_abs] *
                 scaled_df["Energy", "eV"]) ** (1 / R))
        scaled_df["data_derivative", sample_ahv] = smooth_n_derivative(
            scaled_df["ahv_data", sample_ahv],
            scaled_df["Energy", "eV"],
            FILTER_WINDOW, FILTER_ORDER)
    sorted_scaled_df = scaled_df.sort_index(axis=1, level=0)
    return sorted_scaled_df


def smooth_n_derivative(rough_data, energy, window, porder):
    """
    Helper function to smooth data using a Savitzky-Golay filter.

    Input: rough_data (Series)
    Spec data scaled to ahv.
    Returns: derived_data (Series)
    Derivative of smoothed data.
    """
    smooth_data = signal.savgol_filter(rough_data, window, porder)
    derived_data = np.gradient(smooth_data, energy)
    return derived_data


def ahv2extractor(scaled_df):
    """
    Extracts the average ahv^2 value where the derivative is a maximum
    for low temperatures where the band gap presents as a significant
    'bump'. Order should be around the width of the derivative 'bump'.

    Input: scaled_df (DataFrame)
    Dataframe with columns of energy, ahv data and data derivatives.
    Returns: average_ahv2 (float)
    Average ahv^2 value for low temperatures.
    """
    # Approximate width of derivative peak around linear region
    last_bump_height = DERIVATIVE_LINEARITY_REQUIREMENT
    temperature_index = 1
    ahv2list = []
    while (
            last_bump_height >= DERIVATIVE_LINEARITY_REQUIREMENT) and (
            temperature_index < len(scaled_df["data_derivative"].columns)):
        temperature = scaled_df["data_derivative"].columns[temperature_index]
        derivative_array = np.array(scaled_df["data_derivative", temperature])
        # Correct index may be specific to the data
        max_index = signal.argrelextrema(derivative_array,
                                         np.greater,
                                         order=DERIVATIVE_BUMP_WIDTH)[0][-1]
        # # Visually determining bump width
        # plt.plot(scaled_df["Energy", "eV"], derivative_array)
        # plt.vlines(scaled_df["Energy", "eV"][max_index],
        #            0, max(derivative_array))
        # plt.vlines(scaled_df["Energy", "eV"][max_index + bump_width],
        #            0, max(derivative_array), linestyles='dashed')
        # plt.vlines(scaled_df["Energy", "eV"][max_index - bump_width],
        #            0, max(derivative_array), linestyles='dashed')
        # plt.grid(True)
        # plt.show()
        last_bump_height = (
                derivative_array[max_index] -
                derivative_array[max_index + DERIVATIVE_BUMP_WIDTH])
        ahv2list.append(scaled_df["ahv_data", temperature][max_index])
        temperature_index += 1
    average_ahv2 = np.mean(ahv2list)
    return average_ahv2


def linear_region_extrapolater(scaled_df, average_ahv2):
    """
    Calculates the band gap using Tauc-like plots from a line built
    from the max derivative.
    -> change to regression window

    Input : scaled_df (DataFrame)
    Dataframe with columns of energy, ahv data and data derivatives
    for each temperature.
    average_ahv2 (float)
    Average ahv^2 value where derivative is maximum for low temperatures.
    Returns : linear_region_df (DataFrame)
    Dataframe of gradient and y-intercept of lines extrapolated from the
    linear region of the data, energy value at x-intercept is
    the optical band gap.
    """
    linear_region_df = pd.DataFrame()
    for temperature in scaled_df["data_derivative"].columns:
        # Find index of ahv_data value closest to average ahv^2
        max_index = (np.abs(scaled_df["ahv_data", temperature] -
                            average_ahv2)).idxmin()
        regression_slice = slice(max_index - REGRESSION_WINDOW_WIDTH,
                                 max_index + REGRESSION_WINDOW_WIDTH)
        ahv_regression = scaled_df["ahv_data", temperature][regression_slice]
        energy_regression = scaled_df["Energy", "eV"][regression_slice]
        regression_line = linregress(energy_regression, ahv_regression)
        max_slope = regression_line.slope
        linear_y_int = regression_line.intercept
        band_gap = -linear_y_int / max_slope
        # Important to match multi-index
        linear_region_df["ahv_data", temperature] = (max_slope,
                                                     linear_y_int,
                                                     band_gap)
    return linear_region_df


def tauc_plotter(scaled_df, linear_region_df):
    """Plots multiple tauc plots on one set of axes.
    Input:  scaled_df (DataFrame)
    Dataframe with columns of energy, ahv data and data derivatives
    linear_region_df (DataFrame)
    DataFrame with columns of max derivative and y-intercept for each temp
    Returns: None
    Shows plot.
    """
    fig, ax = plt.subplots()
    for temperature in scaled_df["ahv_data"].columns:
        temperature_label = temperature.rstrip("(ahv)")
        ax.plot(scaled_df[("Energy", "eV")],
                scaled_df["ahv_data", temperature],
                label=temperature_label)
        max_derivative, linear_region_y_int, band_gap = linear_region_df[
            "ahv_data", temperature]
        lin_region = max_derivative * scaled_df[
            ("Energy", "eV")] + linear_region_y_int
        ax.plot(scaled_df[("Energy", "eV")], lin_region, "--",
                label=f"{temperature_label} band-gap: {band_gap:.3f}")
    ax.set_xlim(PLOT_X_LIMS)
    ax.set_ylim(PLOT_Y_LIMS)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(r"$(\alpha h v)^{2}$")  # ----------------------
    ax.grid(True)
    ax.legend()
    plt.savefig("Tauc-like plot")


def band_gap_plotter(linear_region_df):
    """
    Plots the band gap as a function of temperature.
    """
    band_gaps = linear_region_df.iloc[2]
    temps = [15, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275,
             300, 303, 325, 325, 350, 375, 400, 425, 450, 475, 500,
             525, 525, 550, 575, 600, 625, 650, 675, 700, 725]
    fig, ax = plt.subplots()
    ax.errorbar(temps, band_gaps,
                xerr=0.5, yerr=STANDARD_ERROR_BAND_GAP*3, fmt="-o")
    ax.grid(True)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Band gap (eV)")
    plt.savefig("Band gap plot")


class InteractivePlot:
    """Interactively shows different smoothing options to show Daniel.
    Probably don't use this"""

    def __init__(self, scaled_df):
        # Initialize data and plot
        self.window, self.p_order = FILTER_WINDOW, FILTER_ORDER
        self.demo_data = scaled_df.iloc[:, 1:]
        self.temperature = self.demo_data.columns[0][1]  # Only uses 1st temp
        self.energy = scaled_df["Energy", "eV"]
        self.fig, self.ax = plt.subplots()

        # plt.subplots_adjust(bottom=0.4)
        # self.ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
        # self.ax_slide_p_ord = plt.axes([0.25, 0.2, 0.65, 0.03])
        # # Window size for Sav-gol filter should be odd
        # self.win_size = Slider(
        #     self.ax_slide, 'Window size',
        #     valmin=3, valmax=151, valinit=FILTER_WINDOW, valstep=2)
        # self.porder_size = Slider(
        #     self.ax_slide_p_ord, 'Poly order',
        #     valmin=0, valmax=10, valinit=FILTER_ORDER, valstep=1)
        self.ax.set_xlim(1.5, 4.5)
        self.ax.set_ylim(0, 5 * 10 ** 12)
        self.ax.set_xlabel("Photon energy (eV)")
        self.ax.set_ylabel(r"d($\alpha E)^2$ / dE")
        self.ax.set_title(self.temperature)
        self.ax.grid(True)

        # Initial calculations
        self.demo_data["data_derivative_raw", self.temperature] = np.gradient(
            self.demo_data["ahv_data", self.temperature],
            self.energy)
        self.raw = self.ax.plot(self.energy,
                                self.demo_data[
                                    "data_derivative_raw", self.temperature],
                                "b.", markersize=1, alpha=0.5)
        self.demo_data[
            "data_derivative", self.temperature] = smooth_n_derivative(
            self.demo_data["ahv_data", self.temperature],
            self.energy, FILTER_WINDOW, FILTER_ORDER)
        self.current, = self.ax.plot(self.energy,
                                     self.demo_data[
                                         "data_derivative", self.temperature],
                                     "r-", linewidth=0.5)
        # self.win_size.on_changed(self.update_win)
        # self.porder_size.on_changed(self.update_porder)
        plt.savefig("Sav-gol demo")
        plt.show()

    def update_win(self, window):
        """Updates the plot with data smoothed with new window size."""
        self.window = window
        self.demo_data[
            "data_derivative", self.temperature] = smooth_n_derivative(
            self.demo_data["ahv_data", self.temperature],
            self.energy,
            window, self.p_order)
        self.current.set_ydata(
            self.demo_data["data_derivative", self.temperature])
        self.fig.canvas.draw()

    def update_porder(self, porder):
        """Updates the plot with data smoothed with new polynomial order."""
        self.p_order = porder
        self.demo_data[
            "data_derivative", self.temperature] = smooth_n_derivative(
            self.demo_data["ahv_data", self.temperature],
            self.energy,
            self.window, porder)
        self.current.set_ydata(
            self.demo_data["data_derivative", self.temperature])
        self.fig.canvas.draw()


def main():  # --------------------------------------------------
    """Main function to take UV-Vis spec data and create Tauc-like plots
    determining band-gap temperature dependence."""
    low_t_data = get_data(LOW_T_FILE)
    high_t_data = get_data(HIGH_T_FILE)
    clean_data, all_baselines = data_cleaner(low_t_data, high_t_data)
    scaled_df = scale_n_differentiate(clean_data)
    # average_ahv2 = ahv2extractor(scaled_df)
    # linear_region_df = linear_region_extrapolater(scaled_df, average_ahv2)
    # tauc_plotter(scaled_df.iloc[:, [0, 1, 33]], linear_region_df)
    # band_gap_plotter(linear_region_df)

    InteractivePlot(scaled_df.iloc[:, [0, 33]])
    plt.show()

    # TODO: iterate regression around point to lower error?,
    #  completely remove multi-index, check band gaps(plot), daniel changes


if __name__ == "__main__":
    main()
