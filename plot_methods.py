# Script for methods for generating figures
# Author: Sahar H. El Abbadi
# Date Created: 2023-02-24
# Date Last Modified: 2023-03-13

# List of methods in this file:
# > rand_jitter
# > plot_parity
# > plot_parity(operator, stage, operator_report, operator_meter)

# Imports
import numpy as np
import pandas as pd
import pathlib
import datetime
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import matplotlib.offsetbox as offsetbox
import matplotlib.ticker as ticker

from methods_source import load_overpass_summary, abbreviate_op_name, classify_histogram_data, \
    load_operator_flight_days, load_daily_releases, calc_meter_uncertainty, load_uncertainty_multipliers, \
    make_logistic_regression
from writing_analysis import calculate_residuals_and_error


# Function: generate jitter for a given array
def rand_jitter(input_list):
    delta = 0.2
    return input_list + np.random.randn(len(input_list)) * delta


# %% Functions for making parity plots

def get_parity_data(operator, stage, error_type='95_CI', strict_discard=False, time_ave=60, gas_comp_source='ms'):
    """

    :param operator: name of operator
    :param stage: stage of analysis
    :param error_type: indicate type of error. Default is 95% CI, alternative is "operator_reported"
    :param strict_discard: boolean, True for strict discard, False for lax discard
    :param time_ave: time average period for meter data
    :param gas_comp_source: source of gas composition for CNG
    :return save_parity_data: dataframe with columns release_rate, release_sigma, operator_report, and operator_sigma
    :return data_description: dictionary with key data parameters (operator, stage, strict_discard, time_ave, gas_comp_source)
    """

    # Load overpass summary csv file
    operator_plot = load_overpass_summary(operator=operator, stage=stage, strict_discard=strict_discard,
                                          time_ave=time_ave, gas_comp_source=gas_comp_source)

    # Apply the following filters to overpass data :
    # Must pass all QC filters:
    operator_plot = operator_plot[(operator_plot.qc_summary == 'pass_all')]

    # For parity plots:
    # All data entries must be a non-zero release
    operator_plot = operator_plot.query('non_zero_release == True')

    # Operator must have quantified the release as non-zero:
    operator_plot = operator_plot.query('operator_quantification > 0')

    # Apply filter for Stage 3 data: remove points for which we gave the team's quantification estimates
    # if phase_iii == 1, then we gave them this release in Phase III dataset
    if stage == 3:
        operator_plot = operator_plot[(operator_plot.phase_iii == 0)]

    # Select values for which operator provided a quantification estimate
    # I think this is old code from when I was including zeros in the parity plot
    # y_index = np.isfinite(operator_plot['operator_quantification'])

    # Select x data
    x_data = operator_plot['release_rate_kgh']
    y_data = operator_plot['operator_quantification']
    x_error = operator_plot['ch4_kgh_sigma']

    ######### Select error bars for operator quantification #########

    if error_type == '95_CI':
        operator_multiplier = load_uncertainty_multipliers(operator)
        legend_error = '95% CI'
    elif error_type == 'operator_reported':
        operator_multiplier = 1
        legend_error = 'Operator Reported Error Bars'
    else:
        operator_multiplier = 1
        legend_error = 'Operator Reported Error Bars'
        print(f'Please correct input error type. Currently using operator reported error bars')

    if operator in ['Kairos', 'Kairos LS23', 'Kairos LS25']:
        legend_error = 'x-error: 95% CI)\n(y-error: measurement range'

    # Kairos individual pods don't report error
    if operator in ['Kairos LS23', 'Kairos LS25']:
        y_error = 0 * len(operator_plot)
    else:
        y_error = (operator_plot['operator_upper'] - operator_plot['operator_quantification']) * operator_multiplier

    # Save data used to make figure
    save_parity_data = pd.DataFrame()
    save_parity_data['release_rate'] = x_data
    save_parity_data['release_sigma'] = x_error
    save_parity_data['operator_report'] = y_data
    save_parity_data['operator_sigma'] = y_error

    # Save data description
    data_description = {
        'operator': operator,
        'stage': stage,
        'strict_discard': strict_discard,
        'time_ave': time_ave,
        'gas_comp_source': gas_comp_source,
        'strict_discard': strict_discard,
        'legend_error': legend_error,
    }

    return save_parity_data, data_description


def make_parity_plot(data, data_description, ax, plot_lim='largest_kgh'):
    """
    :param data: processed data to be plotted
    :param data_description: dictionary with descriptions of data used for plot annotations
    :param ax: subplot ax to plot on
    :param plot_lim: limit of x and y axes
    :return: ax: is the plotted parity chart
    """

    # Stage key for blinded status
    stage_description = {
        1: 'Fully blinded results',
        2: 'Unblinded wind',
        3: 'Partially unblinded',
    }
    ############ Data Preparation and Linear Regression ############

    # Load data description
    operator = data_description['operator']
    stage = data_description['stage']
    time_ave = data_description['time_ave']
    gas_comp_source = data_description['gas_comp_source']
    strict_discard = data_description['strict_discard']
    legend_error = data_description['legend_error']

    # Set x and y data and error values
    x_data = data.release_rate
    y_data = data.operator_report
    x_error = data.release_sigma * 1.96  # value is sigma, multiply by 1.96 for 95% CI
    y_error = data.operator_sigma  # error bars are determined in get_parity_data function

    # Fit linear regression via least squares with numpy.polyfit
    # m is slope, intercept is b
    m, b = np.polyfit(x_data, y_data, deg=1)

    # Calculate R^2 value
    # (using method described here: https://www.askpython.com/python/coefficient-of-determination)
    correlation_matrix = np.corrcoef(x_data, y_data)
    correlation = correlation_matrix[0, 1]
    r2 = correlation ** 2

    # Number of valid overpasses:
    sample_size = len(y_data)

    # Set x and y max values
    # Manually set largest x and y value by changing largest_kgh here to desired value:
    # largest_kgh = max(plot_lim)

    if plot_lim == 'largest_kgh':
        # Filter out NA because operations with NA returns NA
        if np.isnan(max(y_error)) == 1:
            y_error.iloc[:] = 0

        largest_kgh = max(max(x_data), max(y_data)) + max(y_error)
        largest_kgh = math.ceil(largest_kgh / 100) * 100

        # set plot_lim:
        plot_lim = [0, largest_kgh]
    else:
        largest_kgh = max(plot_lim)

    # Create sequence of numbers for plotting linear fit (x)
    x_seq = np.linspace(0, largest_kgh, num=100)

    ############ Generate Figure  ############

    # Add linear regression to in put ax
    ax.plot(x_seq, m * x_seq + b, color='k', lw=2,
            label=f'Best Fit, $R^2 =$ {r2:0.2f}\n$y = {m:0.2f}x+{b:0.2f}$')

    # Add parity line
    # With label:
    # ax.plot(x_seq, x_seq, color='k', lw=2, linestyle='--',
    #          label='Parity Line')
    # Without label:
    ax.plot(x_seq, x_seq, color='k', lw=2, linestyle='--')

    # Add scatter plots with error bars
    ax.errorbar(x_data, y_data,
                xerr=x_error,
                yerr=y_error,
                linestyle='none',
                mfc='white',
                label=f'n = {sample_size}\n({legend_error})',
                fmt='o',
                markersize=5)

    stage_text = stage_description[stage]
    # Set title
    ax.set_title(f'{operator} ({stage_text})')

    # ax.text(100, largest_kgh-100, f'{operator} ({stage_text})', fontsize=15, horizontalalignment='left',
    #          bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))

    # Annotation box
    # text = f'{operator}\n {stage_text}'
    # ob = offsetbox.AnchoredText(text, loc='upper left')
    # ob.set(alpha=0.8)
    # ax.add_artist(ob)

    # Set axes
    ax.set(xlim=plot_lim,
           ylim=plot_lim,
           alpha=0.8)

    # Equalize Axes
    ax.set_aspect('equal', adjustable='box')

    # Set axes and background color to white
    ax.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Axes labels
    # ax.set_xlabel('Methane Release Rate (kgh)', fontsize=14)
    # ax.set_ylabel('Reported Release Rate (kgh)', fontsize=14)
    ax.tick_params(direction='in', right=True, top=True)
    ax.tick_params(labelsize=16)
    ax.minorticks_on()
    ax.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    ax.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)
    ax.grid(False)  # remove grid lines

    # Customize the tick labels with commas at the thousands place
    ax.tick_params(labelsize=16)

    # Define a formatter function to add commas
    def comma_formatter(x, pos):
        return '{:,.0f}'.format(x)  # Add commas to the thousands place

    # Apply the formatter to the tick labels
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(comma_formatter))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(comma_formatter))

    # Legend
    ax.legend(facecolor='white', loc='lower right', fontsize=11)

    return ax


# Inputs:
# > operator
# > stage
# > operator_report
# > operator_meter
def plot_parity(operator, stage, strict_discard=False, time_ave=60, gas_comp_source='ms'):
    """Inputs are operator name, stage of analysis, operator_plot dataframe containing all relevant data"""

    # Generate parity data
    parity_data, parity_notes = get_parity_data(operator=operator, stage=stage, strict_discard=strict_discard,
                                                time_ave=time_ave, gas_comp_source=gas_comp_source)

    # Initialize figure
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Make figure
    ax = make_parity_plot(parity_data, parity_notes, ax)

    # Axes labels
    ax.set_xlabel('Methane Release Rate (kgh)', fontsize=14)
    ax.set_ylabel('Reported Release Rate (kgh)', fontsize=14)

    # Save figure
    if strict_discard == True:
        discard = 'strict'
    else:
        discard = 'lax'

    now = datetime.datetime.now()
    op_ab = abbreviate_op_name(operator)
    save_time = now.strftime("%Y%m%d")
    fig_name = f'{op_ab}_{stage}_{discard}_parity_{save_time}'
    fig_path = pathlib.PurePath('04_figures', 'parity_plots', fig_name)
    plt.savefig(fig_path)
    plt.show()

    # Save data used to make figure

    save_path = pathlib.PurePath('06_results', 'parity_plot_data', f'{op_ab}_{stage}_{discard}_parity_{save_time}.csv')
    parity_data.to_csv(save_path)

    return


# %% Function: plot_detection_limit


def make_detection_limit_df(operator, stage, n_bins, threshold, strict_discard=False, time_ave=60,
                            gas_comp_source='ms'):
    # Load overpass summary for operator, stage, and discard criteria:
    operator_df = load_overpass_summary(operator, stage, strict_discard, time_ave, gas_comp_source)

    # Apply QC filter
    # For SI: Carbon Mapper's QC for determining quantification only, and treat their detection column as applied to all points

    # if (operator == 'Carbon Mapper') or (operator == 'Scientific Aviation'):
    #     operator_df = operator_df[(operator_df.stanford_kept == 1)]
    if (operator == 'Scientific Aviation'):
        # Scientific Aviation explicitly stated that all data could be used in determining detection, their QC only
        # applied to quantification
        operator_df = operator_df[(operator_df.stanford_kept == 1)]
    else:
        operator_df = operator_df[(operator_df.qc_summary == 'pass_all')]

    # Must be non-zero values
    operator_df = operator_df.query('non_zero_release == True')

    # Select release under threshold value
    operator_df = operator_df.query('release_rate_kgh <= @threshold')

    # Create bins for plot
    bins = np.linspace(0, threshold, n_bins + 1)
    detection_probability = np.zeros(n_bins)

    # These variables are for keeping track of values as I iterate through the bins in the for loop below:
    bin_size, bin_num_detected = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
    bin_median = np.zeros(n_bins)
    bin_two_sigma = np.zeros(n_bins)
    two_sigma_upper, two_sigma_lower = np.zeros(n_bins), np.zeros(n_bins)

    # For each bin, find number of data points and detection probability

    for i in range(n_bins):

        # Set boundary of bin
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_median[i] = (bin_min + bin_max) / 2

        # Select data within the bin range
        binned_data = operator_df.loc[operator_df.release_rate_kgh < bin_max].loc[
            operator_df.release_rate_kgh >= bin_min]

        # Count the total number of overpasses detected within each bin
        bin_num_detected[i] = binned_data.operator_detected.sum()

        n = len(binned_data)
        bin_size[i] = n  # this is the y-value for the bin in the plot
        p = binned_data.operator_detected.sum() / binned_data.shape[0]  # df.shape[0] gives number of rows
        detection_probability[i] = p

        # Standard Deviation of a binomial distribution
        sigma = np.sqrt(p * (1 - p) / n)
        bin_two_sigma[i] = 2 * sigma

        # Find the lower and upper bound defined by two sigma
        two_sigma_lower[i] = 2 * sigma
        two_sigma_upper[i] = 2 * sigma
        if 2 * sigma + p > 1:
            two_sigma_upper[i] = 1 - p  # probability cannot exceed 1
        if p - 2 * sigma < 0:
            two_sigma_lower[i] = p  # if error bar includes zero, set lower bound to p?

    detection_prob = pd.DataFrame({
        "bin_median": bin_median,
        "detection_prob_mean": detection_probability,
        "detection_prob_two_sigma_upper": two_sigma_upper,
        "detection_prob_two_sigma_lower": two_sigma_lower,
        "n_data_points": bin_size,
        "n_detected": bin_num_detected})

    return detection_prob, operator_df


def plot_detection_limit(ax, operator, stage, n_bins, threshold, strict_discard=False, time_ave=60,
                         gas_comp_source='ms'):
    """
      :param ax: subplot to plot
      :param logistic_model: boolean, True for including the logistic model, false for not including it
      :param operator: name of operator
      :param stage: stage of testing
      :param threshold: max value for plotting (we are looking at all releases under threshold value)
      :param n_bins: number of bins for plot
      :param strict_discard: Boolean, use Stanford strict discard criteria or lax discard criteria
      :param time_ave: time average for metered data
      :gas_comp_source: source of gas composition, default value is measurement station (ms)
      """

    detection_plot, operator_df = make_detection_limit_df(operator, stage, n_bins, threshold, strict_discard, time_ave,
                                                          gas_comp_source)

    # Set bin width:
    w = threshold / n_bins / 2.5

    # Use n_bins set above
    for i in range(n_bins):
        ax.annotate(f'{detection_plot.n_detected[i]} / {detection_plot.n_data_points[i]}',
                    [detection_plot.bin_median[i] - w / 1.8, 0.03], fontsize=10)

    # for plotting purpose, we don't want a small hyphen indicating zero uncertainty interval
    detection_plot.loc[detection_plot['detection_prob_two_sigma_lower'] == 0, 'detection_prob_two_sigma_lower'] = np.nan
    detection_plot.loc[detection_plot.detection_prob_two_sigma_upper == 0, 'detection_prob_two_sigma_upper'] = np.nan
    detection_plot.loc[detection_plot.detection_prob_mean == 0, 'detection_prob_mean'] = np.nan

    # To avoid RuntimeWarning: All-NaN axis encountered, set yerr to None if all values are np.nan in sigma values
    # (this is the case for Carbon Mapper)

    sigma_lower = detection_plot.detection_prob_two_sigma_lower
    sigma_upper = detection_plot.detection_prob_two_sigma_upper

    if sigma_lower.isnull().all() or sigma_upper.isnull().all():
        y_error = None
    else:
        y_error = [sigma_lower, sigma_upper]

    # Plot bars and detection points
    ax.bar(detection_plot.bin_median,
           detection_plot.detection_prob_mean,
           # yerr=[detection_plot.detection_prob_two_sigma_lower, detection_plot.detection_prob_two_sigma_upper],
           yerr=y_error,
           error_kw=dict(lw=2, capsize=3, capthick=1, alpha=0.3),
           width=threshold / n_bins - 0.5, alpha=0.6, color='#9ecae1', ecolor='black', capsize=2)

    # yulia's color: edgecolor="black",facecolors='none'
    x_data = rand_jitter(operator_df.release_rate_kgh)

    ax.scatter(x_data, np.multiply(operator_df.operator_detected, 1),
               facecolors='black',
               marker='|')

    # Add more room on top and bottom
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, threshold + 0.5])

    # Axes formatting and labels
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=11)
    ax.set_xlabel('Methane Release Rate (kgh)', fontsize=14)
    ax.set_ylabel('Proportion detected', fontsize=14)
    ax.tick_params(direction='in', right=True, top=True)
    ax.tick_params(labelsize=12)
    ax.minorticks_on()
    ax.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    ax.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

    # Set axes and background color to white
    ax.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Set more room on top for annotation
    ax.set_ylim([-0.05, 1.22])
    ax.set_xlim([0, threshold])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=11)

    # Stage key for blinded status
    stage_description = {
        1: 'Fully blinded results',
        2: 'Unblinded wind',
        3: 'Partially unblinded',
    }

    stage_text = stage_description[stage]
    text = f'{operator}\n {stage_text}'

    ax.annotate(text, xy=(0.03, 0.89), xycoords='axes fraction', fontsize=13)

    return ax


def plot_detection_limit_su_filter_only(operator, stage, n_bins, threshold, strict_discard=False, time_ave=60,
                                        gas_comp_source='ms'):
    # Load overpass summary for operator, stage, and discard criteria:
    operator_df = load_overpass_summary(operator, stage, strict_discard, time_ave, gas_comp_source)

    # Apply QC filter
    # For SI: Carbon Mapper's QC for determining quantification only, and treat their detection column as applied to all points

    if (operator == 'Carbon Mapper') or (operator == 'Scientific Aviation'):
        operator_df = operator_df[(operator_df.stanford_kept == 1)]
    # if (operator == 'Scientific Aviation'):
    #     # Scientific Aviation explicitly stated that all data could be used in determining detection, their QC only
    #     # applied to quantification
    #     operator_df = operator_df[(operator_df.stanford_kept == 1)]
    else:
        print(
            f'Your operator is: {operator}.\nOnly use this function with Carbon Mapper or Scientific Aviation, as their QC only applies to quantification.')
        print('For all other operators, please use standard plot_detection_limit function.')
        return

    # Must be non-zero values
    operator_df = operator_df.query('non_zero_release == True')

    # Select release under threshold value
    operator_df = operator_df.query('release_rate_kgh <= @threshold')

    # Create bins for plot
    bins = np.linspace(0, threshold, n_bins + 1)
    detection_probability = np.zeros(n_bins)

    # These variables are for keeping track of values as I iterate through the bins in the for loop below:
    bin_size, bin_num_detected = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
    bin_median = np.zeros(n_bins)
    bin_two_sigma = np.zeros(n_bins)
    two_sigma_upper, two_sigma_lower = np.zeros(n_bins), np.zeros(n_bins)

    # For each bin, find number of data points and detection probability

    for i in range(n_bins):

        # Set boundary of bin
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_median[i] = (bin_min + bin_max) / 2

        # Select data within the bin range
        binned_data = operator_df.loc[operator_df.release_rate_kgh < bin_max].loc[
            operator_df.release_rate_kgh >= bin_min]

        # Count the total number of overpasses detected within each bin
        bin_num_detected[i] = binned_data.operator_detected.sum()

        n = len(binned_data)
        bin_size[i] = n  # this is the y-value for the bin in the plot
        p = binned_data.operator_detected.sum() / binned_data.shape[0]  # df.shape[0] gives number of rows
        detection_probability[i] = p

        # Standard Deviation of a binomial distribution
        sigma = np.sqrt(p * (1 - p) / n)
        bin_two_sigma[i] = 2 * sigma

        # Find the lower and upper bound defined by two sigma
        two_sigma_lower[i] = 2 * sigma
        two_sigma_upper[i] = 2 * sigma
        if 2 * sigma + p > 1:
            two_sigma_upper[i] = 1 - p  # probability cannot exceed 1
        if p - 2 * sigma < 0:
            two_sigma_lower[i] = p  # if error bar includes zero, set lower bound to p?

    detection_prob = pd.DataFrame({
        "bin_median": bin_median,
        "detection_prob_mean": detection_probability,
        "detection_prob_two_sigma_upper": two_sigma_upper,
        "detection_prob_two_sigma_lower": two_sigma_lower,
        "n_data_points": bin_size,
        "n_detected": bin_num_detected})

    # Function will output cm_detection and detection_prob

    detection_plot = detection_prob.copy()
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Set bin width:
    w = threshold / n_bins / 2.5

    # Use n_bins set above
    for i in range(n_bins):
        ax.annotate(f'{detection_plot.n_detected[i]} / {detection_plot.n_data_points[i]}',
                    [detection_plot.bin_median[i] - w / 1.8, 0.03], fontsize=10)

    # for plotting purpose, we don't want a small hyphen indicating zero uncertainty interval
    detection_plot.loc[detection_plot['detection_prob_two_sigma_lower'] == 0, 'detection_prob_two_sigma_lower'] = np.nan
    detection_plot.loc[detection_plot.detection_prob_two_sigma_upper == 0, 'detection_prob_two_sigma_upper'] = np.nan
    detection_plot.loc[detection_plot.detection_prob_mean == 0, 'detection_prob_mean'] = np.nan

    # To avoid RuntimeWarning: All-NaN axis encountered, set yerr to None if all values are np.nan in sigma values
    # (this is the case for Carbon Mapper)

    sigma_lower = detection_plot.detection_prob_two_sigma_lower
    sigma_upper = detection_plot.detection_prob_two_sigma_upper

    if sigma_lower.isnull().all() or sigma_upper.isnull().all():
        y_error = None
    else:
        y_error = [sigma_lower, sigma_upper]

    # Plot bars and detection points
    ax.bar(detection_plot.bin_median,
           detection_plot.detection_prob_mean,
           # yerr=[detection_plot.detection_prob_two_sigma_lower, detection_plot.detection_prob_two_sigma_upper],
           yerr=y_error,
           error_kw=dict(lw=2, capsize=3, capthick=1, alpha=0.3),
           width=threshold / n_bins - 0.5, alpha=0.6, color='#9ecae1', ecolor='black', capsize=2)

    # yulia's color: edgecolor="black",facecolors='none'
    x_data = rand_jitter(operator_df.release_rate_kgh)

    ax.scatter(x_data, np.multiply(operator_df.operator_detected, 1),
               facecolors='black',
               marker='|')

    # Add more room on top and bottom
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, threshold + 0.5])

    # Axes formatting and labels
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=11)
    ax.set_xlabel('Methane Release Rate (kgh)', fontsize=14)
    ax.set_ylabel('Proportion detected', fontsize=14)
    ax.tick_params(direction='in', right=True, top=True)
    ax.tick_params(labelsize=12)
    ax.minorticks_on()
    ax.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    ax.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

    # Set axes and background color to white
    ax.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')

    plt.title(f'{operator} Probability of Detection - Stage {stage}')

    # Save figure

    if strict_discard == True:
        discard = 'strict'
    else:
        discard = 'lax'

    now = datetime.datetime.now()
    save_time = now.strftime("%Y%m%d")
    op_ab = abbreviate_op_name(operator)
    fig_name = f'detect_limit_{op_ab}_stage{stage}_{discard}_{save_time}_su_filter_only'
    fig_path = pathlib.PurePath('04_figures', 'detection_limit', fig_name)
    plt.savefig(fig_path)
    plt.show()

    # Save data used to make plots
    # operator_df.to_csv(pathlib.PurePath('03_results', 'detect_probability_data',
    #                                     f'{op_ab}_{stage}_detect_{save_time}.csv'))
    # detection_prob.to_csv(pathlib.PurePath('03_results', 'detect_probability_data',
    #                                        f'{op_ab}_{stage}_{threshold}kgh_{n_bins}bins_{save_time}.csv'))
    return


def save_pod_plot(operator, stage, strict_discard, time_ave, gas_comp_source):
    if strict_discard == True:
        discard = 'strict'
    else:
        discard = 'lax'

    now = datetime.datetime.now()
    save_time = now.strftime("%Y%m%d")
    op_ab = abbreviate_op_name(operator)
    fig_name = f'detect_limit_{op_ab}_stage{stage}_{discard}_{save_time}'
    fig_path = pathlib.PurePath('04_figures', 'detection_limit', fig_name)
    plt.savefig(fig_path)


# %%

def plot_qc_summary(operators, stage, strict_discard=False):
    # Load saved QC dataframe
    if strict_discard == True:
        file_name = 'all_qc_strict.csv'
        discard = 'strict'
    else:
        file_name = 'all_qc.csv'
        discard = 'lax'
    all_qc = pd.read_csv(pathlib.PurePath('06_results', 'qc_comparison', file_name), index_col=0)
    # Plot

    category = ['fail_stanford_only', 'fail_all_qc', 'fail_operator_only']
    n_operators = len(operators)  # number of operators

    # Determine values for each group, alphabetical order of operators: "Carbon Mapper, GHGSat, Kairos"

    fail_operator = np.zeros(n_operators)
    fail_stanford = np.zeros(n_operators)
    fail_all = np.zeros(n_operators)
    pass_all = np.zeros(n_operators)

    # Height of bars

    for i in range(len(operators)):  # for go through fail stanford only
        op_ab = abbreviate_op_name(operators[i])
        operator_qc = all_qc.loc[all_qc.operator == op_ab]
        operator_stage_qc = operator_qc.loc[operator_qc.stage == stage]
        fail_operator[i] = operator_stage_qc.fail_operator_only
        fail_stanford[i] = operator_stage_qc.fail_stanford_only
        fail_all[i] = operator_stage_qc.fail_all_qc
        pass_all[i] = operator_stage_qc.pass_all_qc

    barWidth = 1
    # Set height of all sets of bars
    # Height of stanford_fail is height of pass_all
    # Height of fail_all is height of fail_stanford and pass_all
    all_fail_height = np.add(fail_stanford, pass_all).tolist()
    # height of fail_operator + fail_all
    operator_height = np.add(all_fail_height, fail_all).tolist()

    # Set color scheme
    pass_color = '#027608'
    fail_op_color = '#0348a1'
    fail_stanford_color = '#c3121e'
    fail_both_color = '#9c5300'

    # pass_color = '#87C27E'
    # fail_op_color = '#FCEFA9'
    # fail_stanford_color = '#B9B5D6'
    # fail_both_color = '#B8ADAA'

    # The position of the bars on the x-axis
    r = np.linspace(0, n_operators, num=n_operators)

    # Bars for fail operator QC (on top of failing Stanford and both)
    plt.bar(r, fail_operator, bottom=operator_height, color=fail_op_color, edgecolor='black', width=barWidth,
            label='Removed by Operator QC')
    # Create bars for failing both QC criteria
    plt.bar(r, fail_all, bottom=all_fail_height, color=fail_both_color, edgecolor='black', width=barWidth,
            label="Removed by Both QC")
    # Create failing Stanford QC only
    plt.bar(r, fail_stanford, bottom=pass_all, color=fail_stanford_color, edgecolor='black', width=barWidth,
            label="Removed by Stanford QC")
    # Creat bars for passing all QC
    plt.bar(r, pass_all, color=pass_color, edgecolor='black', width=barWidth, label='Passed all QC')

    # Custom X axis
    plt.xticks(r, operators, fontweight='bold')
    plt.xlabel("Operator", fontsize=14)
    plt.ylabel("Number of Overpasses", fontsize=14)
    plt.title(f"QC Filtering Summary (SU discard criteria: {discard})")
    plt.tick_params(direction='in', right=True, top=True)
    plt.tick_params(labelsize=12)
    plt.minorticks_on()
    plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    plt.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
    plt.tick_params(direction='in', which='major', length=6, bottom=False, top=False, left=True, right=True)

    plt.legend()

    # Save figure
    now = datetime.datetime.now()
    save_time = now.strftime("%Y%m%d")
    fig_name = f'qc_stage_{stage}_{discard}_{save_time}'
    fig_path = pathlib.PurePath('04_figures', 'qc_summary', fig_name)
    plt.savefig(fig_path)
    plt.show()


# %% Plot daily releases function
def plot_daily_releases(operator, stage, strict_discard=False, time_ave=60, gas_comp_source='ms'):
    """Function to plot daily releases for operators.
    Inputs:
      - Operator is the operator name
      - flight_days is a dataframe with column dates that stores a string for test date of format mm_dd (this can be the operator_flight_days dataframe stored in results
      - operator_releases is a dictionary with a key for each release date (format mm_dd) where corresponding value is a dataframe of Stanford metered flow rates. """

    flight_days = load_operator_flight_days(operator)
    operator_releases = load_daily_releases(operator)
    dates = flight_days.date
    for day in dates:

        # test date and month:
        month_abb = day[0:2]
        date_abb = day[3:5]
        date_string = f'2022-{month_abb}-{date_abb}'

        # Load overpass data:
        operator_stage_overpasses = load_overpass_summary(operator, stage, strict_discard, time_ave, gas_comp_source)
        daily_overpasses = operator_stage_overpasses[operator_stage_overpasses
                                                     ['overpass_datetime'].dt.strftime('%Y-%m-%d') == date_string]

        # Determine date string for title
        if month_abb == '10':
            test_month = 'October'
        elif month_abb == '11':
            test_month = 'November'
        else:
            test_month = 'ERROR! DEBUG!'

        test_date = day[3:5]

        daily_data = operator_releases[day]

        x_data = daily_data['datetime_utc']
        y_data = daily_data[f'methane_kgh']
        kgh_max = math.ceil(max(y_data) / 100) * 100  # Max kgh rounded to nearest 100

        # Initialize Figure
        fig, ax = plt.subplots(1, figsize=(12, 4))
        plt.plot(x_data, y_data, color='black',
                 linewidth=0.5)

        # Set y-axis limits
        ax.set(ylim=(0, kgh_max))

        # Add vertical lines for overpasses

        # set marker height to be 5% below top line
        marker_height = 0.9 * kgh_max

        # create array for y data at marker height
        overpass_y = np.ones(len(daily_overpasses)) * marker_height

        # Set color scheme
        pass_color = '#027608'
        fail_stanford_color = '#c3121e'
        fail_op_color = '#0348a1'
        fail_both_color = '#9c5300'

        overpass_colors = {'pass_all': pass_color,
                           'fail_stanford': fail_stanford_color,
                           'fail_operator': fail_op_color,
                           'fail_all': fail_both_color,
                           }

        overpass_legend = {'Valid Overpass': pass_color,
                           'Stanford Filtered': fail_stanford_color,
                           'Operator Filtered': fail_op_color,
                           'Both Removed': fail_both_color,
                           }

        ax.scatter(x=daily_overpasses['overpass_datetime'],
                   y=daily_overpasses['release_rate_kgh'],
                   # edgecolor='black',
                   color=daily_overpasses['qc_summary'].map(overpass_colors),
                   marker='|',
                   s=2000)

        # Add a legend

        handles = [
            Line2D([0], [0], marker='|', color=v, linestyle='None', label=k, markersize=8) for
            k, v in
            overpass_legend.items()]
        lgd = ax.legend(title='Overpass Key', handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Title
        plt.title(f'{test_month} {test_date}: {operator} Release Rates and Overpasses')

        # Format axes
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.xlabel('Time (UTC)', fontsize=14)
        plt.ylabel('Metered Release Rate (kgh)', fontsize=14)
        plt.tick_params(direction='in', right=True, top=True)
        plt.tick_params(labelsize=12)
        plt.minorticks_on()
        plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
        plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
        plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

        # Save figure

        if strict_discard == True:
            discard = 'strict'
        else:
            discard = 'lax'

        now = datetime.datetime.now()
        op_ab = abbreviate_op_name(operator)
        save_time = now.strftime("%Y%m%d")
        fig_name = f'release_chart_{op_ab}_{day}_{discard}_{save_time}'
        fig_path = pathlib.PurePath('04_figures', 'release_rates', fig_name)
        plt.savefig(fig_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()


# %%

def make_releases_histogram(operator, stage, strict_discard=False, time_ave=60, gas_comp_source='ms'):
    ############## Setup Data ##############

    # Create bins for middle histogram plot
    threshold_lower = 0
    threshold_upper = 50
    n_bins = 10
    op_histogram_low = classify_histogram_data(operator=operator, stage=stage,
                                               threshold_lower=threshold_lower, threshold_upper=threshold_upper,
                                               n_bins=n_bins,
                                               strict_discard=strict_discard, time_ave=time_ave,
                                               gas_comp_source=gas_comp_source)

    # Create bins for right histogram plot
    threshold_lower = 50
    threshold_upper = 1500
    n_bins = 30
    op_histogram_high = classify_histogram_data(operator=operator, stage=stage,
                                                threshold_lower=threshold_lower, threshold_upper=threshold_upper,
                                                n_bins=n_bins,
                                                strict_discard=strict_discard, time_ave=time_ave,
                                                gas_comp_source=gas_comp_source)

    ############## Figure ##############
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3,
                                        figsize=(10, 3),
                                        gridspec_kw={'width_ratios': [0.6, 3, 4]})

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.05,
                        hspace=0.05)

    # Determine max value for the y-axis
    low_height = op_histogram_low.bin_height.max()
    high_height = op_histogram_high.bin_height.max()
    y_height = max(low_height, high_height)
    y_height = math.ceil(y_height / 5) * 5

    # Ram's colors:
    seshadri = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
    #           0sangre,    1neptune,  2pumpkin,  3clover,  4denim,     5cocoa,     6cumin  7berry

    # Color scheme
    tp_color = seshadri[3]
    tn_color = seshadri[1]
    fp_color = seshadri[2]
    fn_color = seshadri[0]
    su_color = seshadri[4]
    op_color = seshadri[5]
    missing_color = seshadri[6]

    ####### Left histogram #######
    bar_width = 0.2
    # add true negatives
    ax1.bar(0, op_histogram_low.true_negative, width=bar_width, edgecolor='black', color=tn_color)

    # Zero release discarded by SU
    su_filter_height0 = op_histogram_low.true_negative
    ax1.bar(0, op_histogram_low.zero_filter_su, bottom=su_filter_height0, width=bar_width, label='Stanford Filtered',
            edgecolor='black', color=su_color)

    # Zero release discarded by operator
    op_filter_height0 = np.add(su_filter_height0, op_histogram_low.zero_filter_su).tolist()
    ax1.bar(0, op_histogram_low.zero_filter_op, bottom=op_filter_height0, width=bar_width, label='Operator Filtered',
            edgecolor='black', color=op_color)

    # Missing data zero release
    missing_height = np.add(op_filter_height0, op_histogram_low.zero_filter_op).tolist()
    ax1.bar(0, op_histogram_low.zero_missing, bottom=op_filter_height0, width=bar_width, label='Operator Filtered',
            edgecolor='black', color=missing_color)

    ####### Middle histogram #######
    bar_width = 4.2
    # Middle plot

    # Add True Positives
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.true_positive, width=bar_width,
            label='True positive', edgecolor='black', color=tp_color)

    # Add False Positives
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.false_positive, bottom=op_histogram_low.true_positive,
            width=bar_width, label='False positive', edgecolor='black', color=fp_color)

    # Add False Negatives
    fn_height = np.add(op_histogram_low.true_positive, op_histogram_low.false_positive).tolist()
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.false_negative, bottom=op_histogram_low.true_positive,
            width=bar_width, label='False Negative', edgecolor='black', color=fn_color)

    # Add Stanford QC
    su_filter_height = np.add(fn_height, op_histogram_low.false_negative).tolist()
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.filter_stanford, bottom=su_filter_height, width=bar_width,
            label='Stanford Filtered', edgecolor='black', color=su_color)

    # Add Carbon Mapper QC
    op_filter_height = np.add(su_filter_height, op_histogram_low.filter_stanford).tolist()
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.filter_operator, bottom=op_filter_height, width=bar_width,
            label='Stanford Filtered', edgecolor='black', color=op_color)

    # Add missing data
    missing_height = np.add(op_filter_height, op_histogram_low.filter_operator).tolist()
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.missing_data, bottom=missing_height, width=bar_width,
            label='Stanford Filtered', edgecolor='black', color=missing_color)

    ####### Right plot #######

    # reset bin width
    bar_width = 40
    # Add True Positives
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.true_positive, width=bar_width, label='True positive',
            edgecolor='black', color=tp_color)

    # Add False Positives
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.false_positive, bottom=op_histogram_high.true_positive,
            width=bar_width, label='False positive', edgecolor='black', color=fp_color)

    # Add False Negatives
    fn_height = np.add(op_histogram_high.true_positive, op_histogram_high.false_positive).tolist()
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.false_negative, bottom=op_histogram_high.true_positive,
            width=bar_width, label='False Negative', edgecolor='black', color=fn_color)

    # Add Stanford QC
    su_filter_height = np.add(fn_height, op_histogram_high.false_negative).tolist()
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.filter_stanford, bottom=su_filter_height, width=bar_width,
            label='Stanford Filtered', edgecolor='black', color=su_color)

    # Add Carbon Mapper QC
    op_filter_height = np.add(su_filter_height, op_histogram_high.filter_stanford).tolist()
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.filter_operator, bottom=op_filter_height, width=bar_width,
            label='Stanford Filtered', edgecolor='black', color=op_color)

    # Add missing data
    missing_height = np.add(op_filter_height, op_histogram_high.filter_operator).tolist()
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.missing_data, bottom=missing_height, width=bar_width,
            label='Stanford Filtered', edgecolor='black', color=missing_color)

    ############ Plot formatting ############
    # Set height of x and y axis limits
    # Left plot only shows zero
    ax1.set_ylim(bottom=0, top=y_height)
    ax1.set_xlim([-0.25, 0.25])

    # Middle plot shows >0 to 50 kgh
    ax2.set_ylim(bottom=0, top=y_height)
    ax2.set_xlim(left=-0.5, right=51)

    # Right plot shows 50 to 1500
    ax3.set_ylim(bottom=0, top=y_height)
    ax3.set_xlim(left=30, right=1500)

    # Common label for x-axis on all suplots
    txt_x_label = fig.text(0.5, -0.08, 'Release Rate (kgh)', ha='center', va='bottom', fontsize=14)

    # Plot title
    txt_title = fig.text(0.5, 1, f'{operator} Stage {stage} Results Classification', ha='center', va='top', fontsize=15)

    # Axes formatting and labels
    ax1.set_xticks([0])  # only have a tick at 0
    ax1.set_ylabel('Number of Releases', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.minorticks_on()
    ax1.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)  # only label left & bottom axis
    ax1.tick_params(direction='in', which='major', axis='y', length=4, left=True, right=True)  # y-axis major
    ax1.tick_params(direction='in', which='minor', length=2, left=True, right=True)  # y-axis minor
    ax1.tick_params(direction='out', axis='x', which='major', length=4, bottom=True, top=False)  # x-axis major

    # Format axes on middle plot
    ax2.tick_params(labelsize=12)
    ax2.minorticks_on()
    ax2.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=False)  # only label bottom axis
    ax2.tick_params(direction='in', which='major', axis='y', length=4, left=True, right=True)  # y-axis major
    ax2.tick_params(direction='in', which='minor', length=2, left=True, right=True)  # y-axis minor
    ax2.tick_params(direction='out', axis='x', which='major', length=4, bottom=True, top=False)  # x-axis major
    ax2.tick_params(which='minor', axis='x', bottom=False, top=False)
    x_ticks = ax2.xaxis.get_major_ticks()
    x_ticks[1].label1.set_visible(False)  # remove label on x=0
    x_ticks[1].set_visible(False)

    # Format axes on right plot
    ax3.tick_params(labelsize=12)
    ax3.minorticks_on()
    ax3.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=False)  # only label on bottom
    ax3.tick_params(axis='y', which='major', direction='in', length=4, left=True, right=True)  # y-axis major
    ax3.tick_params(axis='y', which='minor', direction='in', length=2, left=True, right=True)  # y-axis minor
    ax3.tick_params(direction='out', axis='x', which='major', length=4, bottom=True, top=False)  # x-axis major
    ax3.tick_params(which='minor', axis='x', bottom=False, top=False)

    # Set axes and background color to white
    ax1.set_facecolor('white')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['bottom'].set_color('black')

    # Add legend

    histogram_legend = {
        'True Positive': tp_color,
        'True Negative': tn_color,
        'False Positive': fp_color,
        'False Negative': fn_color,
        'Stanford Filtered': su_color,
        'Operator Filtered': op_color,
        'Missing data': missing_color,
    }

    legend_elements = [Patch(facecolor=v, edgecolor='black', label=k) for k, v in histogram_legend.items()]
    # lgd = ax3.legend(title='Overpass Key', handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    lgd = ax3.legend(title='Overpass Key', handles=legend_elements, loc='upper right')

    ############ Save Data ############

    op_ab = abbreviate_op_name(operator)
    now = datetime.datetime.now()
    save_time = now.strftime("%Y%m%d")

    # Save figure

    if strict_discard == True:
        discard = 'strict'
    else:
        discard = 'lax'

    fig_name = f'histogram_{op_ab}_{discard}_{save_time}'
    fig_path = pathlib.PurePath('04_figures', 'histogram', fig_name)
    plt.savefig(fig_path, bbox_extra_artists=(txt_x_label, txt_title), bbox_inches='tight')

    # Save histogram low kgh inputs
    table_name = f'histogram_low_{op_ab}_{save_time}.csv'
    table_path = pathlib.PurePath('06_results', 'histogram', table_name)
    op_histogram_low.to_csv(table_path)

    # Save histogram high kgh inputs
    table_name = f'histogram_high_{op_ab}_{save_time}.csv'
    table_path = pathlib.PurePath('06_results', 'histogram', table_name)
    op_histogram_high.to_csv(table_path)


# %% Plot meter uncertainty


def plot_meter_uncertainty(select_meter, presentation):
    """Plot uncertainty for meters, input meter abbreviation of 'bc', 'mc', or 'pc' for Baby Corey, Mama Corey, or
     Papa Corey respectively """

    ####### Calculate Uncertainty #######

    meters = ['bc', 'mc', 'pc']
    meter_min = {}
    meter_max = {}
    meter_range = {}
    meter_uncertainty = {}

    meter_min['bc'] = 0.56  # kgh
    meter_max['bc'] = 30  # kgh
    meter_min['mc'] = 3.87  # kgh
    meter_max['mc'] = 300  # kgh
    meter_min['pc'] = 40  # kgh
    meter_max['pc'] = 2000  # kgh

    for meter in meters:
        meter_range[meter] = np.linspace(meter_min[meter], meter_max[meter], 1000)
        store_uncertainty = []
        for flowrate in meter_range[meter]:
            uncertainty = calc_meter_uncertainty(meter, flowrate)
            store_uncertainty.append(uncertainty)

        meter_uncertainty[meter] = store_uncertainty

    ####### Plot #######
    x_data = meter_range[select_meter]
    y_data = meter_uncertainty[select_meter]

    # Name meter for plot
    meter_id = {}
    meter_id['bc'] = 'CMFS015H'
    meter_id['mc'] = 'CMF050M'
    meter_id['pc'] = 'CMFS150M'

    # Initialize figure
    fig, ax = plt.subplots(1, figsize=(6, 4))
    plt.plot(x_data, y_data, color='black',
             linewidth=2)

    ax.set(xlim=(0, meter_max[select_meter]),
           ylim=(0, 2)
           )

    # title
    plt.title(f'{meter_id[select_meter]} Uncertainty\n(Data from Emerson Online Sizing Tool)', fontweight='bold')

    # Format axes
    plt.xlabel('Meter Flow Rate\n(kg/hr whole gas)', fontsize=14, fontweight='bold')
    plt.ylabel('Meter Uncertainty\n(% of flow rate)', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tick_params(direction='in', right=True, top=True)
    plt.tick_params(labelsize=12)
    plt.minorticks_on()
    plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

    ##### Settings for presentations #####
    if presentation is True:
        # change all spines to thicker lineweight
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        # make the ticks bold
        plt.xticks(weight='bold')
        plt.yticks(weight='bold')
        ax.tick_params(which='both', width=2)
        # ax.xaxis.set_tick_params(width=2)
        # ax.yaxis.set_tick_params(width=2)

    # Save figure

    save_name = f'{select_meter}_percent_uncertainty'
    save_location = pathlib.PurePath('04_figures', 'meter_accuracy', save_name)
    plt.savefig(save_location, transparent=True, bbox_inches='tight')

    plt.show()


def plot_selected_release_period(start_t, stop_t, gas_comp_source='ms'):
    """Plot flow rate (kgh CH4) for a given input period."""
    # TODO duplicate code with calc_average_release -> turn into functions later
    if start_t.date() != stop_t.date():
        # End function if start and end t are on different dates
        print(
            "I can't currently plot data with start and end time on different dates. Why are you trying to do this anyway?")
        return
    elif start_t.date() > stop_t.date():
        print('Start time is after end time. Time for *you* to do some debugging!')
        return
    else:
        # Load data
        file_name = start_t.strftime('%m_%d')
        # use subclass Path (instead of PurePath) because we need to check if the file exists
        file_path = pathlib.Path('02_meter_data', 'daily_meter_data', 'whole_gas_clean',
                                 f'{gas_comp_source}', f'{file_name}.csv')
    if not file_path.is_file():
        # If file does not exist, we did not conduct releases on that day.
        # Set all values of results_summary to zero or np.nan
        print('No gas released on selected date.')
        return

    meter_data = pd.read_csv(file_path, index_col=0, parse_dates=['datetime_utc'])
    time_ave_mask = (meter_data['datetime_utc'] > start_t) & (meter_data['datetime_utc'] <= stop_t)
    selected_period = meter_data.loc[time_ave_mask].copy()

    x_data = selected_period.datetime_utc
    y_data = selected_period.methane_kgh

    if max(y_data) > 100:
        kgh_max = math.ceil(max(y_data) / 100) * 100  # Max kgh rounded to nearest 100
    else:
        kgh_max = math.ceil(max(y_data) / 10) * 10  # Max kgh rounded to nearest 100

    # Initialize Figure
    fig, ax = plt.subplots(1, figsize=(5, 5))
    plt.plot(x_data, y_data, color='black',
             linewidth=1)
    # Set y-axis limits
    ax.set(ylim=(0, kgh_max))

    x_interval = (stop_t - start_t) / 5  # x-axis will have 5 intervals
    # Convert datetime to string and back to float
    x_interval = int(str(x_interval)[2:4])

    test_date = start_t.strftime('%b %d')
    start_string = start_t.strftime('%H:%M')
    stop_string = stop_t.strftime('%H:%M')
    # Title
    plt.title(f'{test_date}: {start_string} - {stop_string} (UTC)')

    # Format axes
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xlabel('Time (UTC)', fontsize=14)
    plt.ylabel('Metered Release Rate (kgh)', fontsize=14)
    plt.tick_params(direction='in', right=True, top=True)
    plt.tick_params(labelsize=12)
    plt.minorticks_on()
    plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)

    # Save Fig
    plot_period = start_t.strftime('%m-%d_%H%m')
    now = datetime.datetime.now()
    save_time = now.strftime("%Y%m%d")
    fig_name = f'release_chart_{plot_period}_saved_{save_time}'
    fig_path = pathlib.PurePath('04_figures', 'misc_flow_rate_plots', fig_name)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def set_axes_clean_formatting(ax):
    """Set default axes formats"""
    # Equalize Axes
    ax.set_aspect('equal', adjustable='box')

    # Set axes and background color to white
    ax.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Axes labels
    # ax.set_xlabel('Methane Release Rate (kgh)', fontsize=14)
    # ax.set_ylabel('Reported Release Rate (kgh)', fontsize=14)
    ax.tick_params(direction='in', right=True, top=True)
    ax.tick_params(labelsize=12)
    ax.minorticks_on()
    ax.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    ax.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)
    ax.grid(False)  # remove grid lines

    return ax


def make_generic_plot(x_data, y_data, ax, x_lim='largest_input', y_lim='largest_input'):
    # Add plots to ax
    ax.scatter(x_data, y_data, alpha=0.5)
    ax = set_axes_clean_formatting(ax)

    if x_lim == 'largest_input':
        largest_x = max(x_data)
        x_lim = [0, largest_x + 20]

    if y_lim == 'largest_input':
        smallest_y = min(y_data)
        largest_y = max(y_data)
        # print(f'smallest y: {smallest_y}')
        # print(f'largest y: {largest_y}')
        y_lim = [smallest_y - 50, largest_y + 50]

    # Set axes
    ax.set(xlim=x_lim,
           ylim=y_lim,
           alpha=0.8)

    return ax


def plot_residuals(ax, above_mdl, x_lim, y_lim, operator, stage, qc_status, strict_discard=False, time_ave=60,
                   gas_comp_source='ms'):
    """Plot residuals
    ax is the subplot to be plotted
    above_mdl is True or False, if True then only plots above threshold of 10 kgh
    x_lim is limit of x-axis
    y_lim is limit of y-axis

    """
    error_dataset = calculate_residuals_and_error(operator, stage, qc_status, strict_discard, time_ave, gas_comp_source)
    if above_mdl:  # if True
        error_dataset = error_dataset.loc[error_dataset.meter_data >= 10]

    # Select x and y data from error dataset
    x_data = error_dataset.meter_data
    y_data = error_dataset.residual

    # Initialize figure
    # fig, ax = plt.subplots(1, figsize=(6, 6))
    make_generic_plot(x_data, y_data, ax, x_lim=x_lim, y_lim=y_lim)
    # plt.title(f'{operator} Stage {stage} Residuals')
    ax.set_xlabel('Methane Release Rate (kg/hr)', fontsize=14)
    ax.set_ylabel('Residual (kg/hr)', fontsize=14)

    # Stage key for blinded status
    stage_description = {
        1: 'Fully blinded results',
        2: 'Unblinded wind',
        3: 'Partially unblinded',
    }

    stage_text = stage_description[stage]

    text = f'{operator}\n {stage_text}'
    ob = offsetbox.AnchoredText(text, loc='upper left')
    ob.set(alpha=0.8)
    ax.add_artist(ob)
    ax.set_aspect('auto')



def plot_residual_percent_error(ax, above_mdl, x_lim, y_lim, operator, stage, qc_status, strict_discard=False, time_ave=60,
                   gas_comp_source='ms'):
    """Plot residuals
    ax is the subplot to be plotted
    above_mdl is True or False, if True then only plots above threshold of 10 kgh
    x_lim is limit of x-axis
    y_lim is limit of y-axis

    """
    error_dataset = calculate_residuals_and_error(operator, stage, qc_status, strict_discard, time_ave, gas_comp_source)
    if above_mdl:  # if True
        error_dataset = error_dataset.loc[error_dataset.meter_data >= 10]

    # Select x and y data from error dataset
    x_data = error_dataset.meter_data
    y_data = error_dataset.residual_percent_error

    # Initialize figure
    # fig, ax = plt.subplots(1, figsize=(6, 6))
    make_generic_plot(x_data, y_data, ax, x_lim=x_lim, y_lim=y_lim)
    # plt.title(f'{operator} Stage {stage} Residuals')
    ax.set_xlabel('Methane Release Rate (kg/hr)', fontsize=14)
    ax.set_ylabel('Residual Error (%)', fontsize=14)

    # Stage key for blinded status
    stage_description = {
        1: 'Fully blinded results',
        2: 'Unblinded wind',
        3: 'Partially unblinded',
    }

    stage_text = stage_description[stage]

    text = f'{operator}\n {stage_text}'
    ob = offsetbox.AnchoredText(text, loc='upper left')
    ob.set(alpha=0.8)
    ax.add_artist(ob)
    ax.set_aspect('auto')

    return ax



def plot_quant_error_absolute(ax, above_mdl, x_lim, y_lim, operator, stage, qc_status, strict_discard=False, time_ave=60,
                              gas_comp_source='ms'):
    """Plot absolute error"""
    error_dataset = calculate_residuals_and_error(operator, stage, qc_status, strict_discard, time_ave, gas_comp_source)

    if above_mdl:  # if True
        error_dataset = error_dataset.loc[error_dataset.meter_data >= 10]

    x_data = error_dataset.meter_data
    y_data = error_dataset.quant_error_absolute

    # Initialize figure
    make_generic_plot(x_data, y_data, ax, x_lim=x_lim, y_lim=y_lim)
    plt.title(f'{operator} Stage {stage} Absolute Error')
    ax.set_xlabel('Methane Release Rate (kgh)', fontsize=14)
    ax.set_ylabel('Quantification Error (absolute kg/hr)', fontsize=14)

    # Stage key for blinded status
    stage_description = {
        1: 'Fully blinded results',
        2: 'Unblinded wind',
        3: f'Partially unblinded',
    }

    stage_text = stage_description[stage]

    text = f'{operator}\n {stage_text}'
    ob = offsetbox.AnchoredText(text, loc='upper left')
    ob.set(alpha=0.8)
    ax.add_artist(ob)


def plot_quant_error_percent(ax, above_mdl, x_lim, y_lim, operator, stage, qc_status, strict_discard=False, time_ave=60,
                             gas_comp_source='ms'):
    """Plot percent error"""
    error_dataset = calculate_residuals_and_error(operator, stage, qc_status, strict_discard, time_ave, gas_comp_source)

    if above_mdl:  # if True
        error_dataset = error_dataset.loc[error_dataset.meter_data >= 10]

    x_data = error_dataset.meter_data
    y_data = error_dataset.quant_error_percent

    # Initialize figure
    make_generic_plot(x_data, y_data, ax, x_lim=x_lim, y_lim=y_lim)
    # plt.title(f'{operator} Stage {stage} Percent Error')
    ax.set_xlabel('Methane Release Rate (kg/hr)', fontsize=14)
    ax.set_ylabel('Percent Quantification Error (%)', fontsize=14)

    # Stage key for blinded status
    stage_description = {
        1: 'Fully blinded results',
        2: 'Unblinded wind',
        3: 'Partially unblinded\n',
    }

    stage_text = stage_description[stage]

    text = f'{operator}\n {stage_text}'
    ob = offsetbox.AnchoredText(text, loc='upper right')
    ob.set(alpha=0.8)
    ax.add_artist(ob)
    ax.set_aspect('auto')
    return ax


def plot_logistic_regression(ax, threshold, operator, stage, strict_discard, time_ave, gas_comp_source):
    operator_model = make_logistic_regression(operator=operator, stage=stage, strict_discard=strict_discard,
                                              time_ave=time_ave, gas_comp_source=gas_comp_source)

    # input values for plotting
    x_plot = np.linspace(0, threshold, 500)

    # calculate the probability from the model
    y_plot = operator_model.predict_proba(x_plot.reshape(-1, 1))[:, 1]
    ax.plot(x_plot, y_plot, label='Logistic best fit')
    return ax
