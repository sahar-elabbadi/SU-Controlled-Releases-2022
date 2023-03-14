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

from methods_source import load_overpass_summary, abbreviate_op_name


# Function: generate jitter for a given array
def rand_jitter(input_list):
    delta = 0.2
    return input_list + np.random.randn(len(input_list)) * delta


# %% Function: plot_parity

# Inputs:
# > operator
# > stage
# > operator_report
# > operator_meter
def plot_parity(operator, stage, strict_discard):
    """Inputs are operator name, stage of analysis, operator_plot dataframe containing all relevant data"""
    op_ab = abbreviate_op_name(operator)

    # # Select appropriate path based on whether or not we are using strict QC criteria
    # if strict_discard is True:
    #     path = pathlib.PurePath('03_results', 'overpass_summary', f'{op_ab}_{stage}_overpasses_strict.csv')
    # else:
    #     path = pathlib.PurePath('03_results', 'overpass_summary', f'{op_ab}_{stage}_overpasses.csv')

    # Load overpass summary csv file
    operator_plot = load_overpass_summary(operator, stage, strict_discard)

    # Apply the following filters to overpass data :

    # Must pass all QC filters
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
    y_index = np.isfinite(operator_plot['operator_quantification'])

    # Select x data
    x_data = operator_plot['release_rate_kgh']
    y_data = operator_plot['operator_quantification']
    y_error = operator_plot['operator_upper'] - operator_plot['operator_quantification']

    # Fit linear regression via least squares with numpy.polyfit
    # m is slope, intercept is b
    m, b = np.polyfit(x_data[y_index], y_data[y_index], deg=1)

    # Calculate R^2 value
    # (using method described here: https://www.askpython.com/python/coefficient-of-determination)
    correlation_matrix = np.corrcoef(x_data[y_index], y_data[y_index])
    correlation = correlation_matrix[0, 1]
    r2 = correlation ** 2

    # Number of valid overpasses:
    sample_size = len(y_data)

    # Set x and y max values
    # Manually set largest x and y value by changing largest_kgh here to desired value:
    # largest_kgh = 1200

    # Or, determine largest_kgh by calculating largest value in x_data and y_data
    # Filter out NA because operations with NA returns NA
    if np.isnan(max(y_error)) == 1:
        y_error.iloc[:] = 0

    largest_kgh = max(max(x_data), max(y_data)) + max(y_error)
    largest_kgh = math.ceil(largest_kgh / 100) * 100

    # Create sequence of numbers for plotting linear fit (x)
    x_seq = np.linspace(0, largest_kgh, num=100)

    fig, ax = plt.subplots(1, figsize=(6, 6))
    # Add linear regression
    plt.plot(x_seq, m * x_seq + b, color='k', lw=2,
             label=f'Best Fit ($y = {m:0.2f}x+{b:0.2f}$, $R^2 =$ {r2:0.4f})')

    # Add x = y line
    plt.plot(x_seq, x_seq, color='k', lw=2, linestyle='--',
             label='y = x')

    ax.errorbar(x_data, y_data,
                yerr=y_error,
                linestyle='none',
                mfc='white',
                label=f'{operator} Stage {stage} data',
                fmt='o',
                markersize=5)

    # I don't think I need this, was it hold over from when I was trying to figure it out?
    # If something breaks in the code later, check back to this spot

    # ax.set_xlabel('x-axis')
    # ax.set_ylabel('y-axis')
    # ax.set_title('Line plot with error bars')

    # Set title
    plt.title(f'{operator} Stage {stage} Results ({sample_size} measurements)')

    # Set axes
    ax.set(xlim=(0, largest_kgh),
           ylim=(0, largest_kgh),
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
    plt.xlabel('Methane Release Rate (kgh)', fontsize=14)
    plt.ylabel('Reported Release Rate (kgh)', fontsize=14)
    plt.tick_params(direction='in', right=True, top=True)
    plt.tick_params(labelsize=12)
    plt.minorticks_on()
    plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)
    plt.grid(False)  # remove grid lines

    # Legend
    plt.legend(facecolor='white')

    # Save figure
    now = datetime.datetime.now()
    op_ab = abbreviate_op_name(operator)
    save_time = now.strftime("%Y%m%d")
    fig_name = f'parity_{op_ab}_stage{stage}_{save_time}_test'
    fig_path = pathlib.PurePath('04_figures', 'parity_plots', fig_name)
    plt.savefig(fig_path)
    plt.show()

    # Save data used to make figure
    save_parity_data = pd.DataFrame()
    save_parity_data['release_rate'] = x_data
    save_parity_data['operator_report'] = y_data
    save_parity_data['operator_sigma'] = y_error

    save_path = pathlib.PurePath('03_results', 'parity_plot_data', f'{operator}_{stage}_parity_{save_time}.csv')
    save_parity_data.to_csv(save_path)


# %% Function: plot_detection_limit


# inputs:
# operator: name of operator
# operator_report: operator data report
# operator_meter: operator meter data
# n_bins: number of bins desired in plot
# threshold: highest release rate in kgh to show in detection threshold graph

def plot_detection_limit(operator, stage, operator_report, operator_meter, n_bins, threshold, strict_discard):
    # merge meter and operator reports and apply Stanford QC filter
    operator_df = apply_qc_filter(operator_report, operator_meter, strict_discard)

    # Make column with easier name for coding for now.
    operator_df['release_rate_kgh'] = operator_df['Last 60s (kg/h) - from Stanford']

    # Determine whether each overpass below the threshold value was detected
    operator_detection = pd.DataFrame()
    operator_detection['overpass_id'] = operator_df.PerformerExperimentID
    operator_detection['non_zero_release'] = operator_df.release_rate_kgh != 0  # True if we conducted a release
    operator_detection['operator_detected'] = operator_df.Detected
    operator_detection['release_rate_kgh'] = operator_df.release_rate_kgh
    operator_detection['operator_quantification'] = operator_df.FacilityEmissionRate

    # Select overpasses that are below the threshold of interest AND where release is non-zero
    operator_detection = operator_detection.loc[operator_detection.release_rate_kgh <= threshold].loc[
        operator_detection.non_zero_release == True]

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
        binned_data = operator_detection.loc[operator_detection.release_rate_kgh < bin_max].loc[
            operator_detection.release_rate_kgh >= bin_min]

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

    # Plot bars and detection points
    ax.bar(detection_plot.bin_median,
           detection_plot.detection_prob_mean,
           yerr=[detection_plot.detection_prob_two_sigma_lower, detection_plot.detection_prob_two_sigma_upper],
           error_kw=dict(lw=2, capsize=3, capthick=1, alpha=0.3),
           width=threshold / n_bins - 0.5, alpha=0.6, color='#9ecae1', ecolor='black', capsize=2)

    # yulia's color: edgecolor="black",facecolors='none'
    x_data = rand_jitter(operator_detection.release_rate_kgh)

    ax.scatter(x_data, np.multiply(operator_detection.operator_detected, 1),
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
    now = datetime.datetime.now()
    save_time = now.strftime("%Y%m%d")
    op_ab = abbreviate_op_name(operator)
    fig_name = f'detect_limit_{op_ab}_stage{stage}_{save_time}'
    fig_path = pathlib.PurePath('04_figures', fig_name)
    plt.savefig(fig_path)
    plt.show()

    # Save data used to make plots
    operator_detection.to_csv(pathlib.PurePath('03_results', 'detect_probability_data',
                                               f'{op_ab}_{stage}_detect_{save_time}.csv'))
    detection_prob.to_csv(pathlib.PurePath('03_results', 'detect_probability_data',
                                           f'{op_ab}_{stage}_{threshold}kgh_{n_bins}bins_{save_time}.csv'))
    return


#%%

def plot_qc_summary():

    # Load saved QC dataframe
    all_qc = pd.read_csv(pathlib.PurePath('03_results', 'qc_comparison', 'all_qc.csv'), index_col=0)
    # Plot

    category = ['fail_stanford_only', 'fail_all_qc', 'fail_operator_only']
    stage = 1
    n_operators = 4 # number of operators
    operators = ['Carbon Mapper', 'GHGSat', 'Kairos LS23', 'Kairos LS25']
    # Determine values for each group, alphabetical order of operators: "Carbon Mapper, GHGSat, Kairos"

    fail_operator = np.zeros(n_operators)
    fail_stanford = np.zeros(n_operators)
    fail_all = np.zeros(n_operators)
    pass_all = np.zeros(n_operators)

    # Height of bars

    for i in range(len(operators)): # for go through fail stanford only
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
    pass_color = '#018571'
    fail_op_color = '#a6611a'
    fail_stanford_color = '#dfc27d'
    fail_both_color = '#f5f5f5'

    # pass_color = '#87C27E'
    # fail_op_color = '#FCEFA9'
    # fail_stanford_color = '#B9B5D6'
    # fail_both_color = '#B8ADAA'

    # The position of the bars on the x-axis
    r = [0,1.5,3,4.5]

    # Bars for fail operator QC (on top of failing Stanford and both)
    plt.bar(r, fail_operator, bottom=operator_height, color=fail_op_color, edgecolor='black', width=barWidth, label = 'Removed by Operator QC')
    # Create bars for failing both QC criteria
    plt.bar(r, fail_all, bottom=all_fail_height, color=fail_both_color, edgecolor='black', width=barWidth,  label = "Removed by Both QC")
    # Create failing Stanford QC only
    plt.bar(r, fail_stanford, bottom=pass_all, color=fail_stanford_color, edgecolor='black', width=barWidth, label = "Removed by Stanford QC")
    # Creat bars for passing all QC
    plt.bar(r, pass_all, color=pass_color, edgecolor='black', width=barWidth, label = 'Passed all QC')

    # Custom X axis
    plt.xticks(r, operators, fontweight='bold')
    plt.xlabel("Operator", fontsize = 14)
    plt.ylabel("Number of Overpasses", fontsize = 14)
    plt.title("Summary of Quality Control Filtering")
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
    fig_name = f'qc_stage_{stage}_{save_time}'
    fig_path = pathlib.PurePath('04_figures', 'qc_summary', fig_name)
    plt.savefig(fig_path)
    plt.show()



#%% Plot daily releases function
def plot_daily_releases(operator, flight_days, operator_releases, stage, strict_discard):
    """Function to plot daily releases for operators.
    Inputs:
      - Operator is the operator name
      - flight_days is a dataframe with column dates that stores a string for test date of format mm_dd (this can be the operator_flight_days dataframe stored in results
      - operator_releases is a dictionary with a key for each release date (format mm_dd) where corresponding value is a dataframe of Stanford metered flow rates. """

    dates = flight_days.date
    for day in dates:

        # test date and month:
        month_abb = day[0:2]
        date_abb = day[3:5]
        date_string = f'2022-{month_abb}-{date_abb}'

        # Load overpass data:
        operator_stage_overpasses = load_overpass_summary(operator, stage, strict_discard)
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
        y_data = daily_data['flow_rate']
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

        overpass_colors = {'pass_all': '#018571',
                           'fail_operator': '#a6611a',
                           'fail_stanford': '#dfc27d',
                           'fail_all': '#878787',
                           }

        overpass_legend = {'Valid Overpass': '#018571',
                           'Operator Removed': '#a6611a',
                           'Stanford Removed': '#dfc27d',
                           'Both Removed': '#878787',
                           }

        ax.scatter(x=daily_overpasses['overpass_datetime'],
                   y=daily_overpasses['release_rate_kgh'],
                   # edgecolor='black',
                   color=daily_overpasses['qc_summary'].map(overpass_colors),
                   marker='|',
                   s=2000)

        # add a legend
        # handles for circles
        # handles = [
        #     Line2D([0], [0], marker='o', markerfacecolor=v, color = 'black', linestyle='None', label=k, markersize=8) for
        #     k, v in
        #     overpass_legend.items()]

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
        now = datetime.datetime.now()
        op_ab = abbreviate_op_name(operator)
        save_time = now.strftime("%Y%m%d")
        fig_name = f'release_chart_{op_ab}_{day}'
        fig_path = pathlib.PurePath('04_figures', fig_name)
        plt.savefig(fig_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()
