# Script for methods analyzing results
# Author: Sahar H. El Abbadi
# Date Created: 2023-03-03
# Date Last Modified: 2023-03-03

# List of methods in this file:
# > summarize_qc

# Imports
import pathlib
import math
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from methods_data_cleaning import combine_datetime
from plot_methods import abbreviate_op_name


# %% method summarize_qc
def generate_overpass_summary(operator, stage, operator_report, operator_meter, strict_discard):
    """Generate clean dataframe for each overpass with columns indicating QC status.

    Columns are:
      - overpass_id
      - zero_release
      - non_zero_release
      - operator_kept
      - stanford_kept
      - pass_all_qc
    """

    if strict_discard is True:
        discard_column = 'qc_su_discard_strict'
    else:
        discard_column = 'qc_su_discard'

    op_ab = abbreviate_op_name(operator)

    # Combine operator report and meter data
    combined_df = operator_report.merge(operator_meter, on='overpass_id')

    # Rename columns to be machine-readable
    # Make column with easier name for coding for now.
    combined_df['release_rate_kgh'] = combined_df['kgh_ch4_60']

    # combined_df['time_utc'] = combined_df['Time (UTC) - from Stanford']
    # combined_df['date'] = combined_df['Date']

    # Philippine reports if we discard or not (discard = 1, keep = 0). Change this to have 1 if we keep, 0 if we discard
    combined_df['stanford_kept'] = (1 - combined_df[discard_column])

    # Make dataframe with all relevant info
    operator_qc = pd.DataFrame()
    operator_qc['overpass_id'] = combined_df.overpass_id
    operator_qc['overpass_datetime'] = combined_df.datetime_utc
    operator_qc['zero_release'] = combined_df.release_rate_kgh == 0
    operator_qc['non_zero_release'] = combined_df.release_rate_kgh != 0  # True if we conducted a release
    operator_qc['operator_kept'] = combined_df.OperatorKeep
    operator_qc['stanford_kept'] = combined_df.stanford_kept == 1
    operator_qc['phase_iii'] = combined_df.phase_iii
    operator_qc['pass_all_qc'] = operator_qc.stanford_kept * operator_qc.operator_kept

    # check if overpass failed both stanford and operator
    check_fail = operator_qc['operator_kept'] + operator_qc['stanford_kept']
    operator_qc['fail_all_qc'] = check_fail == 0

    # Include operator results
    operator_qc['operator_detected'] = combined_df.Detected
    operator_qc['release_rate_kgh'] = combined_df.release_rate_kgh
    operator_qc['operator_quantification'] = combined_df.FacilityEmissionRate
    operator_qc['operator_lower'] = combined_df.FacilityEmissionRateLower
    operator_qc['operator_upper'] = combined_df.FacilityEmissionRateUpper

    # Summarize QC results
    # Here is the list of different QC options based on the current QC boolean columns
    qc_conditions = [
        operator_qc['pass_all_qc'] == 1,  # pass all
        operator_qc['fail_all_qc'] == 1,  # fail all
        (operator_qc['stanford_kept'] == 1) & (operator_qc['operator_kept'] == 0),  # fail_operator
        (operator_qc['stanford_kept'] == 0) & (operator_qc['operator_kept'] == 1)  # stanford_fail
    ]

    # Based on the above conditions, the final QC evalation will be one of the following:
    qc_choices = [
        'pass_all',
        'fail_all',
        'fail_operator',
        'fail_stanford',
    ]

    # Apply the conditions to generate a new column for 'qc_summary'
    operator_qc['qc_summary'] = np.select(qc_conditions, qc_choices, 'ERROR')

    # Create save path based on whether or not a strict QC criteria was applied
    if strict_discard is True:
        operator_qc.to_csv(pathlib.PurePath('03_results', 'overpass_summary', f'{op_ab}_{stage}_overpasses_strict.csv'))
    else:
        operator_qc.to_csv(pathlib.PurePath('03_results', 'overpass_summary', f'{op_ab}_{stage}_overpasses.csv'))

    return operator_qc


#%%
def load_overpass_summary(operator, stage, strict_discard):
    op_ab = abbreviate_op_name(operator)

    if strict_discard is True:
        path = pathlib.PurePath('03_results', 'overpass_summary', f'{op_ab}_{stage}_overpasses_strict.csv')
    else:
        path = pathlib.PurePath('03_results', 'overpass_summary', f'{op_ab}_{stage}_overpasses.csv')

    overpass_summary = pd.read_csv(path, index_col=0)

    return overpass_summary


def summarize_qc(operator, stage, operator_report, operator_meter):
    """Summarize QC criteria applied by operator and Stanford"""

    op_ab = abbreviate_op_name(operator)

    # Generate dataframe with all relevant QC information
    operator_qc = load_overpass_summary(operator, stage, operator_report, operator_meter)

    # Determine how many were QC'ed by Stanford
    total_overpasses = len(operator_qc)
    qc_count_stanford_pass = operator_qc.stanford_kept.sum()
    qc_count_operator_pass = operator_qc.operator_kept.sum()
    qc_count_both_pass = operator_qc.pass_all_qc.sum()
    qc_count_stanford_fail = total_overpasses - qc_count_stanford_pass
    qc_count_operator_fail = total_overpasses - qc_count_operator_pass
    qc_count_both_fail = operator_qc.fail_all_qc.sum()

    qc_summary = pd.DataFrame({
        'operator': op_ab,
        'stage': stage,
        'total_overpasses': total_overpasses,
        'pass_stanford_qc': qc_count_stanford_pass,
        f'pass_operator_qc': qc_count_operator_pass,
        'pass_all_qc': qc_count_both_pass,
        'fail_stanford_qc': qc_count_stanford_fail,
        f'fail_operator_qc': qc_count_operator_fail,
        'fail_all_qc': qc_count_both_fail,
        'fail_stanford_only': qc_count_stanford_fail - qc_count_both_fail,
        'fail_operator_only': qc_count_operator_fail - qc_count_both_fail,
    }, index=[0])

    qc_summary.to_csv(pathlib.PurePath('03_results', 'qc_comparison', f'{op_ab}_{stage}_qc.csv'))
    return qc_summary


# %% Load daily meter data

def load_daily_meter_data(date):
    """Load daily meter file saved in format mm_dd.xlsx"""

    # File location
    date_path = pathlib.PurePath('02_meter_data', 'daily_meter_data', f'{date}.xlsx')

    # Import data and rename columns to machine readable format
    date_meter = pd.read_excel(date_path, header=None, names=['datetime_utc', 'flow_rate', 'meter', 'qc_flag'],
                               skiprows=1)
    date_meter.loc[date_meter['meter'] == 'Baby Coriolis', 'meter'] = 'bc'
    date_meter.loc[date_meter['meter'] == 'Mama Coriolis', 'meter'] = 'mc'
    date_meter.loc[date_meter['meter'] == 'Papa Coriolis', 'meter'] = 'pc'

    return date_meter


# %% Load overpass summary for a given operator and a given stage

def load_overpass_summary(operator, stage):
    """Input operator name and stage of analysis. Stage is a number (1, 2, 3). Operator names can be:

      - 'Carbon Mapper'
      - 'GHGSat'
      - 'Kairos LS23'
      - 'Kairos LS25'
      - 'Methane Air'
      - 'Scientific Aviation'

      """
    if operator == 'Kairos':
        pod = input('Please specify pod: LS23 or LS25')
        operator = f'{operator} {pod}'

    op = abbreviate_op_name(operator)
    file_path = pathlib.PurePath('03_results', 'overpass_summary', f'{op}_{stage}_overpasses.csv')
    overpass_summary = pd.read_csv(file_path, parse_dates=['overpass_datetime'])

    return overpass_summary


# %% Load flight days

def load_flight_days():
    cm_flight_days = {
        "date": ['10_10', '10_11', '10_12', '10_28', '10_29', '10_31'],
        "start_time": [
            datetime.datetime(2022, 10, 10, 17, 00, 00),
            datetime.datetime(2022, 10, 11, 17, 16, 13),
            datetime.datetime(2022, 10, 12, 17, 15, 16),
            datetime.datetime(2022, 10, 28, 17, 51, 5),
            datetime.datetime(2022, 10, 29, 17, 13, 30),
            datetime.datetime(2022, 10, 31, 17, 16, 46),
        ],
        "end_time": [
            datetime.datetime(2022, 10, 10, 21, 30, 14),
            datetime.datetime(2022, 10, 11, 21, 16, 36),
            datetime.datetime(2022, 10, 12, 21, 15, 23),
            datetime.datetime(2022, 10, 28, 21, 7, 0),
            datetime.datetime(2022, 10, 29, 21, 11, 34),
            datetime.datetime(2022, 10, 31, 21, 14, 22),
        ],
    }

    # Kairos
    kairos_flight_days = {
        "date": ['10_24', '10_25', '10_26', '10_27', '10_28'],
        "start_time": [
            datetime.datetime(2022, 10, 24, 16, 46, 28),
            datetime.datetime(2022, 10, 25, 16, 36, 27),
            datetime.datetime(2022, 10, 26, 16, 38, 47),
            datetime.datetime(2022, 10, 27, 16, 37, 23),
            datetime.datetime(2022, 10, 28, 16, 41, 12),
        ],
        "end_time": [
            datetime.datetime(2022, 10, 24, 19, 48, 44),
            datetime.datetime(2022, 10, 25, 20, 33, 1),
            datetime.datetime(2022, 10, 26, 20, 40, 55),
            datetime.datetime(2022, 10, 27, 20, 14, 15),
            datetime.datetime(2022, 10, 28, 20, 39, 58),
        ],
    }

    # MAIR
    mair_flight_days = {
        "date": ['10_25', '10_29'],
        "start_time": [
            datetime.datetime(2022, 10, 25, 16, 57, 47),
            datetime.datetime(2022, 10, 29, 16, 25, 1),
        ],
        "end_time": [
            datetime.datetime(2022, 10, 25, 20, 46, 41),
            datetime.datetime(2022, 10, 29, 21, 2, 17),
        ],
    }

    # GHGSat
    ghg_flight_days = {
        "date": ['10_31', '11_02', '11_04', '11_07'],
        "start_time": [
            datetime.datetime(2022, 10, 31, 17, 3, 58),
            datetime.datetime(2022, 11, 2, 16, 38, 30),
            datetime.datetime(2022, 11, 4, 16, 43, 19),
            datetime.datetime(2022, 11, 7, 19, 23, 44),
        ],
        "end_time": [
            datetime.datetime(2022, 10, 31, 20, 59, 3),
            datetime.datetime(2022, 11, 2, 18, 5, 2),
            datetime.datetime(2022, 11, 4, 20, 31, 59),
            datetime.datetime(2022, 11, 7, 22, 9, 39),
        ],
    }

    sciav_flight_days = {
        "date": ['11_08', '11_10', '11_11'],
        "start_time": [
            datetime.datetime(2022, 11, 8, 21, 41, 2),
            datetime.datetime(2022, 11, 10, 18, 5, 1),
            datetime.datetime(2022, 11, 10, 19, 10, 45),
        ],
        "end_time": [
            datetime.datetime(2022, 11, 8, 23, 55, 3),
            datetime.datetime(2022, 11, 10, 22, 5, 52),
            datetime.datetime(2022, 11, 11, 23, 42, 00),
        ],
    }

    # Convert dictionaries to pandas dataframes

    cm_flight_days = pd.DataFrame.from_dict(cm_flight_days)
    ghg_flight_days = pd.DataFrame.from_dict(ghg_flight_days)
    kairos_flight_days = pd.DataFrame.from_dict(kairos_flight_days)
    mair_flight_days = pd.DataFrame.from_dict(mair_flight_days)
    sciav_flight_days = pd.DataFrame.from_dict(sciav_flight_days)

    cm_flight_days.to_csv(pathlib.PurePath('03_results', 'flight_days', f'cm_flight_days.csv'))
    ghg_flight_days.to_csv(pathlib.PurePath('03_results', 'flight_days', f'ghg_flight_days.csv'))
    kairos_flight_days.to_csv(pathlib.PurePath('03_results', 'flight_days', f'kairos_flight_days.csv'))
    mair_flight_days.to_csv(pathlib.PurePath('03_results', 'flight_days', f'mair_flight_days.csv'))
    sciav_flight_days.to_csv(pathlib.PurePath('03_results', 'flight_days', f'sciav_flight_days.csv'))

    return cm_flight_days, ghg_flight_days, kairos_flight_days, mair_flight_days, sciav_flight_days


# %% Generate daily releases

def generate_daily_releases(operator_flight_days):
    """Function to generator a dictionary. Key is flight days in mm_dd format, value is a dataframe with relevant
    metered release rate during periods each airplane was flying"""

    # Initialize dictionary
    operator_releases = {}

    dates = operator_flight_days.date

    for i in range(len(dates)):
        day = dates[i]
        date_meter = load_daily_meter_data(day)

        # Select start and end time on the day in question
        start_t = operator_flight_days.start_time[i]
        end_t = operator_flight_days.end_time[i]

        test_period = date_meter[(date_meter.datetime_utc > start_t) & (date_meter.datetime_utc <= end_t)]
        operator_releases[day] = test_period

    return operator_releases


# %% Plot release days

def plot_daily_releases(operator, flight_days, operator_releases, stage):
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
        operator_stage_overpasses = load_overpass_summary(operator, stage)
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
