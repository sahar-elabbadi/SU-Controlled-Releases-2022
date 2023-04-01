# Script methods for analysis
# Author: Sahar H. El Abbadi
# Date Created: 2023-03-03
# Date Last Modified: 2023-03-03

# Any general function used throughout in any of the notebooks. However, the following are explicitly NOT included:  
# > Any file that specifically outputs a figure will be saved in plot_methods
# > Methods for specifically cleaning the operator reports 

# List of methods in this file:
# > summarize_qc

# Imports
import pathlib
import pandas as pd
import numpy as np
import datetime
import math


# %% Abbreviate operator name
def abbreviate_op_name(operator):
    """Abbreviate operator name for saving files. Use because input to my functions will often be the operator name spelled out in full for plotting purposes,
    while for saving I want the abbreviated name. Input names and corresponding abbreviations are:
     - 'Carbon Mapper': 'cm'
     - 'GHGSat': 'ghg'
     - 'Kairos': 'kairos'
     - 'Kairos LS23': 'kairos_ls23'
     - 'Kairos LS25': 'kairos_ls25'
     - 'Methane Air': 'mair'
    """
    if operator == 'Carbon Mapper' or operator == 'CarbonMapper':
        op_abb = 'cm'
    elif operator == "GHGSat":
        op_abb = 'ghg'
    elif operator == 'Kairos':
        op_abb = 'kairos'
    elif operator == 'Kairos LS23':
        op_abb = 'kairos_ls23'
    elif operator == 'Kairos LS25':
        op_abb = 'kairos_ls25'
    elif operator == 'Methane Air' or operator == 'MethaneAir':
        op_abb = 'mair'
    elif operator == 'Scientific Aviation':
        op_abb = 'sciav'
    else:
        print('Typo in operator name')
        return

    return op_abb


# %% method for loading Philippines summary files (saved in 02_meter_data > summary_files)

def load_summary_files():
    # Carbon Mapper meter data
    cm_path = pathlib.PurePath('02_meter_data', 'summary_files', 'CM.csv')
    cm_meter_raw = pd.read_csv(cm_path)

    # GHGSat meter data
    ghg_path = pathlib.PurePath('02_meter_data', 'summary_files', 'GHGSat.csv')
    ghg_meter_raw = pd.read_csv(ghg_path)

    # Kairos meter data
    kairos_path = pathlib.PurePath('02_meter_data', 'summary_files', 'Kairos.csv')
    kairos_meter_raw = pd.read_csv(kairos_path)

    # MAIR meter data
    mair_path = pathlib.PurePath('02_meter_data', 'summary_files', 'MAIR.csv')
    mair_meter_raw = pd.read_csv(mair_path)

    # Scientific Aviation meter data
    sciav_path = pathlib.PurePath('02_meter_data', 'summary_files', 'SciAv.csv')

    date_columns = ['start_using_sciav', 'end_using_sciav', 'start_using_flightradar', 'end_using_flightradar',
                    'start_using_release_times', 'end_using_release_times']

    sciav_meter_raw = pd.read_csv(sciav_path, index_col=0, parse_dates=date_columns)

    return cm_meter_raw, ghg_meter_raw, kairos_meter_raw, mair_meter_raw, sciav_meter_raw


# %% method summarize_qc
def generate_overpass_summary(operator, stage, operator_report, operator_meter, strict_discard=False,
                              gas_comp_source='km', time_ave=60):
    """Generate clean dataframe for each overpass with columns indicating QC status.

    Inputs:
      - operator: operator name, spelled out in full
      - stage: stage of data: either 1, 2, or 3 (only for identification in saved CSV file, as operator_report is stage specific)
      - operator_report: cleaned operator report to be merged with metered data
      - operator_meter: cleaned operator_meter database, loaded from 02_meter_data > oeprator_meter_data
      - strict_discard: True if using strict discard, False if using lax discard
      - gas_comp_source: either 'su_raw', 'su_normalized', or 'km'. su_raw refers to raw gas composition analysis as
      reported by Eurofins Air Toxins lab, su_normalized refers to using these values normalized such that all values
      reported add to 100%. km refers to using Kinder Morgen gas analysis
      - time_ave: is either '30', '60', or '90', select based on the time average period desired for kgh value

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
    combined_df['release_rate_kgh'] = combined_df[
        f'kgh_ch4_{time_ave}_mean_{gas_comp_source}']
    # combined_df['release_kgh_stdev'] = combined_df[f'kgh_ch4_{time_ave}_std_{gas_comp_source}'] # remove (this is sigma of the meter variability)

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
    operator_qc['altitude_feet'] = combined_df.altitude_feet

    # check if overpass failed both stanford and operator
    check_fail = operator_qc['operator_kept'] + operator_qc['stanford_kept']
    operator_qc['fail_all_qc'] = check_fail == 0

    # Include meter and operator results
    operator_qc['release_rate_kgh'] = combined_df.release_rate_kgh
    operator_qc['kgh_ch4_sigma'] = combined_df.kgh_ch4_sigma
    operator_qc['operator_detected'] = combined_df.Detected
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
        operator_qc.to_csv(pathlib.PurePath('03_results', 'overpass_summary',
                                            f'{op_ab}_{stage}_{time_ave}s_{gas_comp_source}_overpasses_strict.csv'))
    else:
        operator_qc.to_csv(pathlib.PurePath('03_results', 'overpass_summary',
                                            f'{op_ab}_{stage}_{time_ave}s_{gas_comp_source}_overpasses.csv'))

    return operator_qc


# %%
def load_overpass_summary(operator, stage, strict_discard=False, time_ave=60, gas_comp_source='km'):
    """Load overpass summary as a dataframe. Stage is a number (1, 2, 3). Input of "True" for strict_discard sets
    Stanford QC to use strict QC criteria, typical input is False. Operator names can be:

      - 'Carbon Mapper'
      - 'GHGSat'
      - 'Kairos LS23'
      - 'Kairos LS25'
      - 'Methane Air'
      - 'Scientific Aviation'

      """

    op_ab = abbreviate_op_name(operator)

    if operator == 'Scientific Aviation':
        path = pathlib.PurePath('03_results', 'overpass_summary', 'sciav_1_overpasses.csv')
    else:
        if strict_discard is True:
            path = pathlib.PurePath('03_results', 'overpass_summary',
                                    f'{op_ab}_{stage}_{time_ave}s_{gas_comp_source}_overpasses_strict.csv')
        else:
            path = pathlib.PurePath('03_results', 'overpass_summary',
                                    f'{op_ab}_{stage}_{time_ave}s_{gas_comp_source}_overpasses.csv')

    overpass_summary = pd.read_csv(path, index_col=0, parse_dates=['overpass_datetime'])

    return overpass_summary


def summarize_qc(operator, stage, strict_discard=False, time_ave=60, gas_comp_source='km'):
    """Summarize QC criteria applied by operator and Stanford"""

    op_ab = abbreviate_op_name(operator)

    # Generate dataframe with all relevant QC information
    operator_qc = load_overpass_summary(operator, stage, strict_discard, time_ave, gas_comp_source)

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


# %%

def make_qc_table(strict_discard=False, time_ave=60, gas_comp_source='km'):
    """Make a summary table all QC results. Input if strict_discard should be True or False."""

    cm_1_qc = summarize_qc(operator="Carbon Mapper", stage=1, strict_discard=strict_discard, time_ave=time_ave,
                           gas_comp_source=gas_comp_source)
    cm_2_qc = summarize_qc(operator="Carbon Mapper", stage=2, strict_discard=strict_discard, time_ave=time_ave,
                           gas_comp_source=gas_comp_source)
    cm_3_qc = summarize_qc(operator="Carbon Mapper", stage=3, strict_discard=strict_discard, time_ave=time_ave,
                           gas_comp_source=gas_comp_source)

    # GHGSat QC
    ghg_1_qc = summarize_qc(operator="GHGSat", stage=1, strict_discard=strict_discard, time_ave=time_ave,
                            gas_comp_source=gas_comp_source)
    ghg_2_qc = summarize_qc(operator="GHGSat", stage=2, strict_discard=strict_discard, time_ave=time_ave,
                            gas_comp_source=gas_comp_source)
    ghg_3_qc = summarize_qc(operator="GHGSat", stage=3, strict_discard=strict_discard, time_ave=time_ave,
                            gas_comp_source=gas_comp_source)

    # Kairos
    kairos_1_qc = summarize_qc(operator="Kairos", stage=1, strict_discard=strict_discard, time_ave=time_ave,
                               gas_comp_source=gas_comp_source)
    kairos_2_qc = summarize_qc(operator="Kairos", stage=2, strict_discard=strict_discard, time_ave=time_ave,
                               gas_comp_source=gas_comp_source)
    kairos_3_qc = summarize_qc(operator="Kairos", stage=3, strict_discard=strict_discard, time_ave=time_ave,
                               gas_comp_source=gas_comp_source)

    # Kairos LS23
    kairos_ls23_1_qc = summarize_qc(operator="Kairos LS23", stage=1, strict_discard=strict_discard, time_ave=time_ave,
                                    gas_comp_source=gas_comp_source)
    kairos_ls23_2_qc = summarize_qc(operator="Kairos LS23", stage=2, strict_discard=strict_discard, time_ave=time_ave,
                                    gas_comp_source=gas_comp_source)
    kairos_ls23_3_qc = summarize_qc(operator="Kairos LS23", stage=3, strict_discard=strict_discard, time_ave=time_ave,
                                    gas_comp_source=gas_comp_source)

    # Kairos LS25
    kairos_ls25_1_qc = summarize_qc(operator="Kairos LS25", stage=1, strict_discard=strict_discard, time_ave=time_ave,
                                    gas_comp_source=gas_comp_source)
    kairos_ls25_2_qc = summarize_qc(operator="Kairos LS25", stage=2, strict_discard=strict_discard, time_ave=time_ave,
                                    gas_comp_source=gas_comp_source)
    kairos_ls25_3_qc = summarize_qc(operator="Kairos LS25", stage=3, strict_discard=strict_discard, time_ave=time_ave,
                                    gas_comp_source=gas_comp_source)

    # Scientific Aviation
    sciav_1_qc = summarize_qc(operator="Scientific Aviation", stage=1, strict_discard=strict_discard, time_ave=time_ave,
                              gas_comp_source=gas_comp_source)

    # Methane Air
    mair_1_qc = summarize_qc(operator="Methane Air", stage=1, strict_discard=strict_discard, time_ave=time_ave,
                             gas_comp_source=gas_comp_source)

    # Combine all individual QC dataframes

    all_qc = [cm_1_qc, cm_2_qc, cm_3_qc, ghg_1_qc, ghg_2_qc, ghg_3_qc, kairos_1_qc, kairos_2_qc, kairos_3_qc,
              kairos_ls23_1_qc, kairos_ls23_2_qc, kairos_ls23_3_qc, kairos_ls25_1_qc, kairos_ls25_2_qc,
              kairos_ls25_3_qc, sciav_1_qc, mair_1_qc]

    all_qc = pd.concat(all_qc)

    if strict_discard:
        save_name = 'all_qc_strict.csv'
    else:
        save_name = 'all_qc.csv'
    all_qc.to_csv(pathlib.PurePath('03_results', 'qc_comparison', save_name))
    return all_qc


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


# %% Load flight days

def load_flight_days(operator='all'):
    cm_flight_days = {
        "date": ['10_10', '10_11', '10_12', '10_28', '10_29', '10_31'],
        "start_time": [
            datetime.datetime(2022, 10, 10, 16, 59, 00),
            datetime.datetime(2022, 10, 11, 17, 14, 13),
            datetime.datetime(2022, 10, 12, 17, 13, 16),
            datetime.datetime(2022, 10, 28, 17, 49, 5),
            datetime.datetime(2022, 10, 29, 17, 11, 30),
            datetime.datetime(2022, 10, 31, 17, 14, 46),
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
            datetime.datetime(2022, 10, 24, 16, 44, 28),
            datetime.datetime(2022, 10, 25, 16, 34, 27),
            datetime.datetime(2022, 10, 26, 16, 36, 47),
            datetime.datetime(2022, 10, 27, 16, 35, 23),
            datetime.datetime(2022, 10, 28, 16, 39, 12),
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
            datetime.datetime(2022, 10, 25, 16, 55, 47),
            datetime.datetime(2022, 10, 29, 16, 23, 1),
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
            datetime.datetime(2022, 10, 31, 17, 1, 58),
            datetime.datetime(2022, 11, 2, 16, 36, 30),
            datetime.datetime(2022, 11, 4, 16, 41, 19),
            datetime.datetime(2022, 11, 7, 19, 21, 44),
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
            datetime.datetime(2022, 11, 8, 21, 35, 0),
            datetime.datetime(2022, 11, 10, 18, 0, 0),
            datetime.datetime(2022, 11, 11, 19, 0, 0),
        ],
        "end_time": [
            datetime.datetime(2022, 11, 8, 23, 55, 3),
            datetime.datetime(2022, 11, 10, 22, 1, 22),
            datetime.datetime(2022, 11, 11, 23, 39, 33),
        ],
    }

    # TODO add "all_days" to generate meter data for all days using start and end times for each day

    # all_days = {
    #     "date": ['10_10', '10_11', '10_12', '10_13', '10_14', '10_17', '10_18', '10_19', '10_24', '10_25',
    #              '10_26', '10_27', '10_28', '10_29', '10_30', '10_31', '11_01', '11_02', '11_03', '11_04',
    #              '11_07', '11_08', '11_10', '11_11', '11_14', '11_15', '11_16', '11_17', '11_18', '11_21',
    #              '11_22', '11_23', '11_28', '11_29', '11_30']
    #     "start_time": [
    #         datetime.datetime(2022, 10, 10, 16, 59, 00),
    #         datetime.datetime(2022, 10, 11, 17, 14, 13),
    #         datetime.datetime(2022, 10, 12, 17, 13, 16),
    #         datetime.datetime(2022, 10, 13, 20, 45, 00),
    #         datetime.datetime(2022, 10, 14, 16, 22, 30),
    #
    #     ],
    #     "end_time": [
    #         datetime.datetime(2022, 10, 10, 21, 30, 14),
    #         datetime.datetime(2022, 10, 11, 21, 16, 36),
    #         datetime.datetime(2022, 10, 12, 21, 15, 23),
    #         datetime.datetime(2022, 10, 13, 21, 42, 35),
    #         datetime.datetime(2022, 10, 14, 16, 50, 13),
    #     ]
    # }

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

    # Return flight day for the operator
    if operator == 'all':
        return
    else:
        op_ab = abbreviate_op_name(operator)
        if op_ab == 'cm':
            return cm_flight_days
        elif op_ab == 'ghg':
            return ghg_flight_days
        elif op_ab == 'kairos':
            return kairos_flight_days
        elif op_ab == 'mair':
            return mair_flight_days
        elif op_ab == 'sciav':
            return sciav_flight_days
        else:
            print('Typo in operator name ')
            return


def load_operator_flight_days(operator):
    """Load CSV file with operator flight days"""
    op_ab = abbreviate_op_name(operator)
    flight_days = pd.read_csv(pathlib.PurePath('03_results', 'flight_days', f'{op_ab}_flight_days.csv'),
                              index_col=0, parse_dates=['start_time', 'end_time'])
    return flight_days


# %%
def calc_meter_uncertainty(meter, flow_kgh):
    """ Input meter ('bc', 'mc', 'pc') and flow rate to determine the associated uncertainty.
    Uncertainty is a percentage of total gas flow rate, as reported by Emerson. """
    # cutoff dictionary: for all meters, flow rates above a certain cutoff have uncertainty of 0.25% of flow rate
    cutoff = {
        'bc': 4.84,
        'mc': 32.6,
        'pc': 350,
    }

    # Data from Emerson Sizing tool for each meter (see files downloaded from Emerson website in meter_sizing_graphs folder

    # baby corey
    bc_flow_kgh = [4.84, 4.26, 4.12, 3.68, 3.14, 3.1, 2.52, 2.16, 1.94, 1.36, 1.18, 0.78, 0.2, 0.2]
    bc_uncertainty = [0.25, 0.2658, 0.2748, 0.3077, 0.3606, 0.3652, 0.4493, 0.5242, 0.5836, 0.8325, 0.9595, 1.4515,
                      5.6608, 5.6608]

    # mama corey
    mc_flow_kgh = [32.6, 28.9, 25.2, 21.5, 20, 17.8, 14.1, 10.4, 6.7, 3]
    mc_uncertainty = [0.25, 0.2686, 0.308, 0.361, 0.3881, 0.4361, 0.5505, 0.7463, 1.1585, 2.5873]

    # papa corey
    pc_flow_kgh = [350, 318, 300, 286, 254, 222, 190, 158, 126, 94, 62, 30]
    pc_uncertainty = [0.25, 0.2562, 0.2715, 0.2848, 0.3207, 0.3669, 0.4287, 0.5155, 0.6465, 0.8666, 1.3138, 2.7152]

    # Determine best fit for the flow regime below the cutoff:
    bc_power = power_fit(bc_flow_kgh, bc_uncertainty)
    mc_power = power_fit(mc_flow_kgh, mc_uncertainty)
    pc_power = power_fit(pc_flow_kgh, pc_uncertainty)

    meter_best_fit_power = {
        'bc': bc_power,
        'mc': mc_power,
        'pc': pc_power,
    }

    if (flow_kgh == 0) or (meter == 'None'):
        uncertainty_percent = 0  # If release is a zero, meter entry in meter column is 'None', set uncertainty to 0
    else:
        if flow_kgh >= cutoff[meter]:
            uncertainty_percent = 0.25
        else:
            a = meter_best_fit_power[meter]['a']
            b = meter_best_fit_power[meter]['b']
            uncertainty_percent = a * (flow_kgh ** b)

    # meter_sigma = uncertainty_percent/1.96

    return uncertainty_percent


def sum_of_quadrature(*uncertainties):
    """Combine uncertainties using sum of quadrature method"""
    summation = 0  # initialize summation
    for uncertainty in uncertainties:
        summation += uncertainty ** 2

    sigma_final = math.sqrt(summation)
    return sigma_final


# %% Generate daily releases

def generate_daily_releases(gas_comp_sources=['km']):
    """Generates daily metered data for all airplane flight days"""
    operators = ['Carbon Mapper', 'GHGSat', 'Kairos', 'Methane Air', 'Scientific Aviation']

    for operator in operators:
        # Load operator flight days
        operator_flight_days = load_operator_flight_days(operator)
        dates = operator_flight_days.date

        for i in range(len(dates)):
            day = dates[i]
            date_meter = load_daily_meter_data(day)

            print(f'Operator: {operator}')
            print(f'Flight Day: {day}')

            # Select start and end time on the day in question
            start_t = operator_flight_days.start_time[i]
            end_t = operator_flight_days.end_time[i]

            # test_period = date_meter[(date_meter.datetime_utc > start_t) & (date_meter.datetime_utc <= end_t)]
            date_meter = date_meter.query('datetime_utc > @start_t & datetime_utc <= @end_t').copy()

            # gas_comp_sources = ['km', 'su_raw', 'su_normalized']
            for source in gas_comp_sources:

                # Determine uncertainty associated with meter reading
                meter_uncertainty = date_meter.apply(lambda x: calc_meter_uncertainty(x['meter'], x['flow_rate']),
                                                                                      axis=1)
                date_meter['meter_sigma_kgh_gas'] = meter_uncertainty / 100 / 1.96 * date_meter['flow_rate']

                print(f'Assigning {source} methane mole fraction for {operator} on {day}')

                # Determine gas composition mean value and sigma for each datetime
                gas_comp_values = date_meter.apply(lambda x: select_methane_fraction(x['datetime_utc'], source),
                                                   axis=1, result_type='expand')
                date_meter[f'methane_fraction_{source}'] = gas_comp_values[0]
                date_meter[f'gas_comp_sigma_{source}'] = gas_comp_values[1]

                # Calculate kgh CH4 using the methane mole fraction selected
                date_meter[f'kgh_ch4_{source}'] = date_meter['flow_rate'] * date_meter[f'methane_fraction_{source}']

                combined_uncertainty = []
                upper_bound_95 = []
                lower_bound_95 = []

                for index, row in date_meter.iterrows():
                    if row[f'kgh_ch4_{source}'] == 0:
                        combined_sigma_kgh_ch4 = 0
                    else:
                        # Calculate the combined error using sum of quadrature on relative uncertainty
                        # Meter reading relative uncertainty:
                        meter_relative_sigma = row['meter_sigma_kgh_gas'] / row['flow_rate']
                        gas_comp_relative_sigma = row[f'gas_comp_sigma_{source}'] / row[f'methane_fraction_{source}']
                        combined_sigma_kgh_ch4 = row[f'kgh_ch4_{source}'] * sum_of_quadrature(meter_relative_sigma,
                                                                                              gas_comp_relative_sigma)
                    upper_bound = row[f'kgh_ch4_{source}'] + (1.96 * combined_sigma_kgh_ch4)
                    lower_bound = row[f'kgh_ch4_{source}'] - (1.96 * combined_sigma_kgh_ch4)

                    combined_uncertainty.append(combined_sigma_kgh_ch4)
                    upper_bound_95.append(upper_bound)
                    lower_bound_95.append(lower_bound)

                date_meter[f'combined_uncertainty_kgh_ch4_{source}'] = combined_uncertainty
                date_meter[f'CI95_upper_{source}'] = upper_bound_95
                date_meter[f'CI95_lower_{source}'] = lower_bound_95

            # Save the test_period dataframe
            op_ab = abbreviate_op_name(operator)
            save_path = pathlib.PurePath('03_results', 'daily_releases', f'{op_ab}_{day}.csv')
            date_meter.to_csv(save_path)

    return


def load_daily_releases(operator):
    """Load the pre-saved CSV file containing the daily metered release rates for a given operator"""
    # Initialize dictionary
    operator_releases = {}

    op_ab = abbreviate_op_name(operator)
    operator_flight_days = load_operator_flight_days(operator)
    dates = operator_flight_days.date

    for i in range(len(dates)):
        day = dates[i]
        daily_release = pd.read_csv(pathlib.PurePath('03_results', 'daily_releases', f'{op_ab}_{day}.csv'),
                                    index_col=0, parse_dates=['datetime_utc'])
        operator_releases[day] = daily_release

    return operator_releases


# %% Load clean data

def load_clean_operator_reports():
    # Carbon Mapper Stage 1
    cm_path_1 = pathlib.PurePath('01_clean_reports', 'cm_1_clean.csv')
    cm_1 = pd.read_csv(cm_path_1, index_col=0)

    # Carbon Mapper Stage 2
    cm_path_2 = pathlib.PurePath('01_clean_reports', 'cm_2_clean.csv')
    cm_2 = pd.read_csv(cm_path_2, index_col=0)

    # Carbon Mapper Stage 3
    cm_path_3 = pathlib.PurePath('01_clean_reports', 'cm_3_clean.csv')
    cm_3 = pd.read_csv(cm_path_3, index_col=0)

    # GHGSat Stage 1
    ghg_path_1 = pathlib.PurePath('01_clean_reports', 'ghg_1_clean.csv')
    ghg_1 = pd.read_csv(ghg_path_1, index_col=0)

    # GHGSat Stage 2
    ghg_path_2 = pathlib.PurePath('01_clean_reports', 'ghg_2_clean.csv')
    ghg_2 = pd.read_csv(ghg_path_2, index_col=0)

    # GHGSat Stage 3 (same operator report as Phase II)
    ghg_path_2 = pathlib.PurePath('01_clean_reports', 'ghg_2_clean.csv')
    ghg_3 = pd.read_csv(ghg_path_2, index_col=0)

    # Kairos Stage 1 combo
    kairos_path_1 = pathlib.PurePath('01_clean_reports', 'kairos_1_clean.csv')
    kairos_1 = pd.read_csv(kairos_path_1, index_col=0)

    # Kairos Stage 2 combo
    kairos_path_2 = pathlib.PurePath('01_clean_reports', 'kairos_2_clean.csv')
    kairos_2 = pd.read_csv(kairos_path_2, index_col=0)

    # Kairos Stage 3 combo
    kairos_path_3 = pathlib.PurePath('01_clean_reports', 'kairos_3_clean.csv')
    kairos_3 = pd.read_csv(kairos_path_3, index_col=0)

    # Kairos Stage 1 Pod LS23
    kairos_path_1_ls23 = pathlib.PurePath('01_clean_reports', 'kairos_1_ls23_clean.csv')
    kairos_1_ls23 = pd.read_csv(kairos_path_1_ls23, index_col=0)

    # Kairos Stage 1 Pod LS25
    kairos_path_1_ls25 = pathlib.PurePath('01_clean_reports', 'kairos_1_ls25_clean.csv')
    kairos_1_ls25 = pd.read_csv(kairos_path_1_ls25, index_col=0)

    # Kairos Stage 2 Pod LS23
    kairos_path_2_ls23 = pathlib.PurePath('01_clean_reports', 'kairos_2_ls23_clean.csv')
    kairos_2_ls23 = pd.read_csv(kairos_path_2_ls23, index_col=0)

    # Kairos Stage 2 Pod LS25
    kairos_path_2_ls25 = pathlib.PurePath('01_clean_reports', 'kairos_2_ls25_clean.csv')
    kairos_2_ls25 = pd.read_csv(kairos_path_2_ls25, index_col=0)

    # Kairos Stage 3 Pod LS23
    kairos_path_3_ls23 = pathlib.PurePath('01_clean_reports', 'kairos_3_ls23_clean.csv')
    kairos_3_ls23 = pd.read_csv(kairos_path_3_ls23, index_col=0)

    # Kairos Stage 3 Pod LS25
    kairos_path_3_ls25 = pathlib.PurePath('01_clean_reports', 'kairos_3_ls25_clean.csv')
    kairos_3_ls25 = pd.read_csv(kairos_path_3_ls25, index_col=0)

    # Scientific Aviation Stage 1
    sciav_clean_path = pathlib.PurePath('01_clean_reports', 'sciav_1_clean.csv')
    sciav_1 = pd.read_csv(sciav_clean_path)  # do not set index_col to 0, there is none

    # MAIR Stage 1
    mair_clean_path = pathlib.PurePath('01_clean_reports', 'mair_1_clean.csv')
    mair_1 = pd.read_csv(mair_clean_path)  # do not set index_col to 0, there is none

    return cm_1, cm_2, cm_3, ghg_1, ghg_2, ghg_3, kairos_1, kairos_2, kairos_3, kairos_1_ls23, kairos_1_ls25, kairos_2_ls23, kairos_2_ls25, \
        kairos_3_ls23, kairos_3_ls25, sciav_1, mair_1


# %% Load meter data

def load_meter_data(timekeeper='flightradar', gas_comp_source='km', time_ave=60):
    """Input timekeeper. Must be string, all lower case: flightradar, operator, stanford"""
    operators = ['Carbon Mapper', 'GHGSat', 'Kairos', 'Methane Air']
    meter_files = {}
    for operator in operators:
        op_ab = abbreviate_op_name(operator)
        op_path = pathlib.PurePath('02_meter_data', 'operator_meter_data', f'{timekeeper}_timestamp',
                                   f'{op_ab}_{time_ave}s_{gas_comp_source}_meter.csv')
        op_meter = pd.read_csv(op_path, index_col=0, parse_dates=['datetime_utc'])
        meter_files[f'{op_ab}_meter'] = op_meter

    return meter_files


# %% Function: select_valid_overpasses

# Inputs:
# > operator_report: cleaned operator data, loaded from folder 01_clean_reports
# > operator_meter: cleaned metering data, loaded from folder 02_meter_data

def select_stanford_valid_overpasses(operator_report, operator_meter, strict_discard=False):
    """Merge operator report and operator meter dataframes and select overpasses which pass Stanford QC criteria.
    strict_discard is True or False

    Old notes: Operator report dataframe should already have 'nan' values for quantification estimates that do not meet operator
    QC criteria. Returns: y_index, x_data, y_data"""
    # Merge the two data frames
    operator_df = operator_report.merge(operator_meter, on='overpass_id')

    if strict_discard is True:
        discard_column = 'qc_su_discard_strict'
    else:
        discard_column = 'qc_su_discard'

    # Filter based on overpasses that meet Stanford's QC criteria
    operator_df = operator_df[(operator_df[discard_column] == 0)]

    return operator_df


# %% Function to convert 24 hour datetime in AZ local time to 24 hour time in UTC
def convert_utc(dt, delta_t):
    """Convert input time (dt in datetime format) to UTC. detla_t is the time difference between local time and UTC
    time. For Arizona, this value is + 7 hours"""
    local_hr = int(dt.strftime('%H'))
    local_min = int(dt.strftime('%M'))
    local_sec = int(dt.strftime('%S'))

    utc_hr = local_hr + delta_t
    utc_time = datetime.time(utc_hr, local_min, local_sec)
    return utc_time


# %% Function to convert 12 hour time sting into 24 hour datetime
def convert_to_twentyfour(time):
    """Convert a string of format HH:MM AM/PM to 24 hour time. Used for converting GHGSat's reported timestamps
    from 12 hour time to 24 hour time. Output is of class datetime"""
    time_12hr = str(datetime.datetime.strptime(time, '%I:%M:%S %p'))
    time_24hr = time_12hr[-8:]
    time_24hr = datetime.datetime.strptime(time_24hr, '%H:%M:%S').time()
    return time_24hr


# %% Function to combine date string and time string into one datetime object

# date format: YYYY-MM-DD
# time format: HH:MM:DD (UTC)

def combine_datetime(test_date, test_time):
    """Combine two strings for date and time into a datetime object.
    Formatting for inputs:
      - test_date: YYYY-MM-DD
      - test_time: HH:MM:DD

    Use for combing separate columns of date and time in operator report or meter data
     """

    my_time = datetime.datetime.strptime(test_time, '%H:%M:%S').time()

    for fmt in ('%Y-%m-%d', '%m/%d/%y'):
        try:
            my_date = datetime.datetime.strptime(test_date, fmt)
        except ValueError:
            pass

    my_datetime = datetime.datetime.combine(my_date, my_time)
    return my_datetime


# %%

def check_timekeep_capitalization(timekeeper):
    """Check if timekeeper is recorded using all lower-case letters, if so return the corrected value"""
    if timekeeper == 'flightradar':
        timekeeper = 'Flightradar'
    elif timekeeper == 'stanford':
        timekeeper = 'Stanford'
    elif timekeeper == 'operator':
        timekeeper = 'team'

    return timekeeper


# %%
def clean_meter_column_names(operator, operator_meter_raw, overpass_id, timekeeper='flightradar'):
    timekeeper = check_timekeep_capitalization(timekeeper)

    # Relevant columns from Philippine generated meter dataset:
    date = 'Date'
    time = f'Time (UTC) - from {timekeeper}'

    phase_iii = 'PhaseIII'  # 0 indicates the overpass was not provided to team in Phase III, 1 indicates it was
    meter = 'Meter'  # note renaming meter variable used above
    qc_discard = f'Discarded - using {timekeeper}'
    qc_discard_strict = f'Discarded - 1% - using {timekeeper}'
    altitude = 'Average altitude last minute (m)'  # TODO update if Philippine changes this column title for adjusting units

    operator_meter = pd.DataFrame()
    # Populate relevant dataframes

    operator_meter['overpass_id'] = operator_meter_raw[overpass_id]
    operator_meter['phase_iii'] = operator_meter_raw[phase_iii]
    operator_meter['meter'] = operator_meter_raw[meter]
    operator_meter['qc_su_discard'] = operator_meter_raw[qc_discard]
    operator_meter['qc_su_discard_strict'] = operator_meter_raw[qc_discard_strict]
    operator_meter['time'] = operator_meter_raw[time]
    operator_meter['date'] = operator_meter_raw[date]

    # Altitude for Carbon Mapper, GHGSat and Methane Air was obtained using FlightRadar, and is in feet.
    # Flight Radar reports altitude above mean sea level > need to adjust for the elevation of field site
    #
    # Kairos reported flight altitude NOT using Flight Radar, and their units are reported in meters
    #
    # If operator is Kairos, Kairos LS23 or Kairos LS25, convert from meters to feet:
    feet_per_meter = 3.28084
    field_elevation_m = 429
    field_elevation_ft = field_elevation_m * feet_per_meter

    kairos_names = ['Kairos', 'Kairos LS23', 'Kairos LS25', 'kairos']
    if operator in kairos_names:
        operator_meter['altitude_feet'] = (operator_meter_raw[altitude] - field_elevation_m) * feet_per_meter
    else:
        operator_meter['altitude_feet'] = (operator_meter_raw[altitude] - field_elevation_ft)

    # Abbreviate meter names in raw meter file
    names = ['Baby Coriolis', 'Mama Coriolis', 'Papa Coriolis']
    nicknames = ['bc', 'mc', 'pc']
    for meter_name, meter_nickname in zip(names, nicknames):
        operator_meter.loc[operator_meter['meter'] == meter_name, 'meter'] = meter_nickname

    return operator_meter


# %%
def select_methane_fraction(input_datetime, gas_comp_source='km'):
    """Determines the methane mole fraction for a given datetime.
    Inputs:
      - input_datetime is a datetime object
      - gas_comp_source is either 'su_raw', 'su_normalized', or 'km' """

    # Load the clean gas cmoposition data
    gas_comp = pd.read_csv(pathlib.PurePath('02_meter_data', 'gas_comp_clean_su_km.csv'),
                           nrows=15,
                           parse_dates=['start_utc', 'end_utc'])

    # Determine methane mole fraction based on date brackets
    for index, row in gas_comp.iterrows():
        if (input_datetime > row['start_utc']) and (input_datetime <= row['end_utc']):
            methane_mole_fraction = row[gas_comp_source]
            gas_comp_sigma = row[f'{gas_comp_source}_sigma']

    return methane_mole_fraction, gas_comp_sigma


# %%

def power_fit(x_data, y_data):
    """Find equation of best fit using form y = ax^b. Inputs are x_data, y_data: two series of equal length.
    Outputs are a and b from y = ax^b"""

    x_log = np.log(x_data)
    y_log = np.log(y_data)
    slope, intercept = np.polyfit(x_log, y_log, deg=1)
    power_b = slope
    power_a = np.exp(intercept)
    power_constants = {'a': power_a, 'b': power_b}
    return power_constants


# %%
def calc_average_gas_flow_all_overpasses(operator, operator_meter, time_ave=60, comp_source='km'):
    """Calculate average gas flow over the input time period (time_ave) for each operator overpass.
    Specify gas composition source. Output is a dataframe with the following columns (using default values):
       - overpass_id
       - methane_fraction_km
       - methane_fraction_km_sigma
       - kgh_gas_60_mean
       - kgh_gas_60_std
       - kgh_ch4_60_mean_km
       - kgh_ch4_60_std_km
       - meter_uncertainty"""

    op_ab = abbreviate_op_name(operator)

    # Calculate time average period as a Timedelta object
    delta_t = pd.Timedelta(seconds=time_ave)  # the period of time over which we want to average
    overpass_gas_data = []

    # Iterate through each row in the operator_meter dataframe
    for index, row in operator_meter.iterrows():
        overpass_datetime = row.datetime_utc
        flight_date_code = overpass_datetime.strftime('%m_%d')
        flight_data = pd.read_csv(pathlib.PurePath('03_results', 'daily_releases', f'{op_ab}_{flight_date_code}.csv'),
                                  parse_dates=['datetime_utc'], index_col=0)
        # Create the time period for averaging
        start_t = overpass_datetime - delta_t
        select_t_mask = (flight_data['datetime_utc'] > start_t) & (flight_data['datetime_utc'] <= overpass_datetime)
        overpass_period = flight_data.loc[select_t_mask].copy()

        # Calculate mean and standard deviation for gas flow rate and ch4 flow rate
        overpass_gas_mean = overpass_period['flow_rate'].mean()
        overpass_gas_std = overpass_period['flow_rate'].std()
        overpass_ch4_mean = overpass_period[f'kgh_ch4_{comp_source}'].mean()
        overpass_ch4_std = overpass_period[f'kgh_ch4_{comp_source}'].std()

        # Calculate the mean methane fraction, on the off chance that the truck changed during the overpass period
        methane_fraction_mean = overpass_period[f'methane_fraction_{comp_source}'].mean()
        methane_faction_sigma = overpass_period[f'gas_comp_sigma_{comp_source}'].mean()

        # Calculate the uncertainty associated with the meter reading
        meter_uncertainty = calc_meter_uncertainty(row.meter, overpass_gas_mean)
        meter_sigma_percent = meter_uncertainty / 1.96  # convert from 95% CI bound to sigma
        meter_sigma_kgh_gas = meter_sigma_percent / 100 * overpass_gas_mean

        ########### Combine Uncertainties ###########

        if overpass_gas_mean == 0:
            sigma_ch4_kgh = 0  # TODO check if this is ok

        else:
            # First, combine the uncertainties associated with meter reading and gas flow variability using
            # sum of quadrature:
            sigma_gas = sum_of_quadrature(overpass_gas_std, meter_sigma_kgh_gas)

            # Next, calculate the uncertainty inclusive of gas composition.
            # Because we are multiplying gas composition by the whole gas flow rate, we use sum of quadrature
            # on the relative uncertainty: sigma / mean

            relative_gas_sigma = sigma_gas / overpass_gas_mean

            # Relative sigma value for gas compositional analysis:
            relative_comp_sigma = methane_faction_sigma / methane_fraction_mean

            # Combine uncertainties and multiply by the methane flow rate to convert from relative uncertainty to sigma value
            sigma_ch4_kgh = sum_of_quadrature(relative_gas_sigma, relative_comp_sigma) * overpass_ch4_mean

        new_row = {
            'overpass_id': row['overpass_id'],
            f'methane_fraction_{comp_source}': methane_fraction_mean,
            f'methane_fraction_{comp_source}_sigma': methane_faction_sigma,
            f'kgh_gas_{time_ave}_mean': overpass_gas_mean,
            f'kgh_gas_{time_ave}_std': overpass_gas_std,
            f'kgh_ch4_{time_ave}_mean_{comp_source}': overpass_ch4_mean,
            f'kgh_ch4_{time_ave}_std_{comp_source}': overpass_ch4_std,
            f'meter_uncertainty': meter_uncertainty,
            f'meter_sigma_percent': meter_sigma_percent,
            f'meter_sigma_kgh_gas': meter_sigma_kgh_gas,
            f'kgh_ch4_sigma': sigma_ch4_kgh
        }

        overpass_gas_data.append(new_row)

    overpass_gas_data = pd.DataFrame(overpass_gas_data)
    return overpass_gas_data


# %% Function to import Philippine's meter data, select the FlightRadar columns, and with abrename columns to be more
# brief and machine-readable
def make_operator_meter_dataset(operator, operator_summary, timekeeper='flightradar', time_ave=60,
                                gas_comp_source='km'):
    """Function to make a clean dataset for each overpass for a given operator. Input is the full name of operator
    and the operator meter file. Also include the desired timekeeper metric for when the oeprator was overhead: this
    can be one of three options:
      - flightradar: used in analysis, timestamp when FlightRadar GPS coordinates are closest to the stack
      - Stanford: Stanford ground team visual observation of when the airplane was overhead
      - team: participating operator's report of when they were over the source """

    #TODO finish making this code for SciAV
    # # This function is not meant to clean Scientific Aviation data
    if operator == "Scientific Aviation":
        pass
    #     overpass_id = 'overpass_id'
    #     start_t = operator_summary['start_using_sciav']
    #     end_t = operator_summary['end_using_sciav']
    #     operator_meter = operator_summary

    else:
        timekeeper = check_timekeep_capitalization(timekeeper)
        overpass_id = 'PerformerOverpassID'

        # Load summary file that aligns timestamps
        operator_meter = clean_meter_column_names(operator, operator_summary, overpass_id, timekeeper)

        # Drop rows with nan to remove rows with missing values. This is because for some operators, we missed overpasses
        # and timestamps have nan values, which causes issues with downstream code
        operator_meter = operator_meter.dropna(axis='index')  # axis = 'index' means to drop rows with missing values

        # Combine date and time
        overpass_datetime = operator_meter.apply(lambda x: combine_datetime(x['date'], x['time']), axis=1)
        operator_meter.insert(loc=0, column='datetime_utc', value=overpass_datetime)
        # Now that we have removed NA values in time, we can remove date and time columns from operator_meter
        operator_meter = operator_meter.drop(columns=['time', 'date'])

        # Calculate whole gas average for datetime
        gas_flow_rates = calc_average_gas_flow_all_overpasses(operator, operator_meter, time_ave, gas_comp_source)
        operator_meter = operator_meter.merge(gas_flow_rates, on='overpass_id')

        op_ab = abbreviate_op_name(operator)

        # Set save folder
        if timekeeper == 'Flightradar':
            save_folder = 'flightradar_timestamp'
        elif timekeeper == 'Stanford':
            save_folder = 'stanford_timestamp'
        elif timekeeper == 'team':
            save_folder = 'operator_timestamp'

        # Save CSV file
        operator_meter.to_csv(pathlib.PurePath('02_meter_data', 'operator_meter_data',
                                               save_folder, f'{op_ab}_{time_ave}s_{gas_comp_source}_meter.csv'))

    return operator_meter


# %%
def classify_confusion_categories(overpass_summary):
    """Takes input that is an overpass summary dataframe and outputs counts of true positive, false positive, true negative, false negative"""

    true_positives = overpass_summary.query('non_zero_release == True & operator_detected == True')
    false_positives = overpass_summary.query('non_zero_release == False & operator_detected == True')
    false_negatives = overpass_summary.query('non_zero_release == True & operator_detected == False')
    true_negatives = overpass_summary.query('non_zero_release == False & operator_detected == False')

    # Filtered zeros:

    return true_positives, false_positives, true_negatives, false_negatives


def make_histogram_bins(df, threshold_lower, threshold_upper, n_bins):
    bins = np.linspace(threshold_lower, threshold_upper, n_bins + 1)

    # These variables are for keeping track of values as I iterate through the bins in the for loop below:
    bin_count, bin_num_detected = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
    bin_median = np.zeros(n_bins)

    # For each bin, find number of data points and detection probability

    for i in range(n_bins):
        # Set boundary of bin
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_median[i] = (bin_min + bin_max) / 2

        binned_data = df.query('release_rate_kgh < @bin_max & release_rate_kgh >= @bin_min')
        bin_count[i] = len(binned_data)

    detection_prob = pd.DataFrame({
        "bin_median": bin_median,
        "n_data_points": bin_count,
    })

    return detection_prob


def find_missing_data(operator, meter_raw, time_ave=60, gas_comp_source='km'):
    """ Missing data refers to overpasses documented by Stanford that are not reported by the operator"""

    operator_missing_raw = meter_raw.query(
        'PerformerOverpassID.isnull() == True & StanfordOverpassID.isnull() == False')
    operator_missing = clean_meter_column_names(operator, operator_missing_raw, 'FlightradarOverpassID', 'flightradar')

    if operator_missing.empty:
        operator_missing['release_rate_kgh'] = None

    else:
        # Combine date and time
        missing_datetime = operator_missing.apply(lambda x: combine_datetime(x['date'], x['time']), axis=1)
        operator_missing.insert(loc=0, column='datetime_utc', value=missing_datetime)
        # Now that we have removed NA values in time, we can remove date and time columns from operator_meter
        operator_missing = operator_missing.drop(columns=['time', 'date'])

        # Calculate whole gas average for given overpass id
        gas_flow_rates = calc_average_gas_flow_all_overpasses(operator, operator_missing, time_ave, gas_comp_source)
        operator_missing = operator_missing.merge(gas_flow_rates, on='overpass_id')

        # make a column with the desired release_rate_kgh value (so this dataframe can be ready by files that read an overpass summary file)
        operator_missing['release_rate_kgh'] = operator_missing[f'kgh_ch4_{time_ave}_mean_{gas_comp_source}']

    return operator_missing


# %%
def classify_histogram_data(operator, stage, threshold_lower, threshold_upper, n_bins, strict_discard=False,
                            time_ave=60, gas_comp_source='km'):
    # Load operator overpass data
    op_reported = load_overpass_summary(operator=operator, stage=stage, strict_discard=strict_discard,
                                        time_ave=time_ave, gas_comp_source=gas_comp_source)

    # Pass all QC filter
    op_qc_pass = op_reported.query('qc_summary == "pass_all"')

    # Select non-zero releases detected by operator
    tp, fp, tn, fn = classify_confusion_categories(op_qc_pass)

    bin_median = make_histogram_bins(tp, threshold_lower, threshold_upper, n_bins).bin_median
    count_tp = make_histogram_bins(tp, threshold_lower, threshold_upper, n_bins).n_data_points
    count_fp = make_histogram_bins(fp, threshold_lower, threshold_upper, n_bins).n_data_points
    count_fn = make_histogram_bins(fn, threshold_lower, threshold_upper, n_bins).n_data_points
    count_tn = make_histogram_bins(tn, threshold_lower, threshold_upper, n_bins).n_data_points

    # Filtered by Stanford
    # Non-zero SU QC fails
    su_qc_fail = op_reported.query('stanford_kept == False & non_zero_release == True')
    count_su_fail = make_histogram_bins(su_qc_fail, threshold_lower, threshold_upper, n_bins).n_data_points

    # Zero SU QC fails
    zero_su_qc_fail = op_reported.query('stanford_kept == False & non_zero_release == False')
    count_zero_su_fail = make_histogram_bins(zero_su_qc_fail, threshold_lower, threshold_upper, n_bins).n_data_points

    # Filtered by Carbon Mapper
    # if qc_summary is 'fail_operator', this means it passed Stanford QC but not operator QC
    # Non-zero
    op_qc_fail = op_reported.query('qc_summary == "fail_operator" & non_zero_release == True')
    count_op_fail = make_histogram_bins(op_qc_fail, threshold_lower, threshold_upper, n_bins).n_data_points

    # Zero
    zero_op_qc_fail = op_reported.query('qc_summary == "fail_operator" & non_zero_release == False')
    count_zero_op_fail = make_histogram_bins(zero_op_qc_fail, threshold_lower, threshold_upper, n_bins).n_data_points

    # Identify data points where Stanford conducted a release
    # Find data points where we have a flightradar overpass but we do not have an operator overpass

    cm_meter_raw, ghg_meter_raw, kairos_meter_raw, mair_meter_raw, sciav_meter_raw = load_summary_files()

    if operator == 'Carbon Mapper':
        missing = find_missing_data(operator, cm_meter_raw, time_ave=time_ave, gas_comp_source=gas_comp_source)
    elif operator == 'GHGSat':
        missing = find_missing_data(operator, ghg_meter_raw, time_ave=time_ave, gas_comp_source=gas_comp_source)
    elif operator == 'Kairos':
        missing = find_missing_data(operator, kairos_meter_raw, time_ave=time_ave, gas_comp_source=gas_comp_source)
    elif operator == 'Kairos LS23':
        missing = find_missing_data(operator, kairos_meter_raw, time_ave=time_ave, gas_comp_source=gas_comp_source)
    elif operator == 'Kairos LS25':
        missing = find_missing_data(operator, kairos_meter_raw, time_ave=time_ave, gas_comp_source=gas_comp_source)
    elif operator == 'Methane Air':
        missing = find_missing_data(operator, mair_meter_raw, time_ave=time_ave, gas_comp_source=gas_comp_source)
    elif operator == 'Scientific Aviation':
        missing = []

    if operator == 'Scientific Aviation':
        count_missing = 0
        count_missing_zero = 0
    else:
        # Filter missing for non-zero values
        if missing.empty:
            count_missing = 0
            count_missing_zero = 0
        else:
            # Make a subset of the missing non-zero values
            missing_non_zero = missing.query(f'kgh_ch4_{time_ave}_mean_{gas_comp_source} > 0')
            count_missing = make_histogram_bins(missing_non_zero, threshold_lower, threshold_upper,
                                                n_bins).n_data_points

            # Make a subset of the missing zero values
            missing_zero = missing.query(f'kgh_ch4_{time_ave}_mean_{gas_comp_source} == 0')
            count_missing_zero = make_histogram_bins(missing_zero, threshold_lower, threshold_upper,
                                                     n_bins).n_data_points

    # Count total measurements reported by operator
    total_reported = op_reported.max()['overpass_id']

    # Count total releases
    total_releases = total_reported + len(missing)

    ################## store data #########################

    summary = pd.DataFrame({
        'bin_median': bin_median,
        'true_positive': count_tp,
        'false_positive': count_fp,
        'true_negative': count_tn,
        'false_negative': count_fn,
        'filter_stanford': count_su_fail,
        'filter_operator': count_op_fail,
        'missing_data': count_missing,
        'zero_filter_su': count_zero_su_fail,
        'zero_filter_op': count_zero_op_fail,
        'zero_missing': count_missing_zero,
        'total_releases': total_releases,
        'total_reported': total_reported,
    })

    # Determine max bin height for plotting:
    # exclude zeros, zero releases were targeted at 10% of all other releases
    col_for_summing = ['true_positive',
                       'false_positive',
                       'false_negative',
                       'filter_stanford',
                       'filter_operator',
                       'missing_data']

    summary['bin_height'] = summary[col_for_summing].sum(axis=1)

    return summary


def calc_percent_error(observed, expected):
    """Calculate perfect error between an observation and the expected value. Returns value as percent.  """
    # don't divide by zero:
    if expected == 0:
        return np.nan

    if observed == 0:
        return np.nan

    # keep overpasses that aren't quantified in series so it can be aligned later
    if pd.isnull(observed):
        return np.nan
    else:
        return (abs(observed - expected) / expected) * 100


def make_overpass_error_df(operator, stage):
    """Generate a dataframe that calculates percent error for each overpass."""

    ########## Load overpass summary data ##########
    strict_discard = False
    op_lax = load_overpass_summary(operator=operator, stage=stage, strict_discard=strict_discard, time_ave=60,
                                   gas_comp_source='km')

    # Set strict_discard to True
    strict_discard = True
    op_strict = load_overpass_summary(operator=operator, stage=stage, strict_discard=strict_discard, time_ave=60,
                                      gas_comp_source='km')

    ########## Make dataframe for summarizing results with strict and lax QC  ##########

    op_error = pd.DataFrame()

    # Load columns that are the same in both strict and lax overpass summaries:
    op_error['overpass_id'] = op_lax['overpass_id']
    op_error['zero_release'] = op_lax['zero_release']
    op_error['operator_kept'] = op_lax['operator_kept']
    op_error['operator_detected'] = op_lax['operator_detected']
    op_error['operator_quantified'] = op_lax['operator_quantification'].notna()
    op_error['operator_quantification'] = op_lax['operator_quantification']
    op_error['release_rate_kgh'] = op_lax['release_rate_kgh']

    # Load columns that change in strict vs lax overpass summaries
    op_error['stanford_kept_strict'] = op_strict['stanford_kept']
    op_error['qc_summary_strict'] = op_strict['qc_summary']
    op_error['stanford_kept_lax'] = op_lax['stanford_kept']
    op_error['qc_summary_lax'] = op_lax['qc_summary']

    ########## Calculate percent error for each overpass ##########

    # Calculate percent error for all overpasses
    percent_error = op_error.apply(lambda x: calc_percent_error(x['operator_quantification'], x['release_rate_kgh']),
                                   axis=1)
    op_error['percent_error'] = percent_error
    return op_error


def generate_all_overpass_reports(strict_discard=False, timekeeper='flightradar', gas_comp_source='km', time_ave=60):
    """Generate all overpass reports"""
    check_timekeep_capitalization(timekeeper)

    operators = ['Carbon Mapper', 'GHGSat', 'Kairos', 'Kairos LS23', 'Kairos LS25', 'Methane Air']

    # Load clean operator data
    # format for naming: [operator]_stage

    cm_1, cm_2, cm_3, ghg_1, ghg_2, ghg_3, kairos_1, kairos_2, kairos_3, kairos_ls23_1, kairos_ls25_1, kairos_ls23_2, \
        kairos_ls25_2, kairos_ls23_3, kairos_ls25_3, sciav_1, mair_1 = load_clean_operator_reports()

    report_dictionary = {
        'cm_1': cm_1,
        'cm_2': cm_2,
        'cm_3': cm_3,
        'ghg_1': ghg_1,
        'ghg_2': ghg_2,
        'ghg_3': ghg_2,
        'kairos_1': kairos_1,
        'kairos_2': kairos_2,
        'kairos_3': kairos_3,
        'kairos_ls23_1': kairos_ls23_1,
        'kairos_ls25_1': kairos_ls25_1,
        'kairos_ls23_2': kairos_ls23_2,
        'kairos_ls25_2': kairos_ls25_2,
        'kairos_ls23_3': kairos_ls23_3,
        'kairos_ls25_3': kairos_ls25_3,
        'sciav_1': sciav_1,
        'mair_1': mair_1,
    }

    # Load meter data
    meter_dictionary = load_meter_data(timekeeper, gas_comp_source, time_ave)
    # meter_dictionary = {
    #     'cm_meter': cm_meter,
    #     'ghg_meter': ghg_meter,
    #     'kairos_meter': kairos_meter,
    #     'mair_meter': mair_meter,
    # }

    for operator in operators:
        if operator == 'Methane Air':
            stages = [1]  # Methane Air only did one stage
        else:
            stages = [1, 2, 3]

        # For each operator, make a overpass summary for each stage they participated in
        for stage in stages:
            op_ab = abbreviate_op_name(operator)
            operator_report = report_dictionary[f'{op_ab}_{stage}']

            if (operator == 'Kairos LS23') or (operator == 'Kairos LS25'):
                operator_meter = meter_dictionary['kairos_meter']
            else:
                operator_meter = meter_dictionary[f'{op_ab}_meter']
            print(f'Generating operator summary file for {operator} Stage {stage}')
            generate_overpass_summary(operator, stage, operator_report, operator_meter, strict_discard, gas_comp_source,
                                      time_ave)


# %% Compare overpass lenght

def check_overpass_number(operator, max_overpass_id, overpasses_length):
    """Compare the highest value for overpass_id to the length of the overpasses dataframe. Highest overpass_id
    should represent the last overpass conducted, and therefore should be equal to the length of the overpasses
    dataframe. If these values are not equal, then there is likely an error in making overpass dataframe or summary file."""
    if max_overpass_id == overpasses_length:
        print(
            f'The length of the {operator} overpasses dataframe ({overpasses_length:0.0f}) matches the highest value '
            f'for overpass_id ({max_overpass_id:0.0f}).')
    elif max_overpass_id < overpasses_length:
        print(
            f'The maximum {operator} overpass ID is less than the length of the overpass dataframe. Check for '
            f'duplicate values in summary data.')
        print(f'Length of {operator} overpasses dataframe: {overpasses_length:0.0f}')
        print(f'Highest {operator} overpass_id: {max_overpass_id:0.0f}')
    elif max_overpass_id > overpasses_length:
        print(
            f'The maximum {operator} overpass ID is greater than the length of the overpass dataframe. Check for gaps '
            f'in summary data or operator report.')
        print(f'Length of {operator} overpasses dataframe: {overpasses_length:0.0f}')
        print(f'Highest {operator} overpass_id: {max_overpass_id:0.0f}')


# %%
def calc_daily_altitude(operator):
    """Calculate average flight height in feet for each flight day performed by the input operator."""
    # Load operator flight days
    op_days = load_operator_flight_days(operator)

    # Load overpass summary, stage and discard don't matter here because we only care about the altitude column
    op_overpasses = load_overpass_summary(operator=operator, stage=1, strict_discard=False, time_ave=60,
                                          gas_comp_source='km')

    # Set overpass_datetime to index for easier filtering
    op_datetime_index = op_overpasses.set_index('overpass_datetime')

    # Set up dictionary for storing data
    op_daily_altitude = {}

    for i in range(len(op_days)):
        flight_day = op_days.iloc[i].start_time.strftime('%Y-%m-%d')
        daily_overpasses = op_datetime_index.loc[flight_day]
        ave_flight_height = daily_overpasses.altitude_feet.mean()
        op_daily_altitude[flight_day] = ave_flight_height

    # Save data
    op_daily_altitude = pd.DataFrame.from_dict(op_daily_altitude, orient='index')
    op_daily_altitude.columns = ['altitude_feet']
    op_ab = abbreviate_op_name(operator)
    op_daily_altitude.to_csv(pathlib.PurePath('03_results', 'daily_altitude', f'{op_ab}_altitude_feet.csv'))
    return op_daily_altitude
