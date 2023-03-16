# Script for cleaning each of the operator data
# Note: all scripts in this file are meant to be run directly on the operator loaded data
# Author: Sahar H. El Abbadi
# Date Created: 2023-02-22
# Date Last Modified: 2023-02-27
import pathlib

# Methods in this file:
# > clean_cm: clean Carbon Mapper data reports
# > clean_ghgsat: clean GHGSat data reports
# > clean_kairos: clean Kairos data reports

# Imports
import numpy as np
import pandas as pd
import datetime
from methods_source import convert_utc, convert_to_twentyfour


# %% Carbon Mapper Data Cleaning
# In Fall 2022, Carbon Mapper conducted 121 overpasses

def clean_cm(cm_report, cm_overpasses, cm_stage):
    """Clean Carbon Mapper data, with inputs of: cm (Carbon Mapper completed report file), total_overpasses: total
    number of overpasses conducted by Carbon Mapper, and cm_stage: the stage for which the report was submitted."""

    cm_overpasses = np.linspace(1, cm_overpasses, cm_overpasses)  # for indexing for loop
    overpass_list = []  # for generating all new rows
    for overpass in cm_overpasses:

        # Check if the quantification estimate is valid by passing Carbon Mapper's "Good Quality" criteria. Use
        # quantification estimates if valid, otherwise input nan

        # Detected data: if CR plume present = Y, plume is detected
        if cm_report.loc[overpass - 1, "CR plume present (Y/N)"] == "Y":
            detected = True
        else:
            detected = False

        # Quantified data: CR plume present = Y and Good Quality = Y
        if cm_report.loc[overpass - 1, "CR plume present (Y/N)"] == "Y" and cm_report.loc[overpass - 1, "Good Quality " \
                                                                                                        "(Y/N)"] == "Y":
            quantified = True
            emission_rate = cm_report.loc[overpass - 1, "Emission Rate (kg/hr)"]
            emission_upper = cm_report.loc[overpass - 1, "FacilityEmissionRateUpper"]
            emission_lower = cm_report.loc[overpass - 1, "FacilityEmissionRateLower"]

        else:
            quantified = False
            emission_rate = float("nan")
            emission_upper = float("nan")
            emission_lower = float("nan")

        # If the overpass does not pass Carbon Mapper's criteria,
        if cm_report.loc[overpass - 1, "Good Quality (Y/N)"] == "N":
            qc_flag = 'CM-1'
        else:
            qc_flag = 'clear'

        # Set OperatorKeep to True if operator keeps value, False if Operator discards based on QC Flag
        if qc_flag == 'clear':
            operator_kept = True
        else:
            operator_kept = False

        # Convert local time to UTC
        local_time = cm_report.loc[overpass - 1, "Timestamp (hyperspectral technologies only)"]
        utc_time = convert_utc(local_time, 7)

        new_row = {
            'Operator': 'CarbonMapper',
            'Stage': cm_stage,
            'overpass_id': overpass,
            'DateOfSurvey': cm_report.loc[overpass - 1, "DateOfSurvey"].strftime('%Y-%m-%d'),
            'TimestampUTC': utc_time,
            'Detected': detected,
            'QuantifiedPlume': quantified,
            'FacilityEmissionRate': emission_rate,
            'FacilityEmissionRateUpper': emission_upper,
            'FacilityEmissionRateLower': emission_lower,
            'UncertaintyType': '1-sigma',
            'OperatorWindspeed': cm_report.loc[overpass - 1, "WindSpeed (m/s)"],
            'QCFlag': qc_flag,
            'OperatorKeep': operator_kept,
        }
        overpass_list.append(new_row)
    cm_clean = pd.DataFrame(overpass_list)
    return cm_clean


# %% GHGSat Data Cleaning
def clean_ghgsat(ghg_report, ghg_overpasses, ghg_stage):
    # Code variables for iterating in the for loop
    gh_overpasses = np.linspace(1, ghg_overpasses, ghg_overpasses)  # for indexing for loop
    overpass_list = []  # for generating all new rows

    # Replace all "#VALUE!" and "N/A" entries with "nan":
    ghg_report = ghg_report.replace(to_replace=['#VALUE!', 'N/A'], value=float("nan"))

    for overpass in gh_overpasses:

        # Convert local time to UTC
        local_time = ghg_report.loc[overpass - 1, "Timestamp (hyperspectral technologies only)"]
        local_time = convert_to_twentyfour(local_time)
        utc_time = convert_utc(local_time, 7)

        # Determine if a plume was detected or not

        # If QCFlag is 1, conditions were good. Any N/A for quantification is thus a zero
        if ghg_report.loc[overpass - 1, "QC Flag "] == 1:
            if ghg_report.loc[overpass - 1, "FacilityEmissionRate"] > 0:
                detected = True
            else:
                detected = False

        # QCFlag 4 means diffuse emissions visible over site. GHGSat does not provide information on whether these
        # estimates should be considered valid quantifications.
        #
        # If QCFlag == 4, and it is not quantified then I assume no detect. If it is quantified, I assume detect.
        #
        # Note: I compared with our QC criteria, and all QCFlag == 4 will be removed in Stanford's QC process

        elif ghg_report.loc[overpass - 1, "QC Flag "] == 4:
            if ghg_report.loc[overpass - 1, "FacilityEmissionRate"] == 'N/A':
                detected = False
            else:
                detected = True

        # QCFlag is 5 should be removed
        elif ghg_report.loc[overpass - 1, "QC Flag "] == 5:
            detected = False

        # QCFlag of 2 or 3 both indicate a plume was detected, as per GHGSat definition in data submission
        else:
            detected = True

        # Determine if plume was quantified and set emission rate and uncertainty based on the reported QC flags by
        # GHGSat
        if ghg_report.loc[overpass - 1, "QC Flag "] == 1 or ghg_report.loc[overpass - 1, "QC Flag "] == 2:
            quantified = True
            emission_rate = ghg_report.loc[overpass - 1, "FacilityEmissionRate"]

            # In Stage 2, GHGSat FacilityEmissionRateUpper and FacilityEmissionRateLower are importing as strings
            # instead of floats. Convert to float here
            emission_upper = float(ghg_report.loc[overpass - 1, "FacilityEmissionRateUpper"])
            emission_lower = float(ghg_report.loc[overpass - 1, "FacilityEmissionRateLower"])
        else:
            quantified = False
            emission_rate = float("nan")
            emission_upper = float("nan")
            emission_lower = float("nan")

        # Set QC flag:
        ghg_flag = ghg_report.loc[overpass - 1, "QC Flag "]
        qc_flag = f'GH-{ghg_flag:1.0f}'

        # Set OperatorKeep to True if operator keeps value, False if Operator discards based on QC Flag
        if qc_flag in {'GH-1', 'GH-2', 'GH-3'}:
            operator_kept = True
        else:
            operator_kept = False

        # GHGSat changed column  name for WindSpeed in Stage 2 to WindSpeed (LOCAL)
        # I believe this means this is the value for windspeed they are using from the data we provided
        if ghg_stage == 1:
            windspeed = ghg_report.loc[overpass - 1, "WindSpeed"]
        elif ghg_stage == 2:
            windspeed = ghg_report.loc[overpass - 1, "WindSpeed (LOCAL)"]
        elif ghg_stage == 3:
            windspeed = ghg_report.loc[overpass - 1, "WindSpeed (LOCAL)"]

        new_row = {
            'Operator': 'GHGSat-AV',
            'Stage': ghg_stage,
            'overpass_id': overpass,
            'DateOfSurvey': ghg_report.loc[overpass - 1, "DateOfSurvey"],
            'TimestampUTC': utc_time,
            'Detected': detected,
            'QuantifiedPlume': quantified,
            'FacilityEmissionRate': emission_rate,
            'FacilityEmissionRateUpper': emission_upper,
            'FacilityEmissionRateLower': emission_lower,
            'UncertaintyType': '1-sigma',
            'OperatorWindspeed': windspeed,
            'QCFlag': qc_flag,
            'OperatorKeep': operator_kept,

        }
        overpass_list.append(new_row)

    ghg_clean = pd.DataFrame(overpass_list)
    return ghg_clean


# %% Kairos Data Cleaning
def clean_kairos(kairos_report, kairos_overpasses, kairos_stage):
    """Clean Kairos data report. Takes inputs of kairos_report (operator generator report), kairos_overpasses (number
    of overpasses), and kairos_stage (stage of analysis)"""

    # Code variables for iterating in the for loop
    kairos_overpasses = np.linspace(1, kairos_overpasses, kairos_overpasses)  # for indexing for loop
    overpass_list = []  # for generating all new rows

    for overpass in kairos_overpasses:

        # Convert local time to UTC
        local_time = kairos_report.loc[overpass - 1, "Timestamp (hyperspectral technologies only)"]
        local_time = datetime.datetime.strptime(local_time, '%H:%M:%S').time()
        utc_time = convert_utc(local_time, 7)

        # Determine if plume was quantified and set emission rate and uncertainty based on the reported QC flags by
        # Kairos
        kairos_quantified = kairos_report.loc[
            overpass - 1, "Kairos Flag for Dropped Passes or Uncertain Rate Quantification"]
        if pd.isna(kairos_quantified) or kairos_quantified == 'clear':
            quantified = True
            emission_rate = kairos_report.loc[overpass - 1, "FacilityEmissionRate"]
            emission_upper = kairos_report.loc[overpass - 1, "FacilityEmissionRateUpper"]
            emission_lower = kairos_report.loc[overpass - 1, "FacilityEmissionRateLower"]
        else:
            quantified = False
            emission_rate = float("nan")
            emission_upper = float("nan")
            emission_lower = float("nan")

        # Set QC flag:
        kairos_flag = kairos_report.loc[overpass - 1, "Kairos Flag for Dropped Passes or Uncertain Rate Quantification"]
        # check Kairos' QC flags
        if pd.isna(kairos_flag):
            qc_flag = 'clear'
        elif kairos_flag[:5] == 'Plane':
            qc_flag = 'KA-1'
        elif kairos_flag[:5] == 'PARTI':
            qc_flag = 'KA-2'
        elif kairos_flag[:5] == 'Cutof':
            qc_flag = 'KA-3'
        elif kairos_flag[:5] == 'Exces':
            qc_flag = 'KA-4'
        elif kairos_flag[:5] == 'Glare':
            qc_flag = 'KA-5'
        # Check QC flags from combined Kairos LS23 and LS25 dataset
        elif kairos_flag == 'clear':
            qc_flag = 'clear'
        elif kairos_flag == 'fail_both_qc':
            qc_flag = 'KA-Combo'
        else:
            qc_flag = 'ERROR! Identify misclassified QC'

        # Determine if Kairos detected a plume or not Kairos reports zeros (character '0' for some reason) in the
        # quantification column. FacilityEmissionRate is blank if there is a QC Flag. In some instances, there is a
        # qnatification estimate with QC Flag KA-3 (cutoff, low confidence in quantification)

        if qc_flag == 'clear':
            if kairos_report.loc[overpass - 1, "FacilityEmissionRate"] == '0':
                detected = False
            else:
                detected = True
        else:
            detected = False

        # Set OperatorKeep to True if operator keeps value, False if Operator discards based on QC Flag
        if qc_flag == 'clear':
            operator_kept = True
        else:
            operator_kept = False

        new_row = {
            'Operator': 'Kairos - LS23',
            'Stage': kairos_stage,
            'overpass_id': overpass,
            'DateOfSurvey': kairos_report.loc[overpass - 1, "DateOfSurvey"],
            'TimestampUTC': utc_time,
            'Detected': detected,
            'QuantifiedPlume': quantified,
            'FacilityEmissionRate': emission_rate,
            'FacilityEmissionRateUpper': emission_upper,
            'FacilityEmissionRateLower': emission_lower,
            'UncertaintyType': 'nan',
            'OperatorWindspeed': kairos_report.loc[overpass - 1, "WindSpeed"],
            'QCFlag': qc_flag,
            'OperatorKeep': operator_kept,

        }
        overpass_list.append(new_row)

    kairos_clean = pd.DataFrame(overpass_list)
    return kairos_clean


def make_kairos_combo(kairos_overpass, kairos_stage):
    # Generate combined Kairos dataframe
    # Kairos Stage 1 Pod LS23
    ls23_path = pathlib.PurePath('01_clean_reports', f'kairos_{kairos_stage}_ls23_clean.csv')
    ls23 = pd.read_csv(ls23_path, index_col=0)

    # Kairos Stage 1 Pod LS25
    ls25_path = pathlib.PurePath('01_clean_reports', f'kairos_{kairos_stage}_ls25_clean.csv')
    ls25 = pd.read_csv(ls25_path, index_col=0)

    kairos_summary = pd.DataFrame({
        'ls23': ls23.FacilityEmissionRate,
        'ls25': ls25.FacilityEmissionRate,
        'qc_ls23': ls23.QCFlag,
        'qc_ls25': ls25.QCFlag,
    })

    kairos_summary['qc'] = np.where((kairos_summary['qc_ls23'] == 'clear') |
                                    (kairos_summary['qc_ls25'] == 'clear'), 'clear', 'fail_both_qc')

    pod_conditions = [
        (kairos_summary.ls23.notnull() & kairos_summary.ls25.notnull()),  # both pods report data
        (kairos_summary.ls23.notnull() & kairos_summary.ls25.isnull()),  # LS23 reports only
        (kairos_summary.ls23.isnull() & kairos_summary.ls25.notnull()),  # LS25 reports only
        (kairos_summary.ls23.isnull() & kairos_summary.ls25.isnull()),  # both are null
    ]

    pod_choices = [
        (kairos_summary.ls23 + kairos_summary.ls25) / 2,
        kairos_summary.ls23,
        kairos_summary.ls25,
        np.nan

    ]

    kairos_summary['pod_average'] = np.select(pod_conditions, pod_choices, 'ERROR')
    kairos_summary['lower_bound'] = kairos_summary[['ls23', 'ls25']].min(axis=1)
    kairos_summary['upper_bound'] = kairos_summary[['ls23', 'ls25']].max(axis=1)

    kairos_summary.to_csv(pathlib.PurePath('01_clean_reports', f'kairos_{kairos_stage}_combined_data.csv'))

    # Populate summary data into a new dataframe. Use Kairos raw data from LS23 Stage 1

    if kairos_stage == 1:
        kairos_path = pathlib.PurePath('00_raw_reports', 'Kairos_Stage1_podLS23_submitted-2022-11-17.csv')
    elif kairos_stage == 2:
        kairos_path = pathlib.PurePath('00_raw_reports', 'Kairos_Stage2_podLS23_submitted-2022-12-20.csv')
    elif kairos_stage == 3:
        kairos_path = pathlib.PurePath('00_raw_reports', 'Kairos_Stage3_podLS23_submitted-2023-02-23.csv')

    kairos_combo = pd.read_csv(kairos_path)

    # Reset the following columns to be re-populated
    kairos_combo['FacilityEmissionRate'] = np.nan
    kairos_combo['Kairos Flag for Dropped Passes or Uncertain Rate Quantification'] = np.nan
    kairos_combo['OperatorKeep'] = np.nan

    # Set columns to the summary values
    kairos_combo['FacilityEmissionRate'] = kairos_summary['pod_average']
    kairos_combo['FacilityEmissionRateUpper'] = kairos_summary['upper_bound']
    kairos_combo['FacilityEmissionRateLower'] = kairos_summary['lower_bound']
    kairos_combo['Kairos Flag for Dropped Passes or Uncertain Rate Quantification'] = kairos_summary['qc']
    kairos_combo_clean = clean_kairos(kairos_report=kairos_combo, kairos_overpasses=kairos_overpass,
                                      kairos_stage=kairos_stage)

    kairos_combo_clean.to_csv(pathlib.PurePath('01_clean_reports', f'kairos_{kairos_stage}_clean.csv'))

    return kairos_combo_clean
