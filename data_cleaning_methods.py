# Script for cleaning each of the operator data
# Note: all scripts in this file are meant to be run directly on the operator loaded data
# Author: Sahar H. El Abbadi
# Date Created: 2023-02-22
# Date Last Modified: 2023-02-24

# Methods in this file:
# > clean_cm: clean Carbon Mapper data reports
# > clean_ghgsat: clean GHGSat data reports

# Imports
import numpy as np
import pandas as pd
import datetime
from data_manipulation_methods import convert_utc, convert_to_twentyfour


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
        if cm_report.loc[overpass - 1, "CR plume present (Y/N)"] == "Y" and cm_report.loc[
            overpass - 1, "Good Quality (Y/N)"] == "Y":
            quantified = 1
            emission_rate = cm_report.loc[overpass - 1, "Emission Rate (kg/hr)"]
            emission_upper = cm_report.loc[overpass - 1, "FacilityEmissionRateUpper"]
            emission_lower = cm_report.loc[overpass - 1, "FacilityEmissionRateLower"]
        else:
            quantified = 0
            emission_rate = float("nan")
            emission_upper = float("nan")
            emission_lower = float("nan")

        # If the overpass does not pass Carbon Mapper's criteria,
        if cm_report.loc[overpass - 1, "Good Quality (Y/N)"] == "N":
            qc_flag = 'CM-1'
        else:
            qc_flag = 'nan'

        # Convert local time to UTC
        local_time = cm_report.loc[overpass - 1, "Timestamp (hyperspectral technologies only)"]
        utc_time = convert_utc(local_time, 7)

        new_row = {
            'Operator': 'CarbonMapper',
            'Stage': cm_stage,
            'PerformerExperimentID': overpass,
            'DateOfSurvey': cm_report.loc[overpass - 1, "DateOfSurvey"].strftime('%Y-%m-%d'),
            'TimestampUTC': utc_time,
            'QuantifiedPlume': quantified,
            'FacilityEmissionRate': emission_rate,
            'FacilityEmissionRateUpper': emission_upper,
            'FacilityEmissionRateLower': emission_lower,
            'UncertaintyType': '1-sigma',
            'OperatorWindspeed': cm_report.loc[overpass - 1, "WindSpeed (m/s)"],
            'QCFlag': qc_flag,
        }
        overpass_list.append(new_row)
    cm_clean = pd.DataFrame(overpass_list)
    return cm_clean


# %% GHGSat Data Cleaning
def clean_ghgsat(ghg_report, ghg_overpasses, ghg_stage):
    # Code variables for iterating in the for loop
    gh_overpasses = np.linspace(1, ghg_overpasses, ghg_overpasses)  # for indexing for loop
    overpass_list = []  # for generating all new rows
    for overpass in gh_overpasses:

        # Convert local time to UTC
        local_time = ghg_report.loc[overpass - 1, "Timestamp (hyperspectral technologies only)"]
        local_time = convert_to_twentyfour(local_time)
        utc_time = convert_utc(local_time, 7)

        # Determine if plume was quantified and set emission rate and uncertainty based on the reported QC flags by
        # GHGSat
        if ghg_report.loc[overpass - 1, "QC Flag "] == 1 or ghg_report.loc[overpass - 1, "QC Flag "] == 2:
            quantified = 1
            emission_rate = ghg_report.loc[overpass - 1, "FacilityEmissionRate"]
            emission_upper = ghg_report.loc[overpass - 1, "FacilityEmissionRateUpper"]
            emission_lower = ghg_report.loc[overpass - 1, "FacilityEmissionRateLower"]
        else:
            quantified = 0
            emission_rate = float("nan")
            emission_upper = float("nan")
            emission_lower = float("nan")

        # Set QC flag:
        ghg_flag = ghg_report.loc[overpass - 1, "QC Flag "]
        qc_flag = f'GH-{ghg_flag:1.0f}'

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
            'PerformerExperimentID': overpass,
            'DateOfSurvey': ghg_report.loc[overpass - 1, "DateOfSurvey"],
            'TimestampUTC': utc_time,
            'QuantifiedPlume': quantified,
            'FacilityEmissionRate': emission_rate,
            'FacilityEmissionRateUpper': emission_upper,
            'FacilityEmissionRateLower': emission_lower,
            'UncertaintyType': '1-sigma',
            'OperatorWindspeed': windspeed,
            'QCFlag': qc_flag,
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
        if pd.isna(kairos_quantified):
            quantified = 1
            emission_rate = kairos_report.loc[overpass - 1, "FacilityEmissionRate"]
            emission_upper = kairos_report.loc[overpass - 1, "FacilityEmissionRateUpper"]
            emission_lower = kairos_report.loc[overpass - 1, "FacilityEmissionRateLower"]
        else:
            quantified = 0
            emission_rate = float("nan")
            emission_upper = float("nan")
            emission_lower = float("nan")

        # Set QC flag:
        kairos_flag = kairos_report.loc[overpass - 1, "Kairos Flag for Dropped Passes or Uncertain Rate Quantification"]
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
        else:
            qc_flag = 'ERROR! Identify misclassified QC'

        new_row = {
            'Operator': 'Kairos - LS23',
            'Stage': kairos_stage,
            'PerformerExperimentID': overpass,
            'DateOfSurvey': kairos_report.loc[overpass - 1, "DateOfSurvey"],
            'TimestampUTC': utc_time,
            'QuantifiedPlume': quantified,
            'FacilityEmissionRate': emission_rate,
            'FacilityEmissionRateUpper': emission_upper,
            'FacilityEmissionRateLower': emission_lower,
            'UncertaintyType': 'nan',
            'OperatorWindspeed': kairos_report.loc[overpass - 1, "WindSpeed"],
            'QCFlag': qc_flag,
        }
        overpass_list.append(new_row)

    kairos_clean = pd.DataFrame(overpass_list)
    return kairos_clean
