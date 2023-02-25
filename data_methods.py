# Script for methods used in data cleaning and manipulation
# Author: Sahar H. El Abbadi
# Date Created: 2023-02-22
# Date Last Modified: 2023-02-24

# List of methods in this file:
# > convert_utc
# > convert_to_24hr_time
# > load_clean_data


# Notes: Use of the datetime module is very confusing as there is a datetime method and a datetime object. It is
# important to consistently import in the same way, otherwise the two will get confused. See Solution here:
# https://stackoverflow.com/questions/66431493/typeerror-descriptor-date-for-datetime-datetime-objects-doesnt-apply-to-a

# Imports
import datetime
import pandas as pd
import pathlib


# Modules

# %% Load clean data

def load_clean_data():
    # Carbon Mapper Stage 1
    cm_path_1 = pathlib.PurePath('01_clean_data', 'cm_clean.csv')
    cm_1 = pd.read_csv(cm_path_1)

    # GHGSat Stage 1
    ghg_path_1 = pathlib.PurePath('01_clean_data', 'ghg_clean.csv')
    ghg_1 = pd.read_csv(ghg_path_1)

    # Kairos Stage 1 Pod LS23
    kairos_path_1 = pathlib.PurePath('01_clean_data', 'kairos_clean.csv')
    kairos_1_ls23 = pd.read_csv(kairos_path_1)

    return cm_1, ghg_1, kairos_1_ls23

#%% Load meter data

def load_meter_data():
    # Carbon Mapper meter data
    cm_path_meter = pathlib.PurePath('01_meter_data', 'CM_meter.csv')
    cm_meter = pd.read_csv(cm_path_meter)

    # GHGSat Stage 1
    ghg_path_meter = pathlib.PurePath('01_meter_data', 'GHGSat_meter.csv')
    ghg_meter = pd.read_csv(ghg_path_meter)

    # Kairos Stage 1 Pod LS23
    kairos_path_meter = pathlib.PurePath('01_meter_data', 'Kairos_meter.csv')
    kairos_1_ls23_meter = pd.read_csv(kairos_path_meter)

    return cm_meter, ghg_meter, kairos_1_ls23_meter


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
