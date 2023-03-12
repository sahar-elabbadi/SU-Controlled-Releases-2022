# Script for methods used in data manipulating operator reported data and clean data
# Author: Sahar H. El Abbadi
# Date Created: 2023-02-22
# Date Last Modified: 2023-03-02

# List of methods in this file:
# > load_clean_data()
# > load_meter_data()
# > apply_qc_filter(operator_report, operator_meter)
# > convert_utc(dt, delta_t)
# > convert_to_twentyfour(time)


# Notes: Use of the datetime module is very confusing as there is a datetime method and a datetime object. It is
# important to consistently import in the same way, otherwise the two will get confused. See Solution here:
# https://stackoverflow.com/questions/66431493/typeerror-descriptor-date-for-datetime-datetime-objects-doesnt-apply-to-a

# Imports
import datetime
import pandas as pd
import pathlib


# Modules

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
    if operator == "Carbon Mapper":
        op_abb = 'cm'
    elif operator == "GHGSat":
        op_abb = 'ghg'
    elif operator == 'Kairos':
        op_abb = 'kairos'
    elif operator == 'Kairos LS23':
        op_abb = 'kairos_ls23'
    elif operator == 'Kairos LS25':
        op_abb = 'kairos_ls25'
    elif operator == 'Methane Air':
        op_abb = 'mair'
    else:
        print('Typo in operator name')
        return

    return op_abb


# %% Load clean data

def load_clean_data():
    # Carbon Mapper Stage 1
    cm_path_1 = pathlib.PurePath('01_clean_data', 'cm_1_clean.csv')
    cm_1 = pd.read_csv(cm_path_1)

    # Carbon Mapper Stage 2
    cm_path_2 = pathlib.PurePath('01_clean_data', 'cm_2_clean.csv')
    cm_2 = pd.read_csv(cm_path_2)

    # Carbon Mapper Stage 3
    cm_path_3 = pathlib.PurePath('01_clean_data', 'cm_3_clean.csv')
    cm_3 = pd.read_csv(cm_path_3)

    # GHGSat Stage 1
    ghg_path_1 = pathlib.PurePath('01_clean_data', 'ghg_1_clean.csv')
    ghg_1 = pd.read_csv(ghg_path_1)

    # GHGSat Stage 2
    ghg_path_2 = pathlib.PurePath('01_clean_data', 'ghg_2_clean.csv')
    ghg_2 = pd.read_csv(ghg_path_2)

    # Kairos Stage 1 Pod LS23
    kairos_path_1_ls23 = pathlib.PurePath('01_clean_data', 'kairos_1_ls23_clean.csv')
    kairos_1_ls23 = pd.read_csv(kairos_path_1_ls23)

    # Kairos Stage 1 Pod LS25
    kairos_path_1_ls25 = pathlib.PurePath('01_clean_data', 'kairos_1_ls25_clean.csv')
    kairos_1_ls25 = pd.read_csv(kairos_path_1_ls25)

    # Kairos Stage 2 Pod LS23
    kairos_path_2_ls23 = pathlib.PurePath('01_clean_data', 'kairos_2_ls23_clean.csv')
    kairos_2_ls23 = pd.read_csv(kairos_path_2_ls23)

    # Kairos Stage 2 Pod LS25
    kairos_path_2_ls25 = pathlib.PurePath('01_clean_data', 'kairos_2_ls25_clean.csv')
    kairos_2_ls25 = pd.read_csv(kairos_path_2_ls25)

    # Kairos Stage 3 Pod LS23
    kairos_path_3_ls23 = pathlib.PurePath('01_clean_data', 'kairos_3_ls23_clean.csv')
    kairos_3_ls23 = pd.read_csv(kairos_path_3_ls23)

    # Kairos Stage 3 Pod LS25
    kairos_path_3_ls25 = pathlib.PurePath('01_clean_data', 'kairos_3_ls25_clean.csv')
    kairos_3_ls25 = pd.read_csv(kairos_path_3_ls25)

    return cm_1, cm_2, cm_3, ghg_1, ghg_2, kairos_1_ls23, kairos_1_ls25, kairos_2_ls23, kairos_2_ls25, kairos_3_ls23, \
        kairos_3_ls25


# %% Load meter data

def load_meter_data():
    # Carbon Mapper meter data
    cm_path_meter = pathlib.PurePath('02_meter_data', 'CM_meter.csv')
    cm_meter = pd.read_csv(cm_path_meter)

    # GHGSat Stage 1
    ghg_path_meter = pathlib.PurePath('02_meter_data', 'GHGSat_meter.csv')
    ghg_meter = pd.read_csv(ghg_path_meter)

    # Kairos Stage 1 Pod LS23
    kairos_path_meter = pathlib.PurePath('02_meter_data', 'Kairos_meter.csv')
    kairos_1_ls23_meter = pd.read_csv(kairos_path_meter)

    return cm_meter, ghg_meter, kairos_1_ls23_meter


# %%
def apply_qc_filter(operator_report, operator_meter):
    """Merge operator report and operator meter dataframes and select overpasses which pass Stanford QC criteria.
    Operator report dataframe should already have 'nan' values for quantification estimates that do not meet operator
    QC criteria. Returns a combined dataframe matched by PerformerExperimentID"""

    # Merge the two data frames
    combined_df = operator_report.merge(operator_meter, on='PerformerExperimentID')

    # Filter based on overpasses that meet Stanford's QC criteria
    combined_df = combined_df[(combined_df['QC: discard - from Stanford'] == 0)]

    # Filter based on operator QC criteria
    combined_df = combined_df[(combined_df['OperatorKeep'] == 1)]

    return combined_df


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
    my_date = datetime.datetime.strptime(test_date, '%Y-%m-%d')
    my_datetime = datetime.datetime.combine(my_date, my_time)
    return my_datetime


# %% Function to import Philippine's meter data, select the FlightRadar columns, and with abrename columns to be more
# brief and machine-readable
def make_flightradar_operator_dataset(operator, operator_meter_raw, timekeeper):
    """Function to make a clean dataset for each overpass for a given operator. Input is the full name of operator
    and the operator meter file. Also include the desired timekeeper metric for when the oeprator was overhead: this
    can be one of three options:
      - flightradar: used in analysis, timestamp when FlightRadar GPS coordinates are closest to the stack
      - Stanford: Stanford ground team visual observation of when the airplane was overhead
      - team: participating operator's report of when they were over the source """

    if timekeeper == 'flightradar':
        timekeeper = 'Flightradar'
    elif timekeeper == 'stanford':
        timekeeper = 'Stanford'
    elif timekeeper == 'operator':
        timekeeper = 'team'

    # Relevant columns from Philippine generated meter dataset:
    date = 'Date'
    time = f'Time (UTC) - from {timekeeper}'

    if timekeeper == 'Flightradar':
        overpass_id = 'FlightradarOverpassID'
    elif timekeeper == 'Stanford':
        overpass_id = 'StanfordOverpassID'
    elif timekeeper == 'team':
        overpass_id = 'PerformerOverpassID'

    phase_iii = 'PhaseIII'  # 0 indicates the overpass was not provided to team in Phase III, 1 indicates it was
    kgh_gas_30 = f'Last 30s (kg/h) - whole gas measurement - from {timekeeper}'
    kgh_gas_60 = f'Last 60s (kg/h) - whole gas measurement - from {timekeeper}'
    kgh_gas_90 = f'Last 90s (kg/h) - whole gas measurement - from {timekeeper}'
    kgh_ch4_30 = f'Last 30s (kg/h) - from {timekeeper}'
    kgh_ch4_60 = f'Last 60s (kg/h) - from {timekeeper}'
    kgh_ch4_90 = f'Last 90s (kg/h) - from {timekeeper}'
    methane_fraction = 'Percent methane'
    meter = 'Meter'  # note renaming meter variable used above
    qc_discard = f'Discarded - using {timekeeper}'
    qc_discard_strict = f'Discarded - 1% - using {timekeeper}'
    altitude_meters = 'Average altitude last minute (m)'

    operator_meter = pd.DataFrame()
    # Populate relevant dataframes

    operator_meter['overpass_id'] = operator_meter_raw[overpass_id]
    operator_meter['phase_iii'] = operator_meter_raw[phase_iii]
    operator_meter['kgh_gas_30'] = operator_meter_raw[kgh_gas_30]
    operator_meter['kgh_gas_60'] = operator_meter_raw[kgh_gas_60]
    operator_meter['kgh_gas_90'] = operator_meter_raw[kgh_gas_90]
    operator_meter['kgh_ch4_30'] = operator_meter_raw[kgh_ch4_30]
    operator_meter['kgh_ch4_60'] = operator_meter_raw[kgh_ch4_60]
    operator_meter['kgh_ch4_90'] = operator_meter_raw[kgh_ch4_90]
    operator_meter['methane_fraction'] = operator_meter_raw[methane_fraction]
    operator_meter['meter'] = operator_meter_raw[meter]
    operator_meter['qc_discard'] = operator_meter_raw[qc_discard]
    operator_meter['qc_discard_strict'] = operator_meter_raw[qc_discard_strict]
    operator_meter['altitude_meters'] = operator_meter_raw[altitude_meters]
    operator_meter['time'] = operator_meter_raw[time]
    operator_meter['date'] = operator_meter_raw[date]

    # Drop rows with nan to remove rows with missing values. This is because for some operators, we missed overpasses
    # and timestamps have nan values, which causes issues with downstream code

    operator_meter = operator_meter.dropna(axis='index')  # axis = 'index' means to drop rows with missing values

    print(f"currently on {operator} and {timekeeper}")
    overpass_datetime = operator_meter.apply(lambda operator_meter: combine_datetime(operator_meter['date'],
                                                                                              operator_meter['time']),
                                                      axis=1)

    operator_meter.insert(loc=0, column='datetime_utc', value=overpass_datetime)

    # Now that we have removed NA values in time, we can remove date and time columns from operator_meter
    operator_meter = operator_meter.drop(columns=['time', 'date'])

    # Abbreviate meter names in raw meter file
    names = ['Baby Coriolis', 'Mama Coriolis', 'Papa Coriolis']
    nicknames = ['bc', 'mc', 'pc']

    for meter_name, meter_nickname in zip(names, nicknames):
        operator_meter.loc[operator_meter['meter'] == meter_name, 'meter'] = meter_nickname

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
                                           save_folder, f'{op_ab}_meter.csv'))

    return operator_meter
