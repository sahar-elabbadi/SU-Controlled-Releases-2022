# Script for methods used in data cleaning and manipulation
# Author: Sahar H. El Abbadi
# Date Created: 2023-02-22
# Date Last Modified: 2023-02-22

# List of methods in this file:
# > convert_utc
# > convert_to_24hr_time


# Notes: Use of the datetime module is very confusing as there is a datetime method and a datetime object. It is
# important to consistently import in the same way, otherwise the two will get confused. See Solution here:
# https://stackoverflow.com/questions/66431493/typeerror-descriptor-date-for-datetime-datetime-objects-doesnt-apply-to-a

# Imports
import datetime


# Modules
def convert_utc(dt, delta_t):
    """Convert input time (dt in datetime format) to UTC. detla_t is the time difference between local time and UTC
    time. For Arizona, this value is + 7 hours"""
    local_hr = int(dt.strftime('%H'))
    local_min = int(dt.strftime('%M'))
    local_sec = int(dt.strftime('%S'))

    utc_hr = local_hr + delta_t
    utc_time = datetime.time(utc_hr, local_min, local_sec)
    return utc_time


def convert_to_twentyfour(time):
    """Convert a string of format HH:MM AM/PM to 24 hour time. Used for converting GHGSat's reported timestamps
    from 12 hour time to 24 hour time. Output is of class datetime"""
    time_12hr = str(datetime.datetime.strptime(time, '%I:%M:%S %p'))
    time_24hr = time_12hr[-8:]
    time_24hr = datetime.datetime.strptime(time_24hr, '%H:%M:%S').time()
    return time_24hr
