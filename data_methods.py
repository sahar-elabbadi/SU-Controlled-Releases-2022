# Script for methods used in data cleaning and manipulation
# Author: Sahar H. El Abbadi
# Date Created: 2023-02-22
# Date Last Modified: 2023-02-22

# List of methods in this file:
# > convert_utc

# Imports
import datetime


def convert_utc(dt, delta_t):
    """Convert input time (dt) to UTC. detla_t is the time difference between local time and UTC time. For Arizona,
    this value is + 7 hours"""
    local_hr = int(dt.strftime('%H'))
    local_min = int(dt.strftime('%M'))
    local_sec = int(dt.strftime('%S'))

    utc_hr = local_hr + 7
    utc_time = datetime.time(utc_hr, local_min, local_sec)
    return utc_time
