from datetime import datetime
from data_methods import convert_to_twentyfour

from data_methods import convert_to_twentyfour
datetime_str = "1:45 PM"
gh_test = str(datetime.strptime(datetime_str, '%I:%M %p'))
gh_time_only = gh_test[-8:]
gh_time_only = datetime.strptime(gh_time_only, '%H:%M:%S')
# print(gh_test)
# print(gh_test[-8:])
x = convert_to_twentyfour(datetime_str)
print(x)

#test = convert_to_twentyfour("10:02 AM")