# Daily Meter Data 

This folder contains daily meter data used in all calculations.

## Whole Gas Raw
_whole_gas_raw_ files are the output of the Stanford meter clean pipeline. 
This folder contains an .xlsx file for each day of testing throughout the entire experiment, with the naming convention %m_%d.xlsx. 

Each _whole_gas_raw_ file contains the following columns: 

- Datetime (UTC)
- Release Rate (kg/h): 
  - This is the whole gas release rate in kg/hr
  - Value of "NA" indicates system was offline or missing 
  - Value of 0 indicates system was in use with intentional flow rate of 0 kg/hr
- Coriolis Meter in use:
  - 'Baby Coriolis': refers to CMFS015H
  - 'Mama Coriolis': refers to CMF050M
  - 'Papa Coriolis': refers to CMFS150M 
- QC (1: Non-testing period, 2: Non-original data (missing), 3: Below flow accuracy threshold) 
  - #TODO need to clarify this with Philippine

## Whole Gas Clean
This directory contains three sub-directories, one for each source of gas composition used in the study. 

Gas composition sources: 
- km: original source of gas composition is data provided by Kinder Morgan
- su_raw: original source of gas composition is analysis by Eurofins Air Toxics laboratory. Here, we use raw values directly as inputs for methane mole fraction 
- su_normalized: source of gas composition data is the normalized gas composition values from the Eurofins analysis. We noramlized here because total gas constituents in raw data provided by Eurofins included samples where the sum of total constituents added to over 100%. 

We use 'km' as default values for gas composition in all code, and strongly recommend others do so as well. 

Within each gas composition sub-directory are .csv files for each date of testing, formatted using the same naming system as the raw gas files: %m_%d.csv. 

Each files contains the following columns: 
- datetime_utc
- meter: here, we abbreviate meter names to be consistent with methods used throughout the remainder of the analysis  
  - 'bc': refers to CMFS015H (abbreviation of 'Baby Coriolis')
  - 'mc': refers to CMF050M (abbreviation of 'Mama Coriolis')
  - 'pc': refers to CMFS150M (abbreviation of 'Papa Coriolis') 
- meter_percent_uncertainty: this is the percent uncertainty associated with the meter reading, calculated using Emerson Micromotion sizing tool. It is percentage of whole gas flow rate through the meter, assumed to be at 95% confidence interval. Values are calculated using the _calc_meter_uncertainty_ function 
- meter_sigma: this is the sigma (1 standard deviation) value of the meter_percent_uncertainty, converted from percent of flow rate to kg/hr of whole gas using the meter reading of the given flow rate 
- whole_gas_kgh: meter reading, as kg / hr of gas passed through the meter 
- fraction_methane: methane mole fraction, based on the corresponding gas composition selected 
- fraction_methane_sigma: standard deviation associated with variability in gas composition. Details are discussed in _00_gas_comp_data_calculations.ipynb_
- methane_kgh: flow rate as kgh CH4
- methane_kgh_sigma: combined sigma value for kg CH4 / hr, combining the uncertainties associated with the meter reading and the gas composition, using sum of quadratures of the relative uncertainties of the two sources of variability
- data_qc: Stanford meter data QC value 


