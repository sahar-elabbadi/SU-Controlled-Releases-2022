
Submitted spreadsheet:
2021_DataReportingTemplate_AerialAppendix_v5_Expanded_MethaneAIR_AC_20230328.xlsx

##### Decision Tree Flags

# Case 0: No detection
# Case 1: Around the order of magnitude of the observation 
# Case 2: Low SNR 
# Case 3: High SNR & small plume
# Case 4: High SNR & large plume 

# We report the FacilityEmissionRate as the mean between the two methods (E). 

##### Uncertainty Flags

# Confidence Intervals: 
#     The confidence interval for the mIME is derived from a bootstrap of the emission estimate through all of its steps and inputs (except the estimated plume area).
#     The confidence interval for the DI is obtained for the set of DIs in the growing boxes covering at least two eddy scales. 
#     The confidence interval for the average of DI and mIME: We assume the uncertainty of the estimates from the two methods to follow a normal distribution:  the combined s.d. is reported as the root mean square of the sds from the two methods, and the 95% CI is +/- 2 s.d. The new confidence interval equals [E - 2 new s.d., E + 2 new s.d.]. 

# MINIMUM: The observation is below our quantification limit yet above the detection limit.  CI is +/- 100 kg/hr based on stats for quantified plumes.

# DETECTION LIMIT: The observation is below our detection limit. The decision tree suggests Case 0. We report zero emission with confidence intervals of [-100, 100] kg/hr. 

# LOW DT: The decision tree suggests Case 1 or 2. Instead of using the combined confidence interval derived from the two methods, we assume the confidence interval to be [E - 100, E + 100]. 


===================================================================================================================
Supplementary file:
outputs_and_raw_mIME_DI_20230327.csv

Wind direction is obtained from the 2nd principal moment of intertia of the observed plume

If the observed plume is oblate ( "meringue") the pricipal moments aren't distinct and the
direction is taken from the HRRR (mean PBL wind direction, usually very weak for these cases.
