# Script for functions used in paper writing

# setup

from methods_source import find_missing_data, load_overpass_summary, classify_confusion_categories, abbreviate_op_name
import numpy as np
import pandas as pd

def summarize_operator_stages_defaults():
    """Generate dataframe summarizing default conditions. Iterate through this dataframe in order to
    apply functions to all operators across all stages"""

    operators = ['Carbon Mapper', 'GHGSat', 'Kairos', 'MethaneAIR', 'Scientific Aviation']
    stage_dictionary = {
        'cm': 3,
        'ghg': 3,
        'kairos': 3,
        'mair': 1,
        'sciav': 1,
    }

    operator_stage_list =[]
    for operator in operators:
        op_ab = abbreviate_op_name(operator)
        max_stage = stage_dictionary[op_ab]
        for i in range(1, max_stage+1):
            if op_ab == 'sciav':
                qc_strict = True
            else:
                qc_strict = False
            new_row = {
                'operator': operator,
                'stage': i,
                'strict_discard': qc_strict
            }

            # save new row
            operator_stage_list.append(new_row)

    operator_stages = pd.DataFrame(operator_stage_list)

    return operator_stages


def print_overpass_info(release):
    """Print relevant columns (release rate, uncertainty) for a specific overpass from an overpass summary file"""

    relevant_cols = ['release_rate_kgh', 'lower_95CI', 'upper_95CI', 'ch4_kgh_sigma', 'sigma_flow_variability',
                        'sigma_meter_reading', 'sigma_gas_composition']

    print(f'Release Rate: {release.release_rate_kgh:} kg CH4 / hr')
    print(f'[{release.lower_95CI}, {release.upper_95CI}, 95% CI]')
    print(f'(sigma from gas flow: {release.sigma_flow_variability:1f})')
    print(f'(sigma from meter: {release.sigma_meter_reading:1f})')
    print(f'(sigma from gas composition: {release.sigma_gas_composition:1f})')
    print(f'(combined total sigma: {release.ch4_kgh_sigma:1f})\n')

def operator_releases_summary_stats(operator, strict_discard=False):
    """ Function for generating the data used in operator overview paragraphs of Results section"""

    overpasses = load_overpass_summary(operator=operator, strict_discard=strict_discard, stage=1)
    print(f'{operator}: {len(overpasses)} flightlines reported to SU')
    fail_su_qc = overpasses.loc[overpasses.stanford_kept == False]
    print(f'{len(fail_su_qc)} overpasses that fail SU QC')
    pass_su_qc = overpasses.loc[overpasses.stanford_kept == True]
    print(f'{len(pass_su_qc)} overpasses that pass SU QC')
    pass_operator_qc = pass_su_qc.loc[pass_su_qc.operator_kept == True]
    print(f'{len(pass_operator_qc)} overpasses quantified by {operator}')
    fail_operator_qc = pass_su_qc.loc[pass_su_qc.operator_kept == False]
    print(f'{len(fail_operator_qc)} overpasses removed by {operator}\n')
    pass_all_qc = overpasses.loc[overpasses.qc_summary == 'pass_all']

    missing = find_missing_data(operator)
    print(f'{len(missing)} overpasses documented by Stanford but not reported by {operator}\n')
    total_overpasses = len(missing) + len(overpasses)
    print(f'Total releases conducted by Stanford (including missing overpasses not reported by {operator}: {total_overpasses}\n')
    relevant_cols = ['release_rate_kgh', 'lower_95CI', 'upper_95CI', 'ch4_kgh_sigma', 'sigma_flow_variability',
                        'sigma_meter_reading', 'sigma_gas_composition']

    # Find smallest non-zero release to pass SU filtering
    non_zero_overpasses_su = pass_su_qc.loc[pass_su_qc.non_zero_release == True]
    min_su_info = non_zero_overpasses_su.loc[non_zero_overpasses_su.release_rate_kgh.idxmin()][relevant_cols]
    print(f'Smallest non-zero volume overpass for {operator} that passes SU qc:')
    print_overpass_info(min_su_info)

    # Find smallest non-zero release to pass all QC
    non_zero_overpasses = pass_all_qc.loc[pass_all_qc.non_zero_release == True]
    min_info = non_zero_overpasses.loc[non_zero_overpasses.release_rate_kgh.idxmin()][relevant_cols]
    print(f'Smallest non-zero volume overpass for {operator} that passes operator and SU qc:')
    print_overpass_info(min_info)

    # Find largest release given by Stanford that passes all QC
    max_su_info = pass_su_qc.loc[pass_su_qc.release_rate_kgh.idxmax()][relevant_cols]
    print(f'Largest volume overpass for {operator} that passes SU qc:')
    print_overpass_info(max_su_info)

    # Find largest release given by Stanford that passes all QC
    max_info = pass_all_qc.loc[pass_all_qc.release_rate_kgh.idxmax()][relevant_cols]
    print(f'Largest volume overpass for {operator} that passes operator & SU qc:')
    print_overpass_info(max_info)


    # Zero releases
    zero_releases = pass_all_qc.loc[pass_all_qc.zero_release == True]
    print(f'Number of zero releases to {operator}: {len(zero_releases)}\n')

    # Classify confusion matrix
    # For confusoin matrix, use only the ones that pass all QC
    # only apply this to the data that pass SU filtering
    true_positives, false_positives, true_negatives, false_negatives = classify_confusion_categories(pass_all_qc)

    # Were there any false positives?
    if false_positives.empty:
        print(f'No false positives detected, all zero releases were correctly categorized\n')
    else:
        print(f'False positives detected: {len(false_positives)}\n')
        print(false_positives['release_rate_kgh'])

    # Were there any false positives?
    if false_negatives.empty:
        print(f'No false negatives detected, all zero releases were correctly categorized\n')
    else:
        print(f'False negatives detected: {len(false_negatives)}\n')

        # Find largest false negative:
        largest_false_neg = false_negatives.loc[false_negatives.release_rate_kgh.idxmax()][relevant_cols]
        print(f'Largest false negative for {operator}: ')
        print_overpass_info(largest_false_neg)


    # Find smallest plume that operator detected
    detected_non_zero = non_zero_overpasses.loc[non_zero_overpasses.operator_detected == True]
    min_detected = detected_non_zero.loc[detected_non_zero.release_rate_kgh.idxmin()][relevant_cols]
    print(f'Smallest detected plume by {operator}:')
    print_overpass_info(min_detected)

    # Smallest plume quantified by operator
    quantified_non_zero = non_zero_overpasses.loc[non_zero_overpasses.operator_quantification > 0]
    min_quantified = quantified_non_zero.loc[quantified_non_zero.release_rate_kgh.idxmin()]
    print(f'Smallest quantified plume by {operator}:')
    print_overpass_info(min_quantified)


def test_parity(x_value, y_value, y_error):
    """Test if a given y-value and associated error pass the parity line. Returns boolean True or False"""

    # Define upper and lower bounds
    y_upper = y_value + y_error
    y_lower = y_value - y_error

    # Test if x_value is in between y_upper and y_lower
    if (x_value <= y_upper) and (x_value >= y_lower):
        return True
    else:
        return False


def calc_parity_intersection(operator, stage, strict_discard=False):
    """Determine the percent of quantification estimates that cross the parity line"""

    all_overpasses = load_overpass_summary(operator=operator, strict_discard=strict_discard, stage=stage)

    # Only consider points that pass all QC
    overpasses = all_overpasses.loc[all_overpasses.pass_all_qc == True].copy()

    # Only consider overpasses where operator quantification estimate is a real number
    overpasses = overpasses[overpasses.operator_quantification.notnull()]

    # Only consider non-zero releases
    overpasses = overpasses[overpasses.non_zero_release == True]

    # Operator quantification > 0
    overpasses = overpasses[overpasses.operator_quantification > 0]


    # Uncertainty types for each operator
    operator_uncertainty_dictionary = {
        'cm': '1-sigma',
        'ghg': '1-sigma',
        'kairos': 'pod_val',
        'mair': '95_CI',
        'sciav': '1-sigma',
    }

    # Multiplier for concerting uncertainty
    uncertainty_multiplier = {
        '1-sigma': 1.96,
        '95_CI': 1,
    }

    op_ab = abbreviate_op_name(operator)
    operator_multiplier = uncertainty_multiplier[operator_uncertainty_dictionary[op_ab]]
    overpasses['operator_error_bound'] = (overpasses['operator_upper'] - overpasses['operator_quantification'])
    overpasses['operator_95CI_bounds'] = overpasses['operator_error_bound'] * operator_multiplier

    overpasses['intersect_parity_line'] = overpasses.apply(lambda x: test_parity(x['release_rate_kgh'],
                                                                                 x['operator_quantification'],
                                                                                 x['operator_95CI_bounds']),
                                                           axis=1)

    cross_parity = len(overpasses.loc[overpasses.intersect_parity_line == True])
    percent_cross_parity = cross_parity / len(overpasses)
    print(
        f'Fraction of {operator} Stage {stage} overpasses with 95% CI that encompasses parity line: {percent_cross_parity * 100:.0f}%')

    return overpasses


def test_parity_all_stages(operator):
    op_ab = abbreviate_op_name(operator)
    operator_stages = {
        'cm': 3,
        'ghg': 3,
        'kairos': 3,
        'mair': 1,
        'sciav': 1,
    }

    for i in range(operator_stages[op_ab]):
        calc_parity_intersection(operator, stage=(i + 1))

    print('\n')
    return

def calc_residual(x, y, m, b):
    y_fit = m * x + b
    residual = y - y_fit
    return residual

def calc_error_absolute(expected, observed):
    return observed - expected

def calc_error_percent(expected, observed):
    """Calculate perfect error between an observation and the expected value. Returns value as percent.  """

    # Remove zeros, don't divide by zero
    if expected == 0:
        # True zeros where expected and observed values are both zero
        if observed == 0:
            return 0
        else:
            return np.nan

    #
    # if observed == 0:
    #     return np.nan

    # keep overpasses that aren't quantified in series so it can be aligned later
    if pd.isnull(observed):
        return np.nan
    else:
        return (observed - expected) / expected * 100

def calculate_residuals_and_error(operator, stage, qc_status, strict_discard=False, time_ave=60, gas_comp_source='km'):
    """ Calculate the measurement residuals for operator
    qc_status can be: 'pass_all', 'all_points', 'pass_operator'
    """
    # data, description = get_parity_data(operator, stage, strict_discard, time_ave, gas_comp_source)

    overpass_summary = load_overpass_summary(operator, stage, strict_discard, time_ave, gas_comp_source)

    # Remove rows where operator did not quantify
    overpass_summary = overpass_summary.dropna(subset='operator_quantification')

    # Select which QC we want
    if qc_status == 'pass_all':
        qc_mask = (overpass_summary['stanford_kept'] == True) & (overpass_summary['operator_kept'] == True)
    elif qc_status == 'pass_operator':
        qc_mask = (overpass_summary['operator_kept'] == True)
    elif qc_status == 'all_points':
        qc_mask = overpass_summary['operator_quantification'].notna() # generic mask to select all points in dataset

    data = overpass_summary.loc[qc_mask].copy()

    # Set x and y data
    data['meter_data'] = overpass_summary.release_rate_kgh
    data['operator_data'] = overpass_summary.operator_quantification
    data['qc'] = overpass_summary.qc_summary

    # Fit linear regression via least squares with numpy.polyfit
    # m is slope, intercept is b
    m, b = np.polyfit(data.meter_data, data.operator_data, deg=1)

    # Calculate the residual for each row
    data['residual'] = data.apply(lambda dataset:
                                  calc_residual(dataset['meter_data'],
                                                dataset['operator_data'], m, b), axis=1)

    data['quant_error_absolute'] = data.apply(lambda dataset:
                                              calc_error_absolute(dataset['meter_data'],
                                                                            dataset['operator_data']), axis=1)

    data['quant_error_percent'] = data.apply(lambda dataset: calc_error_percent(dataset['meter_data'],
                                                                                dataset['operator_data']), axis=1)

    return data

def determine_relevant_error_ranges(operator, stage, qc_status, strict_discard, time_ave=60, gas_comp_source='km'):
    """ Determine the relevant ranges """
    op_stage = calculate_residuals_and_error(operator, stage, qc_status, strict_discard, time_ave=60, gas_comp_source='km')
    max_residual = op_stage.residual.max()
    min_residual = op_stage.residual.min()
    max_error_percent = op_stage.quant_error_percent.max()
    min_error_percent = op_stage.quant_error_percent.min()
    max_error_absolute = op_stage.quant_error_absolute.max()
    min_error_absolute = op_stage.quant_error_absolute.min()

    relevant_ranges = {
        'operator': operator,
        'stage': stage,
        'max_residual': max_residual,
        'min_residual': min_residual,
        'max_error_percent': max_error_percent,
        'min_error_percent': min_error_percent,
        'max_error_absolute': max_error_absolute,
        'min_error_absolute': min_error_absolute,
    }

    return relevant_ranges
