# File for methods used in calculating error
# Author: Sahar H. El Abbadi
# Date Created: 2023-03-21
import pathlib

# imports

import pandas as pd
from methods_source import make_overpass_error_df, abbreviate_op_name
from scipy.stats.mstats import ttest_ind


def evaluate_error_profile(operator, stage):
    """Evaluate percent error for different Stanford QC criteria"""

    # Load overpass summary with errors calculated. Note: calc_percent_error returns np.nan for percent error on zero
    # releases
    op_error = make_overpass_error_df(operator=operator, stage=stage)

    # Save op_error in results
    op_ab = abbreviate_op_name(operator)
    op_error.to_csv(pathlib.PurePath('03_results', 'overpass_error', f'{op_ab}_error_summary.csv'))

    # Filter to only include overpasses that pass the operator QC criteria:
    op_error = op_error.query('operator_kept == True')

    ######## Evaluate Error Profile ########

    qc_options = ['strict', 'lax']
    error_values = {}
    error_summary = {}
    p_values = {}

    for qc in qc_options:
        # Evaluate error profile for strict Stanford QC
        op_error_pass_qc = op_error.query(f'stanford_kept_{qc} == True')

        # Only consider overpasses that were quantified by the operator
        op_error_pass_qc = op_error_pass_qc.query('operator_quantified == True')
        pass_qc_mean_error = op_error_pass_qc.percent_error.mean()
        print(f'Evaluate {operator} percent error using Stanford {qc} QC criteria: ')
        print(
            f'-There are {len(op_error_pass_qc)} overpasses that {operator} quantified which pass the Stanford {qc} QC criteria')
        print(
            f'-Percent error for {operator} on overpasses that are included in the {qc} criteria is: {pass_qc_mean_error:.2f}%\n')

        # Save error summary for export
        error_summary[qc] = pass_qc_mean_error

        # Save all error values for calculating p-value
        error_values[qc] = op_error_pass_qc.percent_error

    qc = 'lax_not_strict'
    op_error_pass_lax_fail_strict = op_error.query('stanford_kept_lax == True & stanford_kept_strict == False')

    # Only consider overpasses quantified by operator
    op_error_pass_lax_fail_strict = op_error_pass_lax_fail_strict.query('operator_quantified == True')
    pass_lax_fail_strict_mean_error = op_error_pass_lax_fail_strict.percent_error.mean()
    print(
        f'Evaluate {operator} percent error for overpasses that passed the lax criteria and failed the strict criteria:')
    print(
        f'-There are {len(op_error_pass_lax_fail_strict)} overpasses that {operator} quantified which pass the Stanford lax QC criteria but fail the Stanford strict')
    print(
        f'-{operator} mean percent error for overpasses kept in the lax criteria but discarded in strict is: {pass_lax_fail_strict_mean_error:.2f}%\n')
    error_summary[qc] = pass_lax_fail_strict_mean_error
    error_values[qc] = op_error_pass_lax_fail_strict.percent_error

    ######## Calculate P-Values using T-Test ########

    # Compare strict QC to lax adn to those in lax but not strict
    strict_comparisons = ['lax', 'lax_not_strict']

    # Lax vs Strict QC
    op_error_pass_strict = error_values['strict'].dropna()

    p_values = {}
    for comparison in strict_comparisons:
        print(f'Compare the strict discard criteria with the {comparison} criteria:')

        comparison_error = error_values[comparison].dropna()
        t_stat, p_value = ttest_ind(op_error_pass_strict, comparison_error, equal_var=True)
        p_values[f'strict_v_{comparison}'] = p_value
        if p_value <= 0.05:
            print(
                f'-The calculated p-value of {p_value:0.4f} is less than 0.05, the difference is considered statistically significant.\n')
        else:
            print(
                f'-The calculated p-value of {p_value:0.4f} is greater than 0.05, the difference between the two complete overpass sets is not statistically significant.\n')

    return error_summary, p_values
