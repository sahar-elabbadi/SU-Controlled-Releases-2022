# Script for methods analyzing results
# Author: Sahar H. El Abbadi
# Date Created: 2023-03-03
# Date Last Modified: 2023-03-03

# List of methods in this file:
# > evaluate_qc

# Imports
import pathlib
import pandas as pd
from plot_methods import abbreviate_op_name


# %% method evaluate_qc

def evaluate_qc(operator, stage, operator_report, operator_meter):
    """Summarize QC criteria applied by operator and Stanford"""

    # Combine operator report and meter data
    combined_df = operator_report.merge(operator_meter, on='PerformerExperimentID')

    op_ab = abbreviate_op_name(operator)

    # Make column with easier name for coding for now.
    combined_df['release_rate_kgh'] = combined_df['Last 60s (kg/h) - from Stanford']

    # Philippine reports if we discard or not (discard = 1, keep = 0). Change this to have 1 if we keep, 0 if we discard
    combined_df['stanford_kept'] = (1 - combined_df['QC: discard - from Stanford'])

    # Make dataframe with all relevant info
    operator_qc = pd.DataFrame()
    operator_qc['overpass_id'] = combined_df.PerformerExperimentID
    operator_qc['zero_release'] = combined_df.release_rate_kgh == 0
    operator_qc['non_zero_release'] = combined_df.release_rate_kgh != 0  # True if we conducted a release
    operator_qc['operator_kept'] = combined_df.OperatorKeep
    operator_qc['stanford_kept'] = combined_df.stanford_kept == 1
    operator_qc['pass_all_qc'] = operator_qc.stanford_kept * operator_qc.operator_kept

    # check if overpass failed both stanford and operator
    check_fail = operator_qc['operator_kept'] + operator_qc['stanford_kept']
    operator_qc['fail_all_qc'] = check_fail == 0

    # Include operator results
    operator_qc['operator_detected'] = combined_df.Detected
    operator_qc['release_rate_kgh'] = combined_df.release_rate_kgh
    operator_qc['operator_quantification'] = combined_df.FacilityEmissionRate

    # Determine how many were QC'ed by Stanford
    total_overpasses = len(operator_qc)
    qc_count_stanford_pass = operator_qc.stanford_kept.sum()
    qc_count_operator_pass = operator_qc.operator_kept.sum()
    qc_count_both_pass = operator_qc.pass_all_qc.sum()
    qc_count_stanford_fail = total_overpasses - qc_count_stanford_pass
    qc_count_operator_fail = total_overpasses - qc_count_operator_pass
    qc_count_both_fail = operator_qc.fail_all_qc.sum()

    qc_summary = pd.DataFrame({
        'operator': op_ab,
        'stage':stage,
        'total_overpasses': total_overpasses,
        'pass_stanford_qc': qc_count_stanford_pass,
        f'pass_operator_qc': qc_count_operator_pass,
        'pass_all_qc': qc_count_both_pass,
        'fail_stanford_qc': qc_count_stanford_fail,
        f'fail_operator_qc': qc_count_operator_fail,
        'fail_all_qc': qc_count_both_fail,
        'fail_stanford_only': qc_count_stanford_fail - qc_count_both_fail,
        'fail_operator_only': qc_count_operator_fail - qc_count_both_fail,
    }, index=[0])

    qc_summary.to_csv(pathlib.PurePath('03_results', 'qc_comparison', f'{op_ab}_{stage}_qc.csv'))
    return qc_summary
