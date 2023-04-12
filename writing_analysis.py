# Script for functions used in paper writing

# setup

from methods_source import find_missing_data, load_overpass_summary, classify_confusion_categories

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

def operator_releases_summary_stats(operator):
    """ Function for generating the data used in operator overview paragraphs of Results section"""

    overpasses = load_overpass_summary(operator, stage=1)
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


    # Find largest release given by Stanford that passes all QC
    max_su_info = pass_su_qc.loc[pass_su_qc.release_rate_kgh.idxmax()][relevant_cols]
    print(f'Largest volume overpass for {operator} that passes SU qc:')
    print_overpass_info(max_su_info)

    # Find largest release given by Stanford that passes all QC
    max_info = pass_all_qc.loc[pass_all_qc.release_rate_kgh.idxmax()][relevant_cols]
    print(f'Largest volume overpass for {operator} that passes operator & SU qc:')
    print_overpass_info(max_info)

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

    # Largest false negative:
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
