# Sahar making figures for presentation while on airplane and cannot connect to jupyter server

from plot_methods import make_parity_plot, get_parity_data
from methods_source import abbreviate_op_name
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import matplotlib.offsetbox as offsetbox
import math

############# Function to make parity plot #############

def make_f3uel_figures(data, data_description, ax, plot_lim='largest_kgh'):
    """
    :param data: processed data to be plotted
    :param data_description: dictionary with descriptions of data used for plot annotations
    :param ax: subplot ax to plot on
    :param plot_lim: limit of x and y axes
    :return: ax: is the plotted parity charg
    """
    # set font size
    desired_font_size = 22
    # Stage key for blinded status
    stage_description = {
        1: 'Fully blinded results',
        2: 'Unblinded wind',
        3: 'Partially unblinded\n(unblinded releases not included)',
    }
    ############ Data Preparation and Linear Regression ############

    # Load data description
    operator = data_description['operator']
    stage = data_description['stage']
    time_ave = data_description['time_ave']
    gas_comp_source = data_description['gas_comp_source']
    strict_discard = data_description['strict_discard']

    # Set x and y data and error values
    x_data = data.release_rate
    y_data = data.operator_report
    x_error = data.release_sigma * 1.96 # value is sigma, multiply by 1.96 for 95% CI
    y_error = data.operator_sigma

    # Fit linear regression via least squares with numpy.polyfit
    # m is slope, intercept is b
    m, b = np.polyfit(x_data, y_data, deg=1)

    # Calculate R^2 value
    # (using method described here: https://www.askpython.com/python/coefficient-of-determination)
    correlation_matrix = np.corrcoef(x_data, y_data)
    correlation = correlation_matrix[0, 1]
    r2 = correlation ** 2

    # Number of valid overpasses:
    sample_size = len(y_data)

    # Set x and y max values
    # Manually set largest x and y value by changing largest_kgh here to desired value:
    largest_kgh = max(plot_lim)

    if plot_lim == 'largest_kgh':
        # Filter out NA because operations with NA returns NA
        if np.isnan(max(y_error)) == 1:
            y_error.iloc[:] = 0

        largest_kgh = max(max(x_data), max(y_data)) + max(y_error)
        largest_kgh = math.ceil(largest_kgh / 100) * 100

        # set plot_lim:
        plot_lim = [0, largest_kgh]

    # Create sequence of numbers for plotting linear fit (x)
    x_seq = np.linspace(0, largest_kgh, num=100)

    ############ Generate Figure  ############

    # Add linear regression to in put ax
    ax.plot(x_seq, m * x_seq + b, color='k', lw=2,
             label=f'Best Fit, $R^2 =$ {r2:0.2f}\n$y = {m:0.2f}x+{b:0.2f}$')

    # Add parity line
    ax.plot(x_seq, x_seq, color='k', lw=2, linestyle='--',
             label='Parity Line')

    ax.errorbar(x_data, y_data,
                xerr=x_error,
                yerr=y_error,
                linestyle='none',
                mfc='white',
                label=f'n = {sample_size}',
                fmt='o',
                markersize=5)

    # Set title
    # ax.title(f'{operator} Stage {stage} Results ({sample_size} measurements)')
    stage_text = stage_description[stage]
    # ax.annotate(f'{operator}\n {stage_text}', xy=(1, 1), xytext=(-15, -15), fontsize=13,
    #             bbox=dict(boxstyle='square', facecolor='white'))

    # text = f'{operator}\n {stage_text}'
    # ob = offsetbox.AnchoredText(text, loc='upper left')
    # ob.set(alpha=0.8)
    # #TODO figure out how to fix fontsize here
    # # ob.set(fontsize=desired_font_size)
    # ax.add_artist(ob)

    # Set axes
    ax.set(xlim=plot_lim,
           ylim=plot_lim,
           alpha=0.8)

    # Equalize Axes
    ax.set_aspect('equal', adjustable='box')

    # Set axes and background color to white
    ax.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Axes labels
    ax.set_xlabel('Methane Release Rate (kgh)', fontsize=desired_font_size)
    ax.set_ylabel('Reported Release Rate (kgh)', fontsize=desired_font_size)
    ax.tick_params(direction='in', right=True, top=True)

    ax.tick_params(labelsize=(desired_font_size-2))
    ax.minorticks_on()
    ax.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    ax.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)
    ax.grid(False)  # remove grid lines

    # Legend
    # turn off legend and put the fit line in text below the plot
    # ax.legend(facecolor='white', loc='lower right')

    return ax

############# Setup Operators and Stages #############


operators = ['Carbon Mapper', 'GHGSat', 'Kairos', 'MethaneAIR', 'Scientific Aviation']
stage_dictionary = {
    'cm': 3,
    'ghg': 3,
    'kairos': 3,
    'mair': 1,
    'sciav': 1,
}

############# Make Figures #############


for operator in operators:
    op_ab = abbreviate_op_name(operator)
    max_stage = stage_dictionary[op_ab]
    for i in range(1, max_stage+1):
        if op_ab == 'sciav':
            qc_strict = True
        else:
            qc_strict = False
        op_stage_data, op_stage_notes = get_parity_data(operator, stage=i, strict_discard=qc_strict)
        fig, ax = plt.subplots(1, 1, figsize=[10, 10])
        make_f3uel_figures(op_stage_data, op_stage_notes, ax, plot_lim=[0, 2000])
        save_path = pathlib.PurePath('04_figures', '00_presentation_figs', 'f3uel', f'{op_ab}_{i}')
        plt.savefig(save_path)

