# Script for methods used plotting figures
# Author: Sahar H. El Abbadi
# Date Created: 2023-02-24
# Date Last Modified: 2023-02-24
import numpy
# List of methods in this file:
# > plot_parity
# > select_valid_overpasses

# Imports
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import datetime
import math


# %% Function: select_valid_overpasses

# Inputs:
# > operator_report: cleaned operator data, loaded from folder 01_clean_data
# > operator_meter: cleaned metering data, loaded from folder 01_meter_data

def select_valid_overpasses(operator_report, operator_meter):
    """Merge operator report and operator meter dataframes and select overpasses which pass Stanford QC criteria.
    Operator report dataframe should already have 'nan' values for quantification estimates that do not meet operator
    QC criteria. Returns: y_index, x_data, y_data"""
    # Merge the two data frames
    operator_plot = operator_report.merge(operator_meter, on='PerformerExperimentID')

    # Filter based on overpasses that meet Stanford's QC criteria
    operator_plot = operator_plot[(operator_plot['QC: discard - from Stanford'] == 0)]

    # Identify index for overpasses that have valid entries. ie, remove nan values from quantification
    y_index = np.isfinite(operator_plot['FacilityEmissionRate'])

    # Select x data
    x_data = operator_plot['Last 60s (kg/h) - from Stanford']
    y_data = operator_plot['FacilityEmissionRate']

    return y_index, x_data, y_data


# %% Function: plot_parity

# Inputs:
# > operator
# > stage
# > x_data
# > y_data
# > y_index: index for finite values in y_data (ie remove nan)

def plot_parity(operator, stage, x_data, y_data, y_index):
    """Inputs are operator name, stage of analysis, x_data, y_data, y_index (index for finite values in y_data)"""
    # Fit linear regression via least squares with numpy.polyfit
    # m is slope, intercept is b
    m, b = np.polyfit(x_data[y_index], y_data[y_index], deg=1)

    # Calculate R^2 value
    # (using method described here: https://www.askpython.com/python/coefficient-of-determination)
    correlation_matrix = numpy.corrcoef(x_data[y_index], y_data[y_index])
    correlation = correlation_matrix[0,1]
    r2 = correlation ** 2

    # Number of valid overpasses:
    sample_size = len(y_data)

    # Set x and y max values
    # Manually set largest x and y value by changing largest_kgh here to desired value:
    # largest_kgh = 1200

    # Or, determine largest_kgh by calculating largest value in x_data and y_data
    largest_kgh = max(max(x_data), max(y_data))
    largest_kgh = math.ceil(largest_kgh / 100) * 100

    # Create sequence of numbers for plotting linear fit (x)
    x_seq = np.linspace(0, largest_kgh, num=100)

    # Make Figure
    # Initialize layout
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Add linear regression
    plt.plot(x_seq, m * x_seq + b, color='k', lw=2,
             label=f'Best Fit ($y = {m:0.2f}x+{b:0.2f}$, $R^2 =$ {r2:0.4f})')

    # Add x = y line
    plt.plot(x_seq, x_seq, color='k', lw=2, linestyle='--',
             label='y = x')

    # Add data
    plt.plot(x_data, y_data,
             linestyle='none',
             marker='o',
             markersize=5,
             mfc='white',
             label='Stage 1 data')

    # Set title
    plt.title(f'{operator} Stage {stage} Results ({sample_size} measurements)')
    # Set axes

    ax.set(xlim=(0, largest_kgh),
           ylim=(0, largest_kgh),
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
    plt.xlabel('Methane Release Rate (kgh)', fontsize=14)
    plt.ylabel('Reported Release Rate (kgh)', fontsize=14)
    plt.tick_params(direction='in', right=True, top=True)
    plt.tick_params(labelsize=12)
    plt.minorticks_on()
    plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    plt.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)
    plt.grid(False)  # remove grid lines

    # Legend
    plt.legend(facecolor='white')

    # Save figure
    now = datetime.datetime.now()
    save_time = now.strftime("%Y%m%d")
    fig_name = f'{operator}_stage{stage}_{save_time}'
    fig_path = pathlib.PurePath('04_figures', fig_name)
    plt.savefig(fig_path)

    # Show plot
    plt.show()
