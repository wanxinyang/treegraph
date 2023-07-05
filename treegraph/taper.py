import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from pandarallel import pandarallel
from tqdm.autonotebook import tqdm

## wx version 4 (parallel running)
def run(centres, path_ids, tip_radius=None, est_rad='sf_radius', 
        branch_list=None, verbose=False, plot=False):
    '''
    Inputs:
        tip_radius: None or float, unit in mm
    '''
    if 'm_radius' in centres.columns:
        centres = centres.drop(columns=['m_radius'])
    
    if branch_list is None:
        samples = centres.copy()
    else:
        samples = centres[centres.nbranch.isin(branch_list)].copy()

    samples.loc[:, 'm_radius'] = samples[est_rad].copy() * 1e3  # unit in mm

    # estimate mean tip radius from sf_radius if not given
    if tip_radius is None:
        tip_radius = centres[centres.is_tip].sf_radius.mean(skipna=True) * 1e3 # unit in mm

    # run pandarallel on groups of points
    groupby = samples.groupby('nbranch')
    pandarallel.initialize(nb_workers=min(24, len(groupby)), progress_bar=verbose)
    try:
        sent_back = groupby.parallel_apply(radius_correct, centres, path_ids, 
                                           tip_radius, est_rad).values
    except OverflowError:
        print('!pandarallel could not initiate progress bars, running without')
        pandarallel.initialize(progress_bar=False)
        sent_back = groupby.parallel_apply(radius_correct, centres, path_ids, 
                                           tip_radius, est_rad).values

    # create and append clusters and filtered pc
    samples_new = pd.DataFrame()
    for x in sent_back:
        if len(x[0]) == 0: continue
        samples_new = samples_new.append(x[0])

    centres.loc[centres.node_id.isin(samples_new.node_id.values), 
                'm_radius'] = samples_new.m_radius / 1e3  # unit in meter

    return centres


def radius_correct(samples, centres, path_ids, tip_radius, est_rad, 
                   plot=False, xlim=None, ylim=None):
    branch = samples[['nbranch', 'node_id', 'distance_from_base', 'sf_radius', 'm_radius', 'cv']]
    nbranch = np.unique(branch.nbranch)[0]
    if nbranch != 0:
        # ensure child branch radius doesn't exceed twice that of its parent
        parent_node = centres[centres.nbranch == nbranch].parent_node.values[0]
        if len(centres[centres.node_id == parent_node]) != 0: 
            max_radius = centres[centres.node_id == parent_node][est_rad].values[0] * 1e2 # unit in cm
            branch.loc[branch.m_radius > 2*max_radius, 'm_radius'] = 2*max_radius

    # find stem furcation node
    ncyl = centres[centres.ninternode == 0].ncyl.max()
    stem_fur_node = centres[(centres.nbranch == 0) & (centres.ncyl == ncyl)].node_id.values[0]
    # segments from stem furcation node to branch tip
    tip = samples[(samples.nbranch == nbranch)].sort_values('ncyl').node_id.values[-1]
    path = path_ids[tip]
    fur_id = path.index(stem_fur_node)
    path = path[fur_id+1:]
    
    if est_rad == 'sf_radius':
        path = centres.loc[centres.node_id.isin(path)][['node_id', 'distance_from_base', 
                                                        'sf_radius', 'cv']]
        path = path.loc[~np.isnan(path.sf_radius)]
    elif est_rad == 'sm_radius':
        path = centres.loc[centres.node_id.isin(path)][['node_id', 'distance_from_base', 
                                                        'sm_radius', 'cv']]
        path = path.loc[~np.isnan(path.sm_radius)]
    if len(path) < 4:
        samples = samples.loc[~(samples.nbranch == nbranch)]
        centres = centres.loc[~(centres.nbranch == nbranch)]
        return [samples]
    path.loc[:, 'm_radius'] = path[est_rad] * 1e2  # unit in cm

    # segment path into sections and calculate initial upper bound points
    X = np.linspace(path.distance_from_base.min(), path.distance_from_base.max(), 20)
    cut = pd.cut(path.distance_from_base, X)
    bounds = path.groupby(cut).mean().drop(columns=['node_id', est_rad]) 
    bounds.distance_from_base = path.groupby(cut).distance_from_base.max() # distance measured to the end of the branch
    bounds.set_index(np.arange(len(bounds)), inplace=True)
    bounds.loc[:, 'upp'] = bounds.m_radius * 1.2
#         bounds.loc[:, 'low'] = bounds.m_radius * .75
    bounds.loc[bounds.cv>0, 'weight'] = 1. / bounds[bounds.cv>0].cv
    bounds.loc[~(bounds.cv>0), 'weight'] = 0.
    idx = bounds.index.max()
    bounds.loc[idx, 'upp'] = tip_radius
    bounds.loc[idx, 'low'] = tip_radius
    bounds = bounds[~np.isnan(bounds.m_radius)]
    if len(bounds) < 4:
        return [samples]

    # fit an upper bound curve
    L, C = 'upp', 'g'
    f_power = lambda x, a, b, c: a * np.power(x,b) + c  # power 
    f_exp = lambda x, a, b, c, d: a * np.exp(-b * x + c) + d  # exponential
    f_para = lambda x, a, b, c: a + b*x + c*np.power(x,2)  # parabola
    functions = [f_power, f_exp, f_para]

    best_func = None
    best_para = None
    best_err = np.inf

    for func in functions:
        try:
            popt, pcov = optimize.curve_fit(func, bounds.distance_from_base, 
                                            bounds[L], sigma=bounds.weight, maxfev=1000)
            y_pred = func(bounds.distance_from_base, *popt)
            rmse = np.sqrt(np.mean(y_pred - bounds[L])**2)
            if rmse < best_err:
                best_func, best_para, best_err = func, popt, rmse
        except:
            pass

    branch.loc[:, L] = best_func(branch.distance_from_base, *best_para)           
    # branch.loc[branch.upp <= 0, 'upp'] = branch[branch.upp > 0].upp.min()
    branch.loc[branch[L] <= 0, 'upp'] = .0015

    # adjust radii that are NAN or fall beyond upper bound
    branch.loc[np.isnan(branch.m_radius), 'm_radius'] = branch.loc[np.isnan(branch.m_radius)].upp
    branch.loc[branch.m_radius > branch.upp, 'm_radius'] = branch.loc[branch.m_radius > branch.upp].upp  
#         branch.loc[branch.m_radius < branch.low, 'm_radius'] = branch.loc[branch.m_radius < branch.upp].low

    # update centres
    if nbranch == 0:
        samples.loc[samples.node_id.isin(path.node_id.values), 'm_radius'] = branch.m_radius
    else:
        samples.loc[samples.node_id.isin(branch.node_id.values), 'm_radius'] = branch.m_radius 

    if plot:  # plot 
        fig, axs = plt.subplots(1,1,figsize=(8,4))
        ax = [axs]
        ax[0].plot(bounds['distance_from_base'], bounds['upp'], 'go',
                   markerfacecolor='none', markersize=2,
                   label='upper bound candidate pts')
        X = np.linspace(bounds.distance_from_base.min(), bounds.distance_from_base.max(), 20)
        ax[0].plot(X, best_func(X, *best_para), 'g--', linewidth=1, label='fitted upper bound')

        # original radius estimates
        ax[0].plot(branch['distance_from_base'], branch['sf_radius']*1e2, 'r-', 
                   linewidth=1, alpha=0.5, label='Oringal estimates')
        ax[0].plot(branch['distance_from_base'], branch['sf_radius']*1e2, 'ro', 
                   markerfacecolor='none', markersize=1)
        # corrected radius estimates
        ax[0].plot(branch['distance_from_base'], branch['m_radius'], 'b-', 
                   linewidth=1, alpha=0.5, label='Corrected estimates')
        ax[0].plot(branch['distance_from_base'], branch['m_radius'], 'bo', 
                   markerfacecolor='none', markersize=1)

        ax[0].set_xlabel('Distance from base (m)')
        ax[0].set_ylabel('Estimated radius (cm)')
        if xlim is not None:
            ax[0].set_xlim(xlim[0], xlim[1])
            # ax[0].set_xlim([bounds['distance_from_base'].min(), branch['distance_from_base'].max()])
        if ylim is not None:
            ax[0].set_ylim(ylim[0], ylim[1])
        ax[0].set_title(f'Branch {nbranch}')
        ax[0].legend(loc='upper right')

        fig.tight_layout()
        
    return [samples]


def plot_mean_with_CI(centres, branch_list, attr_list, interval=.5,
                      labels=['before correction', 'after correction']):
    """
    This function calculates the mean of given attributes at each distance from base and
    plots the means with a 95% confidence interval (CI).

    Parameters:
    centres (DataFrame): Data of centres with columns ['nbranch', 'distance_from_base'] + attr_list.
    branch_list (list): List of branches to consider.
    attr_list (list): The list of attributes to calculate the mean and CI for.

    Returns:
    None
    """
    # Get a subset of centres belonging to the branches in branch_list
    selected_centres = centres[centres.nbranch.isin(branch_list)].copy()

    # Define bins based on the maximum distance from base, using the defined interval
    bins = np.arange(0, selected_centres.distance_from_base.max() + interval, interval)

    # Assign each centre to a bin based on its distance_from_base
    selected_centres['dfb_bin'] = pd.cut(selected_centres.distance_from_base, bins=bins)

    fig = plt.figure(figsize=(10,5))

    for i, attr in enumerate(attr_list):
        # Group by bin and calculate mean and standard deviation of the selected attribute for each bin
        group_data = selected_centres.groupby(['dfb_bin']).agg({attr: ['mean', 'std']}).reset_index()

        # Rename the columns for better readability
        group_data.columns = ['dfb_bin', f'{attr}_mean', f'{attr}_std']

        # Convert the bin intervals to their midpoints for better representation
        group_data['dfb_bin'] = group_data.dfb_bin.apply(lambda x: x.mid)

        # Calculate upper and lower bounds of the 95% CI for the mean radius
        group_data['upper_bound'] = group_data[f'{attr}_mean'] + 1.96 * group_data[f'{attr}_std']
        group_data['lower_bound'] = group_data[f'{attr}_mean'] - 1.96 * group_data[f'{attr}_std']

        # Plot the mean and CI
        plt.plot(group_data.dfb_bin, group_data[f'{attr}_mean']*100, label=f'{labels[i]}')  # Plot the mean
        plt.fill_between(group_data.dfb_bin, 
                         group_data.lower_bound*100, 
                         group_data.upper_bound*100, alpha=.3)  # Plot 95% CI

    # plot main furcation node location
    stem_fur = centres[centres.ninternode == 0].distance_from_base.max()
    plt.axvline(x=stem_fur, color='green', linestyle='--', alpha=0.3, 
                label='main branching node')

    plt.xlabel('Distance from base (m)', fontsize=12)
    plt.ylabel('Estimated branch radius (cm)', fontsize=12)
    plt.legend()
    plt.show()

    return fig
