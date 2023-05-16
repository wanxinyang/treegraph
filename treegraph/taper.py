import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from pandarallel import pandarallel
from tqdm.autonotebook import tqdm

## updated method
## wx version 4 (parallel running)
def run(centres, path_ids, tip_radius=None, est_rad='sf_radius', 
        branch_list=None, plot=False, verbose=False):
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


def radius_correct(samples, centres, path_ids, tip_radius, est_rad, plot=False):
    branch = samples[['nbranch', 'node_id', 'distance_from_base', 'sf_radius', 'm_radius', 'cv']]
    nbranch = np.unique(branch.nbranch)[0]
    if nbranch != 0:
        # ensure child branch radius doesn't exceed twice that of its parent
        parent_node = centres[centres.nbranch == nbranch].parent_node.values[0]
        if len(centres[centres.node_id == parent_node]) != 0: 
            max_radius = centres[centres.node_id == parent_node][est_rad].values[0] * 1e3
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
    path.loc[:, 'm_radius'] = path[est_rad] * 1e3  # unit in mm

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

        # original radius
        ax[0].plot(branch['distance_from_base'], branch['sf_radius']*1e3, 'r-', 
                   linewidth=1, alpha=0.5, label='Oringal radius')
        ax[0].plot(branch['distance_from_base'], branch['sf_radius']*1e3, 'ro', 
                   markerfacecolor='none', markersize=1)
        # # corrected radius
        ax[0].plot(branch['distance_from_base'], branch['m_radius'], 'b-', 
                   linewidth=1, alpha=0.5, label='Corrected radius')
        ax[0].plot(branch['distance_from_base'], branch['m_radius'], 'bo', 
                   markerfacecolor='none', markersize=1)

        ax[0].set_xlabel('Distance from base (m)')
        ax[0].set_ylabel('Radius (mm)')
        # ax[1].set_xlim([bounds['distance_from_base'].min(), branch['distance_from_base'].max()])
        ax[0].set_title(f'Branch {nbranch}')
        ax[0].legend(loc='upper right')

        fig.tight_layout()
        
    return [samples]


## wx version 3
# def radius_correct(centres, path_ids, tip_radius=None, est_rad='sf_radius', 
#                    branch_list=None, plot=False):
#     '''
#     Inputs:
#         tip_radius: None or float, unit in mm
#     '''
#     if 'm_radius' in centres.columns:
#         centres = centres.drop(columns=['m_radius'])
    
#     if branch_list is None:
#         samples = centres.copy()
#     else:
#         samples = centres[centres.nbranch.isin(branch_list)].copy()

#     samples.loc[:, 'm_radius'] = samples[est_rad].copy() * 1e3  # unit in mm

#     # find stem furcation node
#     ncyl = centres[centres.ninternode == 0].ncyl.max()
#     stem_fur_node = centres[(centres.nbranch == 0) & (centres.ncyl == ncyl)].node_id.values[0]

#     # estimate mean tip radius from sf_radius if not given
#     if tip_radius is None:
#         tip_radius = centres[centres.is_tip].sf_radius.mean(skipna=True) * 1e3 # unit in mm

#     for nbranch in tqdm(samples.nbranch.unique(), total=len(samples.nbranch.unique())):
#         # print(f'nbranch = {nbranch}')
#         branch = samples.loc[samples.nbranch == nbranch][['node_id', 'distance_from_base', 
#                                                           'm_radius', 'cv']]
#         if nbranch != 0:
#             # ensure child branch radius doesn't exceed twice that of its parent
#             parent_node = centres[centres.nbranch == nbranch].parent_node.values[0]
#             if len(centres[centres.node_id == parent_node]) != 0: 
#                 max_radius = centres[centres.node_id == parent_node][est_rad].values[0] * 1e3
#                 branch.loc[branch.m_radius > 2*max_radius, 'm_radius'] = 2*max_radius

#         # segments from stem furcation node to branch tip
#         tip = samples[(samples.nbranch == nbranch)].sort_values('ncyl').node_id.values[-1]
#         path = path_ids[tip]
#         fur_id = path.index(stem_fur_node)
#         path = path[fur_id+1:]
#         if est_rad == 'sf_radius':
#             path = centres.loc[centres.node_id.isin(path)][['node_id', 'distance_from_base', 
#                                                             'sf_radius', 'cv']]
#             path = path.loc[~np.isnan(path.sf_radius)]
#         elif est_rad == 'sm_radius':
#             path = centres.loc[centres.node_id.isin(path)][['node_id', 'distance_from_base', 
#                                                             'sm_radius', 'cv']]
#             path = path.loc[~np.isnan(path.sm_radius)]
#         if len(path) < 4:
#             samples = samples.loc[~(samples.nbranch == nbranch)]
#             centres = centres.loc[~(centres.nbranch == nbranch)]
#             continue
#         path.loc[:, 'm_radius'] = path[est_rad] * 1e3  # unit in mm

#         # segment path into sections and calculate initial upper bound points
#         X = np.linspace(path.distance_from_base.min(), path.distance_from_base.max(), 20)
#         cut = pd.cut(path.distance_from_base, X)
#         bounds = path.groupby(cut).mean().drop(columns=['node_id', est_rad]) 
#         bounds.distance_from_base = path.groupby(cut).distance_from_base.max() # distance measured to the end of the branch
#         bounds.set_index(np.arange(len(bounds)), inplace=True)
#         bounds.loc[:, 'upp'] = bounds.m_radius * 1.2
# #         bounds.loc[:, 'low'] = bounds.m_radius * .75
#         bounds.loc[bounds.cv>0, 'weight'] = 1. / bounds[bounds.cv>0].cv
#         bounds.loc[~(bounds.cv>0), 'weight'] = 0.
#         idx = bounds.index.max()
#         bounds.loc[idx, 'upp'] = tip_radius
#         bounds.loc[idx, 'low'] = tip_radius
#         bounds = bounds[~np.isnan(bounds.m_radius)]
#         if len(bounds) < 4:
#             continue
     
#         # fit a upper bound curve
#         L, C = 'upp', 'r'
#         f_power = lambda x, a, b, c: a * np.power(x,b) + c  # power 
#         f_exp = lambda x, a, b, c, d: a * np.exp(-b * x + c) + d  # exponential
#         f_para = lambda x, a, b, c: a + b*x + c*np.power(x,2)  # parabola
#         functions = [f_power, f_exp, f_para]

#         best_func = None
#         best_para = None
#         best_err = np.inf

#         for func in functions:
#             try:
#                 popt, pcov = optimize.curve_fit(func, bounds.distance_from_base, 
#                                                 bounds[L], sigma=bounds.weight, maxfev=1000)
#                 y_pred = func(bounds.distance_from_base, *popt)
#                 rmse = np.sqrt(np.mean(y_pred - bounds[L])**2)
#                 if rmse < best_err:
#                     best_func, best_para, best_err = func, popt, rmse
#             except:
#                 pass

#         branch.loc[:, L] = best_func(branch.distance_from_base, *best_para)           
#         # branch.loc[branch.upp <= 0, 'upp'] = branch[branch.upp > 0].upp.min()
#         branch.loc[branch[L] <= 0, 'upp'] = .0015

#         if plot:
#             fig, axs = plt.subplots(1,2,figsize=(10,4))
#             ax = axs.flatten()
#             # plot fitted upper boundary
#             ax[0].scatter(bounds['distance_from_base'], bounds['upp'], c='r',
#                           label='pts to fit upp bound')
#             X = np.linspace(bounds.distance_from_base.min(), bounds.distance_from_base.max(), 20)
#             ax[0].plot(X, func(X, *popt), c=C, label='upper bound')
#             ax[0].set_xlabel('Distance from base (m)')
#             ax[0].set_ylabel('Radius (mm)')
#             ax[0].set_title('Bound fitting')
#             ax[0].legend(loc='upper right')
        
#         # adjust radii that are NAN or fall beyond upper bound
#         branch.loc[np.isnan(branch.m_radius), 'm_radius'] = branch.loc[np.isnan(branch.m_radius)].upp
#         branch.loc[branch.m_radius > branch.upp, 'm_radius'] = branch.loc[branch.m_radius > branch.upp].upp  
# #         branch.loc[branch.m_radius < branch.low, 'm_radius'] = branch.loc[branch.m_radius < branch.upp].low

#         # update centres
#         if nbranch == 0:
#             samples.loc[samples.node_id.isin(path.node_id.values), 'm_radius'] = branch.m_radius
#         else:
#             samples.loc[samples.node_id.isin(branch.node_id.values), 'm_radius'] = branch.m_radius 
        
#         if plot:  # plot 
#             ax[1].plot(branch['distance_from_base'], branch['upp'], c='r', label='upper bound')                 
#             adj_nids = branch[branch.m_radius >= branch.upp].node_id.values
#             filt = path[path.node_id.isin(adj_nids)]
# #             adj = branch[branch.node_id.isin(adj_nids)]
# #             ax[1].scatter(adj['distance_from_base'], adj['m_radius'], s=10, c='r', 
# #                           marker='^', label='adjusted radius')
#             ax[1].scatter(filt['distance_from_base'], filt['m_radius'], s=10, c='orange', 
#                           marker='+', label='original radius')
#             ax[1].scatter(branch['distance_from_base'], branch['m_radius'], s=10, c='b', 
#                           marker='o', label='constrained radius')
#             ax[1].set_xlabel('Distance from base (m)')
#             ax[1].set_ylabel('Radius (mm)')
#             ax[1].set_title(f'Constrain: branch {nbranch}')
#             ax[1].legend(loc='upper right')
            
#             fig.tight_layout()

#     centres.loc[centres.node_id.isin(samples.node_id.values), 'm_radius'] = samples.m_radius / 1e3  # unit in meter

#     return centres


## wx version 2
# def radius_correct(centres, path_ids, tip_radius=None, est_rad='sf_radius', 
#                    branch_list=None, plot=False):
#     '''
#     Inputs:
#         tip_radius: None or float, unit in mm
#     '''
#     if branch_list is None:
#         samples = centres.copy()
#     else:
#         samples = centres[centres.nbranch.isin(branch_list)].copy()
#     samples.loc[:, 'm_radius'] = samples[est_rad].copy() * 1e3  # unit in mm
    
#     # find stem furcation node
#     ncyl = centres[centres.ninternode == 0].ncyl.max()
#     stem_fur_node = centres[(centres.nbranch == 0) & (centres.ncyl == ncyl)].node_id.values[0]
    
#     # estimate mean tip radius from sf_radius if not given
#     if tip_radius is None:
#         tip_radius = centres[centres.is_tip].sf_radius.mean(skipna=True) * 1e3 # unit in mm
    
#     for nbranch in tqdm(samples.nbranch.unique(), total=len(samples.nbranch.unique())):
#         # print(f'nbranch = {nbranch}')
#         branch = samples.loc[samples.nbranch == nbranch][['node_id', 'distance_from_base', 'm_radius']]
        
#         if nbranch != 0:
#             # ensure child branch radius doesn't exceed twice that of its parent
#             parent_node = centres[centres.nbranch == nbranch].parent_node.values[0]
#             if len(centres[centres.node_id == parent_node]) != 0: 
#                 max_radius = centres[centres.node_id == parent_node][est_rad].values[0] * 1e3
#                 branch.loc[branch.m_radius > 2*max_radius, 'm_radius'] = 2*max_radius

#         # segments from stem furcation node to branch tip
#         tip = samples[(samples.nbranch == nbranch)].sort_values('ncyl').node_id.values[-1]
#         path = path_ids[tip]
#         fur_id = path.index(stem_fur_node)
#         path = path[fur_id+1:]
#         if est_rad == 'sf_radius':
#             path = centres.loc[centres.node_id.isin(path)][['node_id', 'distance_from_base', 'sf_radius']]
#             path = path.loc[~np.isnan(path.sf_radius)]
#         elif est_rad == 'sf_radius_conv':
#             path = centres.loc[centres.node_id.isin(path)][['node_id', 'distance_from_base', 'sf_radius_conv']]
#             path = path.loc[~np.isnan(path.sf_radius_conv)]
#         if len(path) < 4:
#             samples = samples.loc[~(samples.nbranch == nbranch)]
#             centres = centres.loc[~(centres.nbranch == nbranch)]
#             continue
#         path.loc[:, 'm_radius'] = path[est_rad] * 1e3  # unit in mm
       
#         # segment path into sections and calculate initial upper bound points
#         X = np.linspace(path.distance_from_base.min(), path.distance_from_base.max(), 20)
#         cut = pd.cut(path.distance_from_base, X)
#         bounds = path.groupby(cut).mean().drop(columns=['node_id', est_rad]) 
#         bounds.distance_from_base = path.groupby(cut).distance_from_base.max() # distance measured to the end of the branch
#         bounds.set_index(np.arange(len(bounds)), inplace=True)
#         bounds.loc[:, 'upp'] = bounds.m_radius * 1.2
#         idx = bounds.index.max()
#         bounds.loc[idx, 'upp'] = tip_radius
#         bounds = bounds[~np.isnan(bounds.m_radius)]
#         if len(bounds) < 4:
#             continue

#         # fit a upper bound curve
#         try:
#             func = lambda x, a, b, c: a * np.exp(-b * x + c) + d  # exponential
#             popt, pcov = optimize.curve_fit(func, bounds.distance_from_base, bounds.upp, maxfev=1000)
#         except:
#             func = lambda x, a, b, c: a + b*x + c*np.power(x,2)  # parabola
#             popt, pcov = optimize.curve_fit(func, bounds.distance_from_base, bounds.upp, maxfev=1000)
#         branch.loc[:, 'upp'] = func(branch.distance_from_base, *popt)        
#         # branch.loc[branch.upp <= 0, 'upp'] = branch[branch.upp > 0].upp.min()
#         branch.loc[branch.upp <= 0, 'upp'] = .0015
        
#         # correct radii are NAN or fall beyond upper bound
#         branch.loc[np.isnan(branch.m_radius), 'm_radius'] = branch.loc[np.isnan(branch.m_radius)].upp
#         branch.loc[branch.m_radius > branch.upp, 'm_radius'] = branch.loc[branch.m_radius > branch.upp].upp    

#         # update centres
#         if nbranch == 0:
#             samples.loc[samples.node_id.isin(path.node_id.values), 'm_radius'] = branch.m_radius
#         else:
#             samples.loc[samples.node_id.isin(branch.node_id.values), 'm_radius'] = branch.m_radius 
        
#         if plot:
#             path.loc[:, 'upp'] = func(path.distance_from_base, *popt)
#             ax2 = path.plot.line(x='distance_from_base', y='upp', c='g', label='upper bound')
#             branch.plot.scatter(x='distance_from_base', y='m_radius', s=10, c='r', label='branch nodes', ax=ax2)
#             path.plot.scatter(x='distance_from_base', y='m_radius', s=3, c='grey', ax=ax2, label='path nodes')
#             bounds.plot.scatter(x='distance_from_base', y='upp', s=5, c='g', ax=ax2, label='pts to fit bound')
#             ax2.set_title(f'Branch {nbranch}')
    
#     centres.loc[centres.node_id.isin(samples.node_id.values), 'm_radius'] = samples.m_radius / 1e3  # unit in meter

#     return centres


## wx version 1
# def radius_correct(centres, path_ids, tip_radius=None, plot=False, branch_list=None):
#     '''
#     Inputs:
#         tip_radius: None or float, unit in mm
#     '''
#     if branch_list is None:
#         samples = centres.copy()
#     else:
#         samples = centres[centres.nbranch.isin(branch_list)].copy()
#     samples.loc[:, 'm_radius'] = samples.sf_radius_conv.copy() * 1e3  # unit in mm
    
#     # find stem furcation node
#     ncyl = centres[centres.ninternode == 0].ncyl.max()
#     stem_fur_node = centres[(centres.nbranch == 0) & (centres.ncyl == ncyl)].node_id.values[0]
    
#     # estimate mean tip radius from sf_radius_conv if not given
#     if tip_radius is None:
#         tip_radius = samples[samples.is_tip].sf_radius_conv.mean() * 1e3 # unit in mm
    
#     for nbranch in tqdm(samples.nbranch.unique(), total=len(samples.nbranch.unique())):
#         # print(f'nbranch = {nbranch}')
#         branch = samples.loc[samples.nbranch == nbranch][['node_id', 'distance_from_base', 'm_radius']]
        
#         if nbranch != 0:
#             # ensure no branch has a larger radius than its parent
#             parent_node = centres[centres.nbranch == nbranch].parent_node.values[0]
#             if len(centres[centres.node_id == parent_node]) != 0: 
#                 max_radius = centres[centres.node_id == parent_node].sf_radius_conv.values[0] * 1e3
#                 branch.loc[branch.m_radius > max_radius, 'm_radius'] = max_radius

#         # segments from stem furcation node to branch tip
#         tip = samples[(samples.nbranch == nbranch)].sort_values('ncyl').node_id.values[-1]
#         path = path_ids[tip]
#         fur_id = path.index(stem_fur_node)
#         path = path[fur_id+1:]
#         path = centres.loc[centres.node_id.isin(path)][['node_id', 'distance_from_base', 'sf_radius_conv']]
#         path = path.loc[~np.isnan(path.sf_radius_conv)]
#         path.loc[:, 'm_radius'] = path.sf_radius_conv * 1e3  # unit in mm
       
#         # segment path into sections and calculate initial upper bound points
#         X = np.linspace(path.distance_from_base.min(), path.distance_from_base.max(), 20)
#         cut = pd.cut(path.distance_from_base, X)
#         bounds = path.groupby(cut).mean().drop(columns=['node_id', 'sf_radius_conv']) 
#         bounds.distance_from_base = path.groupby(cut).distance_from_base.max() # distance measured to the end of the branch
#         bounds.set_index(np.arange(len(bounds)), inplace=True)
#         bounds.loc[:, 'upp'] = bounds.m_radius * 1.2
#         idx = bounds.index.max()
#         bounds.loc[idx, 'upp'] = tip_radius
#         bounds = bounds[~np.isnan(bounds.m_radius)]

#         # fit a upper bound curve
#         try:
#             func = lambda x, a, b, c: a * np.exp(-b * x + c) + d  # exponential
#             popt, pcov = optimize.curve_fit(func, bounds.distance_from_base, bounds.upp, maxfev=1000)
#         except:
#             func = lambda x, a, b, c: a + b*x + c*np.power(x,2)  # parabola
#             popt, pcov = optimize.curve_fit(func, bounds.distance_from_base, bounds.upp, maxfev=1000)
#         branch.loc[:, 'upp'] = func(branch.distance_from_base, *popt)   
#         # branch.loc[branch.upp <= 0, 'upp'] = branch[branch.upp > 0].upp.min()
#         branch.loc[branch.upp <= 0, 'upp'] = .0015     

#         # correct radii fall beyond upper bound
#         branch.loc[branch.m_radius > branch.upp, 'm_radius'] = branch.loc[branch.m_radius > branch.upp].upp    

#         # update centres
#         if nbranch == 0:
#             samples.loc[samples.node_id.isin(path.node_id.values), 'm_radius'] = branch.m_radius
#         else:
#             samples.loc[samples.node_id.isin(branch.node_id.values), 'm_radius'] = branch.m_radius 
        
#         if plot:
#             path.loc[:, 'upp'] = func(path.distance_from_base, *popt)
#             ax2 = path.plot.line(x='distance_from_base', y='upp', c='g', label='upper bound')
#             branch.plot.scatter(x='distance_from_base', y='m_radius', s=10, c='r', label='branch nodes', ax=ax2)
#             path.plot.scatter(x='distance_from_base', y='m_radius', s=3, c='grey', ax=ax2, label='path nodes')
#             bounds.plot.scatter(x='distance_from_base', y='upp', s=5, c='g', ax=ax2, label='pts to fit bound')
#             ax2.set_title(f'Branch {nbranch}')
    
#     centres.loc[centres.node_id.isin(samples.node_id.values), 'm_radius'] = samples.m_radius / 1e3  # unit in meter

#     return centres




# # previous method
# def run(centres, path_ids, tip_radius=None, Plot=False, verbose=False):
        
#     """
#     This function is approximately copied from TreeQSM 2.x
#     """
    
#     centres.loc[:, 'm_radius'] = centres.sf_radius.copy()

#     if verbose: print('applying taper function...')
#     for nbranch in tqdm(centres.nbranch.unique(), 
#                         total=len(centres.nbranch.unique())):
        
#         branch = centres.loc[centres.nbranch == nbranch][['distance_from_base', 'm_radius']]
        
#         if nbranch != 0:
#             # ensure no branch has a larger radius than its parent
#             parent_node = centres.loc[(centres.nbranch == nbranch) &
#                                            (~np.isnan(centres.parent_node))].parent_node.unique()[0]

#             max_radius = centres.loc[centres.node_id == parent_node].m_radius.values[0]
#             branch.loc[branch.m_radius > max_radius, 'm_radius'] = max_radius
        
#         # cylinders from base to branch tip
#         tip = centres.loc[(centres.nbranch == nbranch) & (centres.is_tip)].node_id.values[0]
#         path = path_ids[tip]
#         path = centres.loc[centres.node_id.isin(path)].sort_values('distance_from_base')[['distance_from_base', 'm_radius']]
#         path = path.loc[~np.isnan(path.m_radius)]
        
#         if path.distance_from_base.max() < .2: continue
        
#         # calculate upper and lower bounds of cylinder radius as
#         # a function of distance from base
#         X = np.linspace(0, path.distance_from_base.max(), 20)
#         cut = pd.cut(path.distance_from_base, X)
#         bounds = path.groupby(cut).mean()#.reset_index()
#         bounds.distance_from_base = path.groupby(cut).distance_from_base.max() # distance measured to the end of the branch
#         bounds.set_index(np.arange(len(bounds)), inplace=True)
#         bounds.loc[:, 'upp'] = bounds.m_radius * 1.2
#         bounds.loc[:, 'low'] = bounds.m_radius * .75
#         bounds.loc[:, 'avg'] = bounds.m_radius
#         idx = bounds.index.max()
#         if tip_radius is not None:
#             bounds.loc[idx, 'upp'] = tip_radius
#             bounds.loc[idx, 'low'] = tip_radius
#             bounds.loc[idx, 'avg'] = tip_radius
#         bounds = bounds.loc[~np.isnan(bounds.distance_from_base)]
        
#         if Plot and nbranch == 0:
#             ax = bounds.plot.scatter('distance_from_base', 'upp')
#             bounds.plot.scatter('distance_from_base', 'low', ax=ax, c='g')   
#             bounds.plot.scatter('distance_from_base', 'upp', ax=ax, c='r')
        
#         if len(bounds) > 2:
        
#             for L, C in zip(['upp', 'low', 'avg'], ['r', 'g', 'b']):

#                 # weighting polynomial taken from
#                 # https://stackoverflow.com/a/15193360/1414831 

#                 def f(x, *p): return np.poly1d(p)(x)
                
#                 sigma = np.ones(len(bounds.distance_from_base))
#                 if tip_radius is not None: sigma[-1] = .01

#                 p, _ = optimize.curve_fit(f, 
#                                           bounds.distance_from_base, 
#                                           bounds[L], 
#                                           (0, 0, 0),
#                                           sigma=sigma)

#                 branch.loc[:, L] = np.poly1d(p)(branch.distance_from_base)
                
#                 if Plot and nbranch == 0: ax.plot(X, np.poly1d(p)(X), c=C)

#             branch.loc[branch.m_radius > branch.upp, 'm_radius'] = branch.loc[branch.m_radius > branch.upp].upp
#             branch.loc[branch.m_radius < branch.low, 'm_radius'] = branch.loc[branch.m_radius < branch.low].low
#             if tip_radius is not None: branch.loc[branch.m_radius < tip_radius, 'm_radius'] = tip_radius
#             branch.loc[np.isnan(branch.m_radius), 'm_radius'] = np.poly1d(p)(branch.loc[np.isnan(branch.m_radius)].distance_from_base)
#             branch.m_radius = np.abs(branch.m_radius)
            
#             if Plot and nbranch == 0:
#                 branch.plot.scatter('distance_from_base', 'm_radius', s=50, ax=ax, marker='+')
                
#         centres.loc[branch.index, 'm_radius'] = branch.m_radius

#     return centres
