import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from scipy import optimize


def run(centres, path_ids, tip_radius=.005, Plot=False, verbose=False):
        
    """
    This function is approximately copied from TreeQSM 2.x
    """
    
    from tqdm.autonotebook import tqdm
    from scipy import optimize
    
    centres.loc[:, 'm_radius'] = centres.sf_radius.copy()

    if verbose: print('applying taper function...')
    for nbranch in tqdm(centres.nbranch.unique(), 
                        total=len(centres.nbranch.unique())):
        
        branch = centres.loc[centres.nbranch == nbranch][['distance_from_base', 'm_radius']]
        
        if nbranch != 0:
            # ensure no branch has a larger radius than its parent
            parent_node = centres.loc[(centres.nbranch == nbranch) &
                                           (~np.isnan(centres.parent_node))].parent_node.unique()[0]

            max_radius = centres.loc[centres.node_id == parent_node].m_radius.values[0]
            branch.loc[branch.m_radius > max_radius, 'm_radius'] = max_radius
        
        # cylinders from base to branch tip
        tip = centres.loc[(centres.nbranch == nbranch) & (centres.is_tip)].node_id.values[0]
        path = path_ids[tip]
        path = centres.loc[centres.node_id.isin(path)].sort_values('distance_from_base')[['distance_from_base', 'm_radius']]
        path = path.loc[~np.isnan(path.m_radius)]
        
        if path.distance_from_base.max() < .2: continue
        
        # calculate upper and lower bounds of cylinder radius as
        # a function of distance from base
        X = np.linspace(0, path.distance_from_base.max(), 20)
        cut = pd.cut(path.distance_from_base, X)
        bounds = path.groupby(cut).mean()#.reset_index()
        bounds.distance_from_base = path.groupby(cut).distance_from_base.max() # distance measured to the end of the branch
        bounds.set_index(np.arange(len(bounds)), inplace=True)
        bounds.loc[:, 'upp'] = bounds.m_radius * 1.2
        bounds.loc[:, 'low'] = bounds.m_radius * .75
        bounds.loc[:, 'avg'] = bounds.m_radius
        idx = bounds.index.max()
        bounds.loc[idx, 'upp'] = tip_radius
        bounds.loc[idx, 'low'] = tip_radius
        bounds.loc[idx, 'avg'] = tip_radius
        bounds = bounds.loc[~np.isnan(bounds.distance_from_base)]
        
        if Plot and nbranch == 0:
            ax = bounds.plot.scatter('distance_from_base', 'upp')
            bounds.plot.scatter('distance_from_base', 'low', ax=ax, c='g')   
            bounds.plot.scatter('distance_from_base', 'upp', ax=ax, c='r')
        
        if len(bounds) > 2:
        
            for L, C in zip(['upp', 'low', 'avg'], ['r', 'g', 'b']):

                # weighting polynomial taken from
                # https://stackoverflow.com/a/15193360/1414831 

                def f(x, *p): return np.poly1d(p)(x)

                sigma = np.ones(len(bounds.distance_from_base))
                sigma[-1] = .01

                p, _ = optimize.curve_fit(f, 
                                          bounds.distance_from_base, 
                                          bounds[L], 
                                          (0, 0, 0),
                                          sigma=sigma)

                branch.loc[:, L] = np.poly1d(p)(branch.distance_from_base)
                
                if Plot and nbranch == 0: ax.plot(X, np.poly1d(p)(X), c=C)

            branch.loc[branch.m_radius > branch.upp, 'm_radius'] = branch.loc[branch.m_radius > branch.upp].upp
            branch.loc[branch.m_radius < branch.low, 'm_radius'] = branch.loc[branch.m_radius < branch.low].low
            branch.loc[branch.m_radius < tip_radius, 'm_radius'] = tip_radius
            branch.loc[np.isnan(branch.m_radius), 'm_radius'] = np.poly1d(p)(branch.loc[np.isnan(branch.m_radius)].distance_from_base)
            branch.m_radius = np.abs(branch.m_radius)
            
            if Plot and nbranch == 0:
                branch.plot.scatter('distance_from_base', 'm_radius', s=50, ax=ax, marker='+')
                
        centres.loc[branch.index, 'm_radius'] = branch.m_radius

    return centres