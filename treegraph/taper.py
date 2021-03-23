import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from scipy import optimize


def smooth_branches(self, tip_radius=.005, plott=False):
        
    """
    This function is approximately copied from TreeQSM 2.x
    """
    
    from tqdm.autonotebook import tqdm
    from scipy import optimize
    
    self.centres.loc[:, 'm_radius'] = self.centres.sf_radius.copy()

    for nbranch in tqdm(self.centres.nbranch.unique(), 
                        total=len(self.centres.nbranch.unique())):
        
        branch = self.centres.loc[self.centres.nbranch == nbranch][['distance_from_base', 'm_radius']]
        
        if nbranch != 0:
            # ensure no branch has a larger radius than its parent
            parent_node = self.centres.loc[(self.centres.nbranch == nbranch) &
                                           (~np.isnan(self.centres.parent_node))].parent_node.unique()[0]

            max_radius = self.centres.loc[self.centres.node_id == parent_node].m_radius.values[0]
            branch.loc[branch.m_radius > max_radius, 'm_radius'] = max_radius
        
        # cylinders from base to branch tip
        tip = self.centres.loc[(self.centres.nbranch == nbranch) & (self.centres.is_tip)].node_id.values[0]
        path = self.path_ids[tip]
        path = self.centres.loc[self.centres.node_id.isin(path)].sort_values('distance_from_base')[['distance_from_base', 'm_radius']]
        path = path.loc[~np.isnan(path.m_radius)]
        
        # calculate upper and lower bounds of cylinder radius as
        # a function of distance from base
        X = np.linspace(0, path.distance_from_base.max(), 20)
        cut = pd.cut(path.distance_from_base, X)
        bounds = path.groupby(cut).mean()#.reset_index()
        bounds.set_index(np.arange(len(bounds)), inplace=True)
        bounds.loc[:, 'upp'] = bounds.m_radius * 1.1
        bounds.loc[:, 'low'] = bounds.m_radius * .75
        bounds.loc[:, 'avg'] = bounds.m_radius
        idx = bounds.index.max() + 1 # add 
        bounds.loc[idx, 'upp'] = tip_radius
        bounds.loc[idx, 'low'] = tip_radius
        bounds.loc[idx, 'avg'] = tip_radius
        bounds = bounds.loc[~np.isnan(bounds.distance_from_base)]
        
        for L in ['upp', 'low']:
            
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
#                 p = np.polyfit(bounds.distance_from_base, bounds[L], 3)
#                 p = np.poly1d(p)

            branch.loc[:, L] = np.poly1d(p)(branch.distance_from_base)

        branch.m_radius = np.abs(branch.m_radius)
        branch.loc[branch.m_radius > branch.upp, 'm_radius'] = branch.loc[branch.m_radius > branch.upp].upp
        branch.loc[branch.m_radius < branch.low, 'm_radius'] = branch.loc[branch.m_radius < branch.low].low
        branch.loc[branch.m_radius < tip_radius, 'm_radius'] = tip_radius
        branch.loc[np.isnan(branch.m_radius), 'm_radius'] = np.poly1d(p)(branch.loc[np.isnan(branch.m_radius)].distance_from_base)

        self.centres.loc[branch.index, 'm_radius'] = branch.m_radius

