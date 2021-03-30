import math
import numpy as np
import pandas as pd

from treegraph.common import node_angle_f

from tqdm.autonotebook import tqdm

def end_of_branch(l, axis, start):

    # the following part is adjusted from script in Matlab that does rotation
    u = [0,0,1]
    raxis = [u[1]*axis[2]-axis[1]*u[2],
             u[2]*axis[0]-axis[2]*u[0],
             u[0]*axis[1]-axis[0]*u[1]]

    eucl = (axis[0]**2+axis[1]**2+axis[2]**2)**0.5
    euclr = (raxis[0]**2+raxis[1]**2+raxis[2]**2)**0.5

    for i in range(3):
        raxis[i] /= euclr

    angle = math.acos(np.dot(u, axis) / eucl)

    M = rotation_matrix(raxis, angle)
    p = [0.0, 0.0, l]
    x = (p[0]*M[0][0]+p[1]*M[0][1]+p[2]*M[0][2]) + start[0]
    y = (p[0]*M[1][0]+p[1]*M[1][1]+p[2]*M[1][2]) + start[1]
    z = (p[0]*M[2][0]+p[1]*M[2][1]+p[2]*M[2][2]) + start[2]
        
    return pd.Series([x, y, z])

def rotation_matrix(A, angle):
    '''returns the rotation matrix'''
    c = math.cos(angle)
    s = math.sin(angle)
    R = [[A[0]**2+(1-A[0]**2)*c, A[0]*A[1]*(1-c)-A[2]*s, A[0]*A[2]*(1-c)+A[1]*s],
         [A[0]*A[1]*(1-c)+A[2]*s, A[1]**2+(1-A[1]**2)*c, A[1]*A[2]*(1-c)-A[0]*s],
         [A[0]*A[2]*(1-c)-A[1]*s, A[1]*A[2]*(1-c)+A[0]*s, A[2]**2+(1-A[2]**2)*c]]
    return R

def direction_vector(p1, p2):
    return (p2 - p1) / np.linalg.norm(p2 - p1)


def generate_cylinders(self, radius_value='sf_radius'):
    
    self.cyls = pd.DataFrame(columns=['p1', 'p2', 
                                      'sx', 'sy', 'sz', 
                                      'ax', 'ay', 'az', 
                                      'radius', 'length', 'vol', 'surface_area', 'point_density', 
                                      'nbranch', 'ninternode', 'ncyl', 'is_tip', 'branch_order'])

    for ix, row in tqdm(self.centres.sort_values(['nbranch', 'ncyl']).iterrows(), 
                        total=len(self.centres)):
        
            if row.node_id not in self.path_ids.keys(): continue

            # start from the 
            k_path = self.path_ids[row.node_id][::-1]
            k1 = k_path[0]

            if len(k_path) > 1:
                k2 = k_path[1]
                c1 = np.array([row.cx, 
                               row.cy, 
                               row.cz])

                c2 = np.array([self.centres.loc[self.centres.node_id == k2].cx.values[0],
                               self.centres.loc[self.centres.node_id == k2].cy.values[0],
                               self.centres.loc[self.centres.node_id == k2].cz.values[0]])

                correction = 1
                length = np.linalg.norm(c1 - c2)  
                L = length
###             NEEDS FIXING!!!               
#                 if row.ncyl == 0: # i.e. a furcation
                
#                     # parent branch radius
#                     parent_node = self.centres[self.centres.node_id == row.parent_node]
#                     if isinstance(radius_value, int) or isinstance(radius_value, float):
#                         parent_radius = radius_value
#                     else:
#                         parent_radius = parent_node[radius_value].values[0]
#                     if not np.isnan(parent_radius):  # something weird is happening so skip if NaN
                        
#                         parent_branch = parent_node.nbranch.values[0]  

#                         # branch angle
#                         tip_id = self.centres.loc[(self.centres.nbranch == parent_branch) & 
#                                                   (self.centres.is_tip)].node_id.values[0]
#                         branch_path = np.array(self.path_ids[int(tip_id)], dtype=int)
#                         idx = np.where(branch_path == row.parent_node)[0][0]
#                         next_node = branch_path[idx - 1]

#                         A = node_angle_f(row[['cx', 'cy', 'cz']].values.astype(float),
#                                          parent_node[['cx', 'cy', 'cz']].values,
#                                          self.centres[self.centres.node_id == next_node][['cx', 'cy', 'cz']].values)

#                         distance_to_edge =  (parent_radius / np.sin(A))[0][0]
#                         correction = 1 - (distance_to_edge / length)
#                         # calculate new start point of cylinder based upon radius of parent
#                         row[['cx', 'cy', 'cz']] = end_of_branch(correction, c1, c2).values
                  
                length *= correction
                
#                 if np.isnan(length):
#                     print(c1, c2, k1, k2, correction)
                
                if length < 0: continue
#                 if length > 1: continue # remove overly long branches

                if isinstance(radius_value, str):
        
                    radius = self.centres.loc[self.centres.node_id.isin([k1, k2])][radius_value]

                    # mask NaN radius
                    is_null = np.isnan(radius)
                    is_furcation = self.centres.loc[self.centres.node_id.isin([k1, k2])].n_furcation == 0

                    if np.all(is_null):
                        continue
                    if np.all(~is_null) and np.any(is_furcation):
                        # mask furcation node as this leads to overestimation
                        rad = radius.loc[is_furcation].mean()
                    else:
                        rad = radius.loc[~is_null].mean()
                        
                    if row.node_id == 12183: print('node 12183:', rad)
                        
                elif isinstance(radius_value, int) or isinstance(radius_value, float):
                    rad = radius_value
                else:
                    rad = .05

                volume = np.pi * (rad ** 2) * length
                surface_area = 2 * np.pi * rad * length + 2 * np.pi * rad**2
                
                if np.isnan(rad): print(k1, k2)

                direction = direction_vector(c1, c2)
                
                point_density = ((row.n_points + self.centres.loc[self.centres.node_id == k2].n_points.values) / 2) / volume 
                row = row.append(pd.Series(index=['point_density'], data=point_density))
                
                branch_order = len(self.branch_hierarchy[row.nbranch]['all'])

                self.cyls.loc[ix] = [k1, k2, 
                                     c1[0], c1[1], c1[2], 
                                     direction[0], direction[1], direction[2], 
                                     rad, length, volume, surface_area, row.point_density, 
                                     row.nbranch, row.ninternode, row.ncyl, row.is_tip, branch_order] 

                
def smooth_branches_old(self):
    
    from scipy.optimize import curve_fit
    
    self.centres.loc[:, 'm_radius'] = -1

    def func(x, a, b):
        return b + a ** x

    for nbranch in self.centres.nbranch.unique():

        v = self.centres.loc[self.centres.nbranch == nbranch].sort_values('ncyl')[['ncyl', 'sf_radius']]
        v = v.loc[~np.isnan(v.sf_radius)]

        if len(v) <= 2: # if branch length is <= to two cylinders
            self.centres.loc[self.centres.nbranch == nbranch, 'm_radius'] = self.centres.loc[self.centres.nbranch == nbranch].sf_radius
        else:
            (a, b), pcov = curve_fit(func, v.ncyl, v.sf_radius) # 
            self.centres.loc[self.centres.nbranch == nbranch, 'm_radius'] = self.centres.loc[self.centres.nbranch == nbranch].ncyl.apply(func, args=(a, b))  
            

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
        