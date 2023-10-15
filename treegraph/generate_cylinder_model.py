import numpy as np
import pandas as pd
from treegraph.common import node_angle_f
from tqdm.autonotebook import tqdm

# updated version
def run(self, radius_value='m_radius'):
    self.cyls = pd.DataFrame(columns=['p1', 'p2', 
                              'sx', 'sy', 'sz', 
                              'ax', 'ay', 'az', 
                              'radius', 'length', 'vol', 'surface_area', 'point_density', 
                              'nbranch', 'ninternode', 'ncyl', 'is_tip', 'branch_order', 'branch_order2'])
    
    for ix, row in tqdm(self.centres.sort_values(['nbranch', 'ncyl']).iterrows(), 
                        total=len(self.centres)):
        if row.node_id not in self.path_ids.keys(): continue
        # path from current node to the base node
        k_path = self.path_ids[row.node_id][::-1]
        k1 = k_path[0]

        if len(k_path) > 1:
            k2 = k_path[1]
            # current node coords
            c1 = np.array([row.cx, row.cy, row.cz])
            
            if len(self.centres[self.centres.node_id == k2]) == 0: continue
            # previous node coords
            c2 = np.array([self.centres.loc[self.centres.node_id == k2].cx.values[0],
                           self.centres.loc[self.centres.node_id == k2].cy.values[0],
                           self.centres.loc[self.centres.node_id == k2].cz.values[0]])

            correction = 1
            length = np.linalg.norm(c1 - c2)  
            L = length
            length *= correction

            if length < 0: continue

            if isinstance(radius_value, str):
                # rad = self.centres[self.centres.node_id.isin([k1, k2])][radius_value].mean()
                
                if self.centres[self.centres.node_id == k2].n_furcation.values[0] > 0:  
                    # if prev node is a furcation node
                    rad = self.centres[self.centres.node_id == k1][radius_value].values[0]
                else:
                    rad = self.centres[self.centres.node_id == k2][radius_value].values[0]
                
                # mask NaN radius
                is_null = np.isnan(rad)
                if np.all(is_null):
                    continue

            elif isinstance(radius_value, int) or isinstance(radius_value, float):
                rad = radius_value
            else:
                rad = .05

            volume = np.pi * (rad ** 2) * length
            surface_area = 2 * np.pi * rad * length #+ 2 * np.pi * rad**2

            if np.isnan(rad): print(k1, k2)

            direction = direction_vector(c1, c2)

            point_density = ((row.n_points + self.centres.loc[self.centres.node_id == k2].n_points.values) / 2) / volume 
            row = row.append(pd.Series(index=['point_density'], data=point_density))
            
            # branch section order: +1 whenever after a furcation node
            branch_order = row.norder
            # branch order of complete branch (ending at a tip node) = number of its parent branch 
            branch_order2 = len(self.branch_hierarchy[row.nbranch]['parent_branch'])
            

            self.cyls.loc[ix] = [k1, k2, 
                        c1[0], c1[1], c1[2], 
                        direction[0], direction[1], direction[2], 
                        rad, length, volume, surface_area, row.point_density, 
                        row.nbranch, row.ninternode, int(row.ncyl), row.is_tip, branch_order, branch_order2]


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
