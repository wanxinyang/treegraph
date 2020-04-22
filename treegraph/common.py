import os
import struct
import time

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from scipy.spatial.distance import cdist
from scipy.optimize import leastsq
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA 
from sklearn.neighbors import NearestNeighbors

import networkx as nx

import ply_io

#import sys
#sys.path += ['/Users/phil/python/pc2graph', '/Users/phil/python/treegraph/', '/Users/phil/python/treegraph/src']

import pc2graph as p2g


def nn(arr, N):
    
    nbrs = NearestNeighbors(n_neighbors=N+1, algorithm='kd_tree').fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    
    return distances[:, 1]
        
def dbscan_(voxel, eps):

    """
    dbscan implmentation for pc
    fyi eps=.02 for branches
    """
    return DBSCAN(eps=eps, 
                  min_samples=1, 
                  algorithm='kd_tree', 
                  metric='chebyshev',
                  n_jobs=-1).fit(voxel[['x', 'y', 'z']])   

def voxelise(tmp, length):

    binarize = lambda x: struct.pack('i', int((x * 100.) / (length * 100)))

    xb = tmp.x.apply(binarize)
    yb = tmp.y.apply(binarize)
    zb = tmp.z.apply(binarize)
    tmp.loc[:, 'VX'] = xb + yb + zb

    return tmp 

def direction_vector(p1, p2):
    return (p2 - p1) / np.linalg.norm(p2 - p1)

def node_angle_f(self, row):
    
    # create arrays of coordinates 
    a = self.centres[self.centres.node_id == row.child_node][['cx', 'cy', 'cz']].values
    b = self.centres[self.centres.node_id == row.node_id][['cx', 'cy', 'cz']].values
    c = self.centres[self.centres.node_id == row.next_node][['cx', 'cy', 'cz']].values
    
    # normalise distance between coordinate pairs where b is the central coordinate
    ba = a - b
    bc = c - b
  
    # calculate angle between and length of each vector pair 
    angle_pair = lambda ba, bc: np.arccos(np.dot(bc, ba) / (np.linalg.norm(ba) * np.linalg.norm(bc)))

    return angle_pair(bc.T, ba)[0][0]


class treegraph:
    
    def __init__(self, pc, slice_interval=.05, min_pts=10, base_location=None, verbose=False):

        self.pc = pc.copy()        
        self.slice_interval=slice_interval
        self.min_pts = min_pts
        
        if base_location == None:
            self.base_location = self.pc.z.idxmin()
        else:
            self.base_location = base_location
            
        self.verbose = verbose
            
        
def downsample(self, vlength):
    
    """
    Downsamples a point cloud so that there is one point per voxel.
    Points are selected as the point closest to the median xyz value
    
    Parameters
    ----------
    
    pc: pd.DataFrame with x, y, z columns
    vlength: float
    
    
    Returns
    -------
    
    pd.DataFrame with boolean downsample column
    
    """

    self.pc = voxelise(self.pc, vlength)
    groupby = self.pc.groupby('VX')
    self.pc.loc[:, 'mx'] = groupby.x.transform(np.median)
    self.pc.loc[:, 'my'] = groupby.y.transform(np.median)
    self.pc.loc[:, 'mz'] = groupby.z.transform(np.median)
    self.pc.loc[:, 'dist'] = np.linalg.norm(self.pc[['x', 'y', 'z']].values - 
                                            self.pc[['mx', 'my', 'mz']].values, axis=1)
    # need to keep all points for cylinder fitting so when downsampling
    # just adding a column to select by
    self.pc.loc[:, 'downsample'] = False
    self.pc.loc[~self.pc.sort_values(['VX', 'dist']).duplicated('VX'), 'downsample'] = True
    # sorting to base_location index is correct
    self.pc.sort_values('downsample', ascending=False, inplace=True)

    # upadate base_id
    nndist = np.linalg.norm(self.pc.loc[self.base_location][['x', 'y', 'z']] - 
                            self.pc.loc[self.pc.downsample][['x', 'y', 'z']], axis=1)
    self.base_location = nndist.argmin()
    self.pc.reset_index(inplace=True)
    
    
def generate_graph(self, down=False):
    
    c = ['x', 'y', 'z']
    self.G = p2g.array_to_graph(self.pc.loc[self.pc.downsample][c] if 'downsample' in self.pc.columns else self.pc[c], 
                                self.base_location, 
                                3, 
                                100, 
                                self.slice_interval,
                                self.slice_interval / 2)

    self.node_ids, self.distance, path_dict = p2g.extract_path_info(self.G, self.base_location)
    
    # if pc is downsampled to generate graph then reindex downsampled pc 
    # and join distances... 
    if 'downsample' in self.pc.columns:
        dpc = self.pc.loc[self.pc.downsample]
        dpc.reset_index(inplace=True)
        dpc.loc[self.node_ids, 'distance_from_base'] = np.array(list(self.distance))
        self.pc = pd.merge(self.pc, dpc[['VX', 'distance_from_base']], on='VX', how='left')
    # ...or else just join distances to pc
    else:
        self.pc.loc[self.node_ids, 'distance_from_base'] = np.array(list(self.distance))
        
        
def calculate_voxel_length(self, exponent=2, minbin=.005, maxbin=.02):

    # normalise the distance
    self.pc.loc[:, 'normalised_distance'] = self.pc.distance_from_base / self.pc.distance_from_base.max()
    
    # exp function to map smaller bin with increased distance from base
    f, n = np.array([]), 50
    while not self.pc.distance_from_base.max() <= f.sum() < self.pc.distance_from_base.max() * 1.05:
        f = -np.exp(exponent * np.linspace(0, 1, n))
        f = (f - f.min()) / f.ptp() # normalise to multiply by bin width
        f = (((maxbin - minbin) * f) + minbin)
        if f.sum() < self.pc.distance_from_base.max():
            n += 1
        else: n -= 1
    
    self.bin_width = {i: f for i, f in enumerate(f)}

    # generate unique id "slice_id" for bins
    self.pc.loc[:, 'slice_id'] = np.digitize(self.pc.distance_from_base, f.cumsum())
    # colour randomly for vis
    random_c = {sid:i for sid, i in zip(self.pc.slice_id.unique(), 
                                        np.random.choice(self.pc.slice_id.unique(),
                                                         size=len(self.pc.slice_id.unique()),
                                                         replace=False))}
    self.pc.loc[:, 'random_c'] = self.pc.slice_id.map(random_c)
    
    
def skeleton(self, eps=None):

    self.centres = pd.DataFrame(columns=['slice_id', 'centre_id', 'cx', 'cy', 'cz', 'centre_path_dist'])

    for i, s in enumerate(np.sort(self.pc.slice_id.unique())):

        # separate different slice components e.g. different branches
        dslice = self.pc.loc[self.pc.slice_id == s][['x', 'y', 'z']]
        if len(dslice) < self.min_pts: continue
        eps_ = self.bin_width[s] / 2.
        dbscan = dbscan_(dslice, eps=eps_ if eps is None else eps)
        self.pc.loc[dslice.index, 'centre_id'] = dbscan.labels_
        centre_id_max =  dbscan.labels_.max() + 1 # +1 required as dbscan label start at zero

        for c in np.unique(dbscan.labels_):

            # working on each separate branch
            nvoxel = self.pc.loc[dslice.index].loc[self.pc.centre_id == c].copy()
            if len(nvoxel.index) < self.min_pts: continue 
            centre_coords = nvoxel[['x', 'y', 'z']].median()

            self.centres = self.centres.append(pd.Series({'slice_id':int(s), 
                                                          'centre_id':int(c), 
                                                          'cx':centre_coords.x, 
                                                          'cy':centre_coords.y, 
                                                          'cz':centre_coords.z, 
                                                          'centre_path_dist':nvoxel.distance_from_base.mean(),
                                                          'n_points':len(nvoxel)}),
                                               ignore_index=True)

            idx = self.centres.index.values[-1]
            self.pc.loc[(self.pc.slice_id == s) & 
                        (self.pc.centre_id == c), 'node_id'] = idx
    
    self.centres.loc[:, 'node_id'] = self.centres.index
    

def skeleton_path(self, min_dist=1):

    idx = self.centres.centre_path_dist.idxmin() #  
    base_id = self.centres.loc[idx].node_id
    
    edges = pd.DataFrame(columns=['node1', 'node2', 'length'])

    for i, row in enumerate(self.centres.itertuples()):
        
#         if i != 0 and self.verbose and np.isclose(i % (len(self.centres) / 10), 0, atol=1): 
#             print(i // (len(self.centres) / 10) * 10) 

        # first node
        if row.centre_path_dist == self.centres.centre_path_dist.min(): continue
        
        n, nbrs, dist = 2, [], 999
        
        while len(nbrs) == 0 or dist > min_dist:
            # between required incase of small gap in pc
            nbrs = self.centres.loc[self.centres.slice_id.between(row.slice_id - n,
                                                                  row.slice_id - 1)].node_id
            if len(nbrs) > 0:
                nbr_dist = np.linalg.norm(np.array([row.cx, row.cy, row.cz]) - 
                                          self.centres.loc[self.centres.node_id.isin(nbrs)][['cx', 'cy', 'cz']].values, 
                                          axis=1)
                dist = nbr_dist.min()
            n += 1
            if n > 10: 
                print(row)
                break

        nbr_id = nbrs[nbrs.index[np.argmin(nbr_dist)]]        
        edges = edges.append({'node1':int(row.node_id), 
                              'node2':int(nbr_id), 
                              'length':nbr_dist.min()}, ignore_index=True)

    G_skeleton = nx.Graph()
    G_skeleton.add_weighted_edges_from([(int(row.node1), int(row.node2), row.length) 
                                        for row in edges.itertuples()])

    path_distance, self.path_ids = nx.single_source_bellman_ford(G_skeleton, 
                                                                 base_id)
    self.path_distance = {k: v if not isinstance(v, np.ndarray) else v[0] for k, v in path_distance.items()}
    

def attribute_centres(self):
    
    T = time.time()
    
    # nodes before NOT CURRENTLY USED!
    self.centres.loc[:, 'node_before'] = -1
    self.centres.loc[:, 'node_before2'] = -1

#     for node, path in self.path_ids.items():

#         if len(path) < 3:
#             continue
#         elif len(path) == 3:
#             # label the first few nodes
#             self.centres.loc[self.centres.node_id == path[0], 'node_before'] = path[1]
#             self.centres.loc[self.centres.node_id == path[0], 'node_before2'] = path[2]
#             self.centres.loc[self.centres.node_id == path[1], 'node_before'] = path[0]
#             self.centres.loc[self.centres.node_id == path[1], 'node_before2'] = path[2]

#         self.centres.loc[self.centres.node_id == path[-1], 'node_before'] = path[-2]
#         self.centres.loc[self.centres.node_id == path[-1], 'node_before2'] = path[-3]
        
#     print('\tattribute neighbours:', time.time() - T)

    # if node is a tip
    self.centres.loc[:, 'is_tip'] = False
    unique_nodes = np.unique([v for p in self.path_ids.values() for v in p], return_counts=True)
    self.centres.loc[self.centres.node_id.isin(unique_nodes[0][unique_nodes[1] == 1]), 'is_tip'] = True

    if self.verbose: print('\tlocatate tips:', time.time() - T)
    
    # calculate branch lengths and numbers
    self.tip_paths = pd.DataFrame(index=self.centres[self.centres.is_tip].node_id.values, 
                                  columns=['tip2base', 'length', 'nbranch'])
    
    for k, v in self.path_ids.items():
        v = v[::-1]
        if v[0] in self.centres[self.centres.is_tip].node_id.values:
            c1 = self.centres.set_index('node_id').loc[v[:-1]][['cx', 'cy', 'cz']].values
            c2 = self.centres.set_index('node_id').loc[v[1:]][['cx', 'cy', 'cz']].values
            self.tip_paths.loc[self.tip_paths.index == v[0], 'tip2base'] = np.linalg.norm(c1 - c2, axis=1).sum()
            
    if self.verbose: print('\tbranch lengths:', time.time() - T)
            
    self.centres.sort_values('centre_path_dist', inplace=True)
    self.centres.loc[:, 'nbranch'] = -1
    self.centres.loc[:, 'ncyl'] = -1

    for i, row in enumerate(self.tip_paths.sort_values('tip2base', ascending=False).itertuples()):
        
        self.tip_paths.loc[row.Index, 'nbranch'] = i 
        cyls = self.path_ids[row.Index]
        self.centres.loc[(self.centres.node_id.isin(cyls)) & 
                         (self.centres.nbranch == -1), 'nbranch'] = i
        self.centres.loc[self.centres.nbranch == i, 'ncyl'] = np.arange(len(self.centres[self.centres.nbranch == i]))
        v = self.centres.loc[self.centres.nbranch == i].sort_values('ncyl').node_id
        c1 = self.centres.set_index('node_id').loc[v[:-1]][['cx', 'cy', 'cz']].values
        c2 = self.centres.set_index('node_id').loc[v[1:]][['cx', 'cy', 'cz']].values
        self.tip_paths.loc[row.Index, 'length'] = np.linalg.norm(c1 - c2, axis=1).sum()
    
    # reattribute branch numbers starting with the longest
    new_branch_nums = {bn:i for i, bn in enumerate(self.tip_paths.sort_values('length', ascending=False).nbranch)}
    self.tip_paths.loc[:, 'nbranch'] = self.tip_paths.nbranch.map(new_branch_nums)
    self.centres.loc[:, 'nbranch'] = self.centres.nbranch.map(new_branch_nums)
        
    if self.verbose: print('\tbranch and cyl nums:', time.time() - T)

    self.centres.loc[:, 'n_furcation'] = 0        
    self.centres.loc[:, 'parent'] = -1  
    self.centres.loc[:, 'parent_node'] = np.nan
    
    # loop over branch base and identify parent
    for nbranch in self.centres.nbranch.unique():
        
        if nbranch == 0: continue # main branch does not furcate
        furcation_node = -1
        branch_base_idx = self.centres.loc[self.centres.nbranch == nbranch].ncyl.idxmin()
        branch_base_idx = self.centres.loc[branch_base_idx].node_id
        
        for path in self.path_ids.values():    
            if path[-1] == branch_base_idx:
                if len(path) > 2:
                    furcation_node = path[-2]
                else:
                    furcation_node = path[-1]
                self.centres.loc[self.centres.node_id == furcation_node, 'n_furcation'] += 1
                break
        
        if furcation_node != -1:
            parent = self.centres.loc[self.centres.node_id == furcation_node].nbranch.values[0]
            self.centres.loc[self.centres.nbranch == nbranch, 'parent'] = parent
            self.centres.loc[self.centres.nbranch == nbranch, 'parent_node'] = furcation_node
        
    if self.verbose: print('\tidentify parent:', time.time() - T)

    # loop over branches and attribute internode
    self.centres.sort_values(['nbranch', 'ncyl'], inplace=True)
    self.centres.loc[:, 'ninternode'] = -1
    internode_n = 0

    for ix, row in self.centres.iterrows():
        self.centres.loc[self.centres.node_id == row.node_id, 'ninternode'] = internode_n
        if row.n_furcation > 0 or row.is_tip: internode_n += 1
            
    if self.verbose: print('\tidentify internode:', time.time() - T)

    
def cylinder_fit(self):
    
    for c in self.centres.columns:
        if 'sf_radius' in c:
            del self.centres[c]
    
    node_id = self.centres[self.centres.n_points > self.min_pts].sort_values('n_points').node_id.values
    self.radius = self.pc.loc[self.pc.node_id.isin(node_id)].groupby('node_id').apply(cylinderFitting)
    self.centres = pd.merge(self.centres, 
                            pd.DataFrame(self.radius, columns=['sf_radius']), 
                            on='node_id', how='left')
    
#     new_cols = ['sf_radius', 'sf_cx', 'sf_cy', 'sf_cz', 'sf_error']
#     self.centres = self.centres.reindex(columns=list(self.centres.columns) + new_cols)
    
#     for i, row in enumerate(self.centres.itertuples()):

#         v = self.pc.loc[(self.pc.node_id == row.node_id)][['x', 'y', 'z']]

#         # Generates a directional vector along the network path to ensure
#         # that the PCA transformation will not rotate a branch slice
#         # that has diameter larger than slice length.
#         vec = np.vstack([self.centres.loc[self.centres.node_id == row.node_before2][['cx', 'cy', 'cz']].values,
#                          self.centres.loc[self.centres.node_id == row.node_before][['cx', 'cy', 'cz']].values,
#                          self.centres.loc[self.centres.node_id == row.Index][['cx', 'cy', 'cz']].values])

#         center, rad, error = fit_sphere(v.sample(min(100, len(v))), vec)
#         self.centres.loc[self.centres.node_id == row.node_id, 'sf_radius'] = rad
#         self.centres.loc[self.centres.node_id == row.node_id, 'sf_cx'] = center[0]
#         self.centres.loc[self.centres.node_id == row.node_id, 'sf_cy'] = center[1]
#         self.centres.loc[self.centres.node_id == row.node_id, 'sf_cz'] = center[2]
#         self.centres.loc[self.centres.node_id == row.node_id, 'sf_error'] = error

    
def cylinderFitting(xyz):
    
    """
    https://stackoverflow.com/a/44164662/1414831
    
    This is a fitting for a vertical cylinder fitting
    Reference:
    http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf

    xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
    p is initial values of the parameter;
    p[0] = Xc, x coordinate of the cylinder centre
    P[1] = Yc, y coordinate of the cylinder centre
    P[2] = alpha, rotation angle (radian) about the x-axis
    P[3] = beta, rotation angle (radian) about the y-axis
    P[4] = r, radius of the cylinder

    th, threshold for the convergence of the least squares

    """   
    
    def PCA_(xyz):
        pca = PCA(n_components=3, svd_solver='full').fit(xyz)
        return pca, pca.transform(xyz)
    
    def direction(theta, phi):
        '''Return the direction vector of a cylinder defined
        by the spherical coordinates theta and phi.
        '''
        return np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta),
                     np.cos(theta)])
    
    def C(w, Xs):
        '''Calculate the cylinder center given the cylinder direction and 
        a list of data points.
        '''

        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs] 
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        return (np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) /
                np.trace(np.dot(A_hat, A)))
    
    def r(w, Xs):
        '''Calculate the radius given the cylinder direction and a list
        of data points.
        '''
        n = len(Xs)
        P = projection_matrix(w)
        c = C(w, Xs) 
        
        return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)

    def projection_matrix(w):
        '''Return the projection matrix  of a direction w.'''
        return np.identity(3) - np.dot(np.reshape(w, (3,1)), np.reshape(w, (1, 3)))
    
    def calc_A(Ys):
        '''Return the matrix A from a list of Y vectors.'''
        return sum(np.dot(np.reshape(Y, (3,1)), np.reshape(Y, (1, 3))) for Y in Ys)
    
    def calc_A_hat(A, S):
        '''Return the A_hat matrix of A given the skew matrix S'''
        return np.dot(S, np.dot(A, np.transpose(S)))
    
    def skew_matrix(w):
        '''Return the skew matrix of a direction w.'''
        return np.array([[0, -w[2], w[1]],
                         [w[2], 0, -w[0]],
                         [-w[1], w[0], 0]])

    
    try:
        xyz_old = xyz[['x', 'y', 'z']].values
        xyz = xyz[['x', 'y', 'z']].values
        pca, xyz = PCA_(xyz)

        p = np.array([np.median(xyz[:, 0]), # x centre
                      np.median(xyz[:, 1]), # y centre
                      1.5, # alpha, rotation angle (radian) about the x-axis
                      0.03, # beta, rotation angle (radian) about the y-axis
                      xyz[:, 2].ptp() / 2
                      ])

        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]

        fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
        errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

        est_p, success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)

        return est_p[4]
    except Exception as err:
        print(len(xyz), err)
        return -1
    
        
def split_furcation(self, error=.01):
    
    for row in self.centres[self.centres.n_furcation > 0].itertuples():

        c = self.pc[(self.pc.node_id == row.node_id)].copy()
        c.loc[:, 'klabels'] = KMeans(2).fit(c[['x', 'y', 'z']]).labels_
        d = c.groupby('klabels').mean()

        # if new centres are more than error apart
        if nn(d[['x', 'y', 'z']].values, 1).mean() > error:

            for drow in d.itertuples():

                node_id = self.centres.node_id.max() + 1
                nvoxel = c[c.klabels == drow.Index]
                centre_coords = nvoxel[['x', 'y', 'z']].median()
                self.pc.loc[self.pc.index.isin(nvoxel.index), 'node_id'] = node_id

                self.centres = self.centres.append(pd.Series({'slice_id':int(nvoxel.slice_id.mean()), 
                                                              'centre_id':int(nvoxel.centre_id.mean()), 
                                                              'cx':centre_coords.x, 
                                                              'cy':centre_coords.y, 
                                                              'cz':centre_coords.z, 
                                                              'centre_path_dist':nvoxel.distance_from_base.mean(),
                                                              'n_points':len(nvoxel),
                                                              'node_id':node_id}),
                                                   ignore_index=True)

            self.centres = self.centres.loc[self.centres.node_id != row.node_id]

    # if splitting the base node occured - put back together
    if len(self.centres[(self.centres.slice_id == 0) & (self.centres.slice_id == 0)]) > 1:

        print('here;')
        Base = self.centres[(self.centres.slice_id == 0) & (self.centres.slice_id == 0)]
        Mean, Sum = Base.mean(), Base.sum()

        self.centres = self.centres.loc[~self.centres.node_id.isin(Base.node_id.values)]
        self.centres = self.centres.append(pd.Series({'slice_id':0, 
                                                      'centre_id':0, 
                                                      'cx':Mean.cx, 
                                                      'cy':Mean.cy, 
                                                      'cz':Mean.cz, 
                                                      'centre_path_dist':Mean.centre_path_dist,
                                                      'n_points':Sum.n_points,
                                                      'node_id':self.centres.node_id.max() + 1}),
                                           ignore_index=True)

            
def generate_cylinders(self, radius='sf_radius', attribute='point_density'):
    
    self.cyls = pd.DataFrame(columns=['p1', 'p2', 
                                      'sx', 'sy', 'sz', 
                                      'ax', 'ay', 'az', 
                                      'radius', 'length', 'vol', 'surface_area', 'point_density', 
                                      'nbranch', 'ninternode', 'ncyl', 'is_tip'])

    for ix, row in self.centres.sort_values(['nbranch', 'ncyl']).iterrows():
        
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

                length = np.linalg.norm(c1 - c2)  
                if length > 1: continue

                if isinstance(radius, str):
                    # mask nodes as this leads to overestimation
                    is_furcation = self.centres.loc[self.centres.node_id.isin([k1, k2])].n_furcation == 0
                    if np.all(is_furcation) or np.all(is_furcation == False):
                        rad = self.centres.loc[self.centres.node_id.isin([k1, k2])][radius].mean()
                    else:
                        rad = self.centres.loc[self.centres.node_id.isin([k1, k2])].loc[is_furcation][radius].mean()
                elif isinstance(radius, int) or isinstance(radius, float):
                    rad = radius
                else:
                    rad = .05

                volume = np.pi * (rad ** 2) * length
                surface_area = 2 * np.pi * rad * length + 2 * np.pi * rad**2

                direction = direction_vector(c1, c2)
                
                point_density = ((row.n_points + self.centres.loc[self.centres.node_id == k2].n_points.values) / 2) / volume 
                row = row.append(pd.Series(index=['point_density'], data=point_density))

                self.cyls.loc[ix] = [k1, k2, 
                                     c1[0], c1[1], c1[2], 
                                     direction[0], direction[1], direction[2], 
                                     rad, length, volume, surface_area, row.point_density, 
                                     row.nbranch, row.ninternode, row.ncyl, row.is_tip] 
                
def smooth_branches(self):
    
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