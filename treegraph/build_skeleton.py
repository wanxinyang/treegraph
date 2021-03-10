import pandas as pd
import numpy as np
import random
import struct
from sklearn.cluster import DBSCAN
from tqdm.autonotebook import tqdm

from treegraph.third_party import shortpath as p2g
from treegraph.downsample import *

from pandarallel import pandarallel

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


def generate_distance_graph(pc, base_location=None, downsample_cloud=False, knn=100):
    
    if 'downsample' in pc.columns:
        del pc['downsample']
    
    if downsample_cloud:
        pc, base_location = downsample(pc, base_location, downsample_cloud, remove_noise=True)
    
    c = ['x', 'y', 'z']
    G = p2g.array_to_graph(pc.loc[pc.downsample][c] if 'downsample' in pc.columns else pc[c], 
                           base_id=base_location, 
                           kpairs=3, 
                           knn=knn, 
                           nbrs_threshold=.2,
                           nbrs_threshold_step=.1,
#                                 graph_threshold=.05
                            )

    node_ids, distance, path_dict = p2g.extract_path_info(G, base_location)
    
    if 'distance_from_base' in pc.columns:
        del pc['distance_from_base']
    
    # if pc is downsampled to generate graph then reindex downsampled pc 
    # and join distances... 
    if 'downsample' in pc.columns:
        dpc = pc.loc[pc.downsample]
        dpc.reset_index(inplace=True)
        dpc.loc[node_ids, 'distance_from_base'] = np.array(list(distance))
        pc = pd.merge(pc, dpc[['VX', 'distance_from_base']], on='VX', how='left')
    # ...or else just join distances to pc
    else:
        pc.loc[node_ids, 'distance_from_base'] = np.array(list(distance))
        
    return pc
        
        
def calculate_voxel_length(self, exponent=2, minbin=.005, maxbin=.02):

    # normalise the distance
    self.pc.loc[:, 'normalised_distance'] = self.pc.distance_from_base / self.pc.distance_from_base.max()
    
    # exponential function to map smaller bin with increased distance from base
    self.f, n = np.array([]), 50
    while not self.pc.distance_from_base.max() <= self.f.sum() < self.pc.distance_from_base.max() * 1.05:
        self.f = -np.exp(exponent * np.linspace(0, 1, n))
        self.f = (self.f - self.f.min()) / self.f.ptp() # normalise to multiply by bin width
        self.f = (((maxbin - minbin) * self.f) + minbin)
        if self.f.sum() < self.pc.distance_from_base.max():
            n += 1
        else: n -= 1
    
    self.bin_width = {i: f for i, f in enumerate(self.f)}

    # generate unique id "slice_id" for bins
    self.pc.loc[:, 'slice_id'] = np.digitize(self.pc.distance_from_base, self.f.cumsum())


def find_centre(dslice, self, eps):
    
    if len(dslice) < self.min_pts: return []
    
    centres = pd.DataFrame()    
    s = dslice.slice_id.unique()[0]

    # separate different slice components e.g. different branches
    dbscan = dbscan_(dslice[['x', 'y', 'z']], eps=eps)
    dslice.loc[:, 'centre_id'] = dbscan.labels_

    for c in np.unique(dbscan.labels_):

        # working on each separate branch
        nvoxel = dslice.loc[dslice.centre_id == c]
        if len(nvoxel.index) < self.min_pts: 
            dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
            continue # required so centre is added after points are deleted
        centre_coords = nvoxel[['x', 'y', 'z']].median()

        centres = centres.append(pd.Series({'slice_id':int(s), 
                                            'centre_id':int(c), 
                                            'cx':centre_coords.x, 
                                            'cy':centre_coords.y, 
                                            'cz':centre_coords.z, 
                                            'distance_from_base':nvoxel.distance_from_base.mean(),
                                            'n_points':len(nvoxel),
                                            'idx':struct.pack('ii', int(s), int(c))}),
                                           ignore_index=True)
        
        dslice.loc[(dslice.slice_id == s) & 
                   (dslice.centre_id == c), 'idx'] = struct.pack('ii', int(s), int(c))

    if isinstance(centres, pd.DataFrame):
        return [centres, dslice] 
    
def skeleton(self, eps):

    # run pandarallel on groups of points
    groupby = self.pc.groupby('slice_id')
    pandarallel.initialize(nb_workers=min(24, len(groupby)), progress_bar=False)
    sent_back = groupby.parallel_apply(find_centre, self, eps).values

    # create and append clusters and filtered pc
    self.centres = pd.DataFrame()
    self.pc = pd.DataFrame()
    for x in sent_back:
        self.centres = self.centres.append(x[0])
        self.pc = self.pc.append(x[1])

    # reset index as appended df have common values
    self.centres.reset_index(inplace=True)
    self.pc.reset_index(inplace=True)

    # convert binary cluster reference to int
    MAP = {v:i for i, v in enumerate(self.centres.idx.unique())}
    if 'level_0' in self.pc.columns: self.pc = self.pc.drop(columns='level_0')
    if 'index' in self.pc.columns: self.pc = self.pc.drop(columns='index')
    self.pc.loc[:, 'node_id'] = self.pc.idx.map(MAP)
    self.centres.loc[:, 'node_id'] = self.centres.idx.map(MAP)