import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from treegraph.third_party import shortpath as p2g
from treegraph.downsample import *

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


def generate_distance_graph(self, downsample_cloud=False):
    
    if downsample_cloud:
        downsample(self, downsample_cloud, remove_noise=True)
    
    c = ['x', 'y', 'z']
    self.G = p2g.array_to_graph(self.pc.loc[self.pc.downsample][c] if 'downsample' in self.pc.columns else self.pc[c], 
                                base_id=self.base_location, 
                                kpairs=3, 
                                knn=100, 
                                nbrs_threshold=.2,
                                nbrs_threshold_step=.1,
#                                 graph_threshold=.05
                                )

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
    # colour randomly for vis
    random_c = {sid:i for sid, i in zip(self.pc.slice_id.unique(), 
                                        np.random.choice(self.pc.slice_id.unique(),
                                                         size=len(self.pc.slice_id.unique()),
                                                         replace=False))}
    self.pc.loc[:, 'random_c'] = self.pc.slice_id.map(random_c)
    
    
def skeleton(self, eps=None):

    self.centres = pd.DataFrame(columns=['slice_id', 'centre_id', 'cx', 'cy', 'cz', 'centre_path_dist'])

    print('identifying skeleton points...')
    for i, s in tqdm(enumerate(np.sort(self.pc.slice_id.unique())), total=len(self.pc.slice_id.unique())):

        # separate different slice components e.g. different branches
        dslice = self.pc.loc[self.pc.slice_id == s][['x', 'y', 'z']]
        if len(dslice) < self.min_pts: continue
#         eps_ = self.bin_width[s] / 2.
        dbscan = dbscan_(dslice, eps=eps)
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