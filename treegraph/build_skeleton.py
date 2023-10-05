import pandas as pd
import numpy as np
import random
import struct
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from tqdm.autonotebook import tqdm
from treegraph.third_party import shortpath as p2g
from treegraph.downsample import *
from treegraph import common
from pandarallel import pandarallel
from sklearn.decomposition import PCA


def run(self, verbose=False):
    
    columns = self.pc.columns.to_list() + ['node_id']

    # run pandarallel on points grouped by slice_id
    groupby = self.pc.groupby('slice_id')
    pandarallel.initialize(nb_workers=min(24, len(groupby)), progress_bar=verbose)
    try:
        sent_back = groupby.parallel_apply(find_centre, self).values
    except OverflowError:
        if verbose: print('!pandarallel could not initiate progress bars, running without')
        pandarallel.initialize(progress_bar=False)
        sent_back = groupby.parallel_apply(find_centre, self).values

    # create and append clusters and filtered pc
    centres = pd.DataFrame()
    self.pc = pd.DataFrame()
    for x in sent_back:
        if len(x) == 0: continue
        centres = centres.append(x[0])
        self.pc = self.pc.append(x[1])

    # reset index as appended df have common values
    centres.reset_index(inplace=True, drop=True)
    self.pc.reset_index(inplace=True, drop=True)

    if 'node_id' in self.pc.columns: self.pc = self.pc.drop(columns=['node_id'])
    
    # convert binary cluster reference to int
    MAP = {v:i for i, v in enumerate(centres.idx.unique())}
    if 'level_0' in self.pc.columns: self.pc = self.pc.drop(columns='level_0')
    if 'index' in self.pc.columns: self.pc = self.pc.drop(columns='index')
    self.pc.loc[:, 'node_id'] = self.pc.idx.map(MAP)
    centres.loc[:, 'node_id'] = centres.idx.map(MAP)
    
    return centres


def find_centre(dslice, self):
    if len(dslice) < 2: 
        return []
    
    centres = pd.DataFrame()    
    s = dslice.slice_id.unique()[0]
    X = dslice[['x', 'y', 'z']]
    
    group_slice = self.pc[self.pc.slice_id != 0].groupby('slice_id') 
    max_pts_sid = group_slice.apply(lambda x: len(x)).idxmax()
    nn = 10 if s <= max_pts_sid else 5

    results = common.nn_dist(dslice, n_neighbours=nn)
    if type(results) == float:
        return []

    dnn, indices = results
    dists = np.sort(dnn, axis=0)[:,-1]
    idx = np.argsort(np.diff(dists))[::-1]
    knee = dists[idx[1]] if idx[0] == 0 else dists[idx[0]]
    
    mdnn = group_slice.apply(common.mean_dNN, n_neighbours=nn)
    conf_85 = np.nanmean(mdnn) + 1.44 * np.nanstd(mdnn)

    dnn_per_point = np.mean(dnn, axis=1)
    conf95 = np.nanmean(dnn_per_point) + 2 * np.nanstd(dnn_per_point)
    
    def dbscan_cluster(eps):
        dbscan = DBSCAN(eps=eps, min_samples=nn, 
                        algorithm='kd_tree', metric='euclidean', 
                        n_jobs=-1).fit(X)
        labels = np.unique(dbscan.labels_)
        return labels, len(labels[labels >= 0]), dbscan

    labels_knee, c_num_knee, dbscan_knee = dbscan_cluster(knee)
    labels_fix, c_num_fix, dbscan_fix = dbscan_cluster(conf_85)
    labels_conf95, c_num_conf95, dbscan_conf95 = dbscan_cluster(conf95)

    if c_num_knee > 1:   
        # Calculate internal evaluation metrics for the clusters
        clusters = dbscan_knee.fit_predict(X)
        # silhouette score
        knee_s1 = silhouette_score(X, clusters)
        # Calinski-Harabasz index 
        knee_s2 = calinski_harabasz_score(X, clusters)
    
    if c_num_fix > 1:
        # Calculate internal evaluation metrics for the clusters
        clusters = dbscan_fix.fit_predict(X)
        # silhouette score
        fix_s1 = silhouette_score(X, clusters)
        # Calinski-Harabasz index 
        fix_s2 = calinski_harabasz_score(X, clusters)
    

    eps_candidate = [knee, conf_85, conf95]
    cnum = np.array([c_num_knee, c_num_fix, c_num_conf95])
    
    if c_num_knee < 2:  
        eps_ = conf_85 if c_num_fix >= 10 * c_num_knee else knee
    elif c_num_fix < 2: 
        eps_ = knee if c_num_knee >= 10 * c_num_fix else conf_85
    elif c_num_knee <= 10 and s <= max_pts_sid: 
        s1, s2 = max(knee_s1, fix_s1), max(knee_s2, fix_s2)

        if knee_s1 == fix_s1: 
            eps_ = knee if knee_s2 == s2 else conf_85
        elif knee_s1 == s1:
            eps_ = knee
        else:
            eps_ = conf_85
    else:
        eps_ = eps_candidate[np.argmax(cnum)]
        
    dbscan = DBSCAN(eps=eps_, 
                    min_samples=nn, 
                    algorithm='kd_tree', 
                    metric='euclidean',
                    n_jobs=-1).fit(X) 
    dslice.loc[:, 'centre_id'] = dbscan.labels_   
    
    for c in np.unique(dbscan.labels_):
        # working on each cluster
        nvoxel = dslice.loc[dslice.centre_id == c]
        if c == -1:
            dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
        else:
            if len(nvoxel.index) < self.min_pts:
                dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
                continue # required so centre is added after points are deleted
            
            pca = PCA(n_components=3)
            pca.fit(nvoxel[['x', 'y', 'z']].to_numpy())
            ratios = pca.explained_variance_ratio_
            cyl_metric = (ratios[0] / ratios[1] + ratios[0] / ratios[2]) / 2
        
            if (cyl_metric <= 5) & (len(nvoxel) > 100):
                centre_coords = common.CPC(nvoxel[['x', 'y', 'z']]).x
                centre_coords = pd.Series(centre_coords, index=['x', 'y', 'z'])
            else:
                centre_coords = nvoxel[['x', 'y', 'z']].median()

            centres = centres.append(pd.Series({'slice_id':int(s), 
                                                'centre_id':int(c), # id within a cluster
                                                'cx':centre_coords.x, 
                                                'cy':centre_coords.y, 
                                                'cz':centre_coords.z, 
                                                'distance_from_base':nvoxel.distance_from_base.mean(),
                                                'n_points':len(nvoxel),
                                                'idx':struct.pack('ii', int(s), int(c))}),
                                                ignore_index=True)
        
            dslice.loc[(dslice.slice_id == s) & 
                       (dslice.centre_id == c), 'idx'] = struct.pack('ii', int(s), int(c))

    if (len(centres) != 0) & (isinstance(centres, pd.DataFrame)):
        return [centres, dslice] 
    else: 
        return []