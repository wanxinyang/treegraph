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
    # slice that has largest number of points
    max_pts_sid = group_slice.apply(lambda x: len(x)).idxmax()
    if s <= max_pts_sid: # lower part of the tree
        nn = 10
    else:  # upper part of the tree
        nn = 5
        
    # eps candidate 1: largest jump in the k-nearest neighbour distance plot
    results = common.nn_dist(dslice, n_neighbours=nn)
    if type(results) == float:
        return []
    else:
        dnn, indices = results
    # distance to the nn-th nearest neighbour
    dists = np.sort(dnn, axis=0)[:,-1]
    # index of the point at the largest gap
    idx = np.argsort(np.diff(dists))[::-1]
    if idx[0] == 0:
        knee = dists[idx[1]]
    else:
        knee = dists[idx[0]]
        
    dbscan_knee = DBSCAN(eps=knee, 
                        min_samples=nn, 
                        algorithm='kd_tree', 
                        metric='euclidean',
                        n_jobs=-1).fit(X) 
    labels_knee = np.unique(dbscan_knee.labels_) 
    c_num_knee = len(labels_knee[labels_knee >= 0])

    if c_num_knee > 1:   
        # Calculate internal evaluation metrics for the clusters
        clusters = dbscan_knee.fit_predict(X)
        # silhouette score
        knee_s1 = silhouette_score(X, clusters)
        # Calinski-Harabasz index 
        knee_s2 = calinski_harabasz_score(X, clusters)
    
   
    # eps candidate 2: conf_85 for mean dNN of points in all slices
    mdnn = group_slice.apply(common.mean_dNN, n_neighbours=nn)
    conf_85_whole_tree = np.nanmean(mdnn) + 1.44 * np.nanstd(mdnn)
    fix = conf_85_whole_tree
    
    dbscan_fix = DBSCAN(eps=fix, 
                        min_samples=nn, 
                        algorithm='kd_tree', 
                        metric='euclidean',
                        n_jobs=-1).fit(X) 
    labels_fix = np.unique(dbscan_fix.labels_) 
    c_num_fix = len(labels_fix[labels_fix >= 0])

    if c_num_fix > 1:
        # Calculate internal evaluation metrics for the clusters
        clusters = dbscan_fix.fit_predict(X)
        # silhouette score
        fix_s1 = silhouette_score(X, clusters)
        # Calinski-Harabasz index 
        fix_s2 = calinski_harabasz_score(X, clusters)
    
    # eps candidate 3: conf_95 for mean dNN of points in this slice
    dnn_per_point = np.mean(dnn, axis=1)
    conf95 = np.nanmean(dnn_per_point) + 2 * np.nanstd(dnn_per_point)
    
    dbscan_fix = DBSCAN(eps=conf95, 
                        min_samples=nn, 
                        algorithm='kd_tree', 
                        metric='euclidean',
                        n_jobs=-1).fit(X) 
    labels_fix = np.unique(dbscan_fix.labels_) 
    c_num_conf95 = len(labels_fix[labels_fix >= 0])
        
    eps_candidate = [knee, fix, conf95]
    cnum = np.array([c_num_knee, c_num_fix, c_num_conf95])
    
    ## determine eps based on internal evaluation metric
    # if no score due to cluster<2
    if c_num_knee < 2:  # eps=knee DBSCAN cluster number = 1
        if c_num_fix >= (10 * c_num_knee): # eps=knee underestimates too much
            eps_ = fix  
        else:
            eps_ = knee
    elif c_num_fix < 2: # eps=fix DBSCAN cluster number = 1
        if c_num_knee >= (10 * c_num_fix): # esp=fix underestimates too much
            eps_ = knee
        else:
            eps_ = fix
    # cluster number ranges from 2 to 10 and at lower part of the tree 
    elif (c_num_knee <= 10) & (s <= max_pts_sid): 
        s1 = max(knee_s1, fix_s1)
        s2 = max(knee_s2, fix_s2)

        if knee_s1 == fix_s1: # two candidates have same silhouette score 
            if knee_s2 == s2: eps_ = knee # use knee if it has higher Calinski-Harabasz index
            else: eps_ = fix
        elif knee_s1 == s1: # use knee if it has higher silhouette score 
            eps_ = knee
        else: # use fix if it has higher silhouette score 
            eps_ = fix
    # more than 10 clusters or upper part of the tree
    else:
        eps_idx = np.argmax(cnum)
        eps_ = eps_candidate[eps_idx]

        
    dbscan = DBSCAN(eps=eps_, 
                    min_samples=nn, 
                    algorithm='kd_tree', 
                    metric='euclidean',
                    n_jobs=-1).fit(X) 
    dslice.loc[:, 'centre_id'] = dbscan.labels_   
    
    for c in np.unique(dbscan.labels_):
        # working on each cluster
        nvoxel = dslice.loc[dslice.centre_id == c]
        # filter out outliers and clusters with too few points
        if c == -1:
            dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
        else:
            if len(nvoxel.index) < self.min_pts:
                dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
                continue # required so centre is added after points are deleted
            
            # centre_coords = nvoxel[['x', 'y', 'z']].median() 
            
            pca = PCA(n_components=3)
            pca.fit(nvoxel[['x', 'y', 'z']].to_numpy())
            ratios = pca.explained_variance_ratio_
            cyl_metric = (ratios[0] / ratios[1] + ratios[0] / ratios[2]) / 2
        
            # if the pts follow cyl distribution, use CPC to adjust the centre
            if (cyl_metric <= 5) & (len(nvoxel) > 100):
                centre_coords = common.CPC(nvoxel[['x', 'y', 'z']]).x
                centre_coords = pd.Series(centre_coords, index=['x', 'y', 'z'])
            # if not, then use median as centre
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


## selected eps tends to be too large at the upper crown
# def find_centre(dslice, self):
#     if len(dslice) < 2: 
#         return []
    
#     centres = pd.DataFrame()    
#     s = dslice.slice_id.unique()[0]
#     X = dslice[['x', 'y', 'z']]
    
#     group_slice = self.pc[self.pc.slice_id != 0].groupby('slice_id') 
#     # slice that has largest number of points
#     max_pts_sid = group_slice.apply(lambda x: len(x)).idxmax()
#     if s <= max_pts_sid: # lower part of the tree
#         nn = 10
#     else:  # upper part of the tree
#         nn = 5
    
#     # eps candidate 1: largest jump in dnn per point
#     results = common.nn_dist(dslice, n_neighbours=nn)
#     if results is nan:
#         return []
#     else:
#         dists, indices = results
#     # distance to the nn-th nearest neighbour
#     dists = np.sort(dists, axis=0)[:,-1]
#     # index_largest_gap
#     idx = np.argsort(np.diff(dists))[::-1]
#     if idx[0] == 0:
#         knee = dists[idx[1]]
#     else:
#         knee = dists[idx[0]]
        
#     dbscan_knee = DBSCAN(eps=knee, 
#                         min_samples=nn, 
#                         algorithm='kd_tree', 
#                         metric='euclidean',
#                         n_jobs=-1).fit(X) 
#     labels_knee = np.unique(dbscan_knee.labels_) 
#     c_num_knee = len(labels_knee[labels_knee >= 0])

#     if c_num_knee > 1:   
#         # Calculate internal evaluation metrics for the clusters
#         clusters = dbscan_knee.fit_predict(X)
#         # silhouette score
#         knee_s1 = silhouette_score(X, clusters)
#         # Calinski-Harabasz index 
#         knee_s2 = calinski_harabasz_score(X, clusters)
    
   
#     # eps candidate 2: mean dNN of all slices
#     mdnn = group_slice.apply(common.mean_dNN, n_neighbours=nn)
#     conf_85_whole_tree = np.nanmean(mdnn) + 1.44 * np.nanstd(mdnn)
#     fix = conf_85_whole_tree
    
#     dbscan_fix = DBSCAN(eps=fix, 
#                         min_samples=nn, 
#                         algorithm='kd_tree', 
#                         metric='euclidean',
#                         n_jobs=-1).fit(X) 
#     labels_fix = np.unique(dbscan_fix.labels_) 
#     c_num_fix = len(labels_fix[labels_fix >= 0])

#     if c_num_fix > 1:
#         # Calculate internal evaluation metrics for the clusters
#         clusters = dbscan_fix.fit_predict(X)
#         # silhouette score
#         fix_s1 = silhouette_score(X, clusters)
#         # Calinski-Harabasz index 
#         fix_s2 = calinski_harabasz_score(X, clusters)
    
#     # determine eps based on internel evaluation metric
#     if (c_num_knee < 2) or (c_num_fix < 2): # if no score due to cluster<2
#         if s <= max_pts_sid: # lower part of the tree
#             eps_ = knee
#         else:  # upper part of the tree
#             eps_ = min(knee, fix)
#     elif (c_num_knee <= 10) & (s <= max_pts_sid): # cluster number ranges from 2 to 10
#         s1 = max(knee_s1, fix_s1)
#         s2 = max(knee_s2, fix_s2)
#         # lower part of the tree
#         if s <= max_pts_sid: 
#             if knee_s1 == fix_s1: # two candidates have same silhouette score 
#                 if knee_s2 == s2: eps_ = knee # use knee if it has higher Calinski-Harabasz index
#                 else: eps_ = fix
#             elif knee_s1 == s1: # use knee if it has higher silhouette score 
#                 eps_ = knee
#             else: # use fix if it has higher silhouette score 
#                 eps_ = fix
#     # upper part of the tree
#     else:  # more than 10 clusters
#         if c_num_knee >= c_num_fix:
#             eps_ = knee
#         else:
#             eps_ = fix
        
        
#     dbscan = DBSCAN(eps=eps_, 
#                     min_samples=nn, 
#                     algorithm='kd_tree', 
#                     metric='euclidean',
#                     n_jobs=-1).fit(X) 
#     dslice.loc[:, 'centre_id'] = dbscan.labels_   
    
#     for c in np.unique(dbscan.labels_):
#         # working on each cluster
#         nvoxel = dslice.loc[dslice.centre_id == c]
#         # filter out outliers and clusters with too few points
#         if c == -1:
#             dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
#         else:
#             if len(nvoxel.index) < self.min_pts:
#                 dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
#                 continue # required so centre is added after points are deleted
#             centre_coords = nvoxel[['x', 'y', 'z']].median()

#             centres = centres.append(pd.Series({'slice_id':int(s), 
#                                                 'centre_id':int(c), # id within a cluster
#                                                 'cx':centre_coords.x, 
#                                                 'cy':centre_coords.y, 
#                                                 'cz':centre_coords.z, 
#                                                 'distance_from_base':nvoxel.distance_from_base.mean(),
#                                                 'n_points':len(nvoxel),
#                                                 'idx':struct.pack('ii', int(s), int(c))}),
#                                                 ignore_index=True)
        
#             dslice.loc[(dslice.slice_id == s) & 
#                        (dslice.centre_id == c), 'idx'] = struct.pack('ii', int(s), int(c))

#     if (len(centres) != 0) & (isinstance(centres, pd.DataFrame)):
#         return [centres, dslice] 
#     else: 
#         return []
        
        
#     dbscan = DBSCAN(eps=eps_, 
#                     min_samples=nn, 
#                     algorithm='kd_tree', 
#                     metric='euclidean',
#                     n_jobs=-1).fit(X) 
#     dslice.loc[:, 'centre_id'] = dbscan.labels_   
    
#     for c in np.unique(dbscan.labels_):
#         # working on each cluster
#         nvoxel = dslice.loc[dslice.centre_id == c]
#         # filter out outliers and clusters with too few points
#         if c == -1:
#             dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
#         else:
#             if len(nvoxel.index) < self.min_pts:
#                 dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
#                 continue # required so centre is added after points are deleted
#             centre_coords = nvoxel[['x', 'y', 'z']].median()

#             centres = centres.append(pd.Series({'slice_id':int(s), 
#                                                 'centre_id':int(c), # id within a cluster
#                                                 'cx':centre_coords.x, 
#                                                 'cy':centre_coords.y, 
#                                                 'cz':centre_coords.z, 
#                                                 'distance_from_base':nvoxel.distance_from_base.mean(),
#                                                 'n_points':len(nvoxel),
#                                                 'idx':struct.pack('ii', int(s), int(c))}),
#                                                 ignore_index=True)
        
#             dslice.loc[(dslice.slice_id == s) & 
#                        (dslice.centre_id == c), 'idx'] = struct.pack('ii', int(s), int(c))

#     if (len(centres) != 0) & (isinstance(centres, pd.DataFrame)):
#         return [centres, dslice] 
#     else: 
#         return []



## previous method
# def find_centre(dslice, self, eps=None):
    
#     if len(dslice) < self.min_pts: 
#         return []
    
#     centres = pd.DataFrame()    
#     s = dslice.slice_id.unique()[0]

#     if eps is None: 
#         eps_ = self.bins[s] / 2.
#     else: 
#         eps_ = eps

#     # separate different slice components e.g. different branches
#     dbscan = DBSCAN(eps=eps_, 
#                     min_samples=10, 
#                     algorithm='kd_tree', 
#                     metric='euclidean',
#                     n_jobs=-1).fit(dslice[['x', 'y', 'z']]) 
#     dslice.loc[:, 'centre_id'] = dbscan.labels_

#     for c in np.unique(dbscan.labels_):
#         # working on each cluster
#         nvoxel = dslice.loc[dslice.centre_id == c]
#         # filter out noise points and clusters with too few points
#         if c == -1:
#             dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
#         else:
#             if len(nvoxel.index) < self.min_pts:
#                 dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
#                 continue # required so centre is added after points are deleted
#             centre_coords = nvoxel[['x', 'y', 'z']].median()

#             centres = centres.append(pd.Series({'slice_id':int(s), 
#                                                 'centre_id':int(c), # id within a cluster
#                                                 'cx':centre_coords.x, 
#                                                 'cy':centre_coords.y, 
#                                                 'cz':centre_coords.z, 
#                                                 'distance_from_base':nvoxel.distance_from_base.mean(),
#                                                 'n_points':len(nvoxel),
#                                                 'idx':struct.pack('ii', int(s), int(c))}),
#                                                 ignore_index=True)
        
#             dslice.loc[(dslice.slice_id == s) & 
#                     (dslice.centre_id == c), 'idx'] = struct.pack('ii', int(s), int(c))

#     if (len(centres) != 0) & (isinstance(centres, pd.DataFrame)):
#         return [centres, dslice] 
#     else: 
#         return []