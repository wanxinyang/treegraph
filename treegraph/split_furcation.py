import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tqdm

from treegraph.fit_cylinders import *
from treegraph.build_skeleton import *
from treegraph.build_graph import *
from treegraph.attribute_centres import *
from treegraph.common import *

def split_furcation(self, error=.01, max_dist=1):
    
    """
    parameters
    ----------
    
    error; float (default 0.1)
    minimum distance two new nodes have to be apart for them to be included
    
    max_dist; float (default 1)
    used in skeleton path
    """
    
    for ix, row in tqdm(self.centres[self.centres.n_furcation > 0].sort_values('slice_id', ascending=False).iterrows(), 
                        total=self.centres[self.centres.n_furcation > 0].shape[0],
                        disable=False if self.verbose else True):

        # if furcates at the base then ignore
        if row.distance_from_base == self.centres.distance_from_base.min(): continue

        # nodes which are in parent branch (identified from the tip)
        tip_id = self.centres.loc[(self.centres.nbranch == row.nbranch) & 
                                  (self.centres.is_tip)].node_id.values[0]
        branch_path = np.array(self.path_ids[int(tip_id)], dtype=int)
        idx = np.where(branch_path == row.node_id)[0][0]

        # node 1 closer to base which will become new furcation points
        previous_node = branch_path[idx - 1]
        base_coords = self.centres.loc[self.centres.node_id == previous_node][['cx', 'cy', 'cz']].values

        # extract point cloud for current node and run KMeans clustering
        # where K is determined by number of furcations
        c = self.pc[(self.pc.node_id == row.node_id)].copy()
        c.loc[:, 'klabels'] = KMeans(n_clusters=int(row.n_furcation) + 1).fit(c[['x', 'y', 'z']]).labels_
        d = c.groupby('klabels').mean()

        # if new centres are more than error apart
        if nn(d[['x', 'y', 'z']].values, 1).mean() > error:

            points_index = {}

            # create temporary df with which to add new nodes
            child_branch = self.centres[self.centres.parent_node == row.node_id].nbranch.unique()
            child_nodes = self.centres.loc[self.centres.nbranch.isin(child_branch)].node_id.to_list()
            nodes_subset = list(branch_path) + child_nodes
            nodes_subset.remove(row.node_id) # remove existing node as don't want to include in tmp 
            tmp_centres = self.centres.loc[self.centres.node_id.isin(nodes_subset)] 
            node_id = self.centres.node_id.max() + 1

            # add new nodes
            for drow in d.itertuples():
                
                nvoxel = c[c.klabels == drow.Index]
                centre_coords = nvoxel[['x', 'y', 'z']].median()
                tmp_centres = tmp_centres.append({'slice_id':nvoxel.slice_id.unique()[0], 
                                                  'centre_id':nvoxel.centre_id.unique()[0], 
                                                  'cx':centre_coords.x, 
                                                  'cy':centre_coords.y, 
                                                  'cz':centre_coords.z, 
                                                  'distance_from_base':nvoxel.distance_from_base.mean(),
                                                  'n_points':len(nvoxel),
                                                  'node_id':node_id}, ignore_index=True)
                points_index[node_id] = list(nvoxel.index)
                node_id += 1
                
            ### test if new furcation generates cylinders that are within one another
            # compute temporary graph
            path_distance, path_ids = skeleton_path(tmp_centres, max_dist=max_dist, verbose=False)
            tmp_centres = attribute_centres(tmp_centres, path_ids)
            
            # identify and add new node in parent branch
            keep_node_in_path = True
            node_in_parent = tmp_centres.loc[(tmp_centres.node_id.isin(points_index.keys())) & 
                                             (tmp_centres.nbranch == tmp_centres.nbranch.min())] # min nbranch being the parent
            if len(node_in_parent) == 0:
                # if none of the nodes occur in the parent... might be e GOTCHA and needs checking    
                node_in_parent = tmp_centres.loc[(tmp_centres.node_id.isin([list(points_index.keys())[0]]))]
            
            # node_in_parent_radius,  _, _ = cylinderFitting(c.loc[c.klabels.isin(new_nodes.label.loc[new_nodes.parent])])
            node_in_parent_radius,  _, _ = cylinderFitting(c) # worse case i.e. points are wrongly attributed to >1 cluster

            for child_node_id in points_index.keys():
                
                # don't process the node in parent
                if child_node_id in node_in_parent.node_id.values: continue

                # test whether the child is mostly in the parent
                child = tmp_centres.loc[tmp_centres.node_id == child_node_id]
                A = node_angle_f(node_in_parent[['cx', 'cy', 'cz']].values,
                                 base_coords,
                                 child[['cx', 'cy', 'cz']].values) # angle between nodes
                distance_to_edge =  (node_in_parent_radius / np.sin(A))

                dist_between_nodes = np.linalg.norm(base_coords - 
                                                    child[['cx', 'cy', 'cz']].values)

                if distance_to_edge / dist_between_nodes > .8 and keep_node_in_path: # comparing lengths
                    # cylinder overlap is too great, delete new node in parent 
                    # and correctly attribute point cloud with node_id. This tends
                    # to happen where there is already a good fit
                    keep_node_in_path = False
                    points_index[child_node_id] += points_index[node_in_parent.node_id.values[0]]
                     
                self.centres = self.centres.append(tmp_centres.loc[tmp_centres.node_id == child_node_id], 
                                                   ignore_index=True)
                self.pc.loc[self.pc.index.isin(points_index[child_node_id]), 'node_id'] = child_node_id
            
            if keep_node_in_path:
                # new node in parent is valid and should be kept
                self.centres = self.centres.append(node_in_parent, ignore_index=True)
                self.pc.loc[self.pc.index.isin(points_index[node_in_parent.node_id.values[0]]), 'node_id'] = node_in_parent
                # replace
                for k, v in path_ids.items():
                    if k in self.path_ids.keys():
                        self.path_ids[k] = v
                    else: self.path_ids[k] = v       
            else:
                tmp_centres = tmp_centres.loc[tmp_centres.node_id != node_in_parent.node_id.values[0]]
                path_distance, path_ids = skeleton_path(tmp_centres, max_dist=max_dist, verbose=False)
                for k, v in path_ids.items():
                    if k in self.path_ids.keys():
                        self.path_ids[k] = v
                    else: self.path_ids[k] = v
            
            self.centres = self.centres.loc[self.centres.node_id != row.node_id]
            
        # if splitting the base node occured - put back together
        if len(self.centres[(self.centres.slice_id == 0) & (self.centres.slice_id == 0)]) > 1:

            Base = self.centres[(self.centres.slice_id == 0) & (self.centres.slice_id == 0)]
            Mean, Sum = Base.mean(), Base.sum()

            self.centres = self.centres.loc[~self.centres.node_id.isin(Base.node_id.values)]
            self.centres = self.centres.append(pd.Series({'slice_id':0, 
                                                          'centre_id':0, 
                                                          'cx':Mean.cx, 
                                                          'cy':Mean.cy, 
                                                          'cz':Mean.cz, 
                                                          'distance_from_base':Mean.distance_from_base,
                                                          'n_points':Sum.n_points,
                                                          'node_id':self.centres.node_id.max() + 1}),
                                               ignore_index=True)