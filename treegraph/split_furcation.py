import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tqdm

from treegraph.fit_cylinders import *
from treegraph.build_skeleton import *
from treegraph.build_graph import *
from treegraph.attribute_centres import *
from treegraph.common import *

from treegraph.third_party.point2line import *
from treegraph.third_party.closestDistanceBetweenLines import *

def intersection(A0, A1, B0, B1, clampA0=True, clampA=True):
    
    pA, pB, D = closestDistanceBetweenLines(A0, A1, B0, B1, clampA0=clampA, clampA1=clampA)
    if np.isnan(D): D = np.inf
    return pA, pB, D

def split_furcation_new(self):  
    
    """
    split_fucation determines the correct location for a node which furcates.
    This is done by firstly identfying the node which is closest
    to the "child" portion of the parent cluster. Then using the
    point cloud and the first child node, the intersection between
    the parent and child is determined.
    
    This improves on the previous version as no new nodes are added
    which can become complicated.
    """
    
    if self.verbose: print('aligning furcations...')

    for ix, row in tqdm(self.centres[self.centres.n_furcation > 0].sort_values('slice_id', ascending=False).iterrows(), 
                        total=self.centres[self.centres.n_furcation > 0].shape[0],
                        disable=False if self.verbose else True):

        # if furcates at the base then ignore
        if row.distance_from_base == self.centres.distance_from_base.min(): continue

        # the clusters of points that represent the furcation
        cluster = self.pc[(self.pc.node_id == row.node_id)].copy()

        # nodes which are in parent branch (identified from the tip)
        tip_id = self.centres.loc[(self.centres.nbranch == row.nbranch) & 
                                  (self.centres.is_tip)].node_id.values[0]
        branch_path = np.array(self.path_ids[int(tip_id)], dtype=int)
        node_idx = np.where(branch_path == row.node_id)[0][0]

        # nodes either side of furcation
        previous_node = [branch_path[node_idx - 1]]
        subsequent_node = [branch_path[node_idx + 1]]

        # child nodes
        child_nodes = self.centres[(self.centres.parent_node == row.node_id) &
                                   (self.centres.ncyl == 0)].node_id.to_list()

        # label points in cluster
        all_nodes = previous_node + subsequent_node + child_nodes
        all_nodes = self.centres.loc[self.centres.node_id.isin(all_nodes)]
        distances = np.zeros((len(all_nodes), len(cluster)))

        for i, (_, node) in enumerate(all_nodes.iterrows()):
            distances[i, :] = np.linalg.norm(node[['cx', 'cy', 'cz']].astype(float).values - 
                                             cluster[['x', 'y', 'z']], 
                                             axis=1)

        labels = distances.T.argmin(axis=1)
        new_positions = pd.DataFrame(columns=['node_id', 'cx', 'cy', 'cz'])

        for child in child_nodes:
            
            # separaate points
            label = np.where(all_nodes == child)[0]
            child_cluster = cluster.loc[cluster.index[np.where(labels == label)[0]]][['x', 'y', 'z']]
#             child_cluster = cluster.loc[cluster.node_id == child]
            child_centre = child_cluster.mean()
            CHx = self.centres.loc[self.centres.node_id == child][['cx', 'cy', 'cz']].values[0]
            Px = self.centres.loc[self.centres.node_id == previous_node[0]][['cx', 'cy', 'cz']].values[0] 
            Cx = self.centres.loc[self.centres.node_id == row.node_id][['cx', 'cy', 'cz']].values[0]
            Sx = self.centres.loc[self.centres.node_id == subsequent_node[0]][['cx', 'cy', 'cz']].values[0]

            mean_distance = np.zeros(3)

            # calculate distance from point to surrounding nodes 
            # where p, q are the line ends
            for i, q in enumerate([Sx, Cx, Px]):
                mean_distance[i] = d(CHx, q, child_cluster).mean()

            if np.all(np.isnan(mean_distance)): 
                continue # something not right so skip

            if np.argmin(mean_distance) == 0: # closer to subsequent node
                pA, pB, D = intersection(Cx, Sx, child_centre, CHx)
                update_slice_id(self, child, -1)
                nnode = subsequent_node[0]
                
                # update path_ids
                for k, v in self.path_ids.items():
                    if row.node_id in v and child in v:
                        self.path_ids[k] = v[:v.index(child)] + [subsequent_node[0]] +  v[v.index(child):]

            elif np.argmin(mean_distance) == 1: # closer to centre node
                nnode = row.node_id
                A0 = self.centres.loc[self.centres.node_id == subsequent_node[0]][['cx', 'cy', 'cz']].values[0]
                dP, dS = np.linalg.norm(CHx - Px), np.linalg.norm(CHx - Sx) 
                if dP > dS:
                    pA, pB, D = intersection(Cx, Sx, child_centre, CHx)
                else:
                    pA, pB, D = intersection(Cx, Px, child_centre, CHx)      

            else: # closer to previous node
                pA, pB, D = intersection(Cx, Px, child_centre, CHx)
                update_slice_id(self, child, -1)
                nnode = previous_node[0]
                
                # update path_ids
                for k, v in self.path_ids.items():
                    if row.node_id in v and child in v:
                        self.path_ids[k] = v[:v.index(row.node_id)] + v[v.index(child):]

            # add new position to temporary database
            # this is needed if more than one branch joins 
            # to the same node
            new_positions = new_positions.append(pd.Series({'node_id':nnode, 'cx':pA[0], 'cy':pA[1], 'cz':pA[2]}), 
                                                 ignore_index=True)
            
        # update centres with new positions, groupby is required as 
        # 2 branches may be connected to 1 node and therefore the
        # mean position is taken
        if len(new_positions) == 0 or np.all(np.isnan(new_positions[['cx', 'cy', 'cz']])): continue
        for roww in new_positions.groupby('node_id').mean().itertuples():
            if np.any(np.isnan([roww.cx, roww.cy, roww.cz])): continue # something is wrong, leave where is is
            self.centres.loc[self.centres.node_id == roww.Index, 'cx'] = roww.cx 
            self.centres.loc[self.centres.node_id == roww.Index, 'cy'] = roww.cy 
            self.centres.loc[self.centres.node_id == roww.Index, 'cz'] = roww.cz 


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