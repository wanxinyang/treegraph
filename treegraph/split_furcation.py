import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tqdm

from treegraph.fit_cylinders import *
from treegraph.build_skeleton import *
from treegraph.build_graph import *
from treegraph.attribute_centres import *
from treegraph.common import *


def split_furcation(self, error=.01):
    
    for ix, row in tqdm(self.centres[self.centres.n_furcation > 0].iterrows(), 
                        total=self.centres[self.centres.n_furcation > 0].shape[0]):

        if row.centre_path_dist == self.centres.centre_path_dist.min(): continue

        tip_id = self.centres.loc[(self.centres.nbranch == row.nbranch) & 
                                  (self.centres.is_tip)].node_id.values[0]
        branch_path = np.array(self.path_ids[int(tip_id)], dtype=int)
        idx = np.where(branch_path == row.node_id)[0][0]

        previous_node = branch_path[idx - 1]
        base_coords = self.centres.loc[self.centres.node_id == previous_node][['cx', 'cy', 'cz']].values

        # extract point cloud for current node and run KMeans clustering
        # where K is determined by number of furcations
        c = self.pc[(self.pc.node_id == row.node_id)].copy()
        c.loc[:, 'klabels'] = KMeans(n_clusters=int(row.n_furcation) + 1).fit(c[['x', 'y', 'z']]).labels_
        d = c.groupby('klabels').mean()

        # if new centres are more than error apart
        if nn(d[['x', 'y', 'z']].values, 1).mean() > error:

            new_nodes = pd.DataFrame(columns=['label', 'parent'])
            new_nodes.parent = False

            # add new nodes
            for drow in d.itertuples():

                node_id = self.centres.node_id.max() + 1
                nvoxel = c[c.klabels == drow.Index]
                centre_coords = nvoxel[['x', 'y', 'z']].median()
                self.centres = self.centres.append({'slice_id':nvoxel.slice_id.unique()[0], 
                                                    'centre_id':nvoxel.centre_id.unique()[0], 
                                                    'cx':centre_coords.x, 
                                                    'cy':centre_coords.y, 
                                                    'cz':centre_coords.z, 
                                                    'centre_path_dist':nvoxel.distance_from_base.mean(),
                                                    'n_points':len(nvoxel),
                                                    'node_id':node_id}, ignore_index=True)
                new_nodes.loc[node_id] = [drow.Index, False]
                self.pc.loc[self.pc.index.isin(nvoxel.index), 'node_id'] = node_id

            ### test if new generate cylinders that are within one another
            # compute temporary graph
            child_branch = self.centres[self.centres.parent_node == row.node_id].nbranch.unique()[0]
            child_nodes = self.centres.loc[self.centres.nbranch == child_branch].node_id.to_list()
            nodes_subset = list(branch_path) + child_nodes + new_nodes.index.to_list()
            nodes_subset.remove(row.node_id) # remove existing node as don't want to include in tmp 
            tmp_centres = self.centres.loc[self.centres.node_id.isin(nodes_subset)]  
            path_distance, path_ids = skeleton_path(tmp_centres, max_dist=.1)
            tmp_centres = attribute_centres(tmp_centres, path_ids)

            # which new node included in the parent
            node_in_parent = tmp_centres.loc[(tmp_centres.node_id.isin(new_nodes.index.to_list())) & 
                                             (tmp_centres.nbranch == tmp_centres.nbranch.min())]
            new_nodes.loc[node_in_parent.node_id.values[0], 'parent'] = True
#             node_in_parent_radius,  _, _ = cylinderFitting(c.loc[c.klabels.isin(new_nodes.label.loc[new_nodes.parent])])
            node_in_parent_radius,  _, _ = cylinderFitting(c) # worse case i.e. points are wrongly attributed to >1 cluster

            child_nodes = new_nodes.loc[~new_nodes.parent].index

            for child in child_nodes:
                child = self.centres[self.centres.node_id == child]
                # angle between nodes
                A = node_angle_f(node_in_parent[['cx', 'cy', 'cz']].values,
                                 base_coords,
                                 child[['cx', 'cy', 'cz']].values)
                distance_to_edge =  (node_in_parent_radius / np.sin(A))

                dist_between_nodes = np.linalg.norm(base_coords - 
                                                    child[['cx', 'cy', 'cz']].values)

                if distance_to_edge / dist_between_nodes > .8:
                    # cylinder overlap is too great, delete node in parent 
                    # and corrently attribute point cloud with node_idccccccc
                    self.centres = self.centres.loc[self.centres.node_id != node_in_parent.node_id.values[0]]
                    self.pc.loc[self.pc.node_id == node_in_parent.node_id.values[0], 'node_id'] = child.node_id.values[0]
                    break

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
                                                          'centre_path_dist':Mean.centre_path_dist,
                                                          'n_points':Sum.n_points,
                                                          'node_id':self.centres.node_id.max() + 1}),
                                               ignore_index=True)
