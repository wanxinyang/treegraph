import time
import pandas as pd
import numpy as np

from treegraph.build_skeleton import generate_distance_graph

def attribute_centres(centres, path_ids, verbose=False, branch_hierarchy=False):
    
    T = time.time()

    # if node is a tip
    centres.loc[:, 'is_tip'] = False
    unique_nodes = np.unique([v for p in path_ids.values() for v in p], return_counts=True)
    centres.loc[centres.node_id.isin(unique_nodes[0][unique_nodes[1] == 1]), 'is_tip'] = True

    if verbose: print('\tlocatate tips:', time.time() - T)
    
    # calculate branch lengths and numbers
    tip_paths = pd.DataFrame(index=centres[centres.is_tip].node_id.values, 
                             columns=['tip2base', 'length', 'nbranch'])
    
    for k, v in path_ids.items():
        
        v = v[::-1]
        if v[0] in centres[centres.is_tip].node_id.values:
            c1 = centres.set_index('node_id').loc[v[:-1]][['cx', 'cy', 'cz']].values
            c2 = centres.set_index('node_id').loc[v[1:]][['cx', 'cy', 'cz']].values
            tip_paths.loc[tip_paths.index == v[0], 'tip2base'] = np.linalg.norm(c1 - c2, axis=1).sum()
            
    if verbose: print('\tbranch lengths:', time.time() - T)
            
    centres.sort_values('slice_id', inplace=True)
    centres.loc[:, 'nbranch'] = -1
    centres.loc[:, 'ncyl'] = -1

    for i, row in enumerate(tip_paths.sort_values('tip2base', ascending=False).itertuples()):
        
        tip_paths.loc[row.Index, 'nbranch'] = i 
        cyls = path_ids[row.Index]
        centres.loc[(centres.node_id.isin(cyls)) & 
                         (centres.nbranch == -1), 'nbranch'] = i
        centres.loc[centres.nbranch == i, 'ncyl'] = np.arange(len(centres[centres.nbranch == i]))
        v = centres.loc[centres.nbranch == i].sort_values('ncyl').node_id
        c1 = centres.set_index('node_id').loc[v[:-1]][['cx', 'cy', 'cz']].values
        c2 = centres.set_index('node_id').loc[v[1:]][['cx', 'cy', 'cz']].values
        tip_paths.loc[row.Index, 'length'] = np.linalg.norm(c1 - c2, axis=1).sum()
    
    # reattribute branch numbers starting with the longest
    new_branch_nums = {bn:i for i, bn in enumerate(tip_paths.sort_values('length', ascending=False).nbranch)}
    tip_paths.loc[:, 'nbranch'] = tip_paths.nbranch.map(new_branch_nums)
    centres.loc[:, 'nbranch'] = centres.nbranch.map(new_branch_nums)
        
    if verbose: print('\tbranch and cyl nums:', time.time() - T)

    centres.loc[:, 'n_furcation'] = 0        
    centres.loc[:, 'parent'] = -1  
    centres.loc[:, 'parent_node'] = np.nan
    
    # loop over branch base and identify parent
    for nbranch in centres.nbranch.unique():
        
        if nbranch == 0: continue # main branch does not furcate
        furcation_node = -1
        branch_base_idx = centres.loc[centres.nbranch == nbranch].ncyl.idxmin()
        branch_base_idx = centres.loc[branch_base_idx].node_id
        
        for path in path_ids.values():    
            if path[-1] == branch_base_idx:
                if len(path) > 1:
                    furcation_node = path[-2]
                else:
                    furcation_node = path[-1]
                centres.loc[centres.node_id == furcation_node, 'n_furcation'] += 1
                break
        
        if furcation_node != -1:
            parent = centres.loc[centres.node_id == furcation_node].nbranch.values[0]
            centres.loc[(centres.nbranch == nbranch), 'parent'] = parent
            centres.loc[(centres.nbranch == nbranch) & (centres.ncyl == 0), 'parent_node'] = furcation_node
        
    if verbose: print('\tidentify parent:', time.time() - T)

    # loop over branches and attribute internode
    centres.sort_values(['nbranch', 'ncyl'], inplace=True)
    centres.loc[:, 'ninternode'] = -1
    internode_n = 0

    for ix, row in centres.iterrows():
        centres.loc[centres.node_id == row.node_id, 'ninternode'] = internode_n
        if row.n_furcation > 0 or row.is_tip: internode_n += 1
            
    if verbose: print('\tidentify internode:', time.time() - T)
    
    if branch_hierarchy:
        
        branch_hierarchy = {0:{'all':np.array([0]), 'above':centres.nbranch.unique()[1:]}}

        for b in np.arange(centres.nbranch.max() + 1):
            if b == 0: continue
            parent = centres.loc[(centres.nbranch == b) & (centres.ncyl == 0)].parent.values[0]
            branch_hierarchy[b] = {}
            branch_hierarchy[b]['all'] = np.hstack([[b], branch_hierarchy[parent]['all']])

        for b in np.arange(centres.nbranch.max() + 1):
            if b == 0: continue
            ba = set()
            for k, v in branch_hierarchy.items():
                if b not in list(v['all']): continue
                ba.update(set(v['all'][v['all'] > b]))
            if len(ba) > 0: 
                branch_hierarchy[b]['above'] = list(ba)
            else:
                branch_hierarchy[b]['above'] = []

        return centres, branch_hierarchy
    
    else:   
        return centres
    
def distance_from_tip(self, centres, pc, vlength=.005):

    pc.loc[:, 'modified_distance'] = pc.distance_from_base
    PC_nodes = pd.DataFrame(columns=['new_parent'])
    PC_nodes.loc[:, 'parent_node'] = centres.loc[centres.n_furcation != 0].node_id
    PC_nodes = pd.merge(centres.loc[~np.isnan(centres.parent_node)][['node_id', 'parent_node', 'nbranch']], 
                                  PC_nodes, on='parent_node', how='left')
    
    new_pc = pd.DataFrame()
    
    single_node_branch = centres.nbranch.value_counts() # snb 
    snb_nbranch = single_node_branch.loc[single_node_branch == 1].index
    centres.loc[centres.nbranch.isin(snb_nbranch), 'nbranch'] = centres.loc[centres.nbranch.isin(snb_nbranch), 'parent']
    
    for nbranch in np.sort(centres.nbranch.unique()):
        
        # nodes to identify points
        branch_nodes = centres.loc[centres.nbranch == nbranch].node_id.values
        parent_node = list(centres.loc[centres.nbranch == nbranch].parent_node.unique())[0]
        parent_branch = centres.loc[centres.nbranch == nbranch].parent.unique()[0]
        idx = list(pc.loc[pc.node_id.isin(branch_nodes)].index) # index of nodes
        branch_pc = pc.loc[idx]
        
        # correct for some errors in distance_from_base
        if len(branch_pc) > 1000:
            dfb_min = branch_pc['distance_from_base'].min()
            branch_pc = generate_distance_graph(branch_pc, 
                                                base_location=branch_pc.distance_from_base.idxmin(),
                                                downsample_cloud=False if len(branch_pc) <= 100 else vlength,
                                                knn=50)
            branch_pc.distance_from_base += dfb_min
            
        if nbranch == 0:
            branch_pc.loc[:, 'modified_distance'] = branch_pc.distance_from_base
        else:
            # normalising distance so tip is equal to maximum distance
            tip_diff = pc.distance_from_base.max() - branch_pc.distance_from_base.max()
            branch_pc.loc[:, 'modified_distance'] = branch_pc.distance_from_base + tip_diff

        # regenerating slice_ids
        branch_pc.loc[:, 'slice_id'] = np.digitize(branch_pc.modified_distance, self.f.cumsum())
        branch_pc.slice_id = branch_pc.slice_id - branch_pc.slice_id.min()

        # reattribute centres centres
        new_centres = branch_pc.groupby('slice_id')[['x', 'y', 'z']].median().rename(columns={'x':'cx', 'y':'cy', 'z':'cz'})
        centre_path_dist = branch_pc.groupby('slice_id').distance_from_base.mean()
        npoints = branch_pc.groupby('slice_id').x.count()
        npoints.name = 'n_points'                       
        new_centres = new_centres.join(centre_path_dist).join(npoints).reset_index()

        # update pc node_id and slice_id
        new_centres.loc[:, 'node_id'] = np.arange(len(new_centres)) + centres.node_id.max() + 1 
        branch_pc = branch_pc[branch_pc.columns.drop('node_id')].join(new_centres[['slice_id', 'node_id']], 
                                                                      on='slice_id',
                                                                      how='left', 
                                                                      rsuffix='x')

        if nbranch != 0: # main branch does not have a parent
            parent_slice_id = PC_nodes.loc[(PC_nodes.parent_node == parent_node) &
                                           (PC_nodes.nbranch == nbranch)].slice_id.values[0]
            new_centres.slice_id += parent_slice_id + 1
            branch_pc.slice_id += parent_slice_id + 1
        
        # if branch furcates identify new node_id and slice_id
        for _, row in centres.loc[(centres.nbranch == nbranch) & (centres.n_furcation > 0)].iterrows():
            
            new_centres.loc[:, 'dist2fur'] = np.linalg.norm(row[['cx', 'cy', 'cz']] - 
                                                            new_centres[['cx', 'cy', 'cz']],
                                                            axis=1)
            PC_nodes.loc[PC_nodes.parent_node == row.node_id, 'new_parent'] = new_centres.loc[new_centres.dist2fur.idxmin()].node_id
            PC_nodes.loc[PC_nodes.parent_node == row.node_id, 'slice_id'] = new_centres.loc[new_centres.dist2fur.idxmin()].slice_id

        centres = centres.loc[~centres.node_id.isin(branch_nodes)]
        centres = centres.append(new_centres.loc[new_centres.n_points > self.min_pts])

        # update dict that is used to identify new nodes in parent branch
#         node_ids[nbranch] = new_centres.node_id.values
        
        new_pc = new_pc.append(branch_pc)

    new_pc.reset_index(inplace=True)
    centres.reset_index(inplace=True)
    return centres, new_pc