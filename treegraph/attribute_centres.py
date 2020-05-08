import time
import pandas as pd
import numpy as np

def attribute_centres(centres, path_ids, verbose=False):
    
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
            
    centres.sort_values('centre_path_dist', inplace=True)
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
            centres.loc[centres.nbranch == nbranch, 'parent'] = parent
            centres.loc[centres.nbranch == nbranch, 'parent_node'] = furcation_node
        
    if verbose: print('\tidentify parent:', time.time() - T)

    # loop over branches and attribute internode
    centres.sort_values(['nbranch', 'ncyl'], inplace=True)
    centres.loc[:, 'ninternode'] = -1
    internode_n = 0

    for ix, row in centres.iterrows():
        centres.loc[centres.node_id == row.node_id, 'ninternode'] = internode_n
        if row.n_furcation > 0 or row.is_tip: internode_n += 1
            
    if verbose: print('\tidentify internode:', time.time() - T)
        
    return centres