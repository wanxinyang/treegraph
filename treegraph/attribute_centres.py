import time
import pandas as pd
import numpy as np

from tqdm.autonotebook import trange

import treegraph.distance_from_base 

def run(centres, path_ids, verbose=False, branch_hierarchy=False):
    
#     T = time.time()
    with trange(6 if branch_hierarchy else 5, 
                  disable=False if verbose else True,
                  desc='steps') as pbar:
    
        # remove nodes that are not graphed - prob outlying clusters 
        centres = centres.loc[centres.node_id.isin(path_ids.keys())]

        # identify previous node in the graph
        previous_node = lambda nid: np.nan if len(path_ids[nid]) == 1 else path_ids[nid][-2]
        centres.loc[:, 'pnode'] = centres.node_id.apply(previous_node)

        # if node is a tip
        centres.loc[:, 'is_tip'] = False
        unique_nodes = np.unique([v for p in path_ids.values() for v in p], return_counts=True)
        centres.loc[centres.node_id.isin(unique_nodes[0][unique_nodes[1] == 1]), 'is_tip'] = True

        pbar.set_description("identified tips", refresh=True)
        pbar.update(1) # update progress bar

        # calculate branch lengths and numbers
        tip_paths = pd.DataFrame(index=centres[centres.is_tip].node_id.values, 
                                 columns=['tip2base', 'length', 'nbranch'])

        for k, v in path_ids.items():

            v = v[::-1]
            if v[0] in centres[centres.is_tip].node_id.values:
                c1 = centres.set_index('node_id').loc[v[:-1]][['cx', 'cy', 'cz']].values
                c2 = centres.set_index('node_id').loc[v[1:]][['cx', 'cy', 'cz']].values
                tip_paths.loc[tip_paths.index == v[0], 'tip2base'] = np.linalg.norm(c1 - c2, axis=1).sum()

        pbar.set_description("calculated tip to base lengths", refresh=True)
        pbar.update(1)

        centres.sort_values(['slice_id', 'distance_from_base'], inplace=True)
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

        pbar.set_description("idnetified individual branches", refresh=True)
        pbar.update(1)
        
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
                parent = centres.loc[centres.node_id == furcation_node].nbranch.item()
                centres.loc[(centres.nbranch == nbranch), 'parent'] = parent
                centres.loc[(centres.nbranch == nbranch), 'parent_node'] = furcation_node

        pbar.set_description('attributed nodes and identified parents', refresh=True)
        pbar.update(1)

        # loop over branches and attribute internode
        centres.sort_values(['nbranch', 'slice_id', 'distance_from_base'], inplace=True)
        centres.loc[:, 'ninternode'] = -1
        internode_n = 0

        for ix, row in centres.iterrows():
            centres.loc[centres.node_id == row.node_id, 'ninternode'] = internode_n
            if row.n_furcation > 0 or row.is_tip: internode_n += 1

        for internode in centres.ninternode.unique():
            if internode == 0: continue # first internode so ignore
            pnode = centres.loc[centres.ninternode == internode].pnode.min()
            centres.loc[centres.ninternode == internode, 'pinternode'] = centres.loc[centres.node_id == pnode].ninternode.item()

        pbar.set_description('attributed internodes', refresh=True)
        pbar.update(1)
        
        centres = centres.reset_index(drop=True)
        
        if branch_hierarchy:

            branch_hierarchy = {0:{'all':np.array([0]), 'above':centres.nbranch.unique()[1:]}}

            for b in np.sort(centres.nbranch.unique()):
                if b == 0: continue
                parent = centres.loc[(centres.nbranch == b) & (centres.ncyl == 0)].parent.item()    
    #             if b == parent: 
    #                 # need to figure out why some branches are their own parents....
    #                 # think it is because they are isolated 
    #                 nodes = centres.loc[centres.nbranch == b].node_id.values
    #                 centres = centres.loc[~centres.node_id.isin(nodes)]
    #                 pc = pc.loc[~pc.node_id.isin(nodes)]
    #                 print(b)
    #                 continue
                branch_hierarchy[b] = {}
                branch_hierarchy[b]['all'] = np.hstack([[b], branch_hierarchy[parent]['all']])

            for b in centres.nbranch.unique():
                if b == 0: continue
                ba = set()
                for k, v in branch_hierarchy.items():
                    if b not in list(v['all']): continue
                    ba.update(set(v['all'][v['all'] > b]))
                if len(ba) > 0: 
                    branch_hierarchy[b]['above'] = list(ba)
                else:
                    branch_hierarchy[b]['above'] = []
            
            pbar.set_description('created branch hierarchy', refresh=True)
            pbar.update(1)

            return centres, branch_hierarchy

        else:   
            return centres
