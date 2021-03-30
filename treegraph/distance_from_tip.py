import pandas as pd
import numpy as np

from tqdm.autonotebook import tqdm

def run(pc, centres, bins, vlength=.005, verbose=False, min_pts=0):

    pc.loc[:, 'modified_distance'] = pc.distance_from_base
    PC_nodes = pd.DataFrame(columns=['new_parent'])
    PC_nodes.loc[:, 'parent_node'] = centres.loc[centres.n_furcation != 0].node_id
    PC_nodes = pd.merge(centres.loc[~np.isnan(centres.parent_node)][['node_id', 'parent_node', 'nbranch']], 
                                  PC_nodes, on='parent_node', how='left')
    
    new_pc = pd.DataFrame()
    
    single_node_branch = centres.nbranch.value_counts() # snb 
    snb_nbranch = single_node_branch.loc[single_node_branch == 1].index
    centres.loc[centres.nbranch.isin(snb_nbranch), 'nbranch'] = centres.loc[centres.nbranch.isin(snb_nbranch), 'parent']
    
    if verbose: print('reattributing branches...')
    for nbranch in tqdm(np.sort(centres.nbranch.unique()), 
                        total=len(centres.nbranch.unique()), 
                        disable=False if verbose else True):
        
        # nodes to identify points
        branch_nodes = centres.loc[centres.nbranch == nbranch].node_id.values
        parent_node = list(centres.loc[centres.nbranch == nbranch].parent_node.unique())[0]
        parent_branch = centres.loc[centres.nbranch == nbranch].parent.unique()[0]
        idx = list(pc.loc[pc.node_id.isin(branch_nodes)].index) # index of nodes
        branch_pc = pc.loc[idx]
        
        # correct for some errors in distance_from_base
        if len(branch_pc) > 1000:
            dfb_min = branch_pc['distance_from_base'].min()
            try:
                branch_pc = distance_from_base.run(branch_pc, 
                                                   base_location=branch_pc.distance_from_base.idxmin(),
                                                   downsample_cloud=vlength,
                                                   knn=100)
            except: pass
            branch_pc.distance_from_base += dfb_min
            
        if nbranch == 0:
            branch_pc.loc[:, 'modified_distance'] = branch_pc.distance_from_base
        else:
            # normalising distance so tip is equal to maximum distance
            tip_diff = pc.distance_from_base.max() - branch_pc.distance_from_base.max()
            branch_pc.loc[:, 'modified_distance'] = branch_pc.distance_from_base + tip_diff

        # regenerating slice_ids
        branch_pc.loc[:, 'slice_id'] = np.digitize(branch_pc.modified_distance, np.array(list(bins.values())).cumsum())
        
        # check new clusters are not smaller than min_pts, if they
        # are cluster them with the next one
        N = branch_pc.groupby('slice_id').x.count()
        slice_plus = {n:0 if N[n] > min_pts else -1 if n == N.max() else 1 for n in N.index}
        branch_pc.slice_id += branch_pc.slice_id.map(slice_plus)
        
        # normalise slice_id to 0
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
            
            new_centres.loc[:, 'dist2fur'] = np.linalg.norm(row[['cx', 'cy', 'cz']].astype(float) - 
                                                            new_centres[['cx', 'cy', 'cz']],
                                                            axis=1)
            PC_nodes.loc[PC_nodes.parent_node == row.node_id, 'new_parent'] = new_centres.loc[new_centres.dist2fur.idxmin()].node_id
            PC_nodes.loc[PC_nodes.parent_node == row.node_id, 'slice_id'] = new_centres.loc[new_centres.dist2fur.idxmin()].slice_id

        centres = centres.loc[~centres.node_id.isin(branch_nodes)]
        centres = centres.append(new_centres.loc[new_centres.n_points > min_pts])

        # update dict that is used to identify new nodes in parent branch
#         node_ids[nbranch] = new_centres.node_id.values
        
        new_pc = new_pc.append(branch_pc)

    new_pc.reset_index(inplace=True, drop=True)
    centres.reset_index(inplace=True, drop=True)
    
    return centres, new_pc
