import networkx as nx
import pandas as pd
import numpy as np

from tqdm import tqdm

def skeleton_path(centres, max_dist=.1, show_progress=False):

    edges = pd.DataFrame(columns=['node1', 'node2', 'length'])
    
    if show_progress:
        print('generating graph...')
        iterator = tqdm(enumerate(centres.itertuples()), total=len(centres))
    else:
        iterator = enumerate(centres.itertuples())
  
    for i, row in iterator:

        # first node
        if row.centre_path_dist == centres.centre_path_dist.min(): continue
        
        n, nbrs, dist = 3, [], 999
        
        while len(nbrs) == 0 or dist > max_dist:
            
            # between required incase of small gap in pc
            nbrs = centres.loc[centres.slice_id.between(row.slice_id - n,
                                                        row.slice_id - 1)].node_id
            if len(nbrs) > 0:
                nbr_dist = np.linalg.norm(np.array([row.cx, row.cy, row.cz]) - 
                                          centres.loc[centres.node_id.isin(nbrs)][['cx', 'cy', 'cz']].values, 
                                          axis=1)
                dist = nbr_dist.min()

            n += 2
            if n > 10: 
                break

        if dist > max_dist: continue
        nbr_id = nbrs[nbrs.index[np.argmin(nbr_dist)]]  
        edges = edges.append({'node1':int(row.node_id), 
                              'node2':int(nbr_id), 
                              'length':nbr_dist.min()}, ignore_index=True)
        
    idx = centres.centre_path_dist.idxmin() 
    base_id = centres.loc[idx].node_id
    
    G_skeleton = nx.Graph()
    G_skeleton.add_weighted_edges_from([(int(row.node1), int(row.node2), row.length) 
                                        for row in edges.itertuples()])

    path_distance, path_ids = nx.single_source_bellman_ford(G_skeleton, base_id)
    path_distance = {k: v if not isinstance(v, np.ndarray) else v[0] for k, v in path_distance.items()}
    
    return path_distance, path_ids