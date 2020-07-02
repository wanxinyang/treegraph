import networkx as nx
import pandas as pd
import numpy as np

from tqdm.autonotebook import tqdm

def skeleton_path(centres, max_dist=.1, verbose=False):
    
    """
    parameters
    ----------
    max_dist; float (default .1)
    maximum distance between nodes, designed to stop large gaps being spanned
    i.e. it is better to have a disconnection than an unrealistic conncetion
    """
    
    edges = pd.DataFrame(columns=['node1', 'node2', 'length'])
  
    if verbose: print('generating graph...')
    for i, row in tqdm(enumerate(centres.itertuples()), total=len(centres), disable=False if verbose else True):
        
        # first node
            if row.distance_from_base == centres.distance_from_base.min(): continue

            n, dist = 3, np.inf

            while n < 10 and dist > max_dist:

                # between required incase of small gap in pc
                nbrs = centres.loc[centres.slice_id.between(row.slice_id - n, row.slice_id - 1)]
                nbrs.loc[:, 'dist'] = np.linalg.norm(np.array([row.cx, row.cy, row.cz]) - 
                                         nbrs[['cx', 'cy', 'cz']].values, 
                                         axis=1)
                dist = nbrs.dist.min()
                n += 1

            edges = edges.append({'node1':int(row.node_id), 
                                  'node2':int(nbrs.loc[nbrs.dist == nbrs.dist.min()].node_id.values[0]), 
                                  'length':nbrs.dist.min()}, ignore_index=True)
        
    idx = centres.distance_from_base.idxmin() 
    base_id = centres.loc[idx].node_id
    
    G_skeleton = nx.Graph()
    G_skeleton.add_weighted_edges_from([(int(row.node1), int(row.node2), row.length) 
                                        for row in edges.itertuples()])

    path_distance, path_ids = nx.single_source_bellman_ford(G_skeleton, base_id)
    path_distance = {k: v if not isinstance(v, np.ndarray) else v[0] for k, v in path_distance.items()}
    
    return path_distance, path_ids
