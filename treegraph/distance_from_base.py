import numpy as np
import pandas as pd

from treegraph.third_party import shortpath as p2g
from treegraph import downsample

def run(pc, base_location=None, cluster_size=False, knn=100, verbose=False):
    
    """
    Attributes each point with a distance from base
    
    base_location: None of idx (default None)
        index of the base point i.e. where point distance are measured to
    cluster_size: False or float (default False)
        Downsample point cloud to generate skeleton points, this can be
        much quicker for large point clouds.
    knn: int (default 100)
        Refer to pc2graph docs
    """
    
    columns = pc.columns.to_list() + ['distance_from_base']
    
    if cluster_size:
        pc, base_location = downsample.run(pc, cluster_size, 
                                           base_location=base_location, 
                                           remove_noise=True,
                                           keep_columns=['VX'])
    
    c = ['x', 'y', 'z']
    G = p2g.array_to_graph(pc.loc[pc.downsample][c] if 'downsample' in pc.columns else pc[c], 
                           base_id=pc.loc[pc.pid == base_location.values[0]].index[0], 
                           kpairs=3, 
                           knn=knn, 
                           nbrs_threshold=.2,
                           nbrs_threshold_step=.1,
#                                 graph_threshold=.05
                            )

    node_ids, distance, path_dict = p2g.extract_path_info(G, pc.loc[pc.pid == base_location.values[0]].index[0])
    
    if 'distance_from_base' in pc.columns:
        del pc['distance_from_base']
    
    # if pc is downsampled to generate graph then reindex downsampled pc 
    # and join distances... 
    if cluster_size:
        dpc = pc.loc[pc.downsample]
        dpc.reset_index(inplace=True)
        dpc.loc[node_ids, 'distance_from_base'] = np.array(list(distance))
        pc = pd.merge(pc, dpc[['VX', 'distance_from_base']], on='VX', how='left')
    # ...or else just join distances to pc
    else:
        pc.loc[node_ids, 'distance_from_base'] = np.array(list(distance))
    
    return pc[columns]