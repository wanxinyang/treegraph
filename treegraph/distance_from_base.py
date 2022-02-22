import numpy as np
import pandas as pd
import os

from treegraph.third_party import shortpath as p2g
from treegraph import downsample
from treegraph.third_party import cylinder_fitting as cyl_fit

# update run() by adding base fitting correction
def run(pc, base_location=None, cluster_size=False, knn=100, verbose=False,\
        base_correction=True):
    
    """
    Attributes each point with a distance from base
    
    base_location: None or idx (default None)
        Index of the base point i.e. where point distance are measured to.
    cluster_size: False or float (default False)
        Downsample point cloud to generate skeleton points, this can be
        much quicker for large point clouds.
    knn: int (default 100)
        Refer to pc2graph docs
    base_correction: boolean (default True)
        Generate a new base node located at the centre of tree base cross-section.
        Update initial graph by connecting points in the base slice to new base node.
    """
    
    columns = pc.columns.to_list() + ['distance_from_base']
    
    if cluster_size:
        pc, base_location = downsample.run(pc, cluster_size, 
                                           base_location=base_location, 
                                           remove_noise=True,
                                           keep_columns=['VX'])
    
    c = ['x', 'y', 'z']
    # generate initial graph
    G = p2g.array_to_graph(pc.loc[pc.downsample][c] if 'downsample' in pc.columns else pc[c], 
                           base_id=pc.loc[pc.pid == base_location].index[0], 
                           kpairs=3, 
                           knn=knn, 
                           nbrs_threshold=.2,
                           nbrs_threshold_step=.1,
#                                 graph_threshold=.05
                            )
    if base_correction:
        # generate new base node
        _, _, _, new_base = base_fitting(pc, base_slice_len=1.5)
        
        # select points in the lowest slice
        low_slice_len = 0.2  # unit in metre
        if 'downsample' in pc.columns:
            low_slice = pc[(pc.z <= (min(pc.z)+low_slice_len)) &\
                                (pc.downsample == True)]
        else:
            low_slice = pc[pc.z <= (min(pc.z)+low_slice_len)]

        # calculate distance between new_base_node and each point in the lowest slice
        index = []
        distance = []
        for i, row in low_slice.iterrows():
            index.append(i)
            c = row[['x','y','z']].values
            distance.append(np.linalg.norm(new_base - c))

        # add the new base node to the graph    
        new_base_id = np.max(pc.index) + 1
        G.add_node(new_base_id)

        # add edges (weighted by distance) between new base node and the low_slice nodes in graph
        p2g.add_nodes(G, new_base_id, index, distance, np.inf)
        print(f'add new edges: {len(index)}')

        # add new_base_node attributes to pc
        if 'downsample' in pc.columns:
            base_coords = pd.Series({'x':new_base[0], \
                                    'y':new_base[1], \
                                    'z':new_base[2], \
                                    'pid':new_base_id,\
                                    'downsample':True},\
                                    name=new_base_id)
        else:
            base_coords = pd.Series({'x':new_base[0], \
                                    'y':new_base[1], \
                                    'z':new_base[2], \
                                    'pid':new_base_id},\
                                    name=new_base_id)
        pc = pc.append(base_coords)
    
        # extracts shortest path information from the updated initial graph
        node_ids, distance, path_dict = p2g.extract_path_info(G, new_base_id)
 
    else: # do not generate new base node and update the initial graph
        # extracts shortest path information from the initial graph
        node_ids, distance, path_dict = p2g.extract_path_info(G, pc.loc[pc.pid == base_location].index[0])
        new_base = np.nan
        
    if 'distance_from_base' in pc.columns:
        del pc['distance_from_base']
    
    # if pc is downsampled to generate graph then reindex downsampled pc 
    # and join distances... 
    if cluster_size:
        dpc = pc.loc[pc.downsample]
        # dpc.reset_index(inplace=True)
        dpc.loc[node_ids, 'distance_from_base'] = np.array(list(distance))
        pc = pd.merge(pc, dpc[['VX', 'distance_from_base']], on='VX', how='left')
    # ...or else just join distances to pc
    else:
        pc.loc[node_ids, 'distance_from_base'] = np.array(list(distance))
    
    return pc[columns], G, new_base


def base_fitting(pc, base_slice_len=1.5):
    '''
    Find new base node by fitting a cylinder to the bottom slice.
    
    Inputs:
        - pc: point cloud dataframe
        - base_slice_len: float (default: 1.5 meters)
            Vertical length of a slice to be segmented from base, 
            and a cylinder will be fitted to this slice segment.
    
    Ouputs:
        - base_slice: dataframe
            x,y,z coordinates of base slice
        - fit_C: np.array
            x,y,z coordinates of the fitted cylinder fit_C
        - base_node_new: list
            x,y,z coordinates of the new base node  
    '''
    # extract points in the bottom slice    
    if 'downsample' in pc.columns:
        base_slice = pc[(pc.z <= (min(pc.z)+base_slice_len)) &\
                             (pc.downsample == True)][['x','y','z']]
    else:
        base_slice = pc[pc.z <= (min(pc.z)+base_slice_len)][['x','y','z']]
    
    pts = base_slice.to_numpy()
    axis_dir, fit_C, r_fit, fit_err = cyl_fit.fit(pts)
    # print(f'fitted centre coords = {fit_C}')
    # print(f'axis direction = {axis_dir}')
    # print(f'fitted radius = {r_fit}')
    # print(f'fit error = {fit_err}')

    # find the min Z coordinate of the point cloud
    lowest_z = np.array(pc.z[pc.z == min(pc.z)])[0]

    # define new base node as the point located on the cyl axis line
    # with Z coord the same as the lowest point in pc
    # calculate its X,Y coords based on line equation defined by axis direction
    base_x = (lowest_z - fit_C[2]) / axis_dir[2] * axis_dir[0] + fit_C[0]
    base_y = (lowest_z - fit_C[2]) / axis_dir[2] * axis_dir[1] + fit_C[1]
    new_base = [base_x, base_y, lowest_z]
    # print(f'new base node coords: {new_base}')
    
    return base_slice, fit_C, axis_dir, new_base    