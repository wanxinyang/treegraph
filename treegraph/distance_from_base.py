import numpy as np
import pandas as pd
import os

from treegraph.third_party import shortpath as p2g
from treegraph import downsample
from treegraph.third_party.pcd_io import write_pcd  
from subprocess import getoutput  

def run(pc, base_location=None, new_base_coords=None,\
        low_slice_length=.2, cluster_size=False, knn=100, verbose=False):
    
    """
    Attributes each point with a distance from base
    
    base_location: None or idx (default None)
        Index of the base point i.e. where point distance are measured to.
    new_base_node: None or list (default None)
        New base node coordinates [x,y,z] calculated by base_fitting().
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
    # generate initial graph
    G = p2g.array_to_graph(pc.loc[pc.downsample][c] if 'downsample' in pc.columns else pc[c], 
                           base_id=pc.loc[pc.pid == base_location].index[0], 
                           kpairs=3, 
                           knn=knn, 
                           nbrs_threshold=.2,
                           nbrs_threshold_step=.1,
#                                 graph_threshold=.05
                            )

    if new_base_coords is None:
        # extracts shortest path information from the initial graph
        node_ids, distance, path_dict = p2g.extract_path_info(G, pc.loc[pc.pid == base_location].index[0])
    else:
        # select points in the lowest slice
        if 'downsample' in pc.columns:
            low_slice = pc[(pc.z <= (min(pc.z)+low_slice_length)) &\
                                (pc.downsample == True)]
        else:
            low_slice = pc[pc.z <= (min(pc.z)+low_slice_length)]

        # calculate distance between new_base_node and each point in the lowest slice
        index = []
        distance = []
        for i, row in low_slice.iterrows():
            index.append(i)
            c = row[['x','y','z']].values
            distance.append(np.linalg.norm(new_base_coords - c))

        # add the new base node to the graph    
        new_base_id = np.max(pc.index) + 1
        G.add_node(new_base_id)

        # add edges (weighted by distance) between new base node and the low_slice nodes in graph
        p2g.add_nodes(G, new_base_id, index, distance, np.inf)
        print(f'add new edges: {len(index)}')

        # add new_base_node attributes to pc
        if 'downsample' in pc.columns:
            base_coords = pd.Series({'x':new_base_coords[0], \
                                    'y':new_base_coords[1], \
                                    'z':new_base_coords[2], \
                                    'pid':new_base_id,\
                                    'downsample':True},\
                                    name=new_base_id)
        else:
            base_coords = pd.Series({'x':new_base_coords[0], \
                                    'y':new_base_coords[1], \
                                    'z':new_base_coords[2], \
                                    'pid':new_base_id},\
                                    name=new_base_id)
        pc = pc.append(base_coords)
    
        # extracts shortest path information from the updated initial graph
        node_ids, distance, path_dict = p2g.extract_path_info(G, new_base_id)

    
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
    
    return pc[columns], G, path_dict


def base_fitting(self, base_slice_length=2.0, \
                 pc_path='../data/tree1233_ds.ply', output_path='../results/'):
    '''
    Find new base node by ransac cylinder fitting at a bottom slice.
    
    Inputs:
        - self: dataframe
        - base_slice_length: float
            Vertical length of segmented slice from base (unit: meter).
            This slice is the input for ransac cyl fitting.
        - pc_path: str
            Path of the original input point cloud file.
        - output_path: str
            Path of .pcd file which saves base slice points.
    
    Ouputs:
        - base_node_new: list
            X,Y,Z coords of the new base node
        - .pcd file 
            Points coords in base slice
    
    '''
    # extract points in the bottom slice    
    if 'downsample' in self.pc.columns:
        base_slice = self.pc[(self.pc.z <= (min(self.pc.z)+base_slice_length)) &\
                             (self.pc.downsample == True)][['x','y','z']]
    else:
        base_slice = self.pc[self.pc.z <= (min(self.pc.z)+base_slice_length)][['x','y','z']]

    # save slice pc in a .pcd file
    fn = os.path.splitext(pc_path)[0].split('/')[2]
    output_f = output_path + fn + f'_base_slice.pcd'
    write_pcd(base_slice, output_f)
    print(f'Trunk base slice pc has been saved in: \n{output_f}')
    
    # use treeseg ransac cyl fitting method to fit the bottom slice
    output = getoutput(f'/home/ucfaptv/opt/treeseg/build/ransac {output_f}\n')

    # retrive the cyl centre coords, axis direction and radius
    cx = float(output.split(' ')[1])
    cy = float(output.split(' ')[2])
    cz = float(output.split(' ')[3])
    ax = float(output.split(' ')[4])
    ay = float(output.split(' ')[5])
    az = float(output.split(' ')[6])
    r = float(output.split(' ')[7])

    fitted_centre = [cx,cy,cz]
    axis_dir = [ax,ay,az]
    rad = r 
    # print(f'fitted centre coords = {fitted_centre}')
    # print(f'axis direction = {axis_dir}')
    # print(f'fitted radius = {rad}\n')
    
    # find the min Z coordinate of the point cloud
    lowest_z = np.array(self.pc.z[self.pc.z == min(self.pc.z)])[0]

    # define new base node as the point located on the cyl axis 
    # with Z coord the same as the lowest point in pc
    # calculate its X,Y coords based on line equation defined by axis direction
    base_x = (lowest_z - cz) / axis_dir[2] * axis_dir[0] + cx
    base_y = (lowest_z - cz) / axis_dir[2] * axis_dir[1] + cy
    new_base_coords = [base_x, base_y, lowest_z]
    # print(f'new base node coords: {new_base_coords}')
    
    return base_slice, fitted_centre, new_base_coords
