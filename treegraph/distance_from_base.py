import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from treegraph import common
from treegraph.third_party import shortpath as p2g
from treegraph import downsample
from treegraph.third_party import cylinder_fitting as cyl_fit
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def run(pc, base_location=None, cluster_size=False, knn=100, verbose=False,\
        base_correction=True, plot=False):
    
    """
    Purpose: Attributes each point with distance_from_base
    
    Inputs:
        pc: pd.DataFrame
            Input point clouds
        base_location: None or idx (default None)
                       Index of the base point i.e. where point distance are measured to.
        cluster_size: False or float (default False)
                      Downsample points with vlen=cluster_size to speed up graph generation.
                      The redundent points will be remained for later process.
        knn: int (default 100)
             Number of neighbors to search around each point in the neighborhood phase. 
             The higher the better (careful, it's  memory intensive).
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
    
    c = ['x', 'y', 'z', 'pid']
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
        # identify and extract lower part of the stem
        stem, fit_cyl, new_base = identify_stem(pc, plot=plot)
        # fit a cylinder to the lower stem
        fit_C, axis_dir, fit_r, fit_err = fit_cyl

        # select points in the lowest slice
        low_slice_len = 0.2  # unit in metre
        if 'downsample' in pc.columns:
            low_slice = pc[(pc.z <= (min(stem.z)+low_slice_len)) &\
                           (pc.downsample == True)]
        else:
            low_slice = pc[pc.z <= (min(stem.z)+low_slice_len)]


        # calculate distance between new_base_node and each point in the lowest slice
        index = []
        distance = []
        for i, row in low_slice.iterrows():
            index.append(i)
            coor = row[['x','y','z']].values
            distance.append(np.linalg.norm(new_base - coor))

        # add the new base node to the graph    
        new_base_id = np.max(pc.index) + 1
        G.add_node(int(new_base_id), 
                   pos=[float(new_base[0]), float(new_base[1]), float(new_base[2])], 
                   pid=int(new_base_id))

        # add edges (weighted by distance) between new base node and the low_slice nodes in graph
        p2g.add_nodes(G, new_base_id, index, distance, np.inf)
        # print(f'add new edges: {len(index)}')

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
 
    else: # do not generate new base node nor update the initial graph
        # extracts shortest path information from the initial graph
        node_ids, distance, path_dict = p2g.extract_path_info(G, pc.loc[pc.pid == base_location].index[0])
        
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
    
    if base_correction:
        return pc[columns], G, new_base, fit_r
    else:
        return pc[columns], G



def identify_stem(pc, plot=False):
    # create empty df for later stem point collection
    stem = pd.DataFrame(columns=['x','y','z'])
    
    # tree height from point clouds
    h = pc.z.max() - pc.z.min()
    print(f'tree height (from point clouds) = {h:.2f} m')
    
    if h > 20:  # a tall tree
        stop = h / 10.
        step = h / 30.
    else:  # a small tree
        stop = 2.
        step = .5
        
    # loop over slices of point clouds with vertical interval of 'step'
    # loop stop at the 1/10 of the tree height
    for i in np.arange(0, stop, step):
        zmin = pc.z.min()
        if i < (2*step):
            pc_slice = pc[pc.z < (zmin+i+step)]  
        else:
            pc_slice = pc[((zmin+i) <= pc.z) & (pc.z < (zmin+i+step))] 
        pc_coor = pc_slice.reset_index()[['x','y','z']]
        if len(pc_coor) <= 10:
            continue

        # filter out outliers
        nn = NearestNeighbors(n_neighbors=10).fit(pc_coor)
        dnn, indices = nn.kneighbors()
        mean_dnn = np.mean(dnn, axis=1)
        noise = np.where(mean_dnn > 0.05)[0]
        pc_coor = pc_coor.drop(noise)

        # cluster the sliced points
        dbscan = DBSCAN(eps=.1, min_samples=50).fit(pc_coor)
        pc_coor.loc[:, 'clstr'] = dbscan.labels_

        # calculate normal vector of each cluster and find stem points
        if len(np.unique(dbscan.labels_)) > 1:
            # for the lowest slice, stem part is the cluster with most points
            if len(stem) == 0:
                group = pc_coor.groupby(by='clstr').count()
                max_clstr = group[group.x == group.x.max()].index[0]
                stem = pc_coor[pc_coor.clstr == max_clstr][['x','y','z']]
            # for upper slices, ignore slices with multiple clusters
            # to avoid mixed branch points as well as furcation
            else:
                break
        else:
            for c in np.unique(dbscan.labels_):
                xyz = pc_coor[pc_coor.clstr == c][['x','y','z']]
                nv, d = normal_vector(xyz)
                # if the difference between xyz normal and ground normal is small,
                # then regard xyz as stem points
                if d < 0.5:
                    stem = pd.concat([stem, xyz])
                    stem = stem.drop_duplicates()

    # if no stem point is found
    if len(stem) == 0: 
        stem = pc_coor[['x','y','z']]

 
    # fit a cylinder to the extracted stem points
    pts = stem.to_numpy()
    axis_dir, fit_C, r_fit, fit_err = cyl_fit.fit(pts)

    # find the min Z coordinate of the point cloud
    lowest_z = np.array(stem.z[stem.z == min(stem.z)])[0]

    # define new base node as the point located on the cyl axis line
    # with Z coord the same as the lowest point in pc
    # calculate its X,Y coords based on line equation defined by axis direction
    base_x = (lowest_z - fit_C[2]) / axis_dir[2] * axis_dir[0] + fit_C[0]
    base_y = (lowest_z - fit_C[2]) / axis_dir[2] * axis_dir[1] + fit_C[1]
    base_node = [base_x, base_y, lowest_z]
    
    cpc_cen = common.CPC(stem).x
    if stem.x.min() <= base_node[0] <= stem.x.max():
        if stem.y.min() <= base_node[1] <= stem.y.max():
            new_base = base_node
        else:
            new_base = cpc_cen
    else:
        new_base = cpc_cen
        
    # plot extracted stem points and fitted cylinder
    if plot == True:
        fig, axs = plt.subplots(1,3,figsize=(12,4))
        ax = axs.flatten()
        # top view
        stem.plot.scatter(x='x',y='y',s=1,ax=ax[0], c='grey')
        ax[0].scatter(fit_C[0], fit_C[1], s=50, c='blue', label='fitted cyl centre')
        ax[0].scatter(new_base[0], new_base[1], s=50, c='red', label='new base node')

        # front view
        stem.plot.scatter(x='x',y='z',s=1,ax=ax[1], c='grey')
        # fitted cyl centre
        ax[1].scatter(fit_C[0], fit_C[2], s=50, c='blue', label='fitted cyl centre')
        # fitted cyl axis
        z = np.arange(stem.z.min(), stem.z.max(), 0.01)
        x = (axis_dir[0]/axis_dir[2]) * (z - fit_C[2]) + fit_C[0]
        ax[1].plot(x, z, linestyle='dashed', label='fitted cyl axis')
        # new base node
        ax[1].scatter(new_base[0], new_base[2], s=50, c='red', label='new base node')

        # side view
        stem.plot.scatter(x='y',y='z',s=1,ax=ax[2], c='grey')
        # fitted cyl centre
        ax[2].scatter(fit_C[1], fit_C[2], s=50, c='blue', label='fitted cyl centre')
        # fitted cyl axis
        z = np.arange(stem.z.min(), stem.z.max(), 0.01)
        y = (axis_dir[1]/axis_dir[2]) * (z - fit_C[2]) + fit_C[1]
        ax[2].plot(y, z, linestyle='dashed', label='fitted cyl axis')
        # new base node
        ax[2].scatter(new_base[1], new_base[2], s=50, c='red', label='new base node')
        ax[2].legend(bbox_to_anchor=(1.05, 1))

        # fig.suptitle(f'{treeid}')
        fig.tight_layout()

    return stem, [fit_C, axis_dir, r_fit, fit_err], new_base


def normal_vector(pc):
    '''
    Calculate the normal vector of input point cloud,
    and the difference between this normal and ground normal.

    Input:
        - pc: n×3 array, X,Y,Z coordinates of points

    Output:
        - nv: normal vector of input point cloud
        - d: a metric quantifying the difference between pc normal and ground normal
    '''
    # calculating centroid coordinates of points in 'arr'.
    centroid = np.average(pc, axis=0)

    # run SVD on centered points from 'arr'.
    _, evals, evecs = np.linalg.svd(pc - centroid, full_matrices=False)
    
    # normal vector of input pc
    # is the eigenvector associated with the smallest eigenvalue
    nv = evecs[np.argmin(evals)]
    
    # normal vector of the ground
    n0 = np.array([0,0,1])
    
    # difference between pc normal and ground normal
    d = np.abs(np.dot(nv, n0))
    
    return nv, d
