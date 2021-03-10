import pandas as pd
import numpy as np
import string

# from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def voxelise(tmp, length):

    tmp.loc[:, 'xx'] = tmp.x // length * length
    tmp.loc[:, 'yy'] = tmp.y // length * length
    tmp.loc[:, 'zz'] = tmp.z // length * length
    
    code = lambda: ''.join(np.random.choice([x for x in string.ascii_letters], size=8))
    
    xD = {x:code() for x in tmp.xx.unique()}
    yD = {y:code() for y in tmp.yy.unique()}
    zD = {z:code() for z in tmp.zz.unique()}
    
    tmp.loc[:, 'VX'] = tmp.xx.map(xD) + tmp.yy.map(yD) + tmp.zz.map(zD)
    
    return tmp

def downsample(pc, base_location, vlength, remove_noise=False, min_pts=1):
    
    """
    Downsamples a point cloud so that there is one point per voxel.
    Points are selected as the point closest to the median xyz value
    
    Parameters
    ----------
    
    pc: pd.DataFrame with x, y, z columns
    vlength: float
    
    
    Returns
    -------
    
    pd.DataFrame with boolean downsample column
    
    """
    
    pc = voxelise(pc, vlength)

    if remove_noise:
        # dissolve voxels with too few points in to neighbouring voxels
        #     compute N points per voxel
        #     rename to count
        #     join with df of voxel median xyz
        #     reset index
        VX = pd.DataFrame(pc.groupby('VX').x.count()) \
                .rename(columns={'x':'cnt'}) \
                .join(pc.groupby('VX')[['x', 'y', 'z']].median()) \
                .reset_index()
#         dist_between_voxels = calc_distance((VX.loc[VX.cnt > min_pts][['x', 'y', 'z']] * 1000).to_numpy(dtype=np.int8), 
#                                             (VX[['x', 'y', 'z']] * 1000).to_numpy(dtype=np.int8))     # min_pts is 5
        
#         return pc, base_location, VX, dist_between_voxels
#         VX.loc[:, 'VXn'] = VX.loc[VX.index[dist_between_voxels.argmin(axis=0)]].VX.values        
#         VX_map = {row.VX:row.VXn for row in VX[['VX', 'VXn']].itertuples()}
#         pc.VX = pc.VX.map(VX_map)
        nbrs = NearestNeighbors(n_neighbors=10, leaf_size=15, n_jobs=-1).fit(VX[['x', 'y', 'z']])
        distances, indices = nbrs.kneighbors(VX[['x', 'y', 'z']])
        idx = np.argmax(np.isin(indices, VX.loc[VX.cnt > min_pts].index.to_numpy()), axis=1)
        idx = [indices[i, ix] for i, ix in zip(range(len(idx)), idx)]
        VX_map = {vx:vxn for vx, vxn in zip(VX.VX.values, VX.loc[idx].VX.values)}
        pc.VX = pc.VX.map(VX_map)

    # groubpy to find central (closest to median) point
    groupby = pc.groupby('VX')
    pc.loc[:, 'mx'] = groupby.x.transform(np.median)
    pc.loc[:, 'my'] = groupby.y.transform(np.median)
    pc.loc[:, 'mz'] = groupby.z.transform(np.median)
    pc.loc[:, 'dist'] = np.linalg.norm(pc[['x', 'y', 'z']].to_numpy(dtype=np.float32) - 
                                       pc[['mx', 'my', 'mz']].to_numpy(dtype=np.float32), axis=1)
    # need to keep all points for cylinder fitting so when downsampling
    # just adding a column to select by
    pc.loc[:, 'downsample'] = False
    pc.loc[~pc.sort_values(['VX', 'dist']).duplicated('VX'), 'downsample'] = True
    pc.sort_values('downsample', ascending=False, inplace=True) # sorting to base_location index is correct

    # upadate base_id
    nndist = np.linalg.norm(pc.loc[base_location][['x', 'y', 'z']].astype(float) - 
                            pc.loc[pc.downsample][['x', 'y', 'z']].astype(float), axis=1)
    base_location = nndist.argmin()
    pc.reset_index(inplace=True, drop=True)
    
    return pc, base_location