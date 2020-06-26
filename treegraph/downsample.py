import struct
import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist


def voxelise(tmp, length):

    binarize = lambda x: struct.pack('i', int((x * 1000.) / (length * 1000)))

    xb = tmp.x.apply(binarize)
    yb = tmp.y.apply(binarize)
    zb = tmp.z.apply(binarize)
    tmp.loc[:, 'VX'] = xb + yb + zb

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
        dist_between_voxels = cdist(VX.loc[VX.cnt > min_pts][['x', 'y', 'z']], 
                                    VX[['x', 'y', 'z']])     # min_pts is 5
        VX.loc[:, 'VXn'] = VX.loc[VX.index[dist_between_voxels.argmin(axis=0)]].VX.values
        VX_map = {row.VX:row.VXn for row in VX[['VX', 'VXn']].itertuples()}
        pc.VX = pc.VX.map(VX_map)
    
    # groubpy to find central (closest to median) point
    groupby = pc.groupby('VX')
    pc.loc[:, 'mx'] = groupby.x.transform(np.median)
    pc.loc[:, 'my'] = groupby.y.transform(np.median)
    pc.loc[:, 'mz'] = groupby.z.transform(np.median)
    pc.loc[:, 'dist'] = np.linalg.norm(pc[['x', 'y', 'z']].values - 
                                            pc[['mx', 'my', 'mz']].values, axis=1)
    # need to keep all points for cylinder fitting so when downsampling
    # just adding a column to select by
    pc.loc[:, 'downsample'] = False
    pc.loc[~pc.sort_values(['VX', 'dist']).duplicated('VX'), 'downsample'] = True
    pc.sort_values('downsample', ascending=False, inplace=True) # sorting to base_location index is correct

    # upadate base_id
    nndist = np.linalg.norm(pc.loc[base_location][['x', 'y', 'z']] - 
                            pc.loc[pc.downsample][['x', 'y', 'z']], axis=1)
    base_location = nndist.argmin()
    pc.reset_index(inplace=True, drop=True)
    
    return pc, base_location