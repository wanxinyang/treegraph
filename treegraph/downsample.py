import pandas as pd
import numpy as np
import string
import struct

# from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

def voxelise(tmp, length, method='random'):

    tmp.loc[:, 'xx'] = tmp.x // length * length
    tmp.loc[:, 'yy'] = tmp.y // length * length
    tmp.loc[:, 'zz'] = tmp.z // length * length

    if method == 'random':
            
        code = lambda: ''.join(np.random.choice([x for x in string.ascii_letters], size=8))
            
        xD = {x:code() for x in tmp.xx.unique()}
        yD = {y:code() for y in tmp.yy.unique()}
        zD = {z:code() for z in tmp.zz.unique()}
            
        tmp.loc[:, 'VX'] = tmp.xx.map(xD) + tmp.yy.map(yD) + tmp.zz.map(zD)
   
    elif method == 'bytes':
        
        code = lambda row: np.array([row.xx, row.yy, row.zz]).tobytes()
        tmp.loc[:, 'VX'] = self.pc.apply(code, axis=1)
        
    else:
        raise Exception('method {} not recognised: choose "random" or "bytes"')
 
    return tmp 


def run(pc, vlength, 
        base_location=None,  
        remove_noise=False, 
        min_pts=1,
        delete=False,
        keep_columns=[],
        voxel_method='random',
        verbose=False):
    
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

    pc_length = len(pc)
    pc = pc.drop(columns=[c for c in ['downsample', 'VX'] if c in pc.columns])   
 
    if base_location is None:
        base_location = pc.loc[pc.z.idxmin()].pid.values[0]

    columns = pc.columns.to_list() + keep_columns # required for tidy up later
    pc = voxelise(pc, vlength, method=voxel_method)

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
    if base_location not in pc.loc[pc.downsample].pid.values:
        pc.loc[pc.downsample, 'nndist'] = np.linalg.norm(pc.loc[pc.pid == base_location][['x', 'y', 'z']].values - 
                                                         pc.loc[pc.downsample][['x', 'y', 'z']], axis=1)
        base_location = pc.loc[pc.nndist == pc.nndist.min()].pid.values[0]
    
    pc.reset_index(inplace=True, drop=True)

    if delete:
        pc = pc.loc[pc.downsample][columns].reset_index(drop=True)
    else:
        pc = pc[columns + ['downsample']]

    if verbose: print('downsampled point cloud from {} to {} with edge lengthi {}. Points deleted: {}'.format(pc_length, len(pc), vlength, delete))

    return pc, base_location
