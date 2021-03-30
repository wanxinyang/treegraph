import pandas as pd
import numpy as np
import random
import struct
from sklearn.cluster import DBSCAN
from tqdm.autonotebook import tqdm

from treegraph.third_party import shortpath as p2g
from treegraph.downsample import *

from pandarallel import pandarallel


def find_centre(dslice, min_pts, eps):
    
    if len(dslice) < min_pts: return []
    
    centres = pd.DataFrame()    
    s = dslice.slice_id.unique()[0]

    # separate different slice components e.g. different branches
    dbscan = DBSCAN(eps=eps, 
                    min_samples=1, 
                    algorithm='kd_tree', 
                    metric='chebyshev',
                    n_jobs=-1).fit(dslice[['x', 'y', 'z']]) 
    dslice.loc[:, 'centre_id'] = dbscan.labels_

    for c in np.unique(dbscan.labels_):

        # working on each separate branch
        nvoxel = dslice.loc[dslice.centre_id == c]
        if len(nvoxel.index) < min_pts: 
            dslice = dslice.loc[~dslice.index.isin(nvoxel.index)]
            continue # required so centre is added after points are deleted
        centre_coords = nvoxel[['x', 'y', 'z']].median()

        centres = centres.append(pd.Series({'slice_id':int(s), 
                                            'centre_id':int(c), 
                                            'cx':centre_coords.x, 
                                            'cy':centre_coords.y, 
                                            'cz':centre_coords.z, 
                                            'distance_from_base':nvoxel.distance_from_base.mean(),
                                            'n_points':len(nvoxel),
                                            'idx':struct.pack('ii', int(s), int(c))}),
                                           ignore_index=True)
        
        dslice.loc[(dslice.slice_id == s) & 
                   (dslice.centre_id == c), 'idx'] = struct.pack('ii', int(s), int(c))

    if isinstance(centres, pd.DataFrame):
        return [centres, dslice] 
    
def run(pc, eps, min_pts=0, verbose=False):
    
    columns = pc.columns.to_list() + ['node_id']

    # run pandarallel on groups of points
    groupby = pc.groupby('slice_id')
    pandarallel.initialize(nb_workers=min(24, len(groupby)), progress_bar=verbose)
    try:
        sent_back = groupby.parallel_apply(find_centre, min_pts, eps).values
    except OverflowError:
        if verbose: print('!pandarallel could not initiate progress bars, running without')
        pandarallel.initialize(progress_bar=False)
        sent_back = groupby.parallel_apply(find_centre, min_pts, eps).values

    # create and append clusters and filtered pc
    centres = pd.DataFrame()
    pc = pd.DataFrame()
    for x in sent_back:
        if len(x) == 0: continue
        centres = centres.append(x[0])
        pc = pc.append(x[1])

    # reset index as appended df have common values
    centres.reset_index(inplace=True, drop=True)
    pc.reset_index(inplace=True, drop=True)

    if 'node_id' in pc.columns: pc = pc.drop(columns=['node_id'])
    
    # convert binary cluster reference to int
    MAP = {v:i for i, v in enumerate(centres.idx.unique())}
    if 'level_0' in pc.columns: pc = pc.drop(columns='level_0')
    if 'index' in pc.columns: pc = pc.drop(columns='index')
    pc.loc[:, 'node_id'] = pc.idx.map(MAP)
    centres.loc[:, 'node_id'] = centres.idx.map(MAP)
    
    return pc[columns], centres
