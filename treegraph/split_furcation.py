import pandas as pd
import numpy as np
import struct
from tqdm import tqdm
from sklearn.cluster import KMeans
from treegraph.fit_cylinders import *
from treegraph.build_skeleton import *
from treegraph.build_graph import *
from treegraph.attribute_centres import *
from treegraph.common import *
from treegraph.third_party.point2line import *
from treegraph.third_party.closestDistanceBetweenLines import *
from treegraph.third_party import ransac_cyl_fit
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.stats import variation
from numpy import linalg as LA
from sklearn.cluster import SpectralClustering
from pandarallel import pandarallel


def run(pc, centres):
    '''
    Refine skeleton nodes using self-tunning spectral clustering method.
    
    Inputs:
        pc: pd.DataFrame
            original points and their attributes
        centres: pd.DataFrame
            skeleton nodes and their attributes
    
    Outputs:
        pc: pd.DataFrame
            new pc after updating clustering
        centres: pd.DataFrame
            new skeleton nodes after updateing clustering
    '''
    for i, sid in enumerate(np.sort(centres.slice_id.unique())):
        nid = centres[centres.slice_id == sid].node_id.values
        if len(nid) > 1:
            break
    if i == 0:
        sid_start = int(np.sort(centres.slice_id.unique())[i])
    else:
        sid_start = int(np.sort(centres.slice_id.unique())[i-1])

    # run spectral clustering on all nodes except for lower trunk before furcation
    nodes = centres[centres.slice_id >= sid_start].node_id.unique()
    samples = pc[pc.node_id.isin(nodes)]
    
    # run pandarallel on groups of points
    groupby = samples.groupby('node_id')
    sent_back = groupby.apply(stsc_recursive, centres, sid_start=sid_start).values
    

    # update pc and centres
    new_pc = pd.DataFrame()
    new_centres = pd.DataFrame()
    # nid_max = centres.node_id.max() + 1
    nid_max = pc.node_id.max() + 1

    for x in sent_back:
        if len(x) == 0:
            continue
        new_pc = new_pc.append(x[0])
        new_centres = new_centres.append(x[1])
        # remove splitted nodes
        centres = centres.loc[centres.node_id != x[2]]

    new_centres.reset_index(inplace=True)
    if len(new_centres) < 1:
        return pc, centres

    # re-arrange node_id
    MAP = {v : i+nid_max for i, v in enumerate(new_centres.idx.unique())}
    new_pc.loc[:, 'node_id'] = new_pc.idx.map(MAP)
    new_centres.loc[:, 'node_id'] = new_centres.idx.map(MAP)

    # update centres df
    centres = centres.append(new_centres).sort_values(by=['slice_id','node_id'])
    centres.reset_index(inplace=True, drop=True)
    if 'index' in centres.columns:
        centres = centres.drop(columns=['index'])
    # update centre_id
    for s in centres.slice_id.unique():
        sn = len(centres[centres.slice_id == s])
        centres.loc[centres.slice_id == s, 'centre_id'] = np.arange(0, sn)
    # update pc    
    pc.loc[pc.index.isin(new_pc.index), 'idx'] = new_pc.idx
    pc.loc[pc.index.isin(new_pc.index), 'node_id'] = new_pc.node_id
    
    def update_cid(x):
        x.loc[:, 'centre_id'] = centres[centres.node_id == x.node_id.values[0]].centre_id.values[0]
        return x
    pc = pc[pc.node_id.isin(centres.node_id.unique())]
    pc = pc.groupby('node_id').apply(update_cid).reset_index(drop=True)

    return pc, centres



def stsc(clus_pc, centres, nn=None, sid_start=None, plot=False, point_size=8):
    # attr of this cluster
    sid = clus_pc.slice_id.unique()[0]  # slice_id
    nid = clus_pc.node_id.unique()[0]  # node_id
    cid = clus_pc.centre_id.unique()[0]  # centre_id
    pts = len(clus_pc)
    
    if sid_start is None:
        for i, s in enumerate(np.sort(centres.slice_id.unique())):
            nids = centres[centres.slice_id == s].node_id.values
            if len(nids) > 1:
                break
        if i == 0:
            sid_start = int(np.sort(centres.slice_id.unique())[i])
        else:
            sid_start = int(np.sort(centres.slice_id.unique())[i-1])

    if nn is None:
        # determine n_neighbours for spectral clustering
        if pts <= 5: return []
        elif pts <= 20: nn = 5
        elif pts <= 50: nn = 10
        elif pts <= 200: nn = 20
        elif pts <= 500: nn = int(pts*0.1)
        else: nn = int(pts*0.2)
        
    # auto determine cluster number from eigen gaps
    W = getAffinityMatrix(clus_pc[['x','y','z']], k = nn)
    k, eigenvalues, eigenvectors = eigenDecomposition(W, plot=False, topK=3)

    # spectral clustering
    spectral = SpectralClustering(n_clusters=k[0],
                                  random_state=0,
                                  affinity='nearest_neighbors',
                                  n_neighbors=nn).fit(clus_pc[['x', 'y', 'z']])
    clus_pc.loc[:, 'klabels'] = spectral.labels_
    c_num = len(np.unique(spectral.labels_))
    
    # stop if current cluster cannot be segmented into more sub-clusters
    if c_num == 1:
        return []

    # otherwise, try to merge over-segmented sub-clusters that belong to a single branch
    # firstly, calculate dist between cluster points to its centre
    n = np.unique(clus_pc.node_id)[0]
    orig_cen = centres[centres.node_id == n][['cx','cy','cz']].values[0]
    dist_mean = cdist([orig_cen], clus_pc[['x','y','z']]).mean()
    dist_std = cdist([orig_cen], clus_pc[['x','y','z']]).std()
    cv = dist_std / dist_mean

    tmp = pd.DataFrame()
            
    if cv > .4:
        # don't apply merge process if the dispersion of points is too large 
        # to be regarded as part of a single branch
        clus_pc.loc[:, 'adj_k'] = clus_pc.klabels
    
    else:  # apply merge process based on threshold
        count = 1
        # loop over each new cluster
        for klabel in np.unique(spectral.labels_):
            
            # calculate threshold to merge sub-clusters
            # assume threshold decreases exponentially with distance_from_base
            mind = centres[centres.slice_id == sid_start].distance_from_base.values[0]
            maxd = centres.distance_from_base.max()
            currd = centres[centres.node_id == nid].distance_from_base.values[0]
            ratio = 1- ((maxd - currd) / (maxd - mind))
            p = -4 ** (0.5 * ratio) + 2
            threshold = dist_mean * p
#             threshold = dist_mean  # option 2: assume threshold is fixed

            # centre for this sub-cluster
            subclus_pc = clus_pc[clus_pc.klabels == klabel]
            if len(subclus_pc) <= 100:
                new_cen = subclus_pc[['x','y','z']].median().to_numpy()
            else:
                new_cen = CPC(subclus_pc).x

            # dist between sub-cluster centre and the original cluster centre
            d_cc = np.linalg.norm(new_cen - orig_cen)
            if d_cc <= threshold:
                # merge sub-clusters that less than threshold
                tmp = tmp.append(clus_pc[clus_pc.klabels == klabel])   
            else:
                clus_pc.loc[clus_pc.klabels == klabel, 'adj_k'] = int(count)
                count += 1
        # regard all merged sub-clusters as a single cluster
        clus_pc.loc[clus_pc.index.isin(tmp.index), 'adj_k'] = int(count)
    
    c_num_merge = len(clus_pc[clus_pc.adj_k >= 0].adj_k.unique())
    
    if plot:
        pc_stsc = clus_pc
        fig, axs = plt.subplots(1,4,figsize=(15,4.8))
        ax = axs.flatten()
        # spectral clustering results
        ax[0].scatter(pc_stsc.y, pc_stsc.z, c=pc_stsc.klabels, cmap='Pastel1', s=point_size)
        ax[1].scatter(pc_stsc.x, pc_stsc.y, c=pc_stsc.klabels, cmap='Pastel1', s=point_size)
        ax[0].set_xlabel('Y coords (m)')
        ax[0].set_ylabel('Z coords (m)')
        ax[1].set_xlabel('X coords (m)')
        ax[1].set_ylabel('Y coords (m)')
        ax[0].set_title('Before merge')
        ax[1].set_title('Before merge')
        
        # after adjustment based on threshold 
        ax[2].scatter(pc_stsc.y, pc_stsc.z, c=pc_stsc.adj_k, cmap='Pastel1', s=point_size)
        ax[3].scatter(pc_stsc.x, pc_stsc.y, c=pc_stsc.adj_k, cmap='Pastel1', s=point_size)
        ax[2].set_xlabel('Y coords (m)')
        ax[2].set_ylabel('Z coords (m)')
        ax[3].set_xlabel('X coords (m)')
        ax[3].set_ylabel('Y coords (m)')
        ax[2].set_title('After merge')
        ax[3].set_title('After merge')

        fig.suptitle(f'sid={sid}, nid={nid}, pts={pts}, nn={nn}, c_num={c_num}, c_num_m={c_num_merge}, dist_p2c={dist_mean:.2f}m ± {dist_std:.2f}m, cv={cv:.2f}',
                     fontsize=15)
        fig.tight_layout()

    return clus_pc


def stsc_recursive(pc, centres, nn=None, sid_start=None, plot=False):
    # attr of this cluster
    sid = pc.slice_id.unique()[0]  # slice_id
    nid = pc.node_id.unique()[0]  # node_id
    cid = pc.centre_id.unique()[0]  # centre_id
    
    # slice_id where trunk starts furcation
    if sid_start is None:
        for i, s in enumerate(np.sort(centres.slice_id.unique())):
            nids = centres[centres.slice_id == s].node_id.values
            if len(nids) > 1:
                break
        if i == 0:
            sid_start = int(np.sort(centres.slice_id.unique())[i])
        else:
            sid_start = int(np.sort(centres.slice_id.unique())[i-1])
    
    ## 1st stsc
    subclus_1 = stsc(pc, centres, nn=nn, sid_start=sid_start, plot=False, point_size=5)

    if len(subclus_1) < 1: 
        return []  
#     print(f'after 1st stsc, sub-clusters are: {np.sort(subclus_1.adj_k.unique())}\n')
    kmax = subclus_1.adj_k.max()

    if len(np.sort(subclus_1.adj_k.unique())) < 2:  
        # don't do further stsc if only one sub-cluster
        pc.loc[:, 'adj_k'] = subclus_1.adj_k
    else:
        # loop over each sub-cluster
        for k in np.sort(subclus_1.adj_k.unique()):
            subpc = subclus_1[subclus_1.adj_k == k]
            if len(subpc) <= 20:
                continue          
            # sub-cluster new centre
            if len(subpc) <= 100:
                new_cen = subpc[['x','y','z']].median().to_numpy()
            else:
                new_cen = CPC(subpc).x

            # dist between sub-cluster points to its centre
            dist_mean = cdist([new_cen], subpc[['x','y','z']]).mean()
            dist_std = cdist([new_cen], subpc[['x','y','z']]).std()
            cv = dist_std / dist_mean
#             print(f'sid={sid}, nid={nid}, 1st stsc adj_k={int(k)}:')
#             print(f'\tdist_mean={dist_mean:.2f}, dist_std={dist_std:.2f}, cv={cv:.2f}')

            ## try 2nd stsc if pts in sub-cluster are still in high degree of dispersion
            if (cv > .3) or (dist_mean >= .5):
                subclus_2 = stsc(subpc, centres, nn=nn, sid_start=sid_start, plot=False, point_size=10)
                if len(subclus_2) < 1:
                    continue
                subclus_2.loc[:, 'adj_k'] = subclus_2.adj_k + kmax
                kmax += len(np.unique(subclus_2.adj_k))
                pc.loc[pc.index.isin(subclus_2.index), 'adj_k'] = subclus_2.adj_k

                # loop over each sub-sub-cluster
                for kk in np.sort(subclus_2.adj_k.unique()):
                    subpc_ = subclus_2[subclus_2.adj_k == kk]
                    if len(subpc_) <= 20:
                        continue
                    else:
                        ## try 3rd stsc if pts in sub-sub-cluster are still far apart
                        new_cen_ = subpc_[['x','y','z']].median().to_numpy()
                        dist_mean_ = cdist([new_cen_], subpc_[['x','y','z']]).mean()
#                         print(f'\t\t2nd stsc adj_k={int(kk)}: dist_mean={dist_mean:.2f}')

                        if dist_mean >= .5:
                            subclus_3 = stsc(subpc_, centres, nn=nn, sid_start=sid_start, plot=False, point_size=10)
                            if len(subclus_3) < 1:
                                continue
                            subclus_3.loc[:, 'adj_k'] = subclus_3.adj_k + kmax
                            kmax += len(np.unique(subclus_3.adj_k))
                            pc.loc[pc.index.isin(subclus_3.index), 'adj_k'] = subclus_3.adj_k

    pc.loc[:, 'adj_k'] = pc.adj_k.apply(lambda x: np.where(np.array(np.unique(pc.adj_k)) == x)[0][0])
#     print(f'\nFinal subclus labels: {np.sort(pc.adj_k.unique())}')

    
    if plot:
        pc_stsc = pc
        point_size = 10
        pts = len(pc_stsc)
        c_num = len(pc_stsc.adj_k.unique())
        
        fig, axs = plt.subplots(1,4,figsize=(15,4.8))
        ax = axs.flatten()
        # spectral clustering results
        ax[0].scatter(pc_stsc.y, pc_stsc.z, c=pc_stsc.klabels, cmap='Pastel1', s=point_size)
        ax[1].scatter(pc_stsc.x, pc_stsc.y, c=pc_stsc.klabels, cmap='Pastel1', s=point_size)
        ax[0].set_xlabel('Y coords (m)')
        ax[0].set_ylabel('Z coords (m)')
        ax[1].set_xlabel('X coords (m)')
        ax[1].set_ylabel('Y coords (m)')
        ax[0].set_title('Before recursion')
        ax[1].set_title('Before recursion')
        
        # after adjustment based on threshold 
        ax[2].scatter(pc_stsc.y, pc_stsc.z, c=pc_stsc.adj_k, cmap='Pastel1', s=point_size)
        ax[3].scatter(pc_stsc.x, pc_stsc.y, c=pc_stsc.adj_k, cmap='Pastel1', s=point_size)
        ax[2].set_xlabel('Y coords (m)')
        ax[2].set_ylabel('Z coords (m)')
        ax[3].set_xlabel('X coords (m)')
        ax[3].set_ylabel('Y coords (m)')
        ax[2].set_title('After recursion')
        ax[3].set_title('After recursion')
        
        fig.suptitle(f'sid={sid}, nid={nid}, pts={pts}, c_num={c_num}',
                     fontsize=15)
        fig.tight_layout()
    
    
    new_centres = pd.DataFrame()
    # loop over adjusted new clusters
    for kn in np.unique(pc.adj_k):
        # new centres coords and idx
        dcluster = pc[pc.adj_k == kn]
        if len(dcluster) <= 100:
            new_cen_coords = dcluster[['x','y','z']].median().to_numpy()
        else:
            new_cen_coords = CPC(dcluster).x
        new_idx = struct.pack('iii', int(sid), int(cid), int(kn))

        # add new centres to pc
        pc.loc[pc.index.isin(dcluster.index), 'centre_id'] = int(kn)
        pc.loc[pc.index.isin(dcluster.index), 'idx'] = new_idx
        # add new centres to centres
        new_centres = new_centres.append(pd.Series({'slice_id': int(sid), 
                                                    'centre_id':int(kn), # id within a cluster
                                                    'cx':new_cen_coords[0], 
                                                    'cy':new_cen_coords[1], 
                                                    'cz':new_cen_coords[2], 
                                                    'distance_from_base':dcluster.distance_from_base.mean(),
                                                    'n_points':len(dcluster),
                                                    'idx':new_idx}), ignore_index=True)
    pc = pc.drop(columns=['klabels','adj_k'])
    if (len(new_centres) != 0) & (isinstance(new_centres, pd.DataFrame)):
        return [pc, new_centres, nid] 
    else: 
        return []


def split_fur_stsc_cpc(pc, centres):
    '''
    Inputs:
        pc: pd.DataFrame, pc info for an individual node_id
        centres: pd.DataFrame, centres info for either this node_id or all nodes
    Outputs:
        list, [updated_pc, updated_centres, nid]
    '''
    # attr of this cluster
    sid = pc.slice_id.unique()[0]  # slice_id
    nid = pc.node_id.unique()[0]  # node_id
    cid = pc.centre_id.unique()[0]  # centre_id

    # determine n_neighbours for spectral clustering
    pts = len(pc)
    if pts < 20: return []
    if (pts>=20) & (pts<50): nn = 10
    if (pts>=50) & (pts<200): nn = 20
    if pts >= 200: nn = int(pts/5)
    # if (pts >= 200) & (pts < 500): nn = int(pts/5)
    # if pts >= 500: nn = 100

    # auto determine cluster number from eigen gaps
    W = getAffinityMatrix(pc[['x','y','z']], k = nn)
    k, eigenvalues, eigenvectors = eigenDecomposition(W, plot=False, topK=3)

    # spectral clustering
    spectral = SpectralClustering(n_clusters=k[0],
                                  random_state=0,
                                  affinity='nearest_neighbors',
                                  n_neighbors=nn).fit(pc[['x', 'y', 'z']])
    pc.loc[:, 'klabels'] = spectral.labels_

    # calculate dist threshold
    n = np.unique(pc.node_id)[0]
    orig_cen = centres[centres.node_id == n][['cx','cy','cz']].values[0]
    dist_mean = cdist([orig_cen], pc[['x','y','z']]).mean()
    dfb = centres[centres.node_id == n].distance_from_base.values[0]
    dfb_max = centres.distance_from_base.max()
    dfb_ratio = dfb / dfb_max
    threshold = dist_mean * (1 - dfb_ratio) * 1.645
    
    tmp = pd.DataFrame()
    new_centres = pd.DataFrame()

    # current cluster hasn't been further segmented
    if len(np.unique(spectral.labels_)) == 1:
        return []
    # else, loop over each new cluster
    count = 1
    for klabel in np.unique(spectral.labels_):
        # adjust cluster centre coordinates
        opt_cpc = CPC(pc[pc.klabels == klabel])
        new_cen = opt_cpc.x
        
        # dist between new cluster centre and the original centre
        d_cc = np.linalg.norm(new_cen - orig_cen)
        
        # merge sub-clusters that are over-segmented
        if d_cc <= threshold:
            tmp = tmp.append(pc[pc.klabels == klabel])
        # remain sub-clusters that are resonable    
        else:
            pc.loc[pc.klabels == klabel, 'adj_k'] = int(count)
            count += 1
    # regard merged sub-clusters as a single cluster
    pc.loc[pc.index.isin(tmp.index), 'adj_k'] = int(count)
    
    # loop over adjusted new clusters
    for kn in np.unique(pc.adj_k):
        # new centres coords and idx
        dcluster = pc[pc.adj_k == kn]
        if len(dcluster) <= 100:
            new_cen_coords = dcluster[['x','y','z']].median().to_numpy()
        else:
            new_cen_coords = CPC(dcluster).x
        new_idx = struct.pack('iii', int(sid), int(cid), int(kn))

        # add new centres to pc
        pc.loc[pc.index.isin(dcluster.index), 'centre_id'] = int(kn)
        pc.loc[pc.index.isin(dcluster.index), 'idx'] = new_idx
        # add new centres to centres
        new_centres = new_centres.append(pd.Series({'slice_id': int(sid), 
                                                    'centre_id':int(kn), # id within a cluster
                                                    'cx':new_cen_coords[0], 
                                                    'cy':new_cen_coords[1], 
                                                    'cz':new_cen_coords[2], 
                                                    'distance_from_base':dcluster.distance_from_base.mean(),
                                                    'n_points':len(dcluster),
                                                    'idx':new_idx}), ignore_index=True)
    pc = pc.drop(columns=['klabels','adj_k'])
    if (len(new_centres) != 0) & (isinstance(new_centres, pd.DataFrame)):
        return [pc, new_centres, nid] 
    else: 
        return []


def split_fur_stsc(pc):

    # attr of this cluster
    sid = pc.slice_id.unique()[0]  # slice_id
    nid = pc.node_id.unique()[0]  # node_id
    cid = pc.centre_id.unique()[0]  # centre_id

    # determine n_neighbours for spectral clustering
    pts = len(pc)
    if pts < 20: return []
    if (pts>=20) & (pts<50): nn = 10
    if (pts>=50) & (pts<200): nn = 20
    if (pts >= 200) & (pts < 500): nn = int(pts/5)
    if pts >= 500: nn = 100

    # auto determine cluster number from eigen gaps
    W = getAffinityMatrix(pc[['x','y','z']], k = nn)
    k, eigenvalues, eigenvectors = eigenDecomposition(W, plot=False, topK=3)

    # spectral clustering
    spectral = SpectralClustering(n_clusters=k[0],
                                  random_state=0,
                                  affinity='nearest_neighbors',
                                  n_neighbors=nn).fit(pc[['x', 'y', 'z']])
    pc.loc[:, 'klabels'] = spectral.labels_

    new_centres = pd.DataFrame()

    # loop over each new cluster
    if len(np.unique(spectral.labels_)) == 1:
        return []
    for klabel in np.unique(spectral.labels_):
        # new centres coords and idx
        dcluster = pc[pc.klabels == klabel]
        new_cen_coords = dcluster[['x', 'y', 'z']].median()
        new_idx = struct.pack('iii', int(sid), int(cid), int(klabel))

        # add new centres to pc
        pc.loc[pc.index.isin(dcluster.index), 'centre_id'] = int(klabel)
        pc.loc[pc.index.isin(dcluster.index), 'idx'] = new_idx
        # add new centres to centres
        new_centres = new_centres.append(pd.Series({'slice_id': int(sid), 
                                                    'centre_id':int(klabel), # id within a cluster
                                                    'cx':new_cen_coords.x, 
                                                    'cy':new_cen_coords.y, 
                                                    'cz':new_cen_coords.z, 
                                                    'distance_from_base':dcluster.distance_from_base.mean(),
                                                    'n_points':len(dcluster),
                                                    'idx':new_idx}), ignore_index=True)
    pc = pc.drop(columns=['klabels'])
    if (len(new_centres) != 0) & (isinstance(new_centres, pd.DataFrame)):
        return [pc, new_centres, nid] 
    else: 
        return []


def split_fur_fit_cyl(pc):

    # attr of this cluster
    sid = pc.slice_id.unique()[0]  # slice_id
    nid = pc.node_id.unique()[0]  # node_id
    cid = pc.centre_id.unique()[0]  # centre_id

    # determine n_neighbours for spectral clustering
    pts = len(pc)
    if pts < 20: return []
    
    # fit a cyl in this cluster
    if pts > 50:
        pc_xyz = pc[['x','y','z']]
        cyl = ransac_cyl_fit.RANSACcylinderFitting4(pc_xyz.copy(), iterations=50, plot=False)
        rad, cen, err, _ = cyl
        # extract points within the fitted cyl
        pc.loc[:, 'error'] = LA.norm(pc[['x', 'y']] - [cen[0],cen[1]], axis=1) / rad
        idx = pc.loc[pc.error.between(0, 1.3)].index
        pc['cyl'] = 0
        pc.loc[idx, 'cyl'] = 1
        # for points within the cyl, compute coefficient of variation of error
        err = variation(pc[pc.cyl==1].error)
        # print(f'fitting err = {err:.3f}')

        if err <= 0.3:  # fitted cyl represents a sub-cluster
            pc.loc[pc.cyl == 1, 'klabels'] = 0

            # spectral clustering of non-cylinder points
            nc_xyz = pc[pc.cyl == 0][['x','y','z']]
            nc_pts = len(nc_xyz)
            if nc_pts < 20: return []
            if (nc_pts>=20) & (nc_pts<50): test_nn = 10
            if (nc_pts>=50) & (nc_pts<200): test_nn = 20
            if (nc_pts>=200) & (nc_pts<500): test_nn = int(nc_pts/5)
            if nc_pts>=500: test_nn = 100
            
            # calcualte affinity matrix and estimate number of clusters
            W = getAffinityMatrix(nc_xyz, k=test_nn)
            k, eigenvalues, eigenvectors = eigenDecomposition(W, plot=False, topK=3)
            # print(f'\tEst. number of clusters: {k[0]}')
        else:
            nc_xyz = pc[['x','y','z']]
            nc_pts = len(nc_xyz)
            if nc_pts < 20: return []
            if (nc_pts>=20) & (nc_pts<50): test_nn = 10
            if (nc_pts>=50) & (nc_pts<200): test_nn = 20
            if (nc_pts>=200) & (nc_pts<500): test_nn = int(nc_pts/5)
            if nc_pts>=500: test_nn = 100

            # calcualte affinity matrix and estimate number of clusters
            W = getAffinityMatrix(nc_xyz, k=test_nn)
            k, eigenvalues, eigenvectors = eigenDecomposition(W, plot=False, topK=3)
            # print(f'\tEst. number of clusters: {k[0]}')
            
    else:  # this cluster doesn't contain cyl shape 
        nc_xyz = pc[['x','y','z']]
        nc_pts = len(nc_xyz)
        if nc_pts < 20: return []
        if (nc_pts>=20) & (nc_pts<50): test_nn = 10
        if (nc_pts>=50) & (nc_pts<200): test_nn = 20
        if (nc_pts>=200) & (nc_pts<500): test_nn = int(nc_pts/5)
        if nc_pts>=500: test_nn = 100

        # calcualte affinity matrix and estimate number of clusters
        W = getAffinityMatrix(nc_xyz, k=test_nn)
        k, eigenvalues, eigenvectors = eigenDecomposition(W, plot=False, topK=3)
        # print(f'\tEst. number of clusters: {k[0]}')

    # Spectral Clustering using est. n_clusters from eigen gaps
    spectral = SpectralClustering(n_clusters=k[0],
    #                             assign_labels='discretize',  
                                  random_state=0,
                                  affinity='nearest_neighbors',
                                  n_neighbors=test_nn).fit(nc_xyz)
    pc.loc[nc_xyz.index, 'klabels'] = spectral.labels_ + 1

    nlabels = len(pc.klabels.unique())
    if nlabels == 1: return []

    new_centres = pd.DataFrame()
    # loop over each new cluster
    for i, klabel in enumerate(pc.klabels.unique()):
        dcluster = pc[pc.klabels == klabel]
        new_cen_coords = dcluster[['x', 'y', 'z']].median()
        new_idx = struct.pack('iii', int(sid), int(cid), int(i))

        # add new centres to pc
        pc.loc[dcluster.index, 'centre_id'] = int(i)
        pc.loc[dcluster.index, 'idx'] = new_idx
        # add new centres to centres
        new_centres = new_centres.append(pd.Series({'slice_id': int(sid), 
                                                    'centre_id':int(i), # id within a cluster
                                                    'cx':new_cen_coords.x, 
                                                    'cy':new_cen_coords.y, 
                                                    'cz':new_cen_coords.z, 
                                                    'distance_from_base':dcluster.distance_from_base.mean(),
                                                    'n_points':len(dcluster),
                                                    'idx':new_idx}), ignore_index=True)
    for col in ['cyl', 'error', 'klabels']:
        if col in pc.columns:
            pc = pc.drop(columns=col)
    if (len(new_centres) != 0) & (isinstance(new_centres, pd.DataFrame)):
        return [pc, new_centres, nid] 
    else: 
        return []



# https://github.com/ciortanmadalina/high_noise_clustering/blob/master/spectral_clustering.ipynb
def getAffinityMatrix(coordinates, k = 5):
    """
    Calculate affinity matrix based on input coordinates matrix and the numeber
    of nearest neighbours.
    
    Apply local scaling based on the k nearest neighbour
        References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates)) 
    
    # for each row, sort the distances ascendingly and take the index of the 
    #k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T
    
    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


# https://github.com/ciortanmadalina/high_noise_clustering/blob/master/spectral_clustering.ipynb
def eigenDecomposition(A, plot = True, topK=10):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors
    
    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic
    
    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    
    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
    # the euclidean norm of complex numbers.
#     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = LA.eig(L)
    
    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues, s=2)
        plt.grid()
        
    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1
        
    return nb_clusters, eigenvalues, eigenvectors


## updated method but without parallel running
# def run(pc, centres):
#     '''
#     Split furcations using spectral clustering method.
#     '''
#     # furcation nodes and the surrounding points
#     nodes = np.sort(centres[centres.n_furcation > 0].node_id.values)
#     samples = pc[pc.node_id.isin(nodes)]
#     for n in samples.node_id.unique():
#         # cluster points
#         c = samples[samples.node_id == n]

#         # determine n_neighbour 
#         pts = len(c)
#         if pts < 20: continue
#         if (pts>=20) & (pts<50): test_nn = 10
#         if (pts>=50) & (pts<200): test_nn = 20
#         if (pts >= 200) & (pts < 500): test_nn = int(pts/5)
#         if pts >= 500: test_nn = 100

#         # auto determine cluster number from eigen gaps
#         W = getAffinityMatrix(c[['x','y','z']], k = test_nn)
#         k, eigenvalues, eigenvectors = eigenDecomposition(W, plot=False, topK=3)

#         # spectral clustering
#         spectral = SpectralClustering(n_clusters=k[0],
#     #                               assign_labels='discretize',  
#                                   random_state=0,
#                                   affinity='nearest_neighbors',
#                                   n_neighbors=test_nn).fit(c[['x', 'y', 'z']])
#         c.loc[:, 'klabels'] = spectral.labels_
#         d = c.groupby('klabels').median()

#         for drow in d.itertuples():
#             # calculate new centre coords
#             nvoxel = c[c.klabels == drow.Index]
#             centre_coords = nvoxel[['x', 'y', 'z']].median()
#             # add new centres to pc
#             new_node_id = centres.node_id.max() + 1
#             pc.loc[pc.index.isin(nvoxel.index), 'node_id'] = new_node_id
#             # add new centres to centres
#             s = nvoxel.slice_id.unique()[0]
#             centres = centres.append(pd.Series({'slice_id': int(s), 
#                                                 'centre_id':int(drow.Index), 
#                                                 'cx':centre_coords.x, 
#                                                 'cy':centre_coords.y, 
#                                                 'cz':centre_coords.z, 
#                                                 'distance_from_base':nvoxel.distance_from_base.mean(),
#                                                 'n_points':len(nvoxel),
#                                                 'node_id':new_node_id,
#                                                 'idx':struct.pack('ii', int(s), int(drow.Index))}), ignore_index=True)
#         centres = centres.loc[centres.node_id != n] 
#     return pc, centres


def intersection(A0, A1, B0, B1, clampA0=True, clampA=True):
    
    pA, pB, D = closestDistanceBetweenLines(A0, A1, B0, B1, clampA0=clampA, clampA1=clampA)
    if np.isnan(D): D = np.inf
    return pA, pB, D

def split_furcation_2(pc, centres, path_ids, branch_hierarchy, verbose=False):  
    
    """
    split_fucation determines the correct location for a node which furcates.
    This is done by firstly identfying the node which is closest
    to the "child" portion of the parent cluster. Then using the
    point cloud and the first child node, the intersection between
    the parent and child is determined.
    
    This improves on the previous version as no new nodes are added
    which can become complicated.
    """
    
    if verbose: print('aligning furcations...')

    for ix, row in tqdm(centres[centres.n_furcation > 0].sort_values('slice_id', ascending=False).iterrows(), 
                        total=centres[centres.n_furcation > 0].shape[0],
                        disable=False if verbose else True):

        # if furcates at the base then ignore
        if row.distance_from_base == centres.distance_from_base.min(): continue

        # the clusters of points that represent the furcation
        cluster = pc[(pc.node_id == row.node_id)].copy()

        # nodes which are in parent branch (identified from the tip)
        tip_id = centres.loc[(centres.nbranch == row.nbranch) & 
                                  (centres.is_tip)].node_id.values[0]
        branch_path = np.array(path_ids[int(tip_id)], dtype=int)
        node_idx = np.where(branch_path == row.node_id)[0][0]

        # nodes either side of furcation
        previous_node = [branch_path[node_idx - 1]]
        subsequent_node = [branch_path[node_idx + 1]]

        # child nodes
        child_nodes = centres[(centres.parent_node == row.node_id) &
                                   (centres.ncyl == 0)].node_id.to_list()

        # label points in cluster
        all_nodes = previous_node + subsequent_node + child_nodes
        all_nodes = centres.loc[centres.node_id.isin(all_nodes)]
        distances = np.zeros((len(all_nodes), len(cluster)))

        for i, (_, node) in enumerate(all_nodes.iterrows()):
            distances[i, :] = np.linalg.norm(node[['cx', 'cy', 'cz']].astype(float).values - 
                                             cluster[['x', 'y', 'z']], 
                                             axis=1)

        labels = distances.T.argmin(axis=1)
        new_positions = pd.DataFrame(columns=['node_id', 'cx', 'cy', 'cz'])

        for child in child_nodes:
            
            # separate points
            label = np.where(all_nodes == child)[0]
            child_cluster = cluster.loc[cluster.index[np.where(labels == label)[0]]][['x', 'y', 'z']]
#             child_cluster = cluster.loc[cluster.node_id == child]
            child_centre = child_cluster.mean()
            CHx = centres.loc[centres.node_id == child][['cx', 'cy', 'cz']].values[0]
            Px = centres.loc[centres.node_id == previous_node[0]][['cx', 'cy', 'cz']].values[0] 
            Cx = centres.loc[centres.node_id == row.node_id][['cx', 'cy', 'cz']].values[0]
            Sx = centres.loc[centres.node_id == subsequent_node[0]][['cx', 'cy', 'cz']].values[0]

            mean_distance = np.zeros(3)

            # calculate distance from point to surrounding nodes 
            # where p, q are the line ends
            for i, q in enumerate([Sx, Cx, Px]):
                mean_distance[i] = d(CHx, q, child_cluster).mean()

            if np.all(np.isnan(mean_distance)): 
                continue # something not right so skip

            if np.argmin(mean_distance) == 0: # closer to subsequent node
                pA, pB, D = intersection(Cx, Sx, child_centre, CHx)
                centres, branch_hierarchy = update_slice_id(centres, branch_hierarchy, child, -1)
                nnode = subsequent_node[0]
                
                # update path_ids
                for k, v in path_ids.items():
                    if row.node_id in v and child in v:
                        path_ids[k] = v[:v.index(child)] + [subsequent_node[0]] +  v[v.index(child):]

            elif np.argmin(mean_distance) == 1: # closer to centre node
                nnode = row.node_id
                A0 = centres.loc[centres.node_id == subsequent_node[0]][['cx', 'cy', 'cz']].values[0]
                dP, dS = np.linalg.norm(CHx - Px), np.linalg.norm(CHx - Sx) 
                if dP > dS:
                    pA, pB, D = intersection(Cx, Sx, child_centre, CHx)
                else:
                    pA, pB, D = intersection(Cx, Px, child_centre, CHx)      

            else: # closer to previous node
                pA, pB, D = intersection(Cx, Px, child_centre, CHx)
                centres, branch_hierarchy = update_slice_id(centres, branch_hierarchy, child, -1)
                nnode = previous_node[0]
                
                # update path_ids
                for k, v in path_ids.items():
                    if row.node_id in v and child in v:
                        path_ids[k] = v[:v.index(row.node_id)] + v[v.index(child):]

            # add new position to temporary database
            # this is needed if more than one branch joins 
            # to the same node
            new_positions = new_positions.append(pd.Series({'node_id':nnode, 'cx':pA[0], 'cy':pA[1], 'cz':pA[2]}), 
                                                 ignore_index=True)
            
        # update centres with new positions, groupby is required as 
        # 2 branches may be connected to 1 node and therefore the
        # mean position is taken
        if len(new_positions) == 0 or np.all(np.isnan(new_positions[['cx', 'cy', 'cz']])): continue
        for roww in new_positions.groupby('node_id').mean().itertuples():
            if np.any(np.isnan([roww.cx, roww.cy, roww.cz])): continue # something is wrong, leave where is is
            centres.loc[centres.node_id == roww.Index, 'cx'] = roww.cx 
            centres.loc[centres.node_id == roww.Index, 'cy'] = roww.cy 
            centres.loc[centres.node_id == roww.Index, 'cz'] = roww.cz 
            
    return centres, path_ids, branch_hierarchy


def split_furcation_1(self, error=.01, max_dist=1):
    
    """
    parameters
    ----------
    
    error; float (default 0.1)
    minimum distance two new nodes have to be apart for them to be included
    
    max_dist; float (default 1)
    used in skeleton path
    """
    
    for ix, row in tqdm(self.centres[self.centres.n_furcation > 0].sort_values('slice_id', ascending=False).iterrows(), 
                        total=self.centres[self.centres.n_furcation > 0].shape[0],
                        disable=False if self.verbose else True):

        # if furcates at the base then ignore
        if row.distance_from_base == self.centres.distance_from_base.min(): continue

        # nodes which are in parent branch (identified from the tip)
        tip_id = self.centres.loc[(self.centres.nbranch == row.nbranch) & 
                                  (self.centres.is_tip)].node_id.values[0]
        branch_path = np.array(self.path_ids[int(tip_id)], dtype=int)
        idx = np.where(branch_path == row.node_id)[0][0]

        # node 1 closer to base which will become new furcation points
        previous_node = branch_path[idx - 1]
        base_coords = self.centres.loc[self.centres.node_id == previous_node][['cx', 'cy', 'cz']].values

        # extract point cloud for current node and run KMeans clustering
        # where K is determined by number of furcations
        c = self.pc[(self.pc.node_id == row.node_id)].copy()
        c.loc[:, 'klabels'] = KMeans(n_clusters=int(row.n_furcation) + 1).fit(c[['x', 'y', 'z']]).labels_
        d = c.groupby('klabels').mean()

        # if new centres are more than error apart
        if nn(d[['x', 'y', 'z']].values, 1).mean() > error:

            points_index = {}

            # create temporary df with which to add new nodes
            child_branch = self.centres[self.centres.parent_node == row.node_id].nbranch.unique()
            child_nodes = self.centres.loc[self.centres.nbranch.isin(child_branch)].node_id.to_list()
            nodes_subset = list(branch_path) + child_nodes
            nodes_subset.remove(row.node_id) # remove existing node as don't want to include in tmp 
            tmp_centres = self.centres.loc[self.centres.node_id.isin(nodes_subset)] 
            node_id = self.centres.node_id.max() + 1

            # add new nodes
            for drow in d.itertuples():
                
                nvoxel = c[c.klabels == drow.Index]
                centre_coords = nvoxel[['x', 'y', 'z']].median()
                tmp_centres = tmp_centres.append({'slice_id':nvoxel.slice_id.unique()[0], 
                                                  'centre_id':nvoxel.centre_id.unique()[0], 
                                                  'cx':centre_coords.x, 
                                                  'cy':centre_coords.y, 
                                                  'cz':centre_coords.z, 
                                                  'distance_from_base':nvoxel.distance_from_base.mean(),
                                                  'n_points':len(nvoxel),
                                                  'node_id':node_id}, ignore_index=True)
                points_index[node_id] = list(nvoxel.index)
                node_id += 1
                
            ### test if new furcation generates cylinders that are within one another
            # compute temporary graph
            path_distance, path_ids = skeleton_path(tmp_centres, max_dist=max_dist, verbose=False)
            tmp_centres = attribute_centres(tmp_centres, path_ids)
            
            # identify and add new node in parent branch
            keep_node_in_path = True
            node_in_parent = tmp_centres.loc[(tmp_centres.node_id.isin(points_index.keys())) & 
                                             (tmp_centres.nbranch == tmp_centres.nbranch.min())] # min nbranch being the parent
            if len(node_in_parent) == 0:
                # if none of the nodes occur in the parent... might be e GOTCHA and needs checking    
                node_in_parent = tmp_centres.loc[(tmp_centres.node_id.isin([list(points_index.keys())[0]]))]
            
            # node_in_parent_radius,  _, _ = cylinderFitting(c.loc[c.klabels.isin(new_nodes.label.loc[new_nodes.parent])])
            node_in_parent_radius,  _, _ = cylinderFitting(c) # worse case i.e. points are wrongly attributed to >1 cluster

            for child_node_id in points_index.keys():
                
                # don't process the node in parent
                if child_node_id in node_in_parent.node_id.values: continue

                # test whether the child is mostly in the parent
                child = tmp_centres.loc[tmp_centres.node_id == child_node_id]
                A = node_angle_f(node_in_parent[['cx', 'cy', 'cz']].values,
                                 base_coords,
                                 child[['cx', 'cy', 'cz']].values) # angle between nodes
                distance_to_edge =  (node_in_parent_radius / np.sin(A))

                dist_between_nodes = np.linalg.norm(base_coords - 
                                                    child[['cx', 'cy', 'cz']].values)

                if distance_to_edge / dist_between_nodes > .8 and keep_node_in_path: # comparing lengths
                    # cylinder overlap is too great, delete new node in parent 
                    # and correctly attribute point cloud with node_id. This tends
                    # to happen where there is already a good fit
                    keep_node_in_path = False
                    points_index[child_node_id] += points_index[node_in_parent.node_id.values[0]]
                     
                self.centres = self.centres.append(tmp_centres.loc[tmp_centres.node_id == child_node_id], 
                                                   ignore_index=True)
                self.pc.loc[self.pc.index.isin(points_index[child_node_id]), 'node_id'] = child_node_id
            
            if keep_node_in_path:
                # new node in parent is valid and should be kept
                self.centres = self.centres.append(node_in_parent, ignore_index=True)
                self.pc.loc[self.pc.index.isin(points_index[node_in_parent.node_id.values[0]]), 'node_id'] = node_in_parent
                # replace
                for k, v in path_ids.items():
                    if k in self.path_ids.keys():
                        self.path_ids[k] = v
                    else: self.path_ids[k] = v       
            else:
                tmp_centres = tmp_centres.loc[tmp_centres.node_id != node_in_parent.node_id.values[0]]
                path_distance, path_ids = skeleton_path(tmp_centres, max_dist=max_dist, verbose=False)
                for k, v in path_ids.items():
                    if k in self.path_ids.keys():
                        self.path_ids[k] = v
                    else: self.path_ids[k] = v
            
            self.centres = self.centres.loc[self.centres.node_id != row.node_id]
            
        # if splitting the base node occured - put back together
        if len(self.centres[(self.centres.slice_id == 0) & (self.centres.slice_id == 0)]) > 1:

            Base = self.centres[(self.centres.slice_id == 0) & (self.centres.slice_id == 0)]
            Mean, Sum = Base.mean(), Base.sum()

            self.centres = self.centres.loc[~self.centres.node_id.isin(Base.node_id.values)]
            self.centres = self.centres.append(pd.Series({'slice_id':0, 
                                                          'centre_id':0, 
                                                          'cx':Mean.cx, 
                                                          'cy':Mean.cy, 
                                                          'cz':Mean.cz, 
                                                          'distance_from_base':Mean.distance_from_base,
                                                          'n_points':Sum.n_points,
                                                          'node_id':self.centres.node_id.max() + 1}),
                                               ignore_index=True)
            

def split_furcation_w(pc, centres, path_ids, error=.01, verbose=False):
    '''
    split furcation node and its previous node
    add new cluster centre nodes if new centres are more than error apart
    '''
    for ix, row in tqdm(centres[centres.n_furcation > 0].sort_values('slice_id').iterrows(),
                    total=len(centres[centres.n_furcation > 0]),
                    disable=False if verbose else True):
        # if furcates at the base then ignore
        if row.distance_from_base == centres.distance_from_base.min(): continue
        
        ## find the path list of a non-bifurcation part ended with current furcation node
        ncyl_min = centres.ncyl[centres.ninternode == row.ninternode].min()
        # the starting node_id of this sub-branch
        start_n = centres.node_id[(centres.ninternode == row.ninternode) & (centres.ncyl == ncyl_min)].values[0]
        # index of the starting node in the path list
        start_n_idx = np.where(np.array(path_ids[row.node_id]) == start_n)[0][0]
        # path list of this sub-branch
        path_ids = path_ids[row.node_id][start_n_idx:][::-1]

        ## loop over the last two nodes in this sub-branch 
        ## split furcation if new cluster centres are far apart
        for node_id in path_ids[0:2]:        
            # extract self.pc for current node and run KMeans clustering
            # where K is determined by number of furcations
            c = pc[(pc.node_id == node_id)].copy()
            c.loc[:, 'klabels'] = KMeans(n_clusters=int(row.n_furcation) + 1).fit(c[['x', 'y', 'z']]).labels_
            d = c.groupby('klabels').mean()

            # if new centres are more than error apart, then add new nodes
            if nn(d[['x', 'y', 'z']].values, 1).mean() > error:
                for drow in d.itertuples():
                    new_node_id = centres.node_id.max() + 1
                    nvoxel = c[c.klabels == drow.Index]
                    centre_coords = nvoxel[['x', 'y', 'z']].median()

                    pc.loc[pc.index.isin(nvoxel.index), 'node_id'] = new_node_id
                    
                    centres = centres.append(pd.Series({'slice_id':nvoxel.slice_id.unique()[0], 
                                                        'centre_id':nvoxel.centre_id.unique()[0], 
                                                        'cx':centre_coords.x, 
                                                        'cy':centre_coords.y, 
                                                        'cz':centre_coords.z, 
                                                        'distance_from_base':nvoxel.distance_from_base.mean(),
                                                        'n_points':len(nvoxel),
                                                        'node_id':new_node_id}), ignore_index=True)
                centres = centres.loc[centres.node_id != node_id] 
                if len(path_ids) < 2: break
            else: break
                
    return pc, centres, path_ids