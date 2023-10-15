import pandas as pd
import numpy as np
import struct
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.stats import variation
from numpy import linalg as LA
from sklearn.cluster import SpectralClustering
from pandarallel import pandarallel
from treegraph.fit_cylinders import *
from treegraph.build_skeleton import *
from treegraph.build_graph import *
from treegraph.attribute_centres import *
from treegraph.common import *
from treegraph.third_party.point2line import *
from treegraph.third_party.closestDistanceBetweenLines import *
from treegraph.third_party import ransac_cyl_fit


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
    n = np.unique(clus_pc.node_id)[0]
    orig_cen = centres[centres.node_id == n][['cx','cy','cz']].values[0]
    dist_mean = cdist([orig_cen], clus_pc[['x','y','z']]).mean()
    dist_std = cdist([orig_cen], clus_pc[['x','y','z']]).std()
    cv = dist_std / dist_mean

    tmp = pd.DataFrame()
            
    if cv > .4:
        clus_pc.loc[:, 'adj_k'] = clus_pc.klabels 
    else:  
        count = 1
        for klabel in np.unique(spectral.labels_): 
            mind = centres[centres.slice_id == sid_start].distance_from_base.values[0]
            maxd = centres.distance_from_base.max()
            currd = centres[centres.node_id == nid].distance_from_base.values[0]
            ratio = 1- ((maxd - currd) / (maxd - mind))
            p = -4 ** (0.5 * ratio) + 2
            threshold = dist_mean * p

            subclus_pc = clus_pc[clus_pc.klabels == klabel]
            if len(subclus_pc) <= 100:
                new_cen = subclus_pc[['x','y','z']].median().to_numpy()
            else:
                new_cen = CPC(subclus_pc).x

            d_cc = np.linalg.norm(new_cen - orig_cen)
            if d_cc <= threshold:
                tmp = tmp.append(clus_pc[clus_pc.klabels == klabel])   
            else:
                clus_pc.loc[clus_pc.klabels == klabel, 'adj_k'] = int(count)
                count += 1

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
    sid = pc.slice_id.unique()[0]  # slice_id
    nid = pc.node_id.unique()[0]  # node_id
    cid = pc.centre_id.unique()[0]  # centre_id
    
    if sid_start is None:
        for i, s in enumerate(np.sort(centres.slice_id.unique())):
            nids = centres[centres.slice_id == s].node_id.values
            if len(nids) > 1:
                break
        if i == 0:
            sid_start = int(np.sort(centres.slice_id.unique())[i])
        else:
            sid_start = int(np.sort(centres.slice_id.unique())[i-1])
    
    subclus_1 = stsc(pc, centres, nn=nn, sid_start=sid_start, plot=False, point_size=5)

    if len(subclus_1) < 1: 
        return []  

    kmax = subclus_1.adj_k.max()

    if len(np.sort(subclus_1.adj_k.unique())) < 2:  
        pc.loc[:, 'adj_k'] = subclus_1.adj_k
    else:
        for k in np.sort(subclus_1.adj_k.unique()):
            subpc = subclus_1[subclus_1.adj_k == k]
            if len(subpc) <= 20:
                continue          
            if len(subpc) <= 100:
                new_cen = subpc[['x','y','z']].median().to_numpy()
            else:
                new_cen = CPC(subpc).x

            dist_mean = cdist([new_cen], subpc[['x','y','z']]).mean()
            dist_std = cdist([new_cen], subpc[['x','y','z']]).std()
            cv = dist_std / dist_mean

            if (cv > .3) or (dist_mean >= .5):
                subclus_2 = stsc(subpc, centres, nn=nn, sid_start=sid_start, plot=False, point_size=10)
                if len(subclus_2) < 1:
                    continue
                subclus_2.loc[:, 'adj_k'] = subclus_2.adj_k + kmax
                kmax += len(np.unique(subclus_2.adj_k))
                pc.loc[pc.index.isin(subclus_2.index), 'adj_k'] = subclus_2.adj_k

                for kk in np.sort(subclus_2.adj_k.unique()):
                    subpc_ = subclus_2[subclus_2.adj_k == kk]
                    if len(subpc_) <= 20:
                        continue
                    else:
                        new_cen_ = subpc_[['x','y','z']].median().to_numpy()
                        dist_mean_ = cdist([new_cen_], subpc_[['x','y','z']]).mean()

                        if dist_mean >= .5:
                            subclus_3 = stsc(subpc_, centres, nn=nn, sid_start=sid_start, plot=False, point_size=10)
                            if len(subclus_3) < 1:
                                continue
                            subclus_3.loc[:, 'adj_k'] = subclus_3.adj_k + kmax
                            kmax += len(np.unique(subclus_3.adj_k))
                            pc.loc[pc.index.isin(subclus_3.index), 'adj_k'] = subclus_3.adj_k

    pc.loc[:, 'adj_k'] = pc.adj_k.apply(lambda x: np.where(np.array(np.unique(pc.adj_k)) == x)[0][0])

    
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
        dcluster = pc[pc.adj_k == kn]
        if len(dcluster) <= 100:
            new_cen_coords = dcluster[['x','y','z']].median().to_numpy()
        else:
            new_cen_coords = CPC(dcluster).x
        new_idx = struct.pack('iii', int(sid), int(cid), int(kn))

        pc.loc[pc.index.isin(dcluster.index), 'centre_id'] = int(kn)
        pc.loc[pc.index.isin(dcluster.index), 'idx'] = new_idx

        new_centres = new_centres.append(pd.Series({'slice_id': int(sid), 
                                                    'centre_id':int(kn),
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



def intersection(A0, A1, B0, B1, clampA0=True, clampA=True):
    
    pA, pB, D = closestDistanceBetweenLines(A0, A1, B0, B1, clampA0=clampA, clampA1=clampA)
    if np.isnan(D): D = np.inf
    return pA, pB, D
