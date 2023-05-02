import networkx as nx
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from tqdm.autonotebook import tqdm
from pandarallel import pandarallel


def run(pc, centres, n_neighbours=100, verbose=False):
    # find convex hull points of each cluster
    group_pc = pc.groupby('node_id')
    pandarallel.initialize(nb_workers=min(24, len(group_pc)+1), progress_bar=verbose)
    try:
        chull = group_pc.parallel_apply(convexHull)
    except OverflowError:
        if verbose: 
            print('!pandarallel could not initiate progress bars, running without')
        pandarallel.initialize(progress_bar=False)
        chull = group_pc.parallel_apply(convexHull)
    
    # find shortest path from each cluster to the base
    # and build skeleton graph
    path_dist, path_list, G_skel = generate_path(chull, centres, n_neighbours=n_neighbours)
    
    return G_skel, path_dist, path_list


def convexHull(pc):
    if len(pc) > 5:
        try: 
            vertices = ConvexHull(pc[['x', 'y', 'z']]).vertices
            idx = np.random.choice(vertices, size=len(vertices), replace=False)
            return pc.loc[pc.index[idx]]
        except:
            return pc
    else:
        return pc 


def generate_path(samples, centres, n_neighbours=200, max_length=np.inf, not_base=-1):
    # compute nearest neighbours for each vertex in cluster convex hull
    nn = NearestNeighbors(n_neighbors=n_neighbours).fit(samples[['x', 'y', 'z']])
    distances, indices = nn.kneighbors()    
    from_to_all = pd.DataFrame(np.vstack([np.repeat(samples.node_id.values, n_neighbours), 
                                          samples.iloc[indices.ravel()].node_id.values, 
                                          distances.ravel(),
                                          np.repeat(samples.slice_id.values, n_neighbours),
                                          samples.iloc[indices.ravel()].slice_id.values]).T, 
                               columns=['source', 'target', 'length', 's_sliceid', 't_sliceid'])

    # remove X-X connections
    from_to_all = from_to_all.loc[from_to_all.target != from_to_all.source]

    # previous: build edge database where edges with min distance between clusters persist
    # edges = from_to_all.groupby(['source', 'target']).length.min().reset_index()
    
    # updated: build edge list based on min distance and 
    # number of chull pts in nearest neighbour clusters
    groups = from_to_all.groupby(['source', 'target'])
    edges = groups.length.apply(lambda x: x.min() / (np.log10(x.count())+0.001)).reset_index()

    # remove edges that are likely leaps between trees
    edges = edges.loc[edges.length <= max_length]
    
    # removes isolated origin points i.e. > edge.length
    for nid in np.sort(samples.node_id.unique()):
        if nid in edges.source.values:
            origin = [nid]
            break
    # origins = [s for s in origins if s in edges.source.values] ## old method

    # compute graph that connect all clusters
    G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
    # retrieve shortest path list (sp) of each cluster 
    # to the base node and its corresponding distance
    distance, sp = nx.multi_source_dijkstra(G, 
                                            sources=origin,
                                            weight='length')
    # build skeleton graph
    G_skeleton = nx.Graph()
    for i, nid in enumerate(G.nodes()):
        if nid in sp.keys():
            if len(sp[nid]) > 1:       
                x1 = float(centres[centres.node_id == nid].cx)
                y1 = float(centres[centres.node_id == nid].cy)
                z1 = float(centres[centres.node_id == nid].cz)
                node1_coor = np.array([x1,y1,z1])
                sid1 = int(centres[centres.node_id == nid].slice_id)
                G_skeleton.add_node(nid, pos=[x1,y1,z1], node_id=int(nid), slice_id=sid1)

                x2 = float(centres[centres.node_id == sp[nid][-2]].cx)
                y2 = float(centres[centres.node_id == sp[nid][-2]].cy)
                z2 = float(centres[centres.node_id == sp[nid][-2]].cz)
                node2_coor = np.array([x2,y2,z2])
                sid2 = int(centres[centres.node_id == sp[nid][-2]].slice_id)
                G_skeleton.add_node(sp[nid][-2], pos=[x2,y2,z2], 
                                    node_id=int(sp[nid][-2]), slice_id=sid2)

                d = np.linalg.norm(node1_coor - node2_coor)
                G_skeleton.add_weighted_edges_from([(int(sp[nid][-2]), int(nid), float(d))])

    paths = pd.DataFrame(index=distance.keys(), data=distance.values(), columns=['distance'])
    paths.loc[:, 'base'] = not_base
    for p in paths.index: paths.loc[p, 'base'] = sp[p][0]
    paths.reset_index(inplace=True)
    paths.columns = ['node_id', 'distance', 'base_node_id'] # t_node_id is the base node
    
    # identify nodes that are branch tips
    node_occurance = {}
    for v in sp.values():
        for n in v:
            if n in node_occurance.keys(): node_occurance[n] += 1
            else: node_occurance[n] = 1

    tips = [k for k, v in node_occurance.items() if v == 1]

    paths.loc[:, 'is_tip'] = False
    paths.loc[paths.node_id.isin(tips), 'is_tip'] = True

    return paths, sp, G_skeleton


### old version: build graph from nearest neighbours
# def run(centres, verbose=False):
    
#     """
#     parameters
#     ----------
#     max_dist; float (default .1)
#     maximum distance between nodes, designed to stop large gaps being spanned
#     i.e. it is better to have a disconnection than an unrealistic conncetion
#     """
    
#     edges = pd.DataFrame(columns=['node1', 'node2', 'length'])
  
#     if verbose: print('generating graph...')
#     for i, row in tqdm(enumerate(centres.itertuples()), total=len(centres), disable=False if verbose else True):
        
#         # first node
#         # if row.distance_from_base == centres.distance_from_base.min(): continue

#         n, dist = 3, np.inf

#         # while n < 10 and dist > max_dist:
#         while n < 10:

#             # between required incase of small gap in pc
#             nbrs = centres.loc[centres.slice_id.between(row.slice_id - n, row.slice_id - 1)]
#             nbrs.loc[:, 'dist'] = np.linalg.norm(np.array([row.cx, row.cy, row.cz]) - 
#                                      nbrs[['cx', 'cy', 'cz']].values, 
#                                      axis=1)
#             dist = nbrs.dist.min()
#             n += 1          
        
#         if np.isnan(nbrs.dist.min()): # prob an outlying cluster that can be removed
#             continue

#         edges = edges.append({'node1':int(row.node_id), 
#                               'node2':int(nbrs.loc[nbrs.dist == nbrs.dist.min()].node_id.values[0]), 
#                               'length':nbrs.dist.min()}, ignore_index=True)
#         if row.node_id == 0: print(edges)
            
# #     return(edges)
    
#     # to catch centre nodes that are not used
#     centres = centres.loc[centres.node_id.isin(edges.node1.to_list() + edges.node2.to_list())]
    
#     idx = centres.distance_from_base.idxmin() 
#     base_id = centres.loc[idx].node_id
    
#     G_skeleton = nx.Graph()
#     for row in centres.itertuples():
#         G_skeleton.add_node(row.node_id, pos=[float(row.cx), float(row.cy), float(row.cz)], slice_id=int(row.slice_id))
#     G_skeleton.add_weighted_edges_from([(int(row.node1), int(row.node2), float(row.length)) 
#                                         for row in edges.itertuples()])

#     path_distance, path_ids = nx.single_source_bellman_ford(G_skeleton, base_id)
#     path_distance = {k: v if not isinstance(v, np.ndarray) else v[0] for k, v in path_distance.items()}

#     # required as sometimes pc2graph produces strange results
#     centres.distance_from_base = centres.node_id.map(path_distance) 

#     return G_skeleton, path_distance, path_ids