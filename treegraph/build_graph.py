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
    
    # build edge list based on min distance and 
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
