import json
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import *
# from mayavi import mlab  # used for plot_graph_3d(), need to install mayavi first

def save_graph(G, fname):
    '''
    Generate a json file to save the node and edge information of a networkx graph.
    
    Parameters
    ----------
    G : networkx graph
        Graph needs to be saved.
    
    fname : string
        Output path for the saving json file.
        
    '''
    
    dt = datetime.now().strftime('%Y-%m-%d_%H-%M')
    fname = fname + '_' + dt + '.json'
    
    json.dump(dict(nodes=[[int(n), G.nodes[n]] for n in G.nodes()], \
              edges=[[int(u), int(v)] for u,v in G.edges()]),\
              open(fname, 'w'), indent=2)
    
    print(f'Graph has been successfully saved in \n{fname}')
    


def load_graph(fname):
    '''
    Load a networkx graph with nodes and edgse information from a json file.
    
    Parameters
    ----------    
    fname : string
        Path of the file.
    
    
    Returns
    ----------
    G : networkx graph
        Graph with nodes and edges information.
    '''
    
    G = nx.DiGraph()
    d = json.load(open(fname))
    G.add_nodes_from(d['nodes'])
    G.add_edges_from(d['edges'])
    
    return G    



def save_centres_for_graph(centres, fname):
        '''
        Generate a csv file to save centres coordinates and other info.

        Parameters
        -----------
        centres: pandas dataframe
            includes x,y,z coords and node_id of each node 

        fname: string
            output path of the file
        '''
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M')
        fname = fname + '_' + dt + '.csv'
        
        centres.to_csv(fname, index=False)
        
        print(f'\ncentres has been successfully saved in \n{fname}')


def load_centres_for_graph(fname):
    '''
    Load dataframe of centres from a csv file.

    Parameter:
    ----------
    fname: string
        csv file which stores x,y,z coords and node_id of each node
    
    Ouput:
    ----------
    centres: pandas dataframe
    '''
    centres = pd.read_csv(fname)
    
    return centres


    
def plot_graph_3d(G, centres, ofn,
                point_size=0.05,  
                line_width=0.01,
                label=False):
    '''
    Plot 3D graph in mayavi scene, after close the mayavi scene\
    window, plot 2D snapshot figure of the 3D graph.
    
    Paramaters:
    ------------
    G : networkx graph
        graph with node and edge information
    
    centres: pandas dataframe
        for self.pc, needs columns: 'x','y','z','distance_from_base'
        for self.centres, needs columns: 'cx','cy','cz','node_id','slice_id'  
    
    ofn : string
        output path of saved snapshot figure from mayavi scene
    
    point_size (optional) : float 
        point size of nodes in mayavi scene, default is 0.05 
    
    line_width: float
        width of lines connecting the points, default is 0.01

    label: bool
        if True: show point id as label in the plot
    
    '''
    
    print(f'plotting graph...\nThere are {len(G.nodes)} nodes and {len(list(G.edges))} edges in this graph.\n')

    # load edge data 
    edges_table = list(G.edges)
    # build a dict returning xyz coords for each centre node
    nodes_coord = dict()
    slice_id = dict()

    try:
        centres.cx
        for n in G.nodes():
            coord = np.array(list(centres.cx.loc[centres.node_id == n])+
                            list(centres.cy.loc[centres.node_id == n])+
                            list(centres.cz.loc[centres.node_id == n]))
            nodes_coord[n] = coord 
            slice_id[n] = list(centres.slice_id.loc[centres.node_id == n])

    except: 
        for n in G.nodes():
            coord = np.array(list(centres.x.loc[centres.index == n])+
                            list(centres.y.loc[centres.index == n])+
                            list(centres.z.loc[centres.index == n]))
            nodes_coord[n] = coord
            slice_id[n] = list(centres.distance_from_base.loc[centres.index == n])

    # store all coords in a list 
    # and keep track of which node corresponds to a given index in the list
    # connection is specified as connecting the i-th point with the j-th
    nodes = dict()
    coords = list()
    connections = list()
    scalars = list()

    for n1, n2 in edges_table:
        if not n1 in nodes:
            nodes[n1] = len(coords)
            coords.append(nodes_coord[n1])
            scalars.append(slice_id[n1][0])
        if not n2 in nodes:
            nodes[n2] = len(coords)
            coords.append(nodes_coord[n2])
            scalars.append(slice_id[n2][0])

        connections.append((nodes[n1], nodes[n2]))

    # xyz coords array of re-index nodes
    xyz = np.array(coords)

    # plot 3D graph using mayavi.mlab
    # 01-plot 3D points 
    pts = mlab.points3d(xyz[:, 0],
                        xyz[:, 1],
                        xyz[:, 2],
                        scalars,
                        scale_factor = point_size,
                        scale_mode = "none",
                        #color = (0,0,1), # blue point
                        colormap = "flag",
                        resolution = 20)

    # 02-plot edges 
    connections = np.array(connections)
    # add lines between points based on connections
    pts.mlab_source.dataset.lines = connections
    line = mlab.pipeline.tube(pts, tube_radius = line_width)
    mlab.pipeline.surface(line, color = (0.8, 0.8, 0.8))

    # 03-display node names (optional)
    if label:
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        for node, index in nodes.items():
            label = mlab.text(x[index], y[index], str(node), z=z[index],
                            width=0.08, name=str(node))
            label.property.shadow = True 

    # save snapshot figure from mayavi scene
    ofn = ofn + '_snapshot.png'
    mlab.savefig(ofn)  
    print(f'snapshot of mayavi scene has been saved in the file:\n{ofn}\n') 

    # plot 2D snapeshot of the graph 
    I = mpimg.imread(ofn)  
    plt.imshow(I)   

    # show 3D graph in mayavi scene 
    mlab.show()    
