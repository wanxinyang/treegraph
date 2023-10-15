import json
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import *

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
