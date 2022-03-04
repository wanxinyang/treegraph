import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
# %matplotlib inline

def plot_3d_graph(G, pc=pd.DataFrame(), centres=pd.DataFrame(), title='self.G'):
    '''
    Plot 3D networkx graph using Plotly.
    Inputs:
        G: networkx graph
            G.nodes has attributes of 'pos' for x,y,z coordinates and 'pid' for node id.
        pc: pd.DataFrame (optional)
            If 'slice_id' in pc.columns, then plot nodes coloured by slice_id.
            If leave it as blank, then plot all nodes in grey.
        centres: pd.DataFrame (optional)
            Cluster centres in each slice. 
        title: str.
            Title of this plot.
    Output:
        An interactive Plotly figure of a 3D nx graph.
    '''
    # need to separate X,Y,Z coords for Plotly
    x_nodes = []
    y_nodes = []
    z_nodes = []
    labels = []

    for nid in G.nodes:
        if len(G.nodes[nid]) != 0:
            # extract xyz coordinates of the nodes
            x_nodes.append(G.nodes[nid]['pos'][0])
            y_nodes.append(G.nodes[nid]['pos'][1])
            z_nodes.append(G.nodes[nid]['pos'][2])

            # extract slice_id corresponding to the nodes
            if 'slice_id' in pc.columns:
                pid = G.nodes[nid]['pid']
                s = pc[pc.pid == pid].slice_id.values[0]
                labels.append(s)
    
    # create lists that contain the starting and ending coords of each edge
    x_edges = []
    y_edges = []
    z_edges = []
    
    for edge in G.edges():
        nid1 = edge[0]
        nid2 = edge[1]
        if (len(G.nodes[nid1]) != 0) & (len(G.nodes[nid2]) != 0):
            # format: [beginning, ending, None]
            x_coor = [G.nodes[nid1]['pos'][0], G.nodes[nid2]['pos'][0], None]
            x_edges += x_coor
            y_coor = [G.nodes[nid1]['pos'][1], G.nodes[nid2]['pos'][1], None]
            y_edges += y_coor
            z_coor = [G.nodes[nid1]['pos'][2], G.nodes[nid2]['pos'][2], None]
            z_edges += z_coor
    
    #create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x_nodes,
                               y=y_nodes,
                               z=z_nodes,
                               mode='markers',
                               marker=dict(symbol='circle',
                                           size=1.25,
                                           color='grey',
                                           opacity=0.7))
    # trace for nodes with attribute of slice_id
    if 'slice_id' in pc.columns:
        trace_nodes = go.Scatter3d(x=x_nodes,
                                   y=y_nodes,
                                   z=z_nodes,
                                   mode='markers',
                                   marker=dict(symbol='circle',
                                               size=1.25,
                                               opacity=0.7,
                                               color=labels,
                                               colorscale=px.colors.qualitative.Plotly),
                                   text=labels,
                                   hoverinfo='text')

    # create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                               y=y_edges,
                               z=z_edges,
                               mode='lines',
                               line=dict(color='lightgrey', width=0.5),
                               hoverinfo='none')
    
    if len(centres.columns) != 0:
        x_c = [cx for cx in centres.cx]
        y_c = [cy for cy in centres.cy]
        z_c = [cz for cz in centres.cz]
        labels_c = [s for s in centres.slice_id]

        # create a trace for cluster centres
        trace_centres = go.Scatter3d(x=x_c,
                                     y=y_c,
                                     z=z_c,
                                     mode='markers',
                                     marker=dict(symbol='diamond',
                                                 size=4,
                                                 color=labels_c,
                                                 colorscale=px.colors.qualitative.Plotly),
                                     text=labels_c,
                                     hoverinfo='text')
    
    # set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    
    # layout for the plot
    layout = go.Layout(title=title,
                       width=650,
                       height=625,
                       showlegend=False,
                       scene=dict(xaxis=dict(axis),
                                  yaxis=dict(axis),
                                  zaxis=dict(axis)),
                       margin=dict(t=100),
                       hovermode='closest')
    
    # plot the traces in a figure
    if len(centres.columns) != 0:
        data = [trace_nodes, trace_edges, trace_centres]
    else:
        data = [trace_nodes, trace_edges]
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def plot_subgraph(G, pc=pd.DataFrame(), centres=pd.DataFrame(), s_start=0, s_end=-1):
    '''
    Plot subgraph based on given range of slice_id.
    Inputs:
        G: networkx graph
            G.nodes has attributes of 'pos' for x,y,z coordinates and 'pid' for node id.
        pc: pd.DataFrame
            Point cloud info, contains the column 'slice_id'.
        centres: pd.DataFrame (optional)
            Skeleton node info, contains the column 'slice_id'.
        s_start: int.
            Starting index of slice_id, included.
        s_end: int.
            Last index of slice_id, not included.
    Output:
        An interactive Plotly figure of a 3D nx graph.
    '''
    if s_start < 0:
        s_start = len(pc.slice_id.unique()) + s_start + 1
    if s_end < 0:
        s_end = len(pc.slice_id.unique()) + s_end + 2
    slices = [*range(s_start, s_end)]
    pc = pc[pc.slice_id.isin(slices)]
    pids = [pid for pid in pc.pid]

    # select nodes for subgraph
    node_id = []
    for nid in G.nodes():
        if len(G.nodes[nid]) != 0:
            p = G.nodes[nid]['pid']
            if p in pids:
                node_id.append(nid)
    subG = G.subgraph(node_id)
    
    # plot subgraph
    if len(centres.columns) == 0:
        plot_3d_graph(subG, pc=pc, 
                      title=f'Initial graph: slice [{s_start}:{s_end}]')    
    else:
        centres = centres[centres.slice_id.isin(slices)]
        plot_3d_graph(subG, pc=pc, centres=centres,
                      title=f'Initial graph: slice [{s_start}:{s_end}]') 


def plot_slice(pc, centres=None,
               w=10, h=6, 
               start=0, end=-1,
               p_size=0.01, xlim=None, 
               ylim=None, zlim=None):
    '''
    Plot 2D plots of input point cloud (with skeleton nodes).
    Inputs:
        pc: pd.DataFrame
            point cloud info, contains columns 'slice_id'
        centres: pd.DataFrame (optional)
            skeleton nodes info, contains columns 'slice_id'
        w: float
            width of the figure 
        h: float
            height of the figure 
        start: int
            starting index of slice for plotting, included
        end: int
            last index of slice for plotting, not included
        p_size: float
            point size in plotting
        xlim: list, [xmin, xmax]
            range of x coordinates showing on the subplot
        ylim: list, [ymin, ymax]
            range of y coordinates showing on the subplot
        zlim: list, [zmin, zmax]
            range of z coordinates showing on all subplots
    Output:
        Two subplots. 
            Left subplot: x-axis: X coordinates, y-axis: Z coordinates.
            Right subplot: x-axis: Y coordinates, y-axis: Z coordinates.
    '''
    fig, axs = plt.subplots(1,2,figsize=(w,h))
    ax = axs.flatten()
    if start < 0:
        start = len(pc.slice_id.unique()) + start + 1
    if end < 0:
        end = len(pc.slice_id.unique()) + end + 2
    for s in range(start, end):
        slice_pc = pc[pc.slice_id == s]
        ax[0].scatter(slice_pc.x, slice_pc.z, s=p_size, label=f'{s}')
        ax[1].scatter(slice_pc.y, slice_pc.z, s=p_size, label=f'{s}')
        ax[0].set_xlabel('x coordinates (m)', fontsize=12)
        ax[1].set_xlabel('y coordinates (m)', fontsize=12)
        ax[0].set_ylabel('z coordinates (m)', fontsize=12)
        
        if xlim:
            ax[0].set_xlim(xlim[0], xlim[1])
        if ylim:
            ax[1].set_xlim(ylim[0], ylim[1])
        if zlim:
            ax[0].set_ylim(zlim[0], zlim[1])
            ax[1].set_ylim(zlim[0], zlim[1])      
        # plot skeleton nodes coloured by slice
        if centres is None: 
            continue
        else:
            cnode = centres[centres.slice_id == s]
            ax[0].scatter(cnode.cx, cnode.cz, s=10, marker='^', label=f'{s}')
            ax[1].scatter(cnode.cy, cnode.cz, s=10, marker='^', label=f'{s}')
    ax[0].set_title(f'slice_id [{start}:{end}]', fontsize=12) 
    ax[1].set_title(f'slice_id [{start}:{end}]', fontsize=12) 
    fig.tight_layout()


def plot_skeleton(G, pc=pd.DataFrame(), title='self.G_skeleton'):
    '''
    Plot skeleton graph (3D networkx graph) using Plotly.
    Inputs:
        G: networkx graph
            G.nodes has attributes of 'pos' for x,y,z coordinates and 'pid' for node id.
        pc: pd.DataFrame (optional)
            Point cloud info, contains XYZ coordinates. 
        title: str.
            Title of this plot.
    Output:
        An interactive Plotly figure of a skeleton graph.
    '''
    # need to separate X,Y,Z coords for Plotly
    x_nodes = []
    y_nodes = []
    z_nodes = []
    labels = []

    for nid in G.nodes:
        if len(G.nodes[nid]) != 0:
            # extract xyz coordinates of the nodes
            x_nodes.append(G.nodes[nid]['pos'][0])
            y_nodes.append(G.nodes[nid]['pos'][1])
            z_nodes.append(G.nodes[nid]['pos'][2])

            # extract slice_id corresponding to the nodes
            if 'slice_id' in G.nodes[nid].keys():
                s = G.nodes[nid]['slice_id']
                labels.append(s)
    
    # create lists that contain the starting and ending coords of each edge
    x_edges = []
    y_edges = []
    z_edges = []
    
    for edge in G.edges():
        nid1 = edge[0]
        nid2 = edge[1]
        if (len(G.nodes[nid1]) != 0) & (len(G.nodes[nid2]) != 0):
            # format: [beginning, ending, None]
            x_coor = [G.nodes[nid1]['pos'][0], G.nodes[nid2]['pos'][0], None]
            x_edges += x_coor
            y_coor = [G.nodes[nid1]['pos'][1], G.nodes[nid2]['pos'][1], None]
            y_edges += y_coor
            z_coor = [G.nodes[nid1]['pos'][2], G.nodes[nid2]['pos'][2], None]
            z_edges += z_coor
    
    #create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x_nodes,
                               y=y_nodes,
                               z=z_nodes,
                               mode='markers',
                               marker=dict(symbol='circle',
                                           size=2,
                                           color='blue'))
    # trace for nodes with attribute of slice_id
    if 'slice_id' in G.nodes[nid].keys():
        trace_nodes = go.Scatter3d(x=x_nodes,
                                   y=y_nodes,
                                   z=z_nodes,
                                   mode='markers',
                                   marker=dict(symbol='circle',
                                               size=2,
                                               color=labels,
                                               colorscale=px.colors.qualitative.Plotly),
                                   text=labels,
                                   hoverinfo='text')

    # create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                               y=y_edges,
                               z=z_edges,
                               mode='lines',
                               line=dict(color='grey', width=1),
                               hoverinfo='none')
    
    if len(pc.columns) != 0:
        x = [x for x in pc.x]
        y = [y for y in pc.y]
        z = [z for z in pc.z]
        labels_c = [s for s in pc.slice_id]

        # create a trace for cluster centres
        trace_points = go.Scatter3d(x=x,
                                     y=y,
                                     z=z,
                                     mode='markers',
                                     marker=dict(symbol='circle',
                                                 size=0.85,
                                                 color='lightgrey',
                                                 opacity=0.5),
#                                                  colorscale=px.colors.qualitative.Plotly),
                                     text=labels_c,
                                     hoverinfo='text')
    
    # set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    
    # layout for the plot
    layout = go.Layout(title=title,
                       width=800,
                       height=800,
                       showlegend=False,
                       scene=dict(xaxis=dict(axis),
                                  yaxis=dict(axis),
                                  zaxis=dict(axis)),
                       margin=dict(t=100),
                       hovermode='closest')
    
    # plot the traces in a figure
    if len(pc.columns) != 0:
        data = [trace_nodes, trace_edges, trace_points]
    else:
        data = [trace_nodes, trace_edges]
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def plot_subSkeleton(G, pc=pd.DataFrame(), s_start=0, s_end=-1):
    '''
    Plot part of skeleton graph based on given range of slice_id.
    Inputs:
        G: networkx graph
            G.nodes has attributes of 'pos' for x,y,z coordinates and 'pid' for node id.
        pc: pd.DataFrame
            Point cloud info, contains X,Y,Z coords and slice_id.
        s_start: int.
            Starting index of slice_id, included.
        s_end: int.
            Last index of slice_id, not included.
    Output:
        An interactive Plotly figure of a 3D nx graph.
    '''
    if s_start < 0:
        s_start = len(pc.slice_id.unique()) + s_start + 1
    if s_end < 0:
        s_end = len(pc.slice_id.unique()) + s_end + 2
    slices = [*range(s_start, s_end)]

    # select nodes for subgraph
    node_id = []
    for nid in G.nodes():
        if len(G.nodes[nid]) != 0:
            s = G.nodes[nid]['slice_id']
            if s in slices:
                node_id.append(nid)
    subG = G.subgraph(node_id)
    
    # plot subgraph
    if len(pc.columns) == 0:
        plot_skeleton(subG, title=f'Skeleton graph: slice [{s_start}:{s_end}]')    
    else:
        pc = pc[pc.slice_id.isin(slices)]
        plot_skeleton(subG, pc=pc, title=f'Skeleton graph: slice [{s_start}:{s_end}]') 