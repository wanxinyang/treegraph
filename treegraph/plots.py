import math
import networkx as nx 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from treegraph import estimate_radius

def plot_3d_graph(G, pc=pd.DataFrame(), centres=pd.DataFrame(), title='Initial graph'):
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


def plot_subgraph(G, pc=pd.DataFrame(), centres=pd.DataFrame(), slice_id=None,
                  s_start=0, s_end=-1):
    '''
    Plot subgraph based on given range of slice_id.
    Inputs:
        G: networkx graph
            G.nodes has attributes of 'pos' for x,y,z coordinates and 'pid' for node id.
        pc: pd.DataFrame
            Point cloud info, contains the column 'slice_id'.
        centres: pd.DataFrame (optional)
            Skeleton node info, contains the column 'slice_id'.
        slice_id: list
            A list of index of slice_id/node_id for plotting.
        s_start: int
            Starting index of slice_id, included.
        s_end: int
            Last index of slice_id, not included.
    Output:
        An interactive Plotly figure of a 3D nx graph.
    '''
    if slice_id is None:
        if s_start < 0:
            s_start = len(pc.slice_id.unique()) + s_start + 1
        if s_end < 0:
            s_end = len(pc.slice_id.unique()) + s_end + 2
        slice_id = [*range(s_start, s_end)]
    pc = pc[pc.slice_id.isin(slice_id)]
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
        centres = centres[centres.slice_id.isin(slice_id)]
        plot_3d_graph(subG, pc=pc, centres=centres,
                      title=f'Initial graph: slice [{s_start}:{s_end}]') 


def plot_slice(pc, centres=None,
               slice_id=None,
               s_start=0, s_end=-1,
               attr='slice_id',
               w=10, h=6,
               p_size=0.01, xlim=None, 
               ylim=None, zlim=None,
               figtitle=None):
    '''
    Plot 2D plots of input point cloud (with skeleton nodes).
    Inputs:
        pc: pd.DataFrame
            point cloud info, contains columns 'slice_id'
        centres: pd.DataFrame (optional)
            skeleton nodes info, contains columns 'slice_id'
        slice_id: list
            A list of index of slice_id/node_id for plotting.
        s_start: int
            starting index of slice for plotting, included
        s_end: int
            last index of slice for plotting, not included
        attr: str
            Columne name in pc df for colouring point clouds.
            'slice_id': colour points by segment slice_id.
            'node_id': colour points by clusters.
        w: float
            width of the figure 
        h: float
            height of the figure 
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

    if slice_id is None:
        if s_start < 0:
            s_start = len(pc[attr].unique()) + s_start + 1
        if s_end < 0:
            s_end = len(pc[attr].unique()) + s_end + 2
        slice_id = [*range(s_start, s_end)]
        subt = f' {attr} [{s_start}:{s_end}]'
    else:
        subt = f' {attr} = [{slice_id[0]} : {slice_id[-1]}]'

    for s in slice_id:
        if attr == 'slice_id':
            slice_pc = pc[pc.slice_id == s]
        if attr == 'node_id':
            slice_pc = pc[pc.node_id == s]
        ax[0].scatter(slice_pc.x, slice_pc.z, s=p_size, label=f'{s}')
        ax[1].scatter(slice_pc.y, slice_pc.z, s=p_size, label=f'{s}')
        ax[0].set_xlabel('x coordinates (m)', fontsize=12)
        ax[1].set_xlabel('y coordinates (m)', fontsize=12)
        ax[0].set_ylabel('z coordinates (m)', fontsize=12)
        ax[0].axis('equal')
        ax[1].axis('equal')
        
        # plot skeleton nodes coloured by slice
        if centres is None: 
            continue
        else:
            if attr == 'slice_id':
                cnode = centres[centres.slice_id == s]
            if attr == 'node_id':
                cnode = centres[centres.node_id == s]
            ax[0].scatter(cnode.cx, cnode.cz, s=20, marker='^', label=f'{s}')
            ax[1].scatter(cnode.cy, cnode.cz, s=20, marker='^', label=f'{s}')
    
    # set fig title   
    ax[0].set_title(f'Front view ({subt})', fontsize=12) 
    ax[1].set_title(f'Side view ({subt})', fontsize=12) 
    if figtitle is not None:
        fig.suptitle(f'{figtitle}', fontsize=14)
    
    # set X and Y ticks
    if attr == 'slice_id':
        samples = pc[pc.slice_id.isin(slice_id)]
    if attr == 'node_id':
        samples = pc[pc.node_id.isin(slice_id)] 

    fig.tight_layout()


def plot_skeleton(G, pc=pd.DataFrame(), title='Skeleton graph', colour=True, attr='slice_id',
                  fig_w=800, fig_h=800):
    '''
    Plot skeleton graph (3D networkx graph) using Plotly.
    Inputs:
        G: networkx graph
            G.nodes has attributes of 'pos' for x,y,z coordinates and 'slice_id'/'node_id'.
        pc: pd.DataFrame (optional)
            Point cloud info, contains XYZ coordinates. 
        title: str
            Title of this plot.
        colour: bool (default is True)
            If True then plot point cloud coloured based on attr.
            If False then plot point cloud in grey.
        attr: str
            Columne name in pc df for colouring point clouds.
            'slice_id': colour points by segment slice_id.
            'node_id': colour points by clusters.
    Output:
        An interactive Plotly figure of a skeleton graph.
    '''
    # need to separate X,Y,Z coords for Plotly
    x_nodes = []
    y_nodes = []
    z_nodes = []
    labels = []
    slice_labels = []
    node_labels = []

    for nid in G.nodes:
        if len(G.nodes[nid]) != 0:
            # extract xyz coordinates of the nodes
            x_nodes.append(G.nodes[nid]['pos'][0])
            y_nodes.append(G.nodes[nid]['pos'][1])
            z_nodes.append(G.nodes[nid]['pos'][2])

            # extract attr corresponding to the nodes
            if attr in G.nodes[nid].keys():
                s = G.nodes[nid][attr]
                labels.append(s)
            # for test
            if 'slice_id' in G.nodes[nid].keys():
                s = G.nodes[nid]['slice_id']
                slice_labels.append(s)
            if 'node_id' in G.nodes[nid].keys():
                n = G.nodes[nid]['node_id']
                node_labels.append(n)
    
    # trace for nodes with attribute of attr
    if attr in G.nodes[nid].keys():
        trace_nodes = go.Scatter3d(x=x_nodes,
                                   y=y_nodes,
                                   z=z_nodes,
                                   mode='markers',
                                   marker=dict(symbol='circle',
                                               size=2,
                                               color=labels if colour else 'blue',
                                               colorscale=px.colors.qualitative.Plotly),
                                   text=labels,
                                   hoverinfo="text")
    else:
        trace_nodes = go.Scatter3d(x=x_nodes,
                                  y=y_nodes,
                                  z=z_nodes,
                                  mode='markers',
                                  marker=dict(symbol='circle',
                                              size=2,
                                              color='blue'))
    # create lists that contain the starting and ending coords of each edge
    x_edges = []
    y_edges = []
    z_edges = []
    weight = []
    
    
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
            length = round((list(G.edges[(nid1,nid2)].values())[0]), 3)
            weight.append(length)

    # create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                            y=y_edges,
                            z=z_edges,
                            mode='lines',
                            line=dict(color='grey', width=2),
                            text=weight,
                            # hoverinfo='none'
                            hoverinfo='text')
    
    if len(pc.columns) != 0:
        x = [x for x in pc.x]
        y = [y for y in pc.y]
        z = [z for z in pc.z]
        labels_c = [s for s in pc[attr]]

        # create a trace for cluster centres
        trace_points = go.Scatter3d(x=x,
                                    y=y,
                                    z=z,
                                    mode='markers',
                                    marker=dict(symbol='circle',
                                                size=2 if colour else 1,
                                                opacity=0.3,
                                                color=labels_c if colour else 'lightgrey',
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
                       width=fig_w,
                       height=fig_h,
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


def plot_subSkeleton(G, pc=pd.DataFrame(), slice_id=None, s_start=0, s_end=-1,  
                     colour=True, attr='slice_id', title='Skeleton graph',
                     fig_w=800, fig_h=800):
    '''
    Plot part of skeleton graph based on given range of attr.
    Inputs:
        G: networkx graph
            G.nodes has attributes of 'pos' for x,y,z coordinates and 'pid' for node id.
        pc: pd.DataFrame
            Point cloud info, contains X,Y,Z coords and attr.
        slice_id: list
            A list of index of slice_id/node_id for plotting.
        s_start: int
            Starting index of attr, included.
        s_end: int
            Last index of attr, not included.
        colour: bool (default is True)
            If True then plot point cloud coloured in attr.
            If False then plot point cloud in grey.
        attr: str
            Columne name in pc df for colouring point clouds.
            'slice_id': colour points by segment slice_id.
            'node_id': colour points by clusters.
    Output:
        An interactive Plotly figure of a 3D nx graph.
    '''
    if slice_id is None:
        if s_start < 0:
            s_start = len(pc[attr].unique()) + s_start + 1
        if s_end < 0:
            s_end = len(pc[attr].unique()) + s_end + 2
        slice_id = [*range(s_start, s_end)]
        subt = f' {attr} [{s_start}:{s_end}]'
    else:
        subt = f' {attr} = {slice_id}'

    # select nodes for subgraph
    if attr == 'node_id':
        node_id = []
        for nid in G.nodes():
            if nid in slice_id:
                node_id.append(nid)
        subG = G.subgraph(node_id)
    if attr == 'slice_id':
        node_id = []
        for nid in G.nodes():
            if len(G.nodes[nid]) != 0:
                s = G.nodes[nid][attr]
                if s in slice_id:
                    node_id.append(nid)
        subG = G.subgraph(node_id)
    
    # plot subgraph
    if len(pc.columns) == 0:
        plot_skeleton(subG, title = title+subt, colour=colour)  
    else:
        pc = pc[pc[attr].isin(slice_id)]
        plot_skeleton(subG, pc=pc, title = title+subt, colour=colour, attr=attr,
                      fig_w=fig_w, fig_h=fig_h) 


### plot radius change of individual branches ###

def plot_radius_single_plot(centres, radius='m_radius', 
                            attr='distance_from_base', 
                            errorbar=False, branch_list=None, 
                            xlim=None, ylim=None, legend=False, title=None):
    '''
    Inputs:
    - attr: str, 
            'distance_from_base': path distance from current node to base node
            'ncyl': sequence of sliced segment in a branch
    '''
    
    if branch_list is not None:
        centres = centres[centres.nbranch.isin(branch_list)]

    fig, axs = plt.subplots(1, 1, figsize=(10,4))
    ax = [axs]
    
    xmin, ymin = 0, 0
    xmax = centres[attr].max() * 1.03
    ymax = centres[radius].max() * 1.03 * 1e3
    if xlim is not None:
        xmin, xmax = xlim
    if ylim is not None:
        ymin, ymax = ylim        
    # print(f'xmin, xmax = {xmin:.0f}, {xmax:.0f}')
    # print(f'ymin, ymax = {ymin:.0f}, {ymax:.0f}')

    # loop over each branch
    for nbranch in np.unique(centres.nbranch):
        branch = centres[centres.nbranch == nbranch]
        ax[0].plot(branch[attr], branch[radius]*1e3, label=f'branch {nbranch}')
        ax[0].scatter(branch[attr], branch[radius]*1e3, s=3, c='r')
        if attr == 'distance_from_base':
            ax[0].set_xlabel('Distance from tree base (m)', fontsize=12)
            ax[0].set_xlim(xmin, xmax)
            ax[0].set_ylim(ymin, ymax)
        if attr == 'ncyl':
            ax[0].set_xlabel('Sequence of segment in a branch', fontsize=12)
            # plot furcation nodes
            # for j in branch.ninternode.unique()[:-1]:
            #     ncyl = branch[branch.ninternode == j].ncyl.values
            #     nfur = branch[branch.ncyl == ncyl[-1]].n_furcation.values[0]
            #     ax[0].axvline(x=ncyl[-1], ls="--", c="green")
                # yloc = branch.sf_radius.max() * 1e3
                # ax[0].text(ncyl[-1], yloc, f'{nfur+1}')
        if errorbar:
            if 'unc' in centres.columns:
                ax[0].errorbar(branch[attr], branch[radius]*1e3, yerr=branch.unc, ls='none')
            else:
                print('Do not have uncertainty informaiton.')

        ax[0].set_ylabel('Estimated radius (mm)', fontsize=12)
        
        if title is not None:
            ax[0].set_title(f'{title}', fontsize=14)
        if legend:
            ax[0].legend(bbox_to_anchor=(1.2,1))
            # ax[0].legend()

    fig.tight_layout()
    
    return fig


def plot_radius_combine(centres, branch_list=None, xlim=None, ylim=None):
    '''
    Plot branch radius as a function of distance from base.
    Compare raw estimates, smoothed estimates and corrected estimates.
    '''
    
    if branch_list is not None:
        centres = centres[centres.nbranch.isin(branch_list)]
    
    row = 2
    col = 1
    if 'm_radius' in centres.columns: row = 3
        
    fig, axs = plt.subplots(row, col, figsize=(12,row*4))
    ax = axs.flatten()
    
    xmin, ymin = 0, 0
    xmax = centres.distance_from_base.max() * 1.03
    ymax = centres.sf_radius.max() * 1.03 * 1e3
    if xlim is not None:
        xmin, xmax = xlim
    if ylim is not None:
        ymin, ymax = ylim        
    # print(f'xmin, xmax = {xmin:.0f}, {xmax:.0f}')
    # print(f'ymin, ymax = {ymin:.0f}, {ymax:.0f}')

    # loop over each branch
    for nbranch in np.unique(centres.nbranch):
        branch = centres[centres.nbranch == nbranch]
        ax[0].scatter(branch.distance_from_base, branch.sf_radius*1e3, s=5, c='r')
        ax[0].plot(branch.distance_from_base, branch.sf_radius*1e3, label=f'branch {nbranch}')
        ax[0].set_xlabel('distance from base (m)', fontsize=12)
        ax[0].set_ylabel('estimated radius (mm)', fontsize=12)
        ax[0].set_xlim(xmin, xmax)
        ax[0].set_ylim(ymin, ymax)
        # ax[0].set_title(f'Radius estimates from point clouds with interval of {interval} m')
        # ax[0].legend(bbox_to_anchor=(1.13,1))
        
        if 'sm_radius' in centres.columns:
            ax[1].scatter(branch.distance_from_base, branch.sm_radius*1e3, s=5, c='r')
            ax[1].plot(branch.distance_from_base, branch.sm_radius*1e3, label=f'branch {nbranch}')
            ax[1].set_xlabel('distance from base (m)', fontsize=12)
            ax[1].set_ylabel('smoothed radius (mm)', fontsize=12)
            ax[1].set_xlim(xmin, xmax)
            ax[1].set_ylim(ymin, ymax)
            # ax[1].set_title(f'Smooth radius by moving average')
            # ax[1].legend(bbox_to_anchor=(1.13,1))

        if 'sm_radius' in centres.columns:
            ax[1].scatter(branch.distance_from_base, branch.sm_radius*1e3, s=5, c='r')
            ax[1].plot(branch.distance_from_base, branch.sm_radius*1e3, label=f'branch {nbranch}')
            ax[1].set_xlabel('distance from base (m)', fontsize=12)
            ax[1].set_ylabel('smoothed radius (mm)', fontsize=12)
            ax[1].set_xlim(xmin, xmax)
            ax[1].set_ylim(ymin, ymax)
            # ax[1].set_title(f'Smooth radius by moving average')
            # ax[1].legend(bbox_to_anchor=(1.13,1))
        
        if 'm_radius' in centres.columns:
            ax[2].scatter(branch.distance_from_base, branch.m_radius*1e3, s=5, c='r')
            ax[2].plot(branch.distance_from_base, branch.m_radius*1e3, label=f'branch {nbranch}')
            ax[2].set_xlabel('distance from base (m)', fontsize=12)
            ax[2].set_ylabel('corrected radius (mm)', fontsize=12)
            ax[2].set_xlim(xmin, xmax)
            ax[2].set_ylim(ymin, ymax)

    fig.tight_layout()

    return fig


def plot_radius_separate(centres, attr='distance_from_base', branch_list=None, 
                         path2fur=False, path_ids=None, xlim=None, ylim=None):
    '''
    Plot branch radius change for individual branches.
    
    Inputs:
        centres: pd.DataFrame, 
                 with attributes of sf_radius and sm_radius
        attr: str,
              X-axis attribute, default is 'distance_from_base', 
              can also be 'nycl' for sequence of segment in a branch
        branch_list: list, 
                     if None then plot all branches in centres
        path2fur: if False then plot path starting from the branch, 
                  if True then plot path starting from the node after stem furcation node
        path_ids: dict,
                  path node list
        xlim: list, 
              xlim[0] and xlim[1] are xmin and xmax of the figure     
        ylim: list, 
              ylim[0] and ylim[1] are ymin and ymax of the figure
    '''
    
    # segments from stem furcation node to branch tip
    ncyl = centres[centres.ninternode == 0].ncyl.max()
    stem_fur_node = centres[(centres.nbranch == 0) & (centres.ncyl == ncyl)].node_id.values[0]   
    
    samples = centres
    if branch_list is not None:
        samples = centres[centres.nbranch.isin(branch_list)]
    
    # set up figure layout
    n = len(samples.nbranch.unique())
    c = 2
    r = int(math.ceil(n / c))

    fig, axs =  plt.subplots(r,c,figsize=(12, 4*r))
    axs_ = axs.flatten()
    
    xmin, ymin = 0, 0
    xmax = centres.distance_from_base.max() * 1.03
    ymax = centres.sf_radius.max() * 1.03 * 1e3
    if xlim is not None:
        xmin, xmax = xlim
    if ylim is not None:
        ymin, ymax = ylim    
        
    # loop over each branch
    for i, nbranch in enumerate(samples.nbranch.unique()):
        if path2fur:
            tip = centres[centres.nbranch==nbranch].sort_values('ncyl').node_id.values[-1]
            path = path_ids[tip]
            fur_id = path.index(stem_fur_node)
            path = path[fur_id+1:]
            branch = centres[centres.node_id.isin(path)]
        else:
            branch = centres[centres.nbranch == nbranch]
            
        # plot surface fitting radius
        axs_[i].scatter(branch[attr], branch.sf_radius*1e3, s=5, c='grey')
        axs_[i].plot(branch[attr], branch.sf_radius*1e3, c='grey', label='estimated R')

        # plot smoothed radius    
        axs_[i].scatter(branch[attr], branch.sm_radius*1e3, s=5, c='cyan')
        axs_[i].plot(branch[attr], branch.sm_radius*1e3, c='cyan', label='smoothed R')
        
        if 'm_radius' in centres.columns:
            # plot corrected radius    
            axs_[i].scatter(branch[attr], branch.m_radius*1e3, s=5, c='red')
            axs_[i].plot(branch[attr], branch.m_radius*1e3, c='red', label='corrected R')

        # plot settings
        if len(branch) <= 11:
            axs_[i].set_xticks(np.arange(0, len(branch.ncyl)+1))
        axs_[i].set_title(f'Branch No.{nbranch}', fontsize=14)
        
        if attr == 'distance_from_base':
            axs_[i].set_xlabel('Distance from base (m)', fontsize=12)  
            axs_[i].set_xlim(xmin, xmax)
            axs_[i].set_ylim(ymin, ymax)
        if attr == 'ncyl':
            axs_[i].set_xlabel('Sequence of segment in this branch', fontsize=12)  
            # plot furcation nodes
            for j in branch.ninternode.unique()[:-1]:
                ncyl = branch[branch.ninternode == j].ncyl.values
                nfur = branch[branch.ncyl == ncyl[-1]].n_furcation.values[0]
                axs_[i].axvline(x=ncyl[-1], ls="--", c="green")
                yloc = branch.sf_radius.max() * 1e3
                axs_[i].text(ncyl[-1], yloc, f'{nfur+1}')
        if (i % 2) == 0:
            axs_[i].set_ylabel('Modelled Radius (mm)', fontsize=12)
        else:
            axs_[i].legend(bbox_to_anchor=(1.05, 1)) 
            # axs_[i].legend(loc='upper right')   

    fig.tight_layout(w_pad=1.5, h_pad=1.5)   
    
    return fig


### plot point clouds and skeleton nodes of specified branches  ### 

def plot_branches(pc, centres,  branch_list=[*range(0,10)], 
                  title='10 longest branches', point_size=.01, markerscale=50):
    '''
    Plot point clouds of specified branches given in branch_list.

    Inputs:
        - pc: pd.DataFrame, point clouds attributes
        - centres: pd.DataFrame, skeleton node attributes
        - branch_list: list, branch_id 
        - title: str, figure title
    
    Output:
        Return two subplots, front view (X-Z) and side view (Y-Z).
        Each colour represents an individual branch.
    '''
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs = axs.flatten()
    bs = np.sort(centres.nbranch.isin(branch_list))
    fig.suptitle(f'{title}', fontsize=14)
    
    for nbranch in branch_list:
        nodes1 = centres[centres.nbranch == nbranch].node_id
        branch_pc = pc.loc[pc.node_id.isin(nodes1)]
        
        axs[0].scatter(branch_pc.x, branch_pc.z, s=point_size, label=f'branch {nbranch}')
        axs[0].set_xlabel('x coordinate (m)', fontsize=12)
        axs[0].set_ylabel('z coordinate (m)', fontsize=12)
        axs[0].axis('equal')

        axs[1].scatter(branch_pc.y, branch_pc.z, s=point_size, label=f'branch {nbranch}')
        axs[1].set_xlabel('y coordinate (m)', fontsize=12)
        # axs[1].set_ylabel('z coordinate (m)', fontsize=12)
        axs[1].axis('equal')
        axs[1].legend(bbox_to_anchor=(1.05,1), markerscale=markerscale)
        
    
    fig.tight_layout()
    return fig


def plot_single_branch(centres, pc, nbranch, point_size=.01,
                        x_size=4, y_size=8, 
                        xmin=None, xmax=None,
                        ymin=None, ymax=None):
    '''
    Plot point cloud and skeleton nodes of a specific branch.
    
    Inputs:
        - centres: dataframe
        - pc: dataframe
        - nbranch: index of the branch that like to plot
        - x_xize, y_size: the size of the figure
        - xmin, xmax: the range of x coordinates in the plot
        - ymin, ymax: the range of y coordinates in the plot
    
    Output:
        A single figure of a specific branch. Each colour represents 
        a non-bifurcation part separated by a furcation node (red hollow circle).
        Triangle points donote the skeleton nodes.
    '''
    plt.figure(figsize=(x_size, y_size))

    nbranch = nbranch
    for i in centres.ninternode[centres.nbranch == nbranch].unique():
        # node index in this sub-section
        node_ids = centres[(centres.nbranch == nbranch) & (centres.ninternode == i)].node_id
        # branch point cloud
        branch_pc = pc[pc.node_id.isin(node_ids)] 
        # skeleton node properties
        centre_node = centres[centres.node_id.isin(node_ids)] 
        # furcation node index
        fur_node_ids = centres[(centres.nbranch == nbranch) & (centres.n_furcation > 0)].node_id
        # furcation node properties
        fur_node = centres[centres.node_id.isin(fur_node_ids)]

        plt.scatter(centre_node.cx, centre_node.cz, s=10, marker='^') # skeleton node
        # plt.scatter(fur_node.cx, fur_node.cz, s=15, marker='o', color='', edgecolors='r') # furcation node
        plt.scatter(branch_pc.x, branch_pc.z, s=point_size, alpha=0.5) # original point cloud
        plt.title(f'branch {nbranch}', fontsize=14)
        plt.xlabel('x coordinate', fontsize=12)
        plt.ylabel('z coordinate', fontsize=12)
            
        # set X and Y ticks
        centres_samples1 = centres[centres.nbranch == nbranch]
        pc_samples1 = pc[pc.node_id.isin(centres_samples1.node_id.values)]   
        
        if (xmin or xmax):
            plt.xlim(xmin, xmax)
        if (ymin or ymax):
            plt.ylim(ymin, ymax)
        plt.axis('equal')


def plot_single_branch_batch(centres, pc, branch_list=[*range(10)], 
                            point_size=.01, x_size=5, y_size=10, 
                            xmin=None, xmax=None,
                            ymin=None, ymax=None):
    '''
    Plot point cloud and skeleton nodes of specific branches given in branch_list, 
    each subgraph represents a single branch.
    
    Inputs:
        - centres: dataframe
        - pc: dataframe
        - nbranch: index of the branch that like to plot
        - x_xize, y_size: the size of the figure
        - xmin, xmax: the range of x coordinates in the plot
        - ymin, ymax: the range of y coordinates in the plot
    
    Outputs:
        Figure contains subgraphs of invidual branches. Each colour represents 
        a non-bifurcation part separated by a furcation node (red hollow circle).
        Triangle points donote the skeleton nodes.
    '''
    row = int(math.ceil(len(branch_list) / 3.) )
    col = 3
    fig, axs = plt.subplots(row, col, figsize=(x_size*col, y_size*row))
    if (branch_list) == 1:
        ax = [axs]
    else:
        ax = axs.flatten()
    
    colour = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 
    
    for i, nbranch in enumerate(branch_list):
        branch = centres[centres.nbranch == nbranch]
        branch_pc = pc[pc.node_id.isin(branch.node_id.values)]
        
        ax[i].scatter(branch_pc.x, branch_pc.z, s=point_size, c=colour[i%10])
        ax[i].axis('equal')
        ax[i].set_title(f'branch {nbranch}', fontsize=14)
        ax[i].set_xlabel('x coordinate (m)', fontsize=12)
        if i % 3 == 0:
            ax[i].set_ylabel('z coordinate (m)', fontsize=12)

    fig.tight_layout(w_pad=1.5)
    
    return fig

