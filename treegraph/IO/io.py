
import pandas as pd
import treegraph.third_party.ply_io as ply_io
import json
import datetime

from treegraph.third_party.cyl2ply import pandas2ply
from treegraph.common import *
from treegraph.third_party.ply_io import *

def save_centres(centres, path, verbose=False):
    
    drop = [c for c, d in zip(centres.columns, centres.dtypes) if d in ['object']]
    ply_io.write_ply(path, centres.drop(columns=drop).rename(columns={'cx':'x', 'cy':'y', 'cz':'z'}))
    if verbose: print('skeleton points saved to:', path)
    
def save_pc(pc, path, downsample=False, verbose=False):
    
    drop = [c for c, d in zip(pc.columns, pc.dtypes) if d in ['object']]
    ply_io.write_ply(path, pc.drop(columns=drop).loc[pc.downsample if downsample else pc.index])
    if verbose: print('point cloud saved to:', path)

    
def to_ply(cyls, path, attribute='nbranch', verbose=False):
    
    cols = ['length', 'radius', 'sx', 'sy', 'sz', 'ax', 'ay', 'az', attribute]
    pandas2ply(cyls[cols], attribute, path)
    if verbose: print('cylinders saved to:', path)

    
def qsm2json(self, path, name=None):
    
    whole_branch = self.cyls[['length', 'vol', 'surface_area']].sum().to_dict()
    whole_branch['N_terminal_nodes'] = len(self.cyls[self.cyls.is_tip])
    whole_branch['mean_tip_diameter'] = self.cyls[self.cyls.is_tip].radius.mean()
    whole_branch['sd_tip_diameter'] = self.cyls[self.cyls.is_tip].radius.std()
    whole_branch['N_nodes'] = len(self.centres[self.centres.n_furcation > 0])
    whole_branch['N_furcations'] = self.centres.n_furcation.sum()
    if len(self.centres.loc[self.centres.is_tip]) > 1:
        whole_branch['dist_between_tips'] = nn(self.centres.loc[self.centres.is_tip][['cx', 'cy', 'cz']].values, N=1).mean()
    else: whole_branch['dist_between_tips'] = np.nan
    whole_branch['height'] = round((max(self.cyls.sz) - min(self.cyls.sz)),2)
    r = self.cyls.loc[np.abs(self.cyls.sz - min(self.cyls.sz) - 1.3) <= 0.2].radius
    whole_branch['DBH'] = round(np.mean(2*r),2)
    whole_branch['branch_numbers'] = len(self.cyls.nbranch.unique())
    whole_branch['slice_segments'] = len(np.unique(self.centres.slice_id))
    whole_branch['skeleton_points'] = len(np.unique(self.centres.node_id))
    whole_branch['cyl_numbers'] = len(self.cyls)
    
    ### internode data
    self.cyls.ncyl = self.cyls.ncyl.astype(int) 
    self.cyls.loc[:, 'surface_area'] = 2 * np.pi * self.cyls.radius * self.cyls.length + 2 * np.pi * self.cyls.radius**2

    internodes = pd.DataFrame(data=self.cyls.groupby('ninternode').length.sum(),
                              columns=['length', 'volume', 'ncyl', 'mean_radius', 'is_tip',
                                       'distal_radius', 'proximal_radius', 'surface_area'])

    internodes.loc[:, 'ncyl'] = self.cyls.groupby('ninternode').vol.count()
    internodes.loc[:, 'volume'] = self.cyls.groupby('ninternode').vol.sum()
    internodes.loc[:, 'surface_area'] = self.cyls.groupby('ninternode').surface_area.sum()
    internodes.loc[:, 'mean_radius'] = self.cyls.groupby('ninternode').radius.mean()

    self.cyls.loc[:, 'is_tipI'] = self.cyls.is_tip.astype(int)
    internodes.loc[:, 'is_tip'] = self.cyls.groupby('ninternode').is_tipI.max().astype(bool)
    

    first_and_last = self.cyls.groupby('ninternode').ncyl.agg([min, max]).reset_index().rename(columns={'min':'First', 'max':'Last'})

    # distal radius (ends)
    distal_radius_f = lambda row: self.cyls.loc[(self.cyls.ninternode == row.ninternode) & 
                                                (self.cyls.ncyl.isin([row.First, row.Last]))].radius.mean()
    internodes.loc[:, 'distal_radius'] = first_and_last.apply(distal_radius_f, axis=1)


    # proximal radius (centre)
    centre_cyl = first_and_last[['First', 'Last']].mean(axis=1).astype(int).reset_index().rename(columns={'index':'ninternode', 0:'ncyl'})
    proximal_radius_f = lambda row: self.cyls.loc[(self.cyls.ninternode == row.ninternode) & 
                                                  (self.cyls.ncyl == row.ncyl)].radius.mean()
    internodes.loc[:, 'proximal_radius'] = centre_cyl.apply(proximal_radius_f, axis=1)
    
    ### node data
    nodes = self.centres[(self.centres.nbranch != 0) & 
                         (self.centres.ncyl == 0)].set_index('ninternode')[['node_id', 'parent', 'parent_node']]
    nodes.rename(columns={'node_id':'child_node', 'parent':'nbranch', 'parent_node':'node_id'}, inplace=True)
    nodes.reset_index(inplace=True)

    for ix, row in nodes.iterrows():

#         tip_id = self.tip_paths.loc[self.tip_paths.nbranch == row.nbranch].index[0]
        tip_id = self.centres.loc[(self.centres.nbranch == row.nbranch) & 
                          (self.centres.is_tip)].node_id.values[0]
        branch_path = np.array(self.path_ids[int(tip_id)], dtype=int)
        idx = np.where(branch_path == int(row.node_id))[0][0]
        next_node = branch_path[idx + 1]
        row = row.append(pd.Series(index=['next_node'], data=next_node))
        node_angle_f(self.centres[self.centres.node_id == row.child_node][['cx', 'cy', 'cz']].values,
                                                     self.centres[self.centres.node_id == row.node_id][['cx', 'cy', 'cz']].values,
                                                     self.centres[self.centres.node_id == row.next_node][['cx', 'cy', 'cz']].values)[0][0]


        nodes.loc[ix, 'surface_area_b'] = self.cyls[self.cyls.p1.isin(branch_path[idx:])].surface_area.sum()
        nodes.loc[ix, 'length_b'] = self.cyls[self.cyls.p1.isin(branch_path[idx:])].length.sum()
        nodes.loc[ix, 'volums_b'] = self.cyls[self.cyls.p1.isin(branch_path[idx:])].vol.sum()
        nodes.loc[ix, 'child_branch'] = self.centres[self.centres.node_id == row.child_node].nbranch.unique()

    ### input arguments
    args = {'pc_path': self.path, 'base_idx': self.base_location, 
    'attribute': self.attribute, 'radius': self.radius, 'verbose': self.verbose, 
    'cluster_size': self.cluster_size, 'minpts': self.min_pts, 'exponent': self.exponent, 
    'minbin': self.minbin, 'maxbin': self.maxbin, 'output_path': self.output_path}

    ### graph information
    G_init = dict(nodes=[[int(n), self.G.nodes[n]] for n in self.G.nodes()], \
             edges=[[int(u), int(v)] for u,v in self.G.edges()])
    G_init_cent = self.G_centres
    
    G_skel = dict(nodes=[[int(n), self.G_skeleton_splitf.nodes[n]] for n in self.G_skeleton_splitf.nodes()], \
                      edges=[[int(u), int(v)] for u,v in self.G_skeleton_splitf.edges()])
    G_skel_cent = self.G_skeleton_splitf_centres

    ### processing time
    run_time = {'run_time': self.time}

    JSON = {'name':name,
            'created':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'tree':pd.DataFrame(data=whole_branch, index=[0]).to_json(),
            'internode':internodes.to_json(),
            'node':nodes.to_json(),
            'cyls':self.cyls.to_json(),
            'centres':self.centres.to_json(),
            'args':args,
            'G_init':G_init,
            'G_init_cent':G_init_cent.to_json(),
            'G_skel':G_skel,
            'G_skel_cent':G_skel_cent.to_json(),
            'run_time':run_time}

    with open(path, 'w') as fh: fh.write(json.dumps(JSON))

        
def read_json(path, pretty_printing=False, 
              attributes=['tree', 'internode', 'node', 'cyls', 'centres',\
                          'G_init_cent', 'G_skel_cent']):
    
    branch = json.load(open(path))
    name = branch['name']
    tree, internode, node, cyls, centres, G_init_cent, G_skel_cent = [pd.read_json(branch[x]) for x in attributes]
    args = branch['args']
    G_init = branch['G_init']
    G_skel = branch['G_skel']
    run_time = branch['run_time']['run_time']
    
    if pretty_printing:
        print('name:\t\t', name)
        print('date:\t\t', branch['created'])
        print('height:\t\t', '{:.2f} m'.format(tree.loc[0]['height']))
        print('DBH:\t\t', '{:.2f} m'.format(tree.loc[0]['DBH']))
        print('volume:\t\t', '{:.4f} m3 = {:.1f} L'.format(tree.loc[0]['vol'], tree.loc[0]['vol']*1e3))
        print('area:\t\t', '{:.4f} m2'.format(tree.loc[0]['surface_area']))
        print('branch length:\t', '{:.2f} m'.format(tree.loc[0]['length']))
        print('branch number:\t', '{:.0f}'.format(tree.loc[0]['branch_numbers']))
        print('furcation+tip\n nodes:\t\t', '{:.0f}'.format(tree.loc[0]['N_nodes']))
        # print('internodes:\t', '{:.0f}'.format(tree.loc[0]['N_furcations']))
        print('furcations:\t', '{:.0f}'.format(tree.loc[0]['N_furcations']))
        print('tips:\t\t', '{:.0f}'.format(tree.loc[0]['N_terminal_nodes']))
        print('mean tip width:\t', '{:.3f} m'.format(tree.loc[0]['mean_tip_diameter']))
        print('mean distance\n between tips:\t', '{:.3f} m'.format(tree.loc[0]['dist_between_tips']))        
        print('segment slices:\t', '{:.0f}'.format(tree.loc[0]['slice_segments']))
        print('skeleton points:', '{:.0f}'.format(tree.loc[0]['skeleton_points']))
        print('cyl numbers:\t', '{:.0f}'.format(tree.loc[0]['cyl_numbers']))
        m, s = divmod(run_time, 60)
        h, m = divmod(m, 60)
        print('running time:\t', '{:.0f}s = {:.0f}h:{:02.0f}m:{:02.0f}s'.format(run_time, h, m, s))

    return name, tree, internode, node, cyls, centres, args, G_init, G_init_cent, G_skel, G_skel_cent, run_time 

