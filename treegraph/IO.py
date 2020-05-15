import pandas as pd
import ply_io
import json
import datetime

from treegraph.third_party.cyl2ply import pandas2ply
from treegraph.common import *


def save_centres(self, path):
    
    ply_io.write_ply(path, self.centres.rename(columns={'cx':'x', 'cy':'y', 'cz':'z'}))

    
def save_pc(self, path, downsample=False):
    
    ply_io.write_ply(path, self.pc[self.pc.columns.drop('VX')].loc[self.pc.downsample if downsample else self.pc.index])

    
def to_ply(self, path, attribute='nbranch'):
    
    cols = ['length', 'radius', 'sx', 'sy', 'sz', 'ax', 'ay', 'az', attribute]
    pandas2ply(self.cyls[cols], attribute, path)

    
def qsm2json(self, path, name=None):
    
    whole_branch = self.cyls[['length', 'vol', 'surface_area']].sum().to_dict()
    whole_branch['N_terminal_nodes'] = len(self.cyls[self.cyls.is_tip])
    whole_branch['mean_tip_diameter'] = self.cyls[self.cyls.is_tip].radius.mean()
    whole_branch['sd_tip_diameter'] = self.cyls[self.cyls.is_tip].radius.std()
    whole_branch['N_nodes'] = len(self.centres[self.centres.n_furcation > 0])
    whole_branch['N_furcations'] = self.centres.n_furcation.sum()
    whole_branch['dist_between_tips'] = nn(self.centres.loc[self.centres.is_tip][['cx', 'cy', 'cz']].values, N=1).mean()
    
    ### internode data
    self.cyls.ncyl = self.cyls.ncyl.astype(int) 
    self.cyls.loc[:, 'surface_area'] = 2 * np.pi * self.cyls.radius * self.cyls.length + 2 * np.pi * self.cyls.radius**2

    internodes = pd.DataFrame(data=self.cyls.groupby('ninternode').length.sum(),
                              columns=['length', 'volume', 'ncyl', 'mean_radius', 
                                       'distal_radius', 'proximal_radius', 'surface_area'])

    internodes.loc[:, 'ncyl'] = self.cyls.groupby('ninternode').vol.count()
    internodes.loc[:, 'volume'] = self.cyls.groupby('ninternode').vol.sum()
    internodes.loc[:, 'surface_area'] = self.cyls.groupby('ninternode').surface_area.sum()
    internodes.loc[:, 'mean_radius'] = self.cyls.groupby('ninternode').radius.mean()

    first_and_last = self.cyls.groupby('ninternode').ncyl.agg([min, max]).reset_index().rename(columns={'min':'First', 'max':'Last'})

    # distal radius
    distal_radius_f = lambda row: self.cyls.loc[(self.cyls.ninternode == row.ninternode) & 
                                                (self.cyls.ncyl.isin([row.First, row.Last]))].radius.mean()
    internodes.loc[:, 'distal_radius'] = first_and_last.apply(distal_radius_f, axis=1)


    # proximal radius                                                                                                      
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
        
    JSON = {'name':name,
            'created':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'tree':pd.DataFrame(data=whole_branch, index=[0]).to_json(),
            'internode':internodes.to_json(),
            'node':nodes.to_json(),
            'cyls':self.cyls.to_json()}

    with open(path, 'w') as fh: fh.write(json.dumps(JSON))
        
# def pretty_printing():
    