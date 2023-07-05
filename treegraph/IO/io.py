import pandas as pd
import json
import datetime
from treegraph.third_party import ply_io
from treegraph.common import *
from treegraph import estimate_radius
from treegraph.third_party.cyl2ply import pandas2ply

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

    
def qsm2json(self, path, name=None, graph=False):
    # ### tree-level statistics
    # tree = self.cyls[['length', 'vol', 'surface_area']].sum().to_dict()
    # tree['H_from_clouds'] = round((self.pc.z.max() - self.pc.z.min()), 2)
    # tree['H_from_qsm'] = round((self.cyls.sz.max() - self.cyls.sz.min()), 2)
    # tree['N_tip'] = len(self.cyls[self.cyls.is_tip])
    # tree['tip_rad_mean'] = self.cyls[self.cyls.is_tip].radius.mean()
    # tree['tip_rad_std'] = self.cyls[self.cyls.is_tip].radius.std()

    # if len(self.centres.loc[self.centres.is_tip]) > 1:
    #     tree['dist_between_tips'] = nn(self.centres.loc[self.centres.is_tip][['cx', 'cy', 'cz']].values, N=1).mean()
    # else: tree['dist_between_tips'] = np.nan
    
    # ## DBH from point clouds
    # # find stem furcation node
    # ncyl = self.centres[self.centres.ninternode == 0].ncyl.max()
    # stem_fur_node = self.centres[(self.centres.nbranch == 0) & 
    #                              (self.centres.ncyl == ncyl)].node_id.values[0]
    # # stem furcation height above the ground
    # stem_fur_z = self.centres[self.centres.node_id == stem_fur_node].distance_from_base.values[0]
    # # calculate radius change of the trunk
    # _, taper = estimate_radius.p2a(self.pc, self.centres, self.path_ids, 
    #                                zinterval=.2, branch_list=[0], plot=False)

    # if stem_fur_z > 1.3:
    #     dbh = np.nanmean(taper[taper.dfb.between(1.2, 1.4)].p2a_mean) * 2
    # else:  # trunk split less than 1.3m from the ground
    #     dbh = np.nanmean(taper[taper.dfb.between(stem_fur_z-0.2, stem_fur_z)].p2a_mean) * 2
    # tree['DBH_from_clouds'] = round(dbh,3)

    # ## DBH from cylinder model
    # sid = self.pc[np.abs(self.pc.z - self.pc.z.min() -1.3) <=.1].slice_id.unique()
    # nid = self.centres[self.centres['slice_id'].isin(sid)].node_id.unique()
    # r = self.cyls.loc[self.cyls['p1'].isin(nid)].radius
    # dbh = np.nanmean(2*r) 
    # tree['DBH_from_qsm'] = round(dbh,3)

    # ## trunk info
    # trunk_nid = self.centres[self.centres.nbranch == 0].node_id.values
    # for i in range(len(trunk_nid)-1):
    #     if i == 0:
    #         trunk = self.cyls[(self.cyls.p1 == trunk_nid[i+1]) & (self.cyls.p2 == trunk_nid[i])]
    #     else:
    #         trunk = trunk.append(self.cyls[(self.cyls.p1 == trunk_nid[i+1]) & (self.cyls.p2 == trunk_nid[i])]) 
    # tree['trunk_vol'] = trunk.vol.sum()
    # tree['trunk_length'] = trunk.length.sum()

    # ## stem info
    # stem_nid = self.path_ids[stem_fur_node]
    # for i in range(len(stem_nid)-1):
    #     if i == 0:
    #         stem = self.cyls[(self.cyls.p1 == stem_nid[i+1]) & (self.cyls.p2 == stem_nid[i])]
    #     else:
    #         stem = stem.append(self.cyls[(self.cyls.p1 == stem_nid[i+1]) & (self.cyls.p2 == stem_nid[i])])
    # tree['stem_vol'] = stem.vol.sum()
    # tree['stem_length'] = stem.length.sum()
    
    
    ### internode data
    self.cyls.ncyl = self.cyls.ncyl.astype(int) 
    # self.cyls.loc[:, 'surface_area'] = 2 * np.pi * self.cyls.radius * self.cyls.length #+ 2 * np.pi * self.cyls.radius**2

    internodes = pd.DataFrame(data=self.cyls.groupby('ninternode').length.sum(),
                              columns=['length', 'volume', 'ncyl', 'mean_radius', 'is_tip',
                                       'distal_radius', 'proximal_radius', 'surface_area'])

    internodes.loc[:, 'ncyl'] = self.cyls.groupby('ninternode').vol.count()
    internodes.loc[:, 'volume'] = self.cyls.groupby('ninternode').vol.sum()
    internodes.loc[:, 'surface_area'] = self.cyls.groupby('ninternode').surface_area.sum()
    internodes.loc[:, 'mean_radius'] = self.cyls.groupby('ninternode').radius.mean()
    internodes.loc[:, 'parent'] = self.centres.groupby('ninternode').pinternode.min()
    internodes.loc[:, 'is_tip'] = self.cyls.groupby('ninternode').is_tip.max().astype(bool)
    
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

    # radius before furcation ("parent" if measured by hand)
    b4fur_radius = lambda row: self.centres.loc[(self.centres.ninternode == row.ninternode) &
                                                (self.centres.ncyl == row.Last)].m_radius.item()
    internodes.loc[:, 'b4fur_radius'] = first_and_last.apply(b4fur_radius, axis=1)
    internodes.loc[internodes.is_tip, 'b4fur_radius'] = np.nan
    
    # radius after furcation ("child" if measured by hand) 
    after_fur_radius = lambda row: self.centres.loc[(self.centres.ninternode == row.ninternode) &
                                                (self.centres.ncyl == row.First)].m_radius.item()
    internodes.loc[:, 'after_fur_radius'] = first_and_last.apply(after_fur_radius, axis=1)
    
    ### node data
    nodes = self.centres[(self.centres.nbranch != 0) & 
                         (self.centres.ncyl == 0)].set_index('ninternode')[['node_id', 'parent', 'parent_node']]
    nodes.rename(columns={'node_id':'child_node', 'parent':'nbranch', 'parent_node':'node_id'}, inplace=True)
    nodes.reset_index(inplace=True)

    for ix, row in nodes.iterrows():
        if len(self.centres[self.centres.nbranch == row.nbranch]) != 0:
            tip_id = self.centres.loc[(self.centres.nbranch == row.nbranch) & 
                                    (self.centres.is_tip)].node_id.values[0]
            branch_path = np.array(self.path_ids[int(tip_id)], dtype=int)
            idx = np.where(branch_path == int(row.node_id))[0][0]
            next_node = branch_path[idx + 1]
            row = row.append(pd.Series(index=['next_node'], data=next_node))
            angle = node_angle_f(self.centres[self.centres.node_id == row.child_node][['cx', 'cy', 'cz']].values,
                                    self.centres[self.centres.node_id == row.node_id][['cx', 'cy', 'cz']].values,
                                    self.centres[self.centres.node_id == row.next_node][['cx', 'cy', 'cz']].values)[0][0]


            nodes.loc[ix, 'surface_area_b'] = self.cyls[self.cyls.p1.isin(branch_path[idx:])].surface_area.sum()
            nodes.loc[ix, 'length_b'] = self.cyls[self.cyls.p1.isin(branch_path[idx:])].length.sum()
            nodes.loc[ix, 'volums_b'] = self.cyls[self.cyls.p1.isin(branch_path[idx:])].vol.sum()
            nodes.loc[ix, 'child_branch'] = self.centres[self.centres.node_id == row.child_node].nbranch.unique()

            # for test
            nodes.loc[ix, 'angle'] = angle * 180 / np.pi

    ### input arguments
    args = {'data_path': self.data_path, 'output_path': self.output_path, 
    'base_idx': self.base_idx, 'min_pts': self.min_pts, 'cluster_size': self.cluster_size,
    'tip_width': self.tip_width, 'verbose': self.verbose, 'base_corr': self.base_corr,
    'dbh_height': self.dbh_height, 'txt_file': self.txt_file, 'save_graph': self.save_graph}

    ### processing time
    run_time = {'run_time': self.time}

    # final skeleton graph nodes and edges
    G_skel = dict(nodes=[[int(n), self.G_skel_sf.nodes[n]] for n in self.G_skel_sf.nodes()], \
                    edges=[[int(u), int(v), self.G_skel_sf.edges[u,v]] for u,v in self.G_skel_sf.edges()])

    JSON = {'name':name,
            'created':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'args':args,
            'run_time':run_time,
            # 'tree':pd.DataFrame(data=tree, index=[0]).to_json(),
            'tree':self.tree.to_json(),
            'internode':internodes.to_json(),
            'node':nodes.to_json(),
            'cyls':self.cyls.to_json(),
            'centres':self.centres.to_json(),
            'pc':self.pc.to_json(),
            'path_ids':self.path_ids,
            'G_skel':G_skel}
    
    ### if save initial graph information, json file size would be doubled
    if graph:      
        # initial graph nodes and edges
        G_init = dict(nodes=[[int(n), self.G.nodes[n]] for n in self.G.nodes()], \
                edges=[[int(u), int(v), self.G.edges[u,v]] for u,v in self.G.edges()])
 
        
        JSON = {'name':name,
                'created':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'args':args,
                'run_time':run_time,
                'tree':pd.DataFrame(data=tree, index=[0]).to_json(),
                'internode':internodes.to_json(),
                'node':nodes.to_json(),
                'cyls':self.cyls.to_json(),
                'centres':self.centres.to_json(),
                'pc':self.pc.to_json(),
                'path_ids':self.path_ids,
                'G_skel':G_skel,
                'G_init':G_init
                }
    
    with open(path, 'w') as fh: fh.write(json.dumps(JSON))

class read_json:

    def __init__ (self, 
                  path,
                  pretty_printing=False,
                  attributes=['tree', 'internode', 'node', 'cyls', 'centres', 'pc'],
                  graph=False):

        JSON = json.load(open(path))
        setattr(self, 'name', JSON['name'])
        setattr(self, 'args', JSON['args'])
        run_time = JSON['run_time']['run_time']
        setattr(self, 'run_time', run_time)
        setattr(self, 'path_ids', JSON['path_ids'])
        setattr(self, 'G_skel', JSON['G_skel'])


        if pretty_printing:
            
            tree = pd.read_json(JSON['tree'])
            
            print(f"name:\t\t{JSON['name']}")
            print(f"date:\t\t{JSON['created']}")
            print(f"H from clouds:\t{tree.loc[0]['H_from_clouds']:.2f} m")
            print(f"H from qsm:\t{tree.loc[0]['H_from_qsm']:.2f} m")
            print(f"DBH from clouds: {tree.loc[0]['DBH_from_clouds']:.3f} m")
            print(f"DBH from qsm:\t{tree.loc[0]['DBH_from_qsm']:.3f} m")
            print(f"Tot. branch len: {tree.loc[0]['length']:.2f} m")
            print(f"Tot. volume:\t{tree.loc[0]['vol']:.4f} m³ = {tree.loc[0]['vol']*1e3:.1f} L")
            print(f"Tot. surface area: {tree.loc[0]['surface_area']:.4f} m2")
            print(f"Trunk len:\t{tree.loc[0]['trunk_length']:.2f} m")
            print(f"Trunk volume:\t{tree.loc[0]['trunk_vol']:.4f} m³ = {tree.loc[0]['trunk_vol']*1e3:.1f} L")
            # print(f"Stem len:\t{tree.loc[0]['stem_length']:.2f} m")
            # print(f"Stem volume:\t{tree.loc[0]['stem_vol']:.4f} m³ = {tree.loc[0]['stem_vol']*1e3:.1f} L")
            print(f"N tips:\t\t{tree.loc[0]['N_tip']:.0f}")
            print(f"Avg tip width:\t{tree.loc[0]['tip_rad_mean']*2:.3f} ± {tree.loc[0]['tip_rad_std']*2:.3f} m")
            print(f"Avg distance between tips: {tree.loc[0]['dist_between_tips']:.3f} m")        
            m, s = divmod(run_time, 60)
            h, m = divmod(m, 60)
            print(f"Programme running time:\t{h:.0f}h:{m:02.0f}m:{s:02.0f}s")  

        for att in attributes:
            try:
                setattr(self, att, pd.read_json(JSON[att]))
            except:
                raise Exception('Field "{}" not in {}'.format(att, path))

        if graph:
            setattr(self, 'G_init', JSON['G_init'])
