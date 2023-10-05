import os
import yaml
import argparse
import numpy as np
import pandas as pd
import treegraph
from treegraph import downsample
from treegraph import distance_from_base
from treegraph import calculate_voxel_length
from treegraph import build_skeleton
from treegraph import build_graph
from treegraph import attribute_centres
from treegraph import distance_from_tip
from treegraph import split_furcation
from treegraph import estimate_radius
from treegraph import taper
from treegraph import fit_cylinders
from treegraph import generate_cylinder_model
from treegraph import IO
from treegraph import common
from treegraph.third_party import ransac_cyl_fit as fc
from datetime import *


def run(data_path='/path/to/pointclouds.ply', output_path='../results/TreeID/',
        base_idx=None, min_pts=5, cluster_size=.04, tip_width=None, verbose=False,  
        base_corr=True, dbh_height=1.3, txt_file=True, save_graph=False):
    
    self = treegraph.initialise(data_path=data_path,
                                output_path=output_path,
                                base_idx=base_idx,
                                min_pts=min_pts,
                                downsample=.01,
                                cluster_size=cluster_size,
                                tip_width=tip_width,
                                verbose=verbose,  
                                base_corr=base_corr,
                                dbh_height=dbh_height,
                                txt_file=txt_file,
                                save_graph=save_graph)

    ### open a file to store result summary ###
    treeid = os.path.splitext(data_path)[0].split('/')[-1]
    dt = datetime.now()
    print(f'treeid: {treeid}\nProgramme starts at: {dt}')
    cs = f'cs{cluster_size}-'
    tip = f'tip{tip_width}'
    o_f = output_path + treeid + '-' + cs + tip
    
    if txt_file:
        print(f'Outputs are written in a txt file:\n{o_f}.txt')
        inputs = f"data_path = {data_path}\noutput_path = {output_path}\n\
base_idx = {base_idx}\nmin_pts = {min_pts}\ncluster_size = {cluster_size}\n\
tip_width = {tip_width}\nverbose = {verbose}\nbase_corr = {base_corr}\n\
dbh_height = {dbh_height}\ntxt_file = {txt_file}\nsave_graph = {save_graph}"
    
    with open(o_f+'.txt', 'w') as f:
        f.write('='*20 + 'Inputs' + '='*20 + f'\n{inputs}\n\n')
        f.write('='*20 + 'Processing' + '='*20 +'\n')
        f.write(f'treeid: {treeid}\nProgramme starts at: {dt}')


    ### downsample ###
    if self.downsample:
        self.pc, self.base_idx = downsample.run(self.pc, 
                                                self.downsample, 
                                                base_location=self.base_idx,
                                                delete=True, 
                                                verbose=self.verbose)
    else:
        self.pc = downsample.voxelise(self.pc)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write("\n\n----Downsample----")
            f.write(f"\nPoints after downsampling (vlength = {self.downsample} m): {len(np.unique(self.pc.index))}")

    self.pc = self.pc[['x','y','z','pid']]
    
    ### build distance graph and calculate shortest path distance ###
    if base_corr:
        self.pc, self.G, new_base, basal_r = distance_from_base.run(self.pc, 
                                                                    base_location=self.base_idx, 
                                                                    cluster_size=self.cluster_size,
                                                                    knn=100, 
                                                                    verbose=False, 
                                                                    base_correction=base_corr)
        maxbin = basal_r * 1.6  # determine maxbin by basal radius                                                            
    else:
        self.pc, self.G = distance_from_base.run(self.pc, 
                                                base_location=self.base_idx, 
                                                cluster_size=self.cluster_size,
                                                knn=100, verbose=False, 
                                                base_correction=base_corr)    
        maxbin = 0.35                                                                                                    
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Build initial graph----')
            f.write(f'\nInitial graph has {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.') 


    ### identify skeleton nodes ### 
    # slice input point clouds
    minbin = 0.02  # unit in meter
    self.pc, self.bins = calculate_voxel_length.run(self.pc, exponent=1, maxbin=maxbin, minbin=minbin)
    self.pc = self.pc[~self.pc.distance_from_base.isna()]
    

    # identify skeleton nodes by DBSCAN clustering
    self.centres = build_skeleton.run(self, verbose=False)

    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Identify skeleton nodes----')
            f.write(f"\nTotal bin numbers: {len(self.bins)}")
            f.write(f"\nTotal valid slice segments: {len(np.unique(self.pc.slice_id))}")
            f.write('\n\n----Refine skeleton----') 


    ### refine skeleton nodes ###
    self.pc, self.centres = split_furcation.run(self.pc.copy(), self.centres.copy()) 
    
    ### build skeleton graph ###                                                                        
    self.G_skel_sf, self.path_dist, self.path_ids = build_graph.run(self.pc, self.centres, verbose=self.verbose)
    
    ### attribute skeleton ###
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=False)
    
    if base_corr:
        # adjust the coords of the 1st slice centre to the coords of new_base_node
        if self.centres.slice_id.min() == 0:
            idx = self.centres[self.centres.slice_id == 0].index.values[0]
            self.centres.loc[idx, ('cx','cy','cz','distance_from_base')] = [new_base[0], new_base[1], new_base[2], 0]
        else:
            self.centres.loc[:, 'distance_from_base'] = self.centres.distance_from_base - self.centres.distance_from_base.min()
            self.pc.loc[:, 'distance_from_base'] = self.pc.distance_from_base - self.pc.distance_from_base.min()

  
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Rebuild furcation nodes----')
            f.write('\nAttribute of rebuilt skeleton...')  
            f.write(f"\nSlices segments: {len(np.unique(self.centres.slice_id))}")
            f.write(f"\nSkeleton points: {len(np.unique(self.centres.node_id))}")


    ### estimate branch radius ###
    # determine the z-interval for radius estimation (unit: metre)
    trunk = self.centres[self.centres.nbranch == 0]
    dfb_max = trunk.distance_from_base.max() - trunk.distance_from_base.min()
    if dfb_max <= 5: 
        dz2 = dfb_max / 5.
        if dfb_max <= 1.5:
            dz2 = .3
    else: 
        dz2 = 1.
    
    # estimate radius for individual branches
    self.centres = estimate_radius.run(self.pc, self.centres, self.path_ids, 
                                       dz1=.3, dz2=dz2, branch_list=None, plot=False)  
    # apply constrains to smoothed radius to avoid significant overestimated radius
    self.centres = taper.run(self.centres, self.path_ids, tip_radius=None, est_rad='sm_radius', 
                             branch_list=None, plot=False, verbose=False)
    self.centres.loc[:,'sm_radius'] = self.centres.m_radius
    
    # identify outliers in raw estimates based on smoothed radius
    self.centres.loc[:,'zscore'] = (self.centres.sf_radius - self.centres.sm_radius) / self.centres.sm_radius
    threshold = np.nanpercentile(self.centres.zscore, 95)
    outlier_id = self.centres[np.abs(self.centres.zscore) >= threshold].node_id.values
    
    # adjust outliers
    self.centres.loc[self.centres.node_id.isin(outlier_id), 'sf_radius'] = self.centres[self.centres.node_id.isin(outlier_id)].sm_radius
    
    # apply constrains to filtered raw radius to avoid significant overestimated radius
    self.centres = taper.run(self.centres, self.path_ids, tip_radius=tip_width, est_rad='sf_radius', 
                             branch_list=None, plot=False, verbose=False)
    
    # try point-to-node dist to adjust overesitimated twig radius
    # find node_id of last 4 cylinders of each branch
    twig_nid = []
    for nb in self.centres.nbranch.unique():
        branch = self.centres[self.centres.nbranch == nb]
        ncyls = np.sort(branch.ncyl)
        if len(ncyls) <= 4:
            nids = branch[branch.ncyl.isin(ncyls)].node_id.unique()
        else:
            nids = branch[branch.ncyl.isin(ncyls[-4:])].node_id.unique()
        twig_nid.extend(nids)
    # calculate point-to-node distance for all twig nodes
    for n in twig_nid:
        node_pts = np.array(self.pc[self.pc.node_id == n][['x','y','z']])
        node_cen = np.array(self.centres[self.centres.node_id == n][['cx','cy','cz']])
        dist = np.nanmedian(np.linalg.norm((node_pts - node_cen), axis=1))
        self.centres.loc[(self.centres.node_id == n), 'p2c_dist'] = dist
    self.centres.loc[(self.centres.m_radius > self.centres.p2c_dist), 'm_radius'] = self.centres[self.centres.m_radius > self.centres.p2c_dist].p2c_dist

    # delete branch whose sf_rad are all NAN 
    del_nid = np.array([])
    for n in self.centres.nbranch.unique():
        br = self.centres[self.centres.nbranch == n]
        if br.sf_radius.isnull().values.all():
            del_nid = np.append(del_nid, br.node_id.values)
    self.centres = self.centres.loc[~self.centres.node_id.isin(del_nid)]

    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Estimate branch radius----')
            f.write(f"\nEstimated radius:\n{self.centres.m_radius.describe()}")
    
    self.pc = self.pc[self.pc.node_id.isin(self.centres.node_id.values)]
    # re-build skeleton graph                                                                        
    self.G_skel_sf, self.path_dist, self.path_ids = build_graph.run(self.pc, self.centres, verbose=self.verbose)
    # re-attribute skeleton 
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=self.verbose)


    ### generate cylinder model ###
    generate_cylinder_model.run(self, radius_value='m_radius')

    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Generate cylinder model----')
            f.write('\nModelling complete.\n')
 
    
    ### Result Summary ###
    ## tree-level statistics
    tree = self.cyls[['length', 'vol', 'surface_area']].sum().to_dict()
    tree['H_from_clouds'] = round((self.pc.z.max() - self.pc.z.min()), 2)
    tree['H_from_qsm'] = round((self.cyls.sz.max() - self.cyls.sz.min()), 2)
    tree['N_tip'] = len(self.cyls[self.cyls.is_tip])
    tree['tip_rad_mean'] = self.cyls[self.cyls.is_tip].radius.mean()
    tree['tip_rad_std'] = self.cyls[self.cyls.is_tip].radius.std()

    if len(self.centres.loc[self.centres.is_tip]) > 1:
        tree['dist_between_tips'] = common.nn(self.centres.loc[self.centres.is_tip][['cx', 'cy', 'cz']].values, N=1).mean()
    else: tree['dist_between_tips'] = np.nan
        
    ## DBH estimation
    dbh_clouds, dbh_qsm = common.dbh_est(self, h=dbh_height, verbose=False, plot=False)
    tree['DBH_from_clouds'] = dbh_clouds
    tree['DBH_from_qsm'] = dbh_qsm

    ## trunk info
    trunk_nid = self.centres[self.centres.nbranch == 0].node_id.values
    for i in range(len(trunk_nid)-1):
        if i == 0:
            trunk = self.cyls[(self.cyls.p1 == trunk_nid[i+1]) & (self.cyls.p2 == trunk_nid[i])]
        else:
            trunk = trunk.append(self.cyls[(self.cyls.p1 == trunk_nid[i+1]) & (self.cyls.p2 == trunk_nid[i])]) 
    tree['trunk_vol'] = trunk.vol.sum()
    tree['trunk_length'] = trunk.length.sum()

    self.tree = pd.DataFrame(data=tree, index=[0])
    
    ## programme running time
    e_dt = datetime.now()
    self.time = (e_dt - dt).total_seconds()

    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n' + '='*20 + 'Statistical summary' + '='*20 )
            f.write(f"\nH from clouds: {tree['H_from_clouds']:.2f} m")
            f.write(f"\nH from qsm: {tree['H_from_qsm']:.2f} m")
            f.write(f"\nDBH from clouds: {tree['DBH_from_clouds']:.3f} m")
            f.write(f"\nDBH from qsm: {tree['DBH_from_qsm']:.3f} m")
            f.write(f"\nTot. branch len: {tree['length']:.2f} m")
            f.write(f"\nTot. volume: {tree['vol']:.4f} m³ = {tree['vol']*1e3:.1f} L")
            f.write(f"\nTot. surface area: {tree['surface_area']:.4f} m2")
            f.write(f"\nTrunk len: {tree['trunk_length']:.2f} m")
            f.write(f"\nTrunk volume: {tree['trunk_vol']:.4f} m³ = {tree['trunk_vol']*1e3:.1f} L")
            f.write(f"\nN tips: {tree['N_tip']:.0f}")
            f.write(f"\nAvg tip width: {tree['tip_rad_mean']*2:.3f} ± {tree['tip_rad_std']*2:.3f} m")
            f.write(f"\nAvg distance between tips: {tree['dist_between_tips']:.3f} m")     
            f.write(f"\nTotal internodes (furcation nodes + tip nodes): {len(np.unique(self.centres.ninternode))}")
            f.write(f"\n2-children furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 1])}" )
            f.write(f"\n3-children furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 2])}" )
            f.write(f"\n4-children + furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] >= 3])}") 
            f.write('\n' + '='*40 + '\n')
            f.write(f'\nProgramme successfully completed.')
            m, s = divmod(self.time, 60)
            h, m = divmod(m, 60)
            f.write(f'\nTotal running time: {self.time:.0f}s = {h:.0f}h:{m:02.0f}m:{s:02.0f}s\n')

    ### save results ### 
    # save cyl model into a .ply file
    fn_cyls = o_f + '.mesh.ply'
    IO.to_ply(self.cyls, fn_cyls)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Save results----')
            f.write(f'\nMesh (cylinder) model has been saved in:\n{fn_cyls}\n')

    # save skeleton nodes into a .ply file
    fn_centres = o_f + '.centres.ply'
    IO.save_centres(self.centres, fn_centres)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write(f'\nSkeleton points have been saved in:\n{fn_centres}\n')

    # save all results into a json file
    for col in ['idx', 'scalar_intensity', 'pid', 'centre_id', 'zscore']:
        if col in self.centres.columns:
            self.centres = self.centres.drop(columns=[col])
        if col in self.pc.columns:
            self.pc = self.pc.drop(columns=[col])
    self.path_ids = {float(key): [float(i) for i in value] for key, value in self.path_ids.items()}
    
    fn_json = o_f + '.json'
    IO.qsm2json(self, fn_json, name=treeid, graph=save_graph)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write(f'\nJson file:\n{fn_json}\n')
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', '-i', type=str, required=True, help='path to inputs file')
    a = parser.parse_args()

    with open(a.inputs) as f:
        args = yaml.safe_load(f)
        for key, item in args.items():
            print(f'{key}: {item}')
            
    run(data_path=args['data_path'], 
        output_path=args['output_path'],
        base_idx=args['base_idx'],
        min_pts=args['min_pts'], 
        cluster_size=args['cluster_size'], 
        tip_width=args['tip_width'], 
        verbose=args['verbose'],
        base_corr=args['base_corr'],
        dbh_height=args['dbh_height'],
        txt_file=args['txt_file'],
        save_graph=args['save_graph'])
