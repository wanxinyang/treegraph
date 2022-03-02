import os
from networkx.algorithms.distance_measures import radius
import yaml
import argparse
import numpy as np

import treegraph
from treegraph import downsample
from treegraph import distance_from_base
from treegraph import calculate_voxel_length
from treegraph import build_skeleton
from treegraph import build_graph
from treegraph import attribute_centres
from treegraph import distance_from_tip
from treegraph import split_furcation
from treegraph import fit_cylinders
from treegraph import taper
from treegraph import generate_cylinder_model
from treegraph import IO
from datetime import *

def run(path, base_idx=None, attribute='nbranch', radius='m_radius', tip_width=None, verbose=False,
        cluster_size=.02, min_pts=5, exponent=1, minbin=.02, maxbin=.25, output='../results/', 
        txt_file=True, base_corr=True, save_graph=False):

    self = treegraph.initialise(path,
                                base_location=base_idx,
                                min_pts=min_pts,
                                downsample=.001,
                                exponent=exponent,
                                minbin=minbin,
                                maxbin=maxbin,
                                cluster_size=cluster_size,
                                columns=['x', 'y', 'z'],
                                verbose=verbose,
                                attribute=attribute,
                                radius=radius,
                                tip_width=tip_width,
                                output_path=output)
    self.path = path

    ### open a file to store result summary ###
    fn = os.path.splitext(path)[0].split('/')[-1]
    dt = datetime.now()
    print(dt)
    sdt = dt.strftime('%Y-%m-%d_%H-%M')
    cs = f'cs{cluster_size}-'
    e = f'e{exponent}-'
    minb = f'minb{minbin}-'
    maxb = f'maxb{maxbin}-'
    tip = f'tip{tip_width}'
    o_f = output + fn + '-' + cs + e + minb + maxb + tip
    if txt_file:
        inputs = f"path = {path}\nbase_idx = {base_idx}\nattribute = {attribute}\ntip_width = {tip_width}\n\
verbose = {verbose}\ncluster_size = {cluster_size}\nminpts = {min_pts}\nexponent = {exponent}\n\
minbin = {minbin}\nmaxbin = {maxbin}\noutput_path = {output}\ntxt_file = {txt_file}\n\
base_correction = {base_corr}\nsave_graph = {save_graph}"

        with open(o_f+'.txt', 'w') as f:
            f.write(f'************Inputs************\n{inputs}\n')
            f.write('\n************outputs************\n')


    ### downsample ###
    if self.downsample:
        self.pc, self.base_location = downsample.run(self.pc, 
                                                     self.downsample, 
                                                     base_location=self.base_location,
                                                     delete=True, 
                                                     verbose=self.verbose)
    else:
        self.pc = downsample.voxelise(self.pc)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write("----Downsample----")
            f.write("\nPoints after first downsample (vlength=0.001m): {}".format(len(np.unique(self.pc.index))))


    ### build initial graph ###
    self.pc, self.G, new_base = distance_from_base.run(self.pc, self.base_location, 
                                                       cluster_size=self.cluster_size,
                                                       knn=100, verbose=False, 
                                                       base_correction=base_corr)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Build graph----')
            f.write(f'\nInitial graph has {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.') 


    ### identify skeleton and build skeleton graph ###
    self.pc, self.bins = calculate_voxel_length.run(self.pc, exponent=self.exponent, maxbin=self.maxbin, minbin=self.minbin)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Calculate slice segments----')
            f.write(f"\nTotal bin numbers: {len(self.bins)}")
            f.write(f"\nTotal valid slice segments: {len(np.unique(self.pc.slice_id))}")
    
    # identify skeleton nodes
    self.centres = build_skeleton.run(self, verbose=True)

    # build skeleton graph
    self.G_skeleton, self.path_distance, self.path_ids = build_graph.run(self.centres, verbose=self.verbose)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Build skeleton graph----')
            f.write(f'\nInitial skeleton graph has {len(self.G_skeleton.nodes)} nodes and {len(self.G_skeleton.edges)} edges.') 
    
    # attribute skeleton
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Attributes skeleton----')  
            f.write('\nInitial skeleton attributes...')  
            f.write(f"\nSlices segments: {len(np.unique(self.centres.slice_id))}")
            f.write(f"\nSkeleton points: {len(np.unique(self.centres.node_id))}")
            f.write(f"\nTip node numbers: {len(self.centres[self.centres.is_tip == True])}")
            f.write(f"\nTotal internodes (furcation nodes + tip nodes): {len(np.unique(self.centres.ninternode))}")
            f.write(f"\n2-children furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 1])}" )
            f.write(f"\n3-children furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 2])}" )
            f.write(f"\n4-children + furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] >= 3])}")                                                          

    ### rebuild skeleton using distance from tip ###
    self.centres, self.pc = distance_from_tip.run(self.pc, self.centres, self.bins, 
                                                  vlength=self.cluster_size, 
                                                  min_pts=self.min_pts, 
                                                  verbose=True)
    self.G_skeleton_reslice, self.path_distance, self.path_ids = build_graph.run(self.centres, verbose=self.verbose)
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)  
    if base_corr:
        # adjust the coords of the 1st slice centre to the coords of new_base_node
        idx = self.centres[self.centres.slice_id == 0].index.values[0]
        self.centres.loc[idx, ('cx','cy','cz','distance_from_base')] = [new_base[0], new_base[1], new_base[2], 0]
    
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Rebuild skeleton----') 

    ### rebuild furcation nodes ###
    self.centres, self.path_ids, self.branch_hierarchy = split_furcation.run(self.pc.copy(), 
                                                                             self.centres.copy(), 
                                                                             self.path_ids.copy(), 
                                                                             self.branch_hierarchy.copy(),
                                                                             verbose=True)
    self.G_skeleton_splitf, self.path_distance, self.path_ids = build_graph.run(self.centres, verbose=self.verbose)
     
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Rebuild furcation nodes----')
            f.write('\nAttribute of rebuilt skeleton...')  
            f.write(f"\nSlices segments: {len(np.unique(self.centres.slice_id))}")
            f.write(f"\nSkeleton points: {len(np.unique(self.centres.node_id))}")
            f.write(f"\nTip node numbers: {len(self.centres[self.centres.is_tip == True])}")
            f.write(f"\nTotal internodes (furcation nodes + tip nodes): {len(np.unique(self.centres.ninternode))}")
            f.write(f"\n2-children furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 1])}" )
            f.write(f"\n3-children furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 2])}" )
            f.write(f"\n4-children + furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] >= 3])}") 
    
    
    # delete single cylinder branches
    idx = self.centres.loc[(self.centres.ncyl == 0) & (self.centres.is_tip)].index
    self.centres = self.centres.loc[~self.centres.index.isin(idx)]
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write(f'\n\nDelete single cyl branches: {len(idx)}')
    
    
    # generate cylinders and apply taper function
    self.centres = fit_cylinders.run(self.pc.copy(), self.centres.copy(), 
                                     min_pts=self.min_pts, 
                                     ransac_iterations=5,
                                     verbose=self.verbose)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Fit cylinders----')
            f.write(f"\nsf_radius:\n{self.centres.sf_radius.describe()}")

    self.centres.loc[:, 'distance_from_base'] = self.centres.node_id.map(self.path_distance)
    # self.centres = taper.run(self.centres, self.path_ids, tip_radius=.001)
    self.centres = taper.run(self.centres, self.path_ids, tip_radius=None if tip_width is None else tip_width / 2)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Smooth radii based on a taper function----')
            f.write(f"\nm_radius:\n{self.centres.m_radius.describe()}")
    
    
    ### generate cylinder model ###
    generate_cylinder_model.run(self, radius_value=radius)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Generate cylinder model----')

    # del nan values
    index = self.cyls.loc[self.cyls.radius.isnull() == True].index
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write(f'\n{len(index)} self.cyls.radius is nan.')
    self.cyls = self.cyls.drop(index)

    index = self.cyls.loc[self.cyls.ax.isnull() == True].index
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write(f'\n{len(index)} self.cyls.ax is nan.')
    self.cyls = self.cyls.drop(index)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write(f'\n{len(self.cyls)} cylinders are valid.')


    ### save cyl model and skeleton nodes into files ###
    e_dt = datetime.now()
    self.time = (e_dt - dt).total_seconds()

    fn_cyls = o_f + '.mesh.ply'
    IO.to_ply(self.cyls, fn_cyls)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n----Save results----')
            f.write(f'\nMesh (cylinder) model has been saved in:\n{fn_cyls}\n')

    fn_centres = o_f + '.centres.ply'
    IO.save_centres(self.centres, fn_centres)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write(f'\nSkeleton points have been saved in:\n{fn_centres}\n')

    fn_json = o_f + '.json'
    IO.qsm2json(self, fn_json, name=fn, graph=save_graph)
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write(f'\nJson file:\n{fn_json}\n')
 
    
    ### Result Summary ###
    # estimate DBH from cylinder model
    slice_id = self.pc.loc[np.abs(self.pc.z - self.pc.z.min() -1.3) <=.1].slice_id.unique()
    node_id = self.centres.loc[self.centres['slice_id'].isin(slice_id)].node_id.unique()
    r = self.cyls.loc[self.cyls['p1'].isin(node_id)].radius
    dbh = np.nanmean(2*r)

    # tree level attributes
    if txt_file:
        with open(o_f+'.txt', 'a') as f:
            f.write('\n\n==== Summary: Tree Level Attributes ====')
            f.write(f'\nTree volume: {self.cyls.vol.sum():.5f} m³ = {self.cyls.vol.sum()*1e3:.2f} L')
            f.write(f'\nTree height: {round((max(self.cyls.sz) - min(self.cyls.sz)),2)} m')
            f.write(f'\nDBH: {round(dbh,2)} m')
            f.write(f'\nBranch numbers: {len(self.cyls.nbranch.unique())}')
            f.write(f'\nBranch length: {self.cyls.length.sum():.2f} m')
            f.write(f'\nBranch surface area: {self.cyls.surface_area.sum():.2f} ㎡')
            
            f.write(f"\n\nSlices segments: {len(np.unique(self.centres.slice_id))}")
            f.write(f"\nSkeleton points: {len(np.unique(self.centres.node_id))}")
            f.write(f'\nFitted cylinder numbers: {len(self.cyls)}')
            
            f.write(f"\nTip node numbers: {len(self.cyls[self.centres.is_tip == True])}")
            f.write(f"\nmean tip diameter: {self.cyls[self.cyls.is_tip].radius.mean()*100:.2f} cm")
            f.write(f"\nstd tip diameter: {self.cyls[self.cyls.is_tip].radius.std()*100:.2f} cm")
        
            f.write(f"\nTotal internodes (furcation nodes + tip nodes): {len(np.unique(self.centres.ninternode))}")
            f.write(f"\n2-children furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 1])}" )
            f.write(f"\n3-children furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 2])}" )
            f.write(f"\n4-children + furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] >= 3])}") 

        with open(o_f+'.txt', 'a') as f:
            f.write(f'\n\nProgramme successfully completed.')
            m, s = divmod(self.time, 60)
            h, m = divmod(m, 60)
            f.write(f'\nTotal running time: {self.time:.0f}s = {h:.0f}h:{m:02.0f}m:{s:02.0f}s')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', '-i', type=str, required=True, help='path to inputs file')
    a = parser.parse_args()

    with open(a.inputs) as f:
        args = yaml.safe_load(f)
        for key, item in args.items():
            print(f'{key}: {item}')
            
    run(args['data_path'], 
        base_idx=args['base_idx'],
        attribute=args['attribute'], 
        radius=args['radius'],
        tip_width=args['tip_width'], 
        verbose=args['verbose'],
        cluster_size=args['cluster_size'], 
        min_pts=args['minpts'], 
        exponent=args['exponent'], 
        minbin=args['minbin'],
        maxbin=args['maxbin'],
        output=args['output_path'],
        txt_file=args['txt_file'],
        base_corr=args['base_corr'],
        save_graph=args['save_graph'])
