import os
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
from treegraph import graph_process
from datetime import *

def run(path, base_idx=None, attribute='nbranch', radius='m_radius', verbose=False,
        cluster_size=.02, min_pts=5, exponent=1, minbin=.02, maxbin=.25, output='../results/'):

    self = treegraph.initialise(path,
                                base_location=base_idx,
                                min_pts=min_pts,
                                downsample=.001,
                                minbin=minbin,
                                maxbin=maxbin,
                                cluster_size=cluster_size,
                                columns=['x', 'y', 'z'],
                                verbose=verbose)

    inputs = f"path = {path},\nbase_idx = {base_idx},\n\
attribute = {attribute},\nradius = {radius},\n\
verbose = {verbose},\ncluster_size = {cluster_size},\n\
minpts = {min_pts}, \nexponent = {exponent},\n\
minbin = {minbin},\nmaxbin = {maxbin},\n\
output_path = {output}"

    ### open a file to store result summary ###
    fn = os.path.splitext(path)[0].split('/')[2]
    #dt = datetime.datetime.now()
    dt = datetime.now()
    print(dt)
    sdt = dt.strftime('%Y-%m-%d_%H-%M')
    minb = f'minb_{minbin}_'
    maxb = f'maxb_{maxbin}_'
    radius = f'{radius}'
    o_f = output + fn + '_' + minb + maxb + radius
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
    
    with open(o_f+'.txt', 'a') as f:
        f.write("Points after first downsample (vlength=0.001m): {}".format(len(np.unique(self.pc.index))))


    ### build initial graph ###
    base_slice, fitted_centre, new_base_coords = distance_from_base.base_fitting(self, \
        base_slice_length=4.0, pc_path=path, output_path=output)
    
    self.pc, G, self.path_dict = distance_from_base.run(self.pc, self.base_location, \
        new_base_coords, low_slice_length=.3, cluster_size=self.cluster_size)
    
    with open(o_f+'.txt', 'a') as f:
        f.write('\n\nBuild graph...')
        f.write(f'\nInitial graph has {len(G.nodes)} nodes and {len(G.edges)} edges.') 
    ## save initial graph and centres
    # graph_process.save_graph(G, o_f+'_initial_G')
    # graph_process.save_centres_for_graph(self.pc, o_f+'_self.pc')


    ### identify skeleton and build skeleton graph ###
    self.pc, self.bins = calculate_voxel_length.run(self.pc, exponent=1, maxbin=self.maxbin, minbin=self.minbin)
    with open(o_f+'.txt', 'a') as f:
        f.write('\n\nCalculate slice segments...')
        f.write(f"\nTotal bin numbers: {len(self.bins)}")
        f.write(f"\nTotal valid slice segments: {len(np.unique(self.pc.slice_id))}")
    
    # identify skeleton nodes
    self.centres = build_skeleton.run(self, verbose=True)

    # build skeleton graph
    G_skeleton, self.path_distance, self.path_ids = build_graph.run(self.centres, verbose=self.verbose)
    with open(o_f+'.txt', 'a') as f:
        f.write('\n\nBuild skeleton graph...')
        f.write(f'\nInitial skeleton graph has {len(G_skeleton.nodes)} nodes and {len(G_skeleton.edges)} edges.') 
    
    # attribute skeleton
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)
    with open(o_f+'.txt', 'a') as f:
        f.write('\n\nInitial skeleton attributes...')  
        f.write(f"\nSlices segments: {len(np.unique(self.centres.slice_id))}")
        f.write(f"\nSkeleton points: {len(np.unique(self.centres.node_id))}")
        f.write(f"\nTip numbers: {len((np.where(self.centres.is_tip == True)[0]))}")
        f.write(f"\nBranch numbers: {len(np.unique(self.centres.nbranch))}")
        f.write(f"\n1-child furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 1])}")
        f.write(f"\n2-child furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 2])}")
        f.write(f"\n3-child furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 3])}")
        f.write(f"\nTotal internodes (furcation nodes + tip nodes): {len(np.unique(self.centres.ninternode))}")                                                                
                                                                


    ### rebuild skeleton using distance from tip ###
    self.centres, self.pc = distance_from_tip.run(self.pc, self.centres, self.bins, 
                                                  vlength=self.cluster_size, 
                                                  min_pts=self.min_pts, 
                                                  verbose=True)
    G_skeleton_reslice, self.path_distance, self.path_ids = build_graph.run(self.centres, verbose=self.verbose)
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)  
    # adjust the coords of the 1st slice centre with the coords of new_base_node
    idx = self.centres[self.centres.slice_id == 0].index.values[0]
    self.centres.loc[idx, ('cx','cy','cz','distance_from_base')] = [new_base_coords[0], new_base_coords[1], new_base_coords[2], 0]


    ### rebuild furcation nodes ###
    self.centres, self.path_ids, self.branch_hierarchy = split_furcation.run(self.pc.copy(), 
                                                                             self.centres.copy(), 
                                                                             self.path_ids.copy(), 
                                                                             self.branch_hierarchy.copy(),
                                                                             verbose=True)
    G_skeleton_splitf, self.path_distance, self.path_ids = build_graph.run(self.centres, verbose=self.verbose)
    # save skeleton graph and centres
    graph_process.save_graph(G_skeleton_splitf, o_f+'_G_skeleton')
    graph_process.save_centres_for_graph(self.centres, o_f+'_self.centres')
    
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)
    with open(o_f+'.txt', 'a') as f:
        f.write('\n\nAttribute of rebuilt skeleton...')  
        f.write(f"\nSlices segments: {len(np.unique(self.centres.slice_id))}")
        f.write(f"\nSkeleton points: {len(np.unique(self.centres.node_id))}")
        f.write(f"\nTip numbers: {len((np.where(self.centres.is_tip == True)[0]))}")
        f.write(f"\nBranch numbers: {len(np.unique(self.centres.nbranch))}")
        f.write(f"\n1-child furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 1])}")
        f.write(f"\n2-child furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 2])}")
        f.write(f"\n3-child furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 3])}")
        f.write(f"\nTotal internodes (furcation nodes + tip nodes): {len(np.unique(self.centres.ninternode))}")                                                                

    
    # delete single cylinder branches
    idx = self.centres.loc[(self.centres.ncyl == 0) & (self.centres.is_tip)].index
    self.centres = self.centres.loc[~self.centres.index.isin(idx)]
    with open(o_f+'.txt', 'a') as f:
        f.write(f'\n\nDelete single cyl branches: {len(idx)}')
    
    
    # generate cylinders and apply taper function
    self.centres = fit_cylinders.run(self.pc.copy(), self.centres.copy(), 
                                     min_pts=self.min_pts, 
                                     ransac_iterations=5,
                                     verbose=self.verbose)
    with open(o_f+'.txt', 'a') as f:
        f.write('\n\nFit cylinders...')
        f.write(f"\nsf_radius:\n{self.centres.sf_radius.describe()}")

    self.centres.loc[:, 'distance_from_base'] = self.centres.node_id.map(self.path_distance)
    self.centres = taper.run(self.centres, self.path_ids, tip_radius=.001)
    with open(o_f+'.txt', 'a') as f:
        f.write('\n\nSmooth radii based on a taper function...')
        f.write(f"\nm_radius:\n{self.centres.m_radius.describe()}")
    
    
    ### generate cylinder model ###
    generate_cylinder_model.run(self, radius_value='m_radius')
    with open(o_f+'.txt', 'a') as f:
        f.write('\n\nGenerate cylinders...')

    # del nan values
    index = self.cyls.loc[self.cyls.radius.isnull() == True].index
    with open(o_f+'.txt', 'a') as f:
        f.write(f'\n{len(index)} self.cyls.radius is nan.')
    cyls = self.cyls.drop(index)

    index = cyls.loc[cyls.ax.isnull() == True].index
    with open(o_f+'.txt', 'a') as f:
        f.write(f'\n{len(index)} self.cyls.ax is nan.')
    cyls = cyls.drop(index)
    with open(o_f+'.txt', 'a') as f:
        f.write(f'\n{len(cyls)} cylinders are valid.\n')


    ### save cyl model and skeleton nodes into files ###
    fn_cyls = o_f + '.cyls.ply'
    IO.to_ply(cyls, fn_cyls)
    with open(o_f+'.txt', 'a') as f:
        f.write(f'\n\nMesh (cylinder) model has been saved in:\n{fn_cyls}\n')

    fn_centres = o_f + '.centres.ply'
    IO.save_centres(self.centres, fn_centres)
    with open(o_f+'.txt', 'a') as f:
        f.write(f'\nSkeleton points have been saved in:\n{fn_centres}\n')

    fn_json = o_f + '.json'
    IO.qsm2json(self, fn_json)
    with open(o_f+'.txt', 'a') as f:
        f.write(f'\nJson file:\n{fn_json}\n')
 
    
    ### Result Summary ###
    # tree level attributes
    r = self.cyls.loc[np.abs(self.cyls.sz - min(self.cyls.sz) - 1.3) <= 0.2].radius
    with open(o_f+'.txt', 'a') as f:
        f.write('\n-----------------Summary-----------------')
        f.write(f'\nTree volume: {self.cyls.vol.sum():.5f} m³ = {self.cyls.vol.sum()*1e3:.2f} L')
        f.write(f'\nTree height: {round((max(self.cyls.sz) - min(self.cyls.sz)),2)} m')
        f.write(f'\nDBH: {round(np.mean(2*r),2)} m')
        f.write(f'\nBranch length: {self.cyls.length.sum():.2f} m')
        f.write(f'\nBranches numbers: {len(self.cyls.nbranch.unique())}')
        f.write(f'\nFitted cylinder numbers: {len(cyls)}')
        
    # more attributes
    with open(o_f+'.txt', 'a') as f:
        f.write("\n\n-----More attributes-----")
        f.write(f"\nSlices segments: {len(np.unique(self.centres.slice_id))}")
        f.write(f"\nSkeleton points: {len(np.unique(self.centres.node_id))}")
        f.write(f"\nmean number of skeleton nodes in a branch: {int(round(self.centres.ncyl.mean()))}")
        f.write(f"\n1-child furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 1])}")
        f.write(f"\n2-child furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 2])}")
        f.write(f"\n3-child furcation nodes: {len(self.centres.loc[self.centres['n_furcation'] == 3])}")
        f.write(f"\nTotal internodes (furcation nodes + tip nodes): {len(np.unique(self.centres.ninternode))}")      

    e_dt = datetime.now()
    t = (e_dt - dt)
    with open(o_f+'.txt', 'a') as f:
        f.write(f'\n\nProgramme successfully completed.\nTotal running time: {t}')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', '-b', type=str, required=True, help='path to branch')
    parser.add_argument('--bidx', type=int, default=None, required=False, help='index of base point')
    parser.add_argument('--attribute', '-a', default='nbranch', type=str, help='attribute')
    parser.add_argument('--radius', '-r', default='sf_radius', type=str, help='radius attribute or float')
    parser.add_argument('--verbose', action='store_true', help='print something')
    
    parser.add_argument('--vlength', '-vl', default=0.02, type=float, help='voxel length for downsampling before generating initial graph')
    parser.add_argument('--minpts', '-mp', default=5, type=int, help='minimum number of points to pass the filtering')
    parser.add_argument('--exponent', '-e', default=1, type=float, help='the base of the exponential function')
    parser.add_argument('--minbin', '-minbin', default=0.02, type=float, help='minmum length of a bin to segment point cloud')
    parser.add_argument('--maxbin', '-maxbin', default=0.25, type=float, help='maxmum length of a bin to segment point cloud')
    parser.add_argument('--output', '-o', default='../results/', type=str, help='path to output files')


    args = parser.parse_args()
    run(args.branch, 
        base_idx=args.bidx,
        attribute=args.attribute, 
        radius=args.radius, 
        verbose=args.verbose,
        cluster_size=args.vlength, 
        min_pts=args.minpts, 
        exponent=args.exponent, 
        minbin=args.minbin,
        maxbin=args.maxbin,
        output=args.output)
