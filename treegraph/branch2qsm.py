import os
import argparse

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

def run(path, base_idx=None, attribute='nbranch', radius='m_radius', verbose=False):

    self = treegraph.initialise(path,
                                base_location=base_idx,
                                min_pts=5,
                                downsample=.001,
                                minbin=.01,
                                maxbin=.03,
                                cluster_size=.005,
                                columns=['x', 'y', 'z'],
                                verbose=True)

    # downsample 
    if self.downsample:
        self.pc, self.base_location = downsample.run(self.pc, 
                                                     self.downsample, 
                                                     base_location=self.base_location,
                                                     delete=True, 
                                                     verbose=self.verbose)
    else:
        self.pc = downsample.voxelise(self.pc)

    # build and attribute initial skeleton
    base_slice, fitted_centre, new_base_coords = distance_from_base.base_fitting(self, \
        base_slice_length=0.5, pc_path=path, output_path='../results/tree1233/')
    
    # self.pc = distance_from_base.run(self.pc, self.base_location, cluster_size=self.cluster_size)
    self.pc, G, self.path_dict = distance_from_base.run(self.pc, self.base_location, \
        new_base_coords, low_slice_length=.3, cluster_size=self.cluster_size)
    
    self.pc, self.bins = calculate_voxel_length.run(self.pc, exponent=1, maxbin=self.maxbin, minbin=self.minbin)
    # self.pc, self.centres = build_skeleton.run(self.pc, eps=None, min_pts=self.min_pts, verbose=True)
    
    self.centres = build_skeleton.run(self, verbose=True)
    
    # adjust the coords of the 1st slice centre with the coords of new_base_node
    idx = self.centres[self.centres.slice_id == 0].index.values[0]
    self.centres.loc[idx, ('cx','cy','cz','distance_from_base')] = [new_base_coords[0], new_base_coords[1], new_base_coords[2], 0]

    G_skeleton, self.path_distance, self.path_ids = build_graph.run(self.centres, verbose=self.verbose)
    
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)

    # rebuild using distance from tip
    self.centres, self.pc = distance_from_tip.run(self.pc, self.centres, self.bins, 
                                                  vlength=self.cluster_size, 
                                                  min_pts=self.min_pts, 
                                                  verbose=True)
    G_skeleton_reslice, self.path_distance, self.path_ids = build_graph.run(self.centres, verbose=self.verbose)
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)

    # rebuild furcation nodes
    self.centres, self.path_ids, self.branch_hierarchy = split_furcation.run(self.pc.copy(), 
                                                                             self.centres.copy(), 
                                                                             self.path_ids.copy(), 
                                                                             self.branch_hierarchy.copy(),
                                                                             verbose=True)
    G_skeleton_splitf, self.path_distance, self.path_ids = build_graph.run(self.centres, verbose=self.verbose)
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)

    
    # delete single cylinder branches
    idx = self.centres.loc[(self.centres.ncyl == 0) & (self.centres.is_tip)].index
    self.centres = self.centres.loc[~self.centres.index.isin(idx)]
    
    
    # generate cylinders and apply taper function
    self.centres = fit_cylinders.run(self.pc.copy(), self.centres.copy(), 
                                     min_pts=self.min_pts, 
                                     ransac_iterations=5,
                                     verbose=self.verbose)
    self.centres.loc[:, 'distance_from_base'] = self.centres.node_id.map(self.path_distance)


    self.centres = taper.run(self.centres, self.path_ids, tip_radius=.001)
    
    
    # generate cylinder model and export
    generate_cylinder_model.run(self, radius_value='m_radius')

    # del nan values
    index = self.cyls.loc[self.cyls.radius.isnull() == True].index
    cyls = self.cyls.drop(index)

    index = cyls.loc[cyls.ax.isnull() == True].index
    cyls = cyls.drop(index)

    # save cyl model and skeleton nodes into files
    IO.to_ply(cyls, os.path.splitext(os.path.split(path)[1])[0] + '.cyls.ply', verbose=True)
    IO.save_centres(self.centres, os.path.split(path)[1])[0] + '.centres.ply')
    IO.qsm2json(self, os.path.splitext(os.path.split(path)[1])[0] + '.json')
 
 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', '-b', type=str, required=True, help='path to branch')
    parser.add_argument('--bidx', type=int, default=None, required=False, help='index of base point')
    parser.add_argument('--attribute', '-a', default='nbranch', type=str, help='attribute')
    parser.add_argument('--radius', '-r', default='sf_radius', type=str, help='radius attribute or float')
    parser.add_argument('--verbose', action='store_true', help='print something')
    args = parser.parse_args()
    run(args.branch, 
        base_idx=args.bidx,
        attribute=args.attribute, 
        radius=args.radius, 
        verbose=args.verbose)
