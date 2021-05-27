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
                                downsample=.02,
                                minbin=.1,
                                maxbin=.5,
                                cluster_size=.1,
                                columns=['x', 'y', 'z'],
                                verbose=True)

    # downsample to 
    if self.downsample:
        self.pc, self.base_location = downsample.run(self.pc, 
                                                     self.downsample, 
                                                     base_location=self.base_location,
                                                     delete=True, 
                                                     verbose=self.verbose)
    else:
        self.pc = downsample.voxelise(self.pc)

    # build and attribute initial skeleton
    self.pc = distance_from_base.run(self.pc, self.base_location, cluster_size=self.cluster_size)
    self.pc, self.bins = calculate_voxel_length.run(self.pc, exponent=1, maxbin=self.maxbin, minbin=self.minbin)
    self.pc, self.centres = build_skeleton.run(self.pc, eps=.05, min_pts=self.min_pts, verbose=True)
    self.path_distance, self.path_ids = build_graph.run(self.centres, max_dist=.5, verbose=self.verbose)
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)

    # rebuild using distance from tip
    self.centres, self.pc = distance_from_tip.run(self.pc, self.centres, self.bins, 
                                                  vlength=.1, 
                                                  min_pts=self.min_pts, 
                                                  verbose=True)
    self.path_distance, self.path_ids = build_graph.run(self.centres, max_dist=.5, verbose=self.verbose)
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)

    # rebuild furcation nodes
    self.centres, self.path_ids, self.branch_hierarchy = split_furcation.run(self.pc.copy(), 
                                                                             self.centres.copy(), 
                                                                             self.path_ids.copy(), 
                                                                             self.branch_hierarchy.copy(),
                                                                             verbose=True)
    self.path_distance, self.path_ids = build_graph.run(self.centres, max_dist=.5, verbose=self.verbose)
    self.centres, self.branch_hierarchy = attribute_centres.run(self.centres, self.path_ids, 
                                                                branch_hierarchy=True, verbose=True)

    # %autoreload 2
    # generate cylinders and apply taper function
    self.centres = fit_cylinders.run(self.pc.copy(), self.centres.copy(), 
                                     min_pts=self.min_pts, 
                                     ransac_iterations=20,
                                     verbose=self.verbose)
    self.centres.loc[:, 'distance_from_base'] = self.centres.node_id.map(self.path_distance)

    self.centres = taper.run(self.centres, self.path_ids, Plot=True)

    # generate cylinder model and export
    generate_cylinder_model.run(self, radius_value='m_radius')    
    IO.to_ply(self.cyls, os.path.splitext(path)[0] + '.mesh.ply', verbose=self.verbose)
    IO.qsm2json(self, os.path.splitext(path)[0] + '.json', name=os.path.splitext(path)[0])
    
    if self.verbose: print('completed:', path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', '-t', type=str, required=True, help='path to branch')
    parser.add_argument('--bidx', type=int, default=None, required=False, help='index of base point')
    parser.add_argument('--attribute', '-a', default='nbranch', type=str, help='attribute')
    parser.add_argument('--radius', '-r', default='m_radius', type=str, help='radius attribute or float, default is "m_radius" which corrects radius using a taper function')
    parser.add_argument('--verbose', action='store_true', help='print something')
    args = parser.parse_args()
    
    run(args.tree, 
        base_idx=args.bidx,
        attribute=args.attribute, 
        radius=args.radius, 
        verbose=args.verbose)
