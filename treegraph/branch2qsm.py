import datetime
import sys
import os
import argparse

from treegraph.common import *
from treegraph.IO import *
from treegraph.downsample import *
from treegraph.build_skeleton import *
from treegraph.build_graph import *
from treegraph.attribute_centres import *
from treegraph.split_furcation import *
from treegraph.fit_cylinders import *
from treegraph.connected_cylinders import *


def run(path, base_idx=None, attribute='nbranch', radius='m_radius', 
        split_furcations=1, verbose=False):

    # read in pc
    pc = ply_io.read_ply(path)
    pc.rename(columns={x:x.replace('scalar_', '') for x in pc.columns}, inplace=True)

    if verbose: print('begin:', datetime.datetime.now())
    self = treegraph(pc, slice_interval=.2, min_pts=10, base_location=base_idx, verbose=verbose)

    # downsample branches before running treegraph
    self.pc, self.base_idx = downsample(self.pc, 
                                        self.pc.z.idxmin() if base_idx == None else base_idx, 
                                        .001, remove_noise=False)
    self.pc = self.pc[self.pc.downsample]
    del self.pc['downsample']
    self.pc.reset_index(inplace=True)

    # run treegraph
    self.pc = generate_distance_graph(self.pc, self.base_idx, downsample_cloud=.005)
    if verbose: print('generate_graph:', datetime.datetime.now())
    calculate_voxel_length(self, exponent=1, maxbin=.03, minbin=.01)
    if verbose: print('calculate_voxel_length:', datetime.datetime.now())
    skeleton(self, eps=.005)
    if verbose: print('skeleton:', datetime.datetime.now())
    self.path_distance, self.path_ids = skeleton_path(self.centres, max_dist=.1, verbose=self.verbose)
    if verbose: print('skeleton_path:', datetime.datetime.now())
    self.centres, self.branch_hierarchy = attribute_centres(self.centres, self.path_ids, 
                                                            branch_hierarchy=True)
    if verbose: print('attribute_centres:', datetime.datetime.now())
    
    # recalculate slice_id based on distance from tip and rebuild graph
    self.centres, self.pc = distance_from_tip(self, self.centres, self.pc)
    if verbose: print('skeleton:', datetime.datetime.now())
    self.path_distance, self.path_ids = skeleton_path(self.centres, max_dist=.1, verbose=self.verbose)
    if verbose: print('skeleton_path:', datetime.datetime.now())
    self.centres, self.branch_hierarchy = attribute_centres(self.centres, self.path_ids, 
                                                            branch_hierarchy=True)
    if verbose: print('attribute_centres:', datetime.datetime.now()) 

    # split furcations and reattribute
    split_furcation(self)
    if verbose: print('\tattribute_centres:', datetime.datetime.now())
    self.centres, self.branch_hierarchy = attribute_centres(self.centres, self.path_ids, 
                                                                branch_hierarchy=True)

    # delete single cylinder branches    
    idx = self.centres.loc[(self.centres.ncyl == 0) & (self.centres.is_tip)].index
    self.centres = self.centres.loc[~self.centres.index.isin(idx)]
        
    # fit cylinders
    if verbose: print('fitting cylinders..' )
    cylinder_fit(self)
    
    # smooth cylinders
    smooth_branches(self)
    
    # generate cyls
    generate_cylinders(self, radius_value=radius)
    
    # save data
    qsm2json(self, os.path.splitext(path)[0] + '.json', name=path)
    to_ply(self, os.path.splitext(path)[0] + '.cyls.ply')
        
    return self


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', '-b', type=str, required=True, help='path to branch')
    parser.add_argument('--bidx', type=int, default=None, required=False, help='index of base point')
    parser.add_argument('--attribute', '-a', default='nbranch', type=str, help='attribute')
    parser.add_argument('--radius', '-r', default='sf_radius', type=str, help='radius attribute or float')
    parser.add_argument('--splits', '-s', type=int, default=1, help='number of iterations for splitting furcations')
    parser.add_argument('--verbose', action='store_true', help='print something')
    args = parser.parse_args()
    run(args.branch, 
        base_idx=args.bidx,
        attribute=args.attribute, 
        radius=args.radius, 
        split_furcations=args.splits, 
        verbose=args.verbose)
