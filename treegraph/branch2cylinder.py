import datetime
import sys
import os

from treegraph.common import *
from treegraph.cyl2ply import pandas2ply


def run(path, base_idx=None):

    # read in pc
    pc = ply_io.read_ply(path)
    pc.rename(columns={x:x.replace('scalar_', '') for x in pc.columns}, inplace=True)

    print('begin:', datetime.datetime.now())
    self = treegraph(pc, slice_interval=.2, min_pts=10, base_location=base_idx)
    
    # downsample branches before running treegraph
    downsample(self, vlength=.001)
    self.pc = self.pc[self.pc.downsample]
    del self.pc['downsample']
    self.pc.reset_index(inplace=True)
    
    # run treegraph
    print('downsample:', datetime.datetime.now())
    generate_graph(self)
    print('generate_graph:', datetime.datetime.now())
    calculate_voxel_length(self, exponent=1, maxbin=.02, minbin=.005)
    print('calculate_voxel_length:', datetime.datetime.now())
    skeleton(self, eps=.005)
    print('skeleton:', datetime.datetime.now())
    skeleton_path(self)
    print('skeleton_path:', datetime.datetime.now())
    attribute_centres(self)
    print('attribute_centres:', datetime.datetime.now())

    # split furcations and reattribute
    split_furcation(self)
    print('split_furcation:', datetime.datetime.now())
    skeleton_path(self, counter=True)
    print('skeleton_path:', datetime.datetime.now())
    attribute_centres(self)
    print('attribute_centres:', datetime.datetime.now())
    
    split_furcation(self)
    print('split_furcation:', datetime.datetime.now())
    skeleton_path(self, counter=True)
    print('skeleton_path:', datetime.datetime.now())
    attribute_centres(self)
    print('attribute_centres:', datetime.datetime.now())

    # fit cylinders
    cylinder_fit(self)
    
    # smooth cylinders
    smooth_branches(self)
    
    return self

    # generate cylinder df
#     generate_cylinders(self, attribute='nbranch')

#    # save to ply
#    cols = ['length', 'radius', 'sx', 'sy', 'sz', 'ax', 'ay', 'az', 'nbranch']
#    pandas2ply(self.cyls[cols], 
#               'nbranch', 
#               os.path.splitext()[0] + '.cyls.ply')

    return self

if __name__ == "__main__":

    run()
