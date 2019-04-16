# Copyright (c) 2019, Matheus Boni Vicari, treestruct
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2019, treestruct"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "0.11"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"


import numpy as np
from scripts import (generate_tree_struct, generate_branch_struct,
                     struct2ply)
from reporting import pdf_report


def test_full_tree():

    # dist_threshold is the maximum length allowed for a cylinder. Cylinders
    # longer than dist_threshold will be removed.
    dist_threshold = 0.5
    # min_pts is the minimum number of points around each skeleton point that
    # should be used to fit a cylinder. Local neighborhoods containing less 
    # than min_pts will be ignored in the fitting step.
    min_pts = 5
    # slice_interval sets the slicing of the wood skeleton, used only in the
    # graph building step.
    slice_interval = 0.05
    # Setting up downsampling distance, used to reduce the number of points 
    # and speed up processing.
    down_size = 0.1
    # Setting up minimum and maximum distances to use in the connected
    # component analysis (part of the skeletonization process).
    min_cc_dist = 0.03
    max_cc_dist = 0.2

    # Loads wood-only point cloud.
    wood = np.loadtxt('../data/test_tree.txt')
    wood = wood[:, :3]
    # Generate branch and cylinder data and saves as a custom "struct"
    # file (nested Python dictionaries).
    struct = generate_tree_struct('../data/test_tree.struct', wood, 
                                  slice_interval, min_pts, down_size,
                                  min_cc_dist, max_cc_dist)
    # Generates 'ply' mesh file from branches/cylinders in struct.
    struct2ply('../data/test_tree.ply', struct, dist_threshold)
    
    # Generates a report of the structural reconstruction.
    try:
        pdf_report('../data/test_tree.pdf', '../data/test_tree.txt',
                   '../data/test_tree.struct', '../data/test_tree.ply')    
    except:
        pass  # Mayavi not installed.

    return


def test_small_branch():

    # dist_threshold is the maximum length allowed for a cylinder. Cylinders
    # longer than dist_threshold will be removed.
    dist_threshold = 0.1
    # min_pts is the minimum number of points around each skeleton point that
    # should be used to fit a cylinder. Local neighborhoods containing less 
    # than min_pts will be ignored in the fitting step.
    min_pts = 5
    # slice_interval sets the slicing of the wood skeleton, used only in the
    # graph building step.
    slice_interval = 0.01
    # Setting up downsampling distance, used to reduce the number of points 
    # and speed up processing.
    down_size = 0.002
    # Setting up minimum and maximum distances to use in the connected
    # component analysis (part of the skeletonization process).
    min_cc_dist = 0.005
    max_cc_dist = 0.01

    # Loads wood-only point cloud.
    wood = np.loadtxt('../data/test_branch.txt', delimiter=',')
    wood = wood[:, :3]
    # Generate branch and cylinder data and saves as a custom "struct"
    # file (nested Python dictionaries).
    struct = generate_branch_struct('../data/test_branch.struct', wood,
                                    slice_interval, min_pts, down_size,
                                    min_cc_dist, max_cc_dist)
    # Generates 'ply' mesh file from branches/cylinders in struct.
    struct2ply('../data/test_branch.ply', struct, dist_threshold)

    # Generates a report of the structural reconstruction.
    try:
        pdf_report('../data/test_branch.pdf', '../data/test_branch.txt',
                   '../data/test_branch.struct', '../data/test_branch.ply')
    except:
        pass  # Mayavi not installed.
    
    return

