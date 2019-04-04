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
__version__ = "0.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"


import numpy as np
from geometry import cylinder_from_spheres
from data_utils import (save_ply, save_struct, load_struct)
from treestruct import main


def generate_struct(filename, point_cloud, slice_interval, min_pts):
    struct_data = main(point_cloud, slice_interval, min_pts)
    save_struct(filename, struct_data)
    return


def struct2ply(filename, struct_data, dist_threshold):

    branch_data = struct_data['branches']
    cyl_data = struct_data['cylinders']

    vt = []
    ft = []
    ids = []
    for k, v in branch_data.iteritems():
        for vi in v['cylinder_ids']:
            cyl = cyl_data[vi]
            if cyl['length'] <= dist_threshold:
                vertices, facets = cylinder_from_spheres(cyl['p1'], cyl['p2'],
                                                         cyl['rad'], 10)
                vt.append(vertices)
                ft.append(facets)
                ids.append(np.full(vertices.shape[0], k))

    new_vv = np.concatenate(vt)
    new_ids = np.concatenate(ids)
    new_ft = np.array(ft[0])
    for ff in ft[1:]:
        tf = np.array(ff)
        new_ft = np.vstack((new_ft, (tf + np.max(new_ft) + 1)))

    save_ply(filename, new_vv, new_ft, scalar_array=new_ids)
    return



if __name__ == '__main__':

    # RUNNING A SIMPLE EXAMPLE.
    # dist_threshold is the maximum length allowed for a cylinder. Cylinders
    # longer than dist_threshold will be removed.
    dist_threshold = 0.5
    # min_pts is the minimum number of points around each skeleton point that
    # should be used to fit a cylinder. Local neighborhoods containing less 
    # than min_pts will be ignored in the fitting step.
    min_pts = 10
    # slice_interval sets the slicing of the wood skeleton, used only in the
    # graph building step.
    slice_interval = 0.1

    # Loads wood-only point cloud.
    wood = np.loadtxt('../data/test_data_wood.txt')
    # Generate branch and cylinder data and saves as a custom "struct"
    # file (nested Python dictionaries).
    generate_struct('../data/test.struct', wood, slice_interval, min_pts)
    # Loads struct file.
    struct = load_struct('../data/test.struct')
    # Generates 'ply' mesh file from branches/cylinders in struct.
    struct2ply('../data/test.ply', struct, dist_threshold)
    
    
