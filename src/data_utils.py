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
import pickle


def save_struct(filename, dictionary):
    f = open(filename, "wb")
    pickle.dump(dictionary, f)
    f.close()
    return


def load_struct(filename):
    f = open(filename, "rb")
    struct = pickle.load(f)
    f.close()
    return struct


def save_ply(filename, vertex_array, facets_array, scalar_array=None):

    if scalar_array is not None:
        BASE_HEADER = ("ply\nformat ascii 1.0\nelement vertex %s\n\
property float32 x\nproperty float32 y\nproperty float32 z\n\
property float32 s\nelement face %s\nproperty list uint8 int32 \
vertex_indices\nend_header\n" % (vertex_array.shape[0], facets_array.shape[0]))

        tri_str = ''
        for i, v in enumerate(vertex_array):
            s = scalar_array[i]
            tri_str = tri_str + '%s %s %s %s\n' % (v[0], v[1], v[2], s)

        for f in facets_array:
            tri_str = tri_str + '3 %s %s %s\n' % (f[0], f[1], f[2])

        ply_str = BASE_HEADER + tri_str

    else:
        BASE_HEADER = ("ply\nformat ascii 1.0\nelement vertex %s\n\
property float32 x\nproperty float32 y\nproperty float32 z\nelement face %s\n\
property list uint8 int32 vertex_indices\nend_header\n" %
                       (vertex_array.shape[0], facets_array.shape[0]))

        tri_str = ''
        for v in vertex_array:
            tri_str = tri_str + '%s %s %s\n' % (v[0], v[1], v[2])

        for f in facets_array:
            tri_str = tri_str + '3 %s %s %s\n' % (f[0], f[1], f[2])

        ply_str = BASE_HEADER + tri_str

    with open(filename, 'w') as ply_file:
        ply_file.write(ply_str)

    return


def load_ply(filename):

    with open(filename, 'r') as f:
        ply_str = f.readlines()
        
    for p in ply_str:
        if 'element vertex' in p:
            n_vertex = int(p.split(' ')[-1])
            break
        
    for p in ply_str:
        if 'element face' in p:
            n_facets = int(p.split(' ')[-1])
            break
        
    for i, p in enumerate(ply_str):
        if 'end_header' in p:
            vertex_start = i + 1
            break
        
    vertex_end = vertex_start + n_vertex
    facets_end = vertex_end + n_facets
    vertex_str = ply_str[vertex_start:vertex_end]
    facets_str = ply_str[vertex_end:facets_end]
    
    vertex_data = np.array([i.split(' ') for i in vertex_str]).astype(float)
    facets_data = np.array([i.split(' ') for i in facets_str]).astype(int)
    
    if vertex_data.shape[1] == 4:
        vertex_coords = vertex_data[:, :3]
        vertex_ids = vertex_data[:, 3]
    else:
        vertex_coords = vertex_data[:, :3]
        vertex_ids = np.zeros(vertex_coords.shape[0], dtype=int)
    
    facets_tri = facets_data[:, 1:]
    
    return vertex_coords, vertex_ids, facets_tri
    

