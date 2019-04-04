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
