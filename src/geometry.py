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
from sklearn.neighbors import NearestNeighbors
from cylinder_fitting.geometry import rotation_matrix_from_axis_and_angle


def cylinder_from_spheres(sph1, sph2, rad, n_components=40):

    direction = direction_vector(sph1, sph2)

    theta = np.arccos(np.dot(direction, np.array([0, 0, 1])))
    if np.isnan(theta):
        theta = 0
    phi = np.arctan2(direction[1], direction[0])
    if np.isnan(phi):
        phi = 0

    M = np.dot(rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), phi),
               rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), theta))

    length = np.linalg.norm(sph1 - sph2)

    vertices, facets = cylinder_mesh(length, rad, n_components)
    rotated_vertices = np.dot(vertices, M.T)
    translated_vertices = rotated_vertices + sph1

    return translated_vertices, facets


def cylinder_mesh(length, radius, n_components=40):

    tv = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    tf = np.array([[0, 1, 2], [0, 2, 3]])

    vertices = []
    facets = []
    step_size = float(length) / n_components
    steps = np.arange(0, length, step_size)
    for i in steps:
        for j in steps:
            dv = (tv * step_size) + np.array([[i, j]])
            df = tf + (len(vertices) * 4)
            vertices.append(dv)
            facets.append(df)

    vv = np.asarray([j for i in vertices for j in i])
    ff = np.asarray([j for i in facets for j in i])
    X = vv[:, 0]
    ZZ = vv[:, 1]

    delta_pi = ((2 * np.pi - 0)/(np.max(X) - np.min(X)) *
                (X - np.max(X)) + (2 * np.pi))

    XX = radius * np.cos(delta_pi)
    YY = radius * np.sin(delta_pi)

    bottom_rim_x = top_rim_x = radius * np.cos(np.unique(delta_pi))
    bottom_rim_y = top_rim_y = radius * np.sin(np.unique(delta_pi))
    bottom_rim_z = np.zeros(bottom_rim_x.shape[0])
    top_rim_z = np.zeros(top_rim_x.shape[0]) + length

    bx = np.concatenate((bottom_rim_x, [0]))
    by = np.concatenate((bottom_rim_y, [0]))
    bz = np.concatenate((bottom_rim_z, [0]))
    tx = np.concatenate((top_rim_x, [0]))
    ty = np.concatenate((top_rim_y, [0]))
    tz = np.concatenate((top_rim_z, [length]))

    botf = []
    for i in range(len(bx) - 1):
        if (i + 1) < (len(bx) - 1):
            botf.append([i, i + 1, len(bx) - 1])
        else:
            botf.append([i, 0, len(bx) - 1])

    topf = []
    for i in range(len(tx) - 1):
        if (i + 1) < (len(tx) - 1):
            topf.append([i, i + 1, len(tx) - 1])
        else:
            topf.append([i, 0, len(tx) - 1])

    XX = np.concatenate((XX, bx))
    YY = np.concatenate((YY, by))
    ZZ = np.concatenate((ZZ, bz))

    botf = np.array(botf) + (np.max(ff) + 1)
    ff = np.vstack((ff, botf))

    XX = np.concatenate((XX, tx))
    YY = np.concatenate((YY, ty))
    ZZ = np.concatenate((ZZ, tz))

    topf = np.array(topf) + (np.max(ff) + 1)
    ff = np.vstack((ff, topf))

    vert = np.vstack((XX, YY, ZZ)).T
    new_vertices, new_facets = remove_duplicated_vertices(vert, ff)

    return new_vertices, new_facets


def remove_duplicated_vertices(vertices, facets):

    uf = {}
    for i, f in enumerate(facets):
        for j, fi in enumerate(f):
            if fi in uf:
                uf[fi].append([i, j])
            else:
                uf[fi] = [[i, j]]

    nbrs = NearestNeighbors(radius=0.001).fit(vertices)
    ids = nbrs.radius_neighbors(vertices, return_distance=False)

    umask = np.zeros(vertices.shape[0])
    for i in ids:
        for j in i:
            if j != np.min(i):
                fids = uf[j]
                for f in fids:
                    facets[f[0], f[1]] = np.min(i)
            else:
                umask[j] = 1

    fnew = {}
    c = 0
    for i, ui in enumerate(umask):
        if ui > 0:
            fnew[i] = c
            c += 1

    new_facets = []
    for f in facets:
        fftemp = []
        for fi in f:
            fftemp.append(fnew[fi])
        new_facets.append(fftemp)

    new_facets = np.array(new_facets)
    new_vertices = vertices[umask.astype(bool)]

    return new_vertices, new_facets


def direction_vector(p1, p2):
    return (p2 - p1) / np.linalg.norm(p2 - p1)
